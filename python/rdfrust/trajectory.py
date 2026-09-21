"""RDFs of MD trajectories: per frame, averaged, and animated."""
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._rdfrust import Structure, read_xyz_trajectory

Pair = Tuple[str, str]


def _duration(sec):
    """3.2 s, 4 min 05 s, 1 h 02 min."""
    if sec < 60:
        return f"{sec:.1f} s"
    if sec < 3600:
        m, s = divmod(int(round(sec)), 60)
        return f"{m} min {s:02d} s"
    h, m = divmod(int(round(sec / 60)), 60)
    return f"{h} h {m:02d} min"


class _Progress:
    """Periodic 'k / n, elapsed, time left' lines on stderr.

    Reports at every 10 % of the work, and at least every 30 s on slow runs,
    so a long job shows it is alive without flooding the log. The time left
    is extrapolated from the average time per item so far.
    """

    def __init__(self, label, total, say, every_s=30.0, quiet_below_s=2.0):
        import time

        self._time = time.perf_counter
        self.label, self.total, self.say, self.every_s = label, total, say, every_s
        # A step that finishes within a couple of seconds only prints its final
        # line; intermediate estimates would be noise.
        self.quiet_below_s = quiet_below_s
        self.t0 = self._time()
        self.last = self.t0
        self.step = max(1, -(-total // 10))      # ceil(total / 10)

    def update(self, done):
        now = self._time()
        if done < self.total and (now - self.t0 < self.quiet_below_s
                                  or (done % self.step and now - self.last < self.every_s)):
            return
        self.last = now
        elapsed = now - self.t0
        pct = 100.0 * done / self.total
        if done < self.total:
            left = elapsed / done * (self.total - done)
            self.say(f"{self.label} {done:>{len(str(self.total))}}/{self.total} "
                     f"({pct:3.0f}%)  {_duration(elapsed)} elapsed, ~{_duration(left)} left")
        else:
            self.say(f"{self.label} {done}/{self.total} (100%)  done in {_duration(elapsed)}")


@dataclass
class TrajectoryRDF:
    """Result of :func:`trajectory_prdf`.

    ``partials`` and ``total`` are the averages over the frames used;
    ``*_std`` is the frame-to-frame standard deviation, and ``*_per_frame``
    hold every frame, shaped (n_frames, n_bins).
    """

    r: np.ndarray
    partials: Dict[Pair, np.ndarray]
    total: np.ndarray
    partials_std: Dict[Pair, np.ndarray]
    total_std: np.ndarray
    partials_per_frame: Dict[Pair, np.ndarray]
    total_per_frame: np.ndarray
    frames: List[int]
    density: float
    files: Dict[str, str] = field(default_factory=dict)


def read_trajectory(path, every=1, start=0, stop=None, cell=None):
    """Read frames ``start, start + every, ...`` (before ``stop``) of an XYZ file.

    Extended XYZ frames carry their own ``Lattice="..."``, so a box that
    changes during the run is followed frame by frame. For plain XYZ, which
    has no cell, pass ``cell`` as a 3x3 matrix or as three lengths for an
    orthorhombic box.

    Returns ``(structures, frame_indices)``.
    """
    frames, indices, _ = read_xyz_trajectory(
        str(path), every=every, start=start, stop=stop, cell=_as_matrix(cell)
    )
    return frames, indices


def _as_matrix(cell):
    if cell is None:
        return None
    c = np.asarray(cell, dtype=float)
    if c.shape == (3,):
        c = np.diag(c)
    if c.shape != (3, 3):
        raise ValueError("cell must be a 3x3 matrix or three box lengths")
    return c.tolist()


def trajectory_prdf(source, cutoff=10.0, bin_size=0.1, every=1, start=0, stop=None,
                    cell=None, n_threads=0, save_plots=None, animation=True,
                    animation_format="gif", html=True, fps=8, max_animation_frames=200,
                    title=None, verbose=True):
    """Partial and total RDF of a trajectory, frame by frame and averaged.

    Parameters
    ----------
    source : str or list of Structure
        Path to a multi-frame XYZ file, or structures already in memory.
    every : int
        Use every n-th frame (``every=5``: frames 0, 5, 10, ...).
    start, stop : int
        First frame considered, and the frame to stop before.
    cell : optional
        Box for plain XYZ files without ``Lattice="..."``.
    save_plots : str, optional
        Folder for the averaged plots, the animations and the CSV files.
        None, the default, saves nothing.
    animation : bool
        Also write an animation of the frames (needs ``save_plots``).
    animation_format : "gif" or "mp4"
        GIF works with plain matplotlib; MP4 needs ffmpeg.
    html : bool
        Also write ``rdf_animation.html``: an offline player with play/pause,
        speed control and a frame slider, driving both animations together.
    max_animation_frames : int
        Longer runs are sampled evenly down to this many animation frames,
        which keeps the file small. The averages always use every frame read.
    verbose : bool
        Report progress on stderr: frames done, time elapsed, time left.

    Returns
    -------
    TrajectoryRDF
    """
    import time

    def say(msg):
        if verbose:
            print(f"[rdfrust] {msg}", file=sys.stderr, flush=True)

    t_start = time.perf_counter()
    if isinstance(source, (str, os.PathLike)):
        name = os.path.basename(os.fspath(source))
        say(f"reading {name} ...")
        structures, frames, n_in_file = read_xyz_trajectory(
            os.fspath(source), every=every, start=start, stop=stop, cell=_as_matrix(cell)
        )
        say(f"{name}: using {len(structures)} of {n_in_file} frame(s) "
            f"(every {every}, from frame {start}"
            + (f", before {stop}" if stop is not None else "")
            + f"), read in {_duration(time.perf_counter() - t_start)}")
    else:
        structures = list(source)[start:stop:every]
        frames = list(range(start, start + every * len(structures), every))
    if not structures:
        raise ValueError("no frames to analyse")

    if structures:
        n_atoms = structures[0].n_sites
        say(f"computing RDF of {len(structures)} frame(s), {n_atoms} atoms each ...")
    progress = _Progress("frames", len(structures), say)
    r = None
    total_rows, per_pair, densities = [], {}, []
    for k, s in enumerate(structures):
        rk, partials, total = s.prdf(cutoff=cutoff, bin_size=bin_size, n_threads=n_threads)
        if r is None:
            r = np.asarray(rk)
        total_rows.append(np.asarray(total))
        for pair, g in partials.items():
            per_pair.setdefault(pair, {})[k] = np.asarray(g)
        densities.append(s.n_atoms / s.volume)
        progress.update(k + 1)

    n = len(structures)
    total_per_frame = np.vstack(total_rows)
    partials_per_frame = {}
    for pair, rows in per_pair.items():
        # An element absent from a frame contributes zero to that frame.
        arr = np.zeros((n, len(r)))
        for k, g in rows.items():
            arr[k] = g
        partials_per_frame[pair] = arr

    ddof = 1 if n > 1 else 0
    result = TrajectoryRDF(
        r=r,
        partials={p: a.mean(axis=0) for p, a in partials_per_frame.items()},
        total=total_per_frame.mean(axis=0),
        partials_std={p: a.std(axis=0, ddof=ddof) for p, a in partials_per_frame.items()},
        total_std=total_per_frame.std(axis=0, ddof=ddof),
        partials_per_frame=partials_per_frame,
        total_per_frame=total_per_frame,
        frames=list(frames),
        density=float(np.mean(densities)),
    )

    if save_plots:
        from . import plotting

        t_plot = time.perf_counter()
        say("writing averaged plots and CSV files ...")
        label = f"average of {n} frame" + ("s" if n != 1 else "")
        label = f"{title} · {label}" if title else label
        result.files.update(plotting.save_prdf_plots(
            r, result.partials, result.total, save_plots, bin_size=bin_size,
            title=label, density=result.density,
            partials_std=result.partials_std if n > 1 else None,
            total_std=result.total_std if n > 1 else None,
        ))
        # The averaged CSV is rdf.csv; the per-frame totals go alongside it.
        path = os.path.join(save_plots, "total_rdf_per_frame.csv")
        x = r + bin_size / 2.0
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame"] + [f"{v:.6g}" for v in x])
            for fr, row in zip(result.frames, total_per_frame):
                w.writerow([fr] + row.tolist())
        result.files["per_frame_csv"] = path

        if animation and n > 1:
            idx = np.arange(n)
            if n > max_animation_frames:
                idx = np.unique(np.linspace(0, n - 1, max_animation_frames).astype(int))
                say(f"animation shows {len(idx)} of the {n} frames; the averages use all {n}")
            kinds = [animation_format.upper()] + (["HTML"] if html else [])
            say(f"rendering animations ({' + '.join(kinds)}) ...")
            anim_progress = _Progress("animation frames", 2 * len(idx) * (2 if html else 1), say)
            src = os.path.basename(os.fspath(source)) if isinstance(source, (str, os.PathLike)) else "structures"
            info = (f"{src} · {n} frames analysed (every {every})"
                    + (f", {len(idx)} shown" if len(idx) < n else "")
                    + f" · cutoff {cutoff} Å, bin {bin_size} Å")
            result.files.update(plotting.save_animations(
                r, [result.frames[i] for i in idx], total_per_frame[idx],
                {p: a[idx] for p, a in partials_per_frame.items()}, save_plots,
                bin_size=bin_size, total_mean=result.total,
                partials_mean=result.partials, title=title, fps=fps,
                fmt=animation_format, html=html,
                progress=lambda done, total: anim_progress.update(done),
                html_info=info,
            ))
        say(f"saved {len(result.files)} file(s) to {save_plots} "
            f"in {_duration(time.perf_counter() - t_plot)}")
        if "html" in result.files:
            say(f"open {result.files['html']} in a browser to play the animation")
    say(f"finished in {_duration(time.perf_counter() - t_start)}")
    return result
