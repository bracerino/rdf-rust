"""RDFs of MD trajectories: per frame, averaged, and animated."""
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._rdfrust import Structure, read_xyz_trajectory

Pair = Tuple[str, str]


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
                    animation_format="gif", fps=8, max_animation_frames=200,
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
    max_animation_frames : int
        Longer runs are sampled evenly down to this many animation frames,
        which keeps the file small. The averages always use every frame read.

    Returns
    -------
    TrajectoryRDF
    """
    say = (lambda msg: print(f"[rdfrust] {msg}", file=sys.stderr)) if verbose else (lambda msg: None)

    if isinstance(source, (str, os.PathLike)):
        structures, frames = read_trajectory(source, every, start, stop, cell)
        say(f"{os.fspath(source)}: using {len(structures)} frame(s) "
            f"(every {every}, from frame {start}"
            + (f", before {stop}" if stop is not None else "") + ")")
    else:
        structures = list(source)[start:stop:every]
        frames = list(range(start, start + every * len(structures), every))
    if not structures:
        raise ValueError("no frames to analyse")

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
            result.files.update(plotting.save_animations(
                r, [result.frames[i] for i in idx], total_per_frame[idx],
                {p: a[idx] for p, a in partials_per_frame.items()}, save_plots,
                bin_size=bin_size, total_mean=result.total,
                partials_mean=result.partials, title=title, fps=fps,
                fmt=animation_format,
            ))
        say(f"saved {len(result.files)} file(s) to {save_plots}")
    return result
