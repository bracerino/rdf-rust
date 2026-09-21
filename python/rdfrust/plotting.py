"""Publication-style figures of partial and total RDFs.

Called by ``Structure.prdf(save_plots=...)`` and ``trajectory_prdf(...)``.
Figures are drawn on matplotlib ``Figure`` objects directly rather than through
``pyplot``, so saving plots never switches the caller's backend or leaves
windows open.
"""
import csv
import os

import numpy as np

# Validated categorical palette, assigned in this order and never cycled: past
# eight element pairs the overlay becomes a grid of small panels instead.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BAND_ALPHA = 0.18
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 16, 14, 12, 12


def _mpl():
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "saving plots needs matplotlib, which rdf-rust normally installs; "
            "reinstall it with pip install matplotlib"
        ) from exc
    return Figure, FigureCanvasAgg


def _new_figure(width=8.0, height=5.2, nrows=1, ncols=1, sharex=False):
    Figure, Canvas = _mpl()
    fig = Figure(figsize=(width, height), facecolor="white")
    Canvas(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=sharex)
    for ax in axes.flat:
        _style(ax)
    return fig, axes


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b3")
    ax.tick_params(colors=INK, labelsize=FS_TICK)


def _label(ax, title=None, ylabel="g(r)", xlabel="r (Å)"):
    ax.set_xlabel(xlabel, color=INK, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, color=INK, fontsize=FS_LABEL)
    if title:
        ax.set_title(title, color=INK, fontsize=FS_TITLE, pad=10)


def _legend(ax, **kw):
    leg = ax.legend(fontsize=FS_LEGEND, frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")


def unique_pairs(partials):
    """(A, B) pairs with A <= B: g_BA carries the same information as g_AB."""
    elements = sorted({a for a, _ in partials} | {b for _, b in partials})
    return [(a, b) for i, a in enumerate(elements) for b in elements[i:]
            if (a, b) in partials]


def pair_colors(pairs):
    """Colour per pair, following the pair rather than its position in a plot.

    Past eight pairs a colour can no longer identify a pair, so every pair
    gets the first slot and the plots identify them by panel instead.
    """
    if len(pairs) <= len(PALETTE):
        return {p: PALETTE[i] for i, p in enumerate(pairs)}
    return {p: PALETTE[0] for p in pairs}


def _centers(r, bin_size):
    r = np.asarray(r, dtype=float)
    if bin_size is None:
        bin_size = float(r[1] - r[0]) if len(r) > 1 else 0.0
    return r + bin_size / 2.0


def _grid_shape(n):
    ncols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    return int(np.ceil(n / ncols)), ncols


def save_prdf_plots(r, partials, total, out_dir, bin_size=None, title=None,
                    density=None, partials_std=None, total_std=None,
                    fmt="png", dpi=200):
    """Write the total RDF, the partials and a CSV of the numbers.

    Parameters
    ----------
    r : array
        Left edge of each bin, as returned by ``Structure.prdf``. The plots use
        bin centres.
    partials : dict
        ``(A, B) -> g_AB(r)``.
    total : array
        Total RDF.
    out_dir : str
        Folder to write into; created if needed.
    density : float, optional
        Number density; drawn as the level the total RDF approaches.
    partials_std, total_std : optional
        Spread to draw as a shaded band (used for trajectory averages).

    Returns
    -------
    dict
        What was written: ``{"total": path, "partials": path, ...}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    x = _centers(r, bin_size)
    total = np.asarray(total, dtype=float)
    pairs = unique_pairs(partials)
    colors = pair_colors(pairs)
    tag = f" · {title}" if title else ""
    written = {}

    # Total RDF.
    fig, ax = _new_figure()
    ax = ax[0, 0]
    if total_std is not None:
        sd = np.asarray(total_std, dtype=float)
        ax.fill_between(x, np.maximum(total - sd, 0), total + sd,
                        color=PALETTE[0], alpha=BAND_ALPHA, lw=0)
    ax.plot(x, total, color=PALETTE[0], lw=1.8, label="total RDF")
    if density:
        # In the legend rather than on the plot, where it would sit on the curve.
        ax.axhline(density, color=MUTED, lw=1.0, ls="--",
                   label=f"number density ρ = {density:.4f} Å⁻³")
        _legend(ax, loc="upper right")
    ax.set_ylim(bottom=0)
    _label(ax, "Total RDF" + tag, ylabel="G(r)")
    path = os.path.join(out_dir, f"total_rdf.{fmt}")
    _save(fig, path, dpi)
    written["total"] = path

    # All pairs together: overlaid while colour can tell them apart, otherwise
    # one small panel per pair on a shared scale.
    if len(pairs) <= len(PALETTE):
        fig, ax = _new_figure(width=8.6)
        ax = ax[0, 0]
        # Lines only: several spread bands overlaid bury each other. Each pair's
        # own file below carries its band.
        for p in pairs:
            ax.plot(x, np.asarray(partials[p], dtype=float), color=colors[p], lw=1.6,
                    label=f"{p[0]}–{p[1]}")
        ax.set_ylim(bottom=0)
        _label(ax, "Partial RDFs" + tag)
        if len(pairs) > 1:
            _legend(ax, ncol=2 if len(pairs) > 4 else 1)
    else:
        nrows, ncols = _grid_shape(len(pairs))
        fig, axes = _new_figure(width=3.4 * ncols, height=2.6 * nrows + 0.6,
                                nrows=nrows, ncols=ncols, sharex=True)
        ymax = max(float(np.max(partials[p])) for p in pairs) * 1.08 or 1.0
        for k, ax in enumerate(axes.flat):
            if k >= len(pairs):
                ax.set_visible(False)
                continue
            p = pairs[k]
            ax.plot(x, partials[p], color=PALETTE[0], lw=1.3)
            ax.set_ylim(0, ymax)
            ax.set_title(f"{p[0]}–{p[1]}", color=INK, fontsize=FS_LABEL, pad=4)
            ax.tick_params(labelsize=FS_TICK - 2)
        fig.suptitle("Partial RDFs" + tag, color=INK, fontsize=FS_TITLE)
        fig.supxlabel("r (Å)", color=INK, fontsize=FS_LABEL)
        fig.supylabel("g(r)", color=INK, fontsize=FS_LABEL)
    path = os.path.join(out_dir, f"partial_rdfs.{fmt}")
    _save(fig, path, dpi)
    written["partials"] = path

    # One file per pair.
    for p in pairs:
        fig, ax = _new_figure()
        ax = ax[0, 0]
        g = np.asarray(partials[p], dtype=float)
        if partials_std is not None and p in partials_std:
            sd = np.asarray(partials_std[p], dtype=float)
            ax.fill_between(x, np.maximum(g - sd, 0), g + sd,
                            color=colors[p], alpha=BAND_ALPHA, lw=0)
        ax.plot(x, g, color=colors[p], lw=1.8)
        ax.set_ylim(bottom=0)
        _label(ax, f"Partial RDF {p[0]}–{p[1]}" + tag, ylabel=f"g$_{{{p[0]}{p[1]}}}$(r)")
        path = os.path.join(out_dir, f"partial_{p[0]}-{p[1]}.{fmt}")
        _save(fig, path, dpi)
        written[f"{p[0]}-{p[1]}"] = path

    # The numbers.
    path = os.path.join(out_dir, "rdf.csv")
    header = ["r_center_A", "total"] + [f"g_{a}-{b}" for a, b in pairs]
    cols = [x, total] + [np.asarray(partials[p]) for p in pairs]
    if total_std is not None:
        header.append("total_std")
        cols.append(np.asarray(total_std))
    if partials_std is not None:
        for a, b in pairs:
            if (a, b) in partials_std:
                header.append(f"g_{a}-{b}_std")
                cols.append(np.asarray(partials_std[(a, b)]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(np.column_stack(cols).tolist())
    written["csv"] = path
    return written




# --------------------------------------------------------------------------
# Animations
# --------------------------------------------------------------------------

def _total_figure(x, tot, frames, tag, total_mean):
    fig, ax = _new_figure()
    ax = ax[0, 0]
    if total_mean is not None:
        ax.plot(x, total_mean, color=MUTED, lw=1.2, ls="--", label="trajectory average")
    (line,) = ax.plot(x, tot[0], color=PALETTE[0], lw=1.8, label="this frame")
    ax.set_ylim(0, float(np.max(tot)) * 1.08 or 1.0)
    ax.set_xlim(x[0], x[-1])
    _label(ax, None, ylabel="G(r)")
    # Laid out with the real title text, or tight_layout leaves no room for it.
    head = ax.set_title(f"Total RDF · frame {frames[-1]}{tag}", color=INK,
                        fontsize=FS_TITLE, pad=10)
    _legend(ax, loc="upper right")
    fig.tight_layout()

    def update(k):
        line.set_ydata(tot[k])
        head.set_text(f"Total RDF · frame {frames[k]}{tag}")

    return fig, update


def _partials_figure(x, per_frame, frames, tag, partials_mean):
    pairs = unique_pairs(per_frame)
    colors = pair_colors(pairs)
    lines = {}
    if len(pairs) <= len(PALETTE):
        fig, ax = _new_figure(width=8.6)
        ax = ax[0, 0]
        for p in pairs:
            if partials_mean is not None and p in partials_mean:
                ax.plot(x, partials_mean[p], color=colors[p], lw=1.0, ls="--", alpha=0.55)
            (lines[p],) = ax.plot(x, per_frame[p][0], color=colors[p], lw=1.6,
                                  label=f"{p[0]}–{p[1]}")
        top = max(float(np.max(per_frame[p])) for p in pairs)
        ax.set_ylim(0, top * 1.08 or 1.0)
        ax.set_xlim(x[0], x[-1])
        _label(ax, None)
        head = ax.set_title(f"Partial RDFs · frame {frames[-1]}{tag}", color=INK,
                            fontsize=FS_TITLE, pad=10)
        if len(pairs) > 1:
            _legend(ax, ncol=2 if len(pairs) > 4 else 1, loc="upper right")
        fig.tight_layout()
    else:
        nrows, ncols = _grid_shape(len(pairs))
        fig, axes = _new_figure(width=3.4 * ncols, height=2.6 * nrows + 0.6,
                                nrows=nrows, ncols=ncols, sharex=True)
        top = max(float(np.max(per_frame[p])) for p in pairs) * 1.08 or 1.0
        for k, ax in enumerate(axes.flat):
            if k >= len(pairs):
                ax.set_visible(False)
                continue
            p = pairs[k]
            if partials_mean is not None and p in partials_mean:
                ax.plot(x, partials_mean[p], color=MUTED, lw=0.9, ls="--")
            (lines[p],) = ax.plot(x, per_frame[p][0], color=PALETTE[0], lw=1.3)
            ax.set_ylim(0, top)
            ax.set_title(f"{p[0]}–{p[1]}", color=INK, fontsize=FS_LABEL, pad=4)
            ax.tick_params(labelsize=FS_TICK - 2)
        head = fig.suptitle(f"Partial RDFs · frame {frames[-1]}{tag}", color=INK,
                            fontsize=FS_TITLE)
        fig.supxlabel("r (Å)", color=INK, fontsize=FS_LABEL)
        fig.supylabel("g(r)", color=INK, fontsize=FS_LABEL)
        fig.tight_layout()

    def update(k):
        for p, ln in lines.items():
            ln.set_ydata(per_frame[p][k])
        head.set_text(f"Partial RDFs · frame {frames[k]}{tag}")

    return fig, update


def _render(fig, update, n, dpi, tick=None):
    """PNG bytes of every frame. Rendered once, then reused for GIF and HTML."""
    import io

    pngs = []
    for k in range(n):
        update(k)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        pngs.append(buf.getvalue())
        if tick:
            tick()
    return pngs


def _write_gif(pngs, path, fps):
    import io

    from PIL import Image

    images = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=max(1, round(1000 / fps)), loop=0)


def save_animations(r, frames, total_per_frame, partials_per_frame, out_dir,
                    bin_size=None, total_mean=None, partials_mean=None,
                    title=None, fps=8, fmt="gif", html=True, dpi=110,
                    progress=None, html_info=None):
    """Animate the RDF frame by frame, with the trajectory average for reference.

    Writes ``total_rdf_animation`` and ``partial_rdfs_animation`` (GIF, or MP4
    with ``fmt="mp4"``, which needs ffmpeg) and, with ``html=True``,
    ``rdf_animation.html``: an offline player that steps both animations in
    step with play/pause, speed and a frame slider.

    ``total_per_frame`` is (n_frames, n_bins); ``partials_per_frame`` maps
    ``(A, B)`` to (n_frames, n_bins). ``progress(done, total)`` is called as
    frames are rendered.
    """
    os.makedirs(out_dir, exist_ok=True)
    x = _centers(r, bin_size)
    tag = f" · {title}" if title else ""
    tot = np.asarray(total_per_frame, dtype=float)
    n = len(frames)
    written = {}

    done = [0]
    # Rendering both animations, plus shrinking every frame for the HTML page.
    total_work = 2 * n * (2 if html else 1)

    def tick():
        done[0] += 1
        if progress:
            progress(done[0], total_work)

    fig_t, upd_t = _total_figure(x, tot, frames, tag, total_mean)
    fig_p, upd_p = _partials_figure(x, partials_per_frame, frames, tag, partials_mean)

    if fmt == "mp4":
        # ffmpeg encodes straight from the figures.
        from matplotlib import animation

        for key, fig, upd in (("total_animation", fig_t, upd_t),
                              ("partials_animation", fig_p, upd_p)):
            name = "total_rdf_animation" if key == "total_animation" else "partial_rdfs_animation"
            anim = animation.FuncAnimation(fig, lambda k, u=upd: u(k), frames=n, blit=False)
            path = os.path.join(out_dir, f"{name}.mp4")
            anim.save(path, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
            written[key] = path
        if not html:
            return written

    png_t = _render(fig_t, upd_t, n, dpi, tick)
    png_p = _render(fig_p, upd_p, n, dpi, tick)

    if fmt == "gif":
        path = os.path.join(out_dir, "total_rdf_animation.gif")
        _write_gif(png_t, path, fps)
        written["total_animation"] = path
        path = os.path.join(out_dir, "partial_rdfs_animation.gif")
        _write_gif(png_p, path, fps)
        written["partials_animation"] = path

    if html:
        path = os.path.join(out_dir, "rdf_animation.html")
        _write_player(path, png_p, png_t, frames, fps, title, html_info, tick)
        written["html"] = path
    return written


def _palette_png(png, colors=128):
    """Same frame as a palette PNG: line plots need few colours, and this is
    several times smaller than the full-colour original."""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("RGB").quantize(colors=colors)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _write_player(path, png_partials, png_total, frames, fps, title, info, tick=None):
    """Self-contained HTML player: no network access, no external files."""
    import base64
    import html as _html
    import json

    def uris(pngs):
        out = []
        for p in pngs:
            out.append("data:image/png;base64,"
                       + base64.b64encode(_palette_png(p)).decode("ascii"))
            if tick:
                tick()
        return out

    heading = _html.escape(f"RDF animation · {title}" if title else "RDF animation")
    sub = _html.escape(info or f"{len(frames)} frames")
    payload = json.dumps({
        "partials": uris(png_partials),
        "total": uris(png_total),
        "frames": [int(f) for f in frames],
        "fps": fps,
    })
    page = _PLAYER.replace("__HEADING__", heading).replace("__SUB__", sub) \
                  .replace("__DATA__", payload)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)


_PLAYER = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__HEADING__</title>
<style>
  :root { --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; --bg:#f7f7f5; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { padding: 20px 24px 8px; max-width: 1500px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; font-weight: 600; }
  .sub { color: var(--muted); margin: 0; }
  .bar { position: sticky; top: 0; z-index: 2; background: rgba(247,247,245,.96);
         border-bottom: 1px solid var(--line); backdrop-filter: blur(4px); }
  .controls { max-width: 1500px; margin: 0 auto; padding: 10px 24px;
              display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; }
  button { display: inline-flex; align-items: center; justify-content: center;
           width: 38px; height: 38px; border: 1px solid var(--line); border-radius: 8px;
           background: #fff; color: var(--ink); cursor: pointer; padding: 0; }
  button:hover { border-color: var(--accent); color: var(--accent); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; width: 46px; }
  button.primary:hover { filter: brightness(1.08); color: #fff; }
  button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  svg { width: 18px; height: 18px; fill: currentColor; }
  .group { display: inline-flex; gap: 6px; }
  .slider { flex: 1 1 260px; display: flex; align-items: center; gap: 10px; min-width: 200px; }
  input[type=range] { flex: 1; accent-color: var(--accent); }
  .readout { font-variant-numeric: tabular-nums; min-width: 150px; color: var(--muted); }
  .readout b { color: var(--ink); font-weight: 600; }
  label { color: var(--muted); display: inline-flex; align-items: center; gap: 6px; }
  select { height: 34px; border: 1px solid var(--line); border-radius: 8px; background: #fff;
           color: var(--ink); padding: 0 8px; font: inherit; }
  main { max-width: 1500px; margin: 0 auto; padding: 16px 24px 32px;
         display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 18px; }
  figure { margin: 0; background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 8px; }
  figure img { display: block; width: 100%; height: auto; }
  .keys { color: var(--muted); font-size: 13px; max-width: 1500px; margin: 0 auto; padding: 0 24px 24px; }
  kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px;
        padding: 0 5px; background: #fff; font-size: 12px; }
  @media (max-width: 560px) { main { grid-template-columns: 1fr; } .readout { min-width: 0; } }
</style>
</head>
<body>
<header>
  <h1>__HEADING__</h1>
  <p class="sub">__SUB__</p>
</header>
<div class="bar"><div class="controls">
  <div class="group">
    <button id="first" title="First frame" aria-label="First frame"><svg viewBox="0 0 24 24"><path d="M6 5h2v14H6zM20 5v14L9 12z"/></svg></button>
    <button id="prev" title="Previous frame (←)" aria-label="Previous frame"><svg viewBox="0 0 24 24"><path d="M15 5v14L6 12z"/></svg></button>
    <button id="play" class="primary" title="Play / pause (space)" aria-label="Play"><svg viewBox="0 0 24 24"><path id="playicon" d="M8 5v14l11-7z"/></svg></button>
    <button id="next" title="Next frame (→)" aria-label="Next frame"><svg viewBox="0 0 24 24"><path d="M9 5v14l9-7z"/></svg></button>
    <button id="last" title="Last frame" aria-label="Last frame"><svg viewBox="0 0 24 24"><path d="M16 5h2v14h-2zM4 5v14l11-7z"/></svg></button>
  </div>
  <div class="slider">
    <input id="pos" type="range" min="0" value="0" step="1" aria-label="Frame">
    <span class="readout" id="readout"></span>
  </div>
  <label>Speed
    <select id="speed" aria-label="Playback speed">
      <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
      <option value="1" selected>1×</option><option value="2">2×</option>
      <option value="4">4×</option><option value="8">8×</option>
    </select>
  </label>
  <label><input id="loop" type="checkbox" checked> Loop</label>
</div></div>
<main>
  <figure><img id="img_p" alt="Partial RDFs of the current frame"></figure>
  <figure><img id="img_t" alt="Total RDF of the current frame"></figure>
</main>
<p class="keys"><kbd>Space</kbd> play / pause &nbsp; <kbd>←</kbd> <kbd>→</kbd> step &nbsp;
  <kbd>+</kbd> <kbd>−</kbd> speed &nbsp; <kbd>Home</kbd> <kbd>End</kbd> first / last.
  Dashed lines are the average over the whole trajectory.</p>
<script>
const D = __DATA__;
const N = D.frames.length;
const imgP = document.getElementById("img_p"), imgT = document.getElementById("img_t");
const pos = document.getElementById("pos"), readout = document.getElementById("readout");
const playBtn = document.getElementById("play"), playIcon = document.getElementById("playicon");
const speed = document.getElementById("speed"), loop = document.getElementById("loop");
const PLAY = "M8 5v14l11-7z", PAUSE = "M6 5h4v14H6zM14 5h4v14h-4z";
pos.max = N - 1;
let k = 0, timer = null;

function show(i) {
  k = Math.max(0, Math.min(N - 1, i));
  imgP.src = D.partials[k]; imgT.src = D.total[k];
  pos.value = k;
  readout.innerHTML = "frame <b>" + D.frames[k] + "</b> &nbsp;(" + (k + 1) + " / " + N + ")";
}
function interval() { return 1000 / (D.fps * parseFloat(speed.value)); }
function tick() {
  if (k >= N - 1) { if (loop.checked) show(0); else { stop(); return; } }
  else show(k + 1);
}
function start() {
  if (timer) return;
  if (k >= N - 1 && !loop.checked) show(0);
  timer = setInterval(tick, interval());
  playIcon.setAttribute("d", PAUSE); playBtn.setAttribute("aria-label", "Pause");
}
function stop() {
  clearInterval(timer); timer = null;
  playIcon.setAttribute("d", PLAY); playBtn.setAttribute("aria-label", "Play");
}
function toggle() { timer ? stop() : start(); }
function restartIfPlaying() { if (timer) { stop(); start(); } }
function step(d) { stop(); show(k + d); }
function nudgeSpeed(d) {
  const i = Math.max(0, Math.min(speed.options.length - 1, speed.selectedIndex + d));
  speed.selectedIndex = i; restartIfPlaying();
}

playBtn.onclick = toggle;
document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);
document.getElementById("first").onclick = () => { stop(); show(0); };
document.getElementById("last").onclick = () => { stop(); show(N - 1); };
pos.oninput = () => { stop(); show(parseInt(pos.value, 10)); };
speed.onchange = restartIfPlaying;
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); toggle(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  else if (e.key === "+" || e.key === "=") nudgeSpeed(1);
  else if (e.key === "-" || e.key === "_") nudgeSpeed(-1);
  else if (e.key === "Home") { stop(); show(0); }
  else if (e.key === "End") { stop(); show(N - 1); }
});
show(0);
</script>
</body>
</html>
"""
