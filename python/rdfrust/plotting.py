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
            "saving plots needs matplotlib: pip install matplotlib "
            "(or pip install \"rdf-rust[plot]\")"
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


def save_animations(r, frames, total_per_frame, partials_per_frame, out_dir,
                    bin_size=None, total_mean=None, partials_mean=None,
                    title=None, fps=8, fmt="gif", dpi=110):
    """Animate the RDF frame by frame, with the trajectory average for reference.

    ``total_per_frame`` is (n_frames, n_bins); ``partials_per_frame`` maps
    ``(A, B)`` to (n_frames, n_bins).
    """
    from matplotlib import animation

    os.makedirs(out_dir, exist_ok=True)
    x = _centers(r, bin_size)
    tag = f" · {title}" if title else ""
    written = {}
    writer = (animation.PillowWriter(fps=fps) if fmt == "gif"
              else animation.FFMpegWriter(fps=fps))

    # Total RDF.
    tot = np.asarray(total_per_frame, dtype=float)
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

    def draw_total(k):
        line.set_ydata(tot[k])
        head.set_text(f"Total RDF · frame {frames[k]}{tag}")
        return (line, head)

    anim = animation.FuncAnimation(fig, draw_total, frames=len(frames), blit=False)
    path = os.path.join(out_dir, f"total_rdf_animation.{fmt}")
    anim.save(path, writer=writer, dpi=dpi)
    written["total_animation"] = path

    # Partials.
    pairs = unique_pairs(partials_per_frame)
    colors = pair_colors(pairs)
    if len(pairs) <= len(PALETTE):
        fig, ax = _new_figure(width=8.6)
        ax = ax[0, 0]
        lines = {}
        for p in pairs:
            if partials_mean is not None and p in partials_mean:
                ax.plot(x, partials_mean[p], color=colors[p], lw=1.0, ls="--", alpha=0.55)
            (lines[p],) = ax.plot(x, partials_per_frame[p][0], color=colors[p], lw=1.6,
                                  label=f"{p[0]}–{p[1]}")
        top = max(float(np.max(partials_per_frame[p])) for p in pairs)
        ax.set_ylim(0, top * 1.08 or 1.0)
        ax.set_xlim(x[0], x[-1])
        _label(ax, None)
        head = ax.set_title(f"Partial RDFs · frame {frames[-1]}{tag}", color=INK,
                            fontsize=FS_TITLE, pad=10)
        if len(pairs) > 1:
            _legend(ax, ncol=2 if len(pairs) > 4 else 1, loc="upper right")
        fig.tight_layout()
        artists = list(lines.values())
    else:
        nrows, ncols = _grid_shape(len(pairs))
        fig, axes = _new_figure(width=3.4 * ncols, height=2.6 * nrows + 0.6,
                                nrows=nrows, ncols=ncols, sharex=True)
        top = max(float(np.max(partials_per_frame[p])) for p in pairs) * 1.08 or 1.0
        lines = {}
        for k, ax in enumerate(axes.flat):
            if k >= len(pairs):
                ax.set_visible(False)
                continue
            p = pairs[k]
            if partials_mean is not None and p in partials_mean:
                ax.plot(x, partials_mean[p], color=MUTED, lw=0.9, ls="--")
            (lines[p],) = ax.plot(x, partials_per_frame[p][0], color=PALETTE[0], lw=1.3)
            ax.set_ylim(0, top)
            ax.set_title(f"{p[0]}–{p[1]}", color=INK, fontsize=FS_LABEL, pad=4)
            ax.tick_params(labelsize=FS_TICK - 2)
        head = fig.suptitle(f"Partial RDFs · frame {frames[-1]}{tag}", color=INK,
                            fontsize=FS_TITLE)
        fig.supxlabel("r (Å)", color=INK, fontsize=FS_LABEL)
        fig.supylabel("g(r)", color=INK, fontsize=FS_LABEL)
        fig.tight_layout()
        artists = list(lines.values())

    def draw_partials(k):
        for p, ln in lines.items():
            ln.set_ydata(partials_per_frame[p][k])
        head.set_text(f"Partial RDFs · frame {frames[k]}{tag}")
        return artists + [head]

    anim = animation.FuncAnimation(fig, draw_partials, frames=len(frames), blit=False)
    path = os.path.join(out_dir, f"partial_rdfs_animation.{fmt}")
    anim.save(path, writer=writer, dpi=dpi)
    written["partials_animation"] = path
    return written
