"""Do the three backends draw the same curves? Renders the evidence.

Figure 1: ordered structure, all three backends overlaid plus residuals.
Figure 2: disordered structure -- rdfrust's occupancy-weighted curve against
          the average over explicitly sampled random supercells.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from pymatgen.core import Structure as PmgStructure, Lattice
from matminer.featurizers.structure import PartialRadialDistributionFunction
from prdf_vesin import compute_prdf_vesin
import rdfrust

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated slots 1-3
CUT, BS = 10.0, 0.1


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e4e3df", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cfcec9")
    ax.tick_params(colors=MUTED, labelsize=8)


def figure_ordered(out):
    pm = PmgStructure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"],
                      [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    pm.make_supercell([4, 4, 4])
    f = PartialRadialDistributionFunction(cutoff=CUT, bin_size=BS); f.fit([pm])
    r, mp = f.compute_prdf(pm)
    _, vp = compute_prdf_vesin(pm, CUT, BS)
    _, rp, rtot = rdfrust.from_pymatgen(pm).prdf(cutoff=CUT, bin_size=BS)

    pairs = [("Sr", "Sr"), ("Sr", "O"), ("Ti", "O"), ("O", "O")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    fig.patch.set_facecolor(SURFACE)
    for col, key in enumerate(pairs):
        ax = axes[0, col]; style(ax)
        # Decreasing width so all three stay visible where they coincide;
        # the dash pattern is the secondary encoding the palette WARN requires.
        ax.plot(r, mp[key], color=S1, lw=5.0, alpha=0.40, solid_capstyle="round",
                label="matminer", zorder=2)
        ax.plot(r, vp[key], color=S2, lw=2.2, ls=(0, (5, 3)), label="vesin + NumPy", zorder=3)
        ax.plot(r, np.asarray(rp[key]), color=S3, lw=1.2, label="rdfrust", zorder=4)
        ax.set_title(f"g$_{{{key[0]}-{key[1]}}}$(r)", color=INK, fontsize=11, pad=6)
        if col == 0:
            ax.set_ylabel("g(r)", color=MUTED, fontsize=9)

        ax2 = axes[1, col]; style(ax2)
        d_v = np.abs(np.asarray(vp[key]) - mp[key])
        d_r = np.abs(np.asarray(rp[key]) - mp[key])
        ax2.plot(r, np.maximum(d_v, 1e-20), color=S2, lw=1.6, ls=(0, (5, 3)))
        ax2.plot(r, np.maximum(d_r, 1e-20), color=S3, lw=1.2)
        ax2.set_yscale("log"); ax2.set_ylim(1e-19, 1e-10)
        ax2.axhline(1e-15, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax2.set_xlabel("r (Å)", color=MUTED, fontsize=9)
        if col == 0:
            ax2.set_ylabel("|diff| vs matminer", color=MUTED, fontsize=8)
            ax2.text(0.4, 1.6e-15, "double precision", color=MUTED, fontsize=7, va="bottom")
            # vesin reproduces matminer bit for bit here, so its residual is
            # identically zero and has no place on a log axis -- say so rather
            # than let the missing trace read as a plotting bug.
            ax2.text(0.4, 3e-19, "vesin: exactly 0 (off scale)", color=S2,
                     fontsize=7, va="bottom")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.005), labelcolor=INK)
    fig.suptitle("Ordered SrTiO$_3$, 320 atoms — the three backends agree to machine precision",
                 color=INK, fontsize=13, y=0.93)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    return max(np.max(np.abs(np.asarray(rp[k]) - mp[k])) for k in mp)


def figure_disordered(out):
    from test_disorder_physics import disordered_bsto, ensemble_average
    cut, bs = 8.0, 0.2
    tmpl = disordered_bsto(5, (("Ba", 0.7), ("Sr", 0.3)))
    r, weighted, _ = tmpl.prdf(cutoff=cut, bin_size=bs)
    avg = ensemble_average(tmpl, 400, cut, bs, seed=1)

    keys = [("Ba", "Ba"), ("Ba", "Sr"), ("Sr", "Sr"), ("Ba", "O")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key in zip(axes, keys):
        style(ax)
        ax.plot(r, avg[key], color=S1, lw=5.0, alpha=0.40, solid_capstyle="round",
                label="mean of 400 random supercells", zorder=2)
        ax.plot(r, np.asarray(weighted[key]), color=S3, lw=1.6,
                label="rdfrust, occupancy-weighted", zorder=3)
        ax.set_title(f"g$_{{{key[0]}-{key[1]}}}$(r)", color=INK, fontsize=11, pad=6)
        ax.set_xlabel("r (Å)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("g(r)", color=MUTED, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.01), labelcolor=INK)
    fig.suptitle("Disordered Ba$_{0.7}$Sr$_{0.3}$TiO$_3$ — the weighted curve is the ensemble average, "
                 "computed once instead of sampled", color=INK, fontsize=13, y=0.90)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    d = figure_ordered(os.path.join(here, "curves_ordered.png"))
    print(f"  max |rdfrust - matminer| over all pairs: {d:.2e}")
    figure_disordered(os.path.join(here, "curves_disordered.png"))
