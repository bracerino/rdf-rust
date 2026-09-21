"""Combine the per-thread benchmark runs into one comparison.

Assumes the runs already happened:

    python benchmarks/compare_methods.py --threads 1 --only size
    python benchmarks/compare_methods.py --threads 4 --only size

then:

    python benchmarks/compare_threads.py
"""
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_methods import COLOR, FS_LABEL, FS_LEGEND, FS_TICK, FS_TITLE, GRID, INK, SURFACE

# 1 and 4 threads, in the validated categorical order.
RUNS = [("results_t1.json", 1), ("results_t4.json", 4)]
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
MARK = ["o", "s", "D"]


def load():
    out = []
    for name, threads in RUNS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"  skipping {name} (not found)")
            continue
        rows = json.load(open(path))["rows"]
        rr = sorted([r for r in rows if r["backend"] == "rdfrust"
                     and r.get("group") == "size"], key=lambda r: r["atoms"])
        if not rr:
            continue
        n = threads or max(r.get("threads", 1) for r in rr)
        out.append((n, {r["atoms"]: r["seconds"] * 1000 for r in rr}))
        if threads == 1:
            ve = sorted([r for r in rows if r["backend"] == "vesin"
                         and r.get("group") == "size"], key=lambda r: r["atoms"])
            out.append(("vesin", {r["atoms"]: r["seconds"] * 1000 for r in ve}))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.9)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b8b3")
    ax.tick_params(colors=INK, labelsize=FS_TICK)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load()
    series = [(k, v) for k, v in data if k != "vesin"]
    vesin = dict(data).get("vesin", {})
    if not series:
        print("no results to combine"); return
    sizes = sorted(series[0][1])

    # ---- absolute time ----
    fig, ax = plt.subplots(figsize=(8.0, 6.0)); fig.patch.set_facecolor(SURFACE); style(ax)
    for i, (n, d) in enumerate(series):
        ax.plot(sizes, [d[s] for s in sizes if s in d], marker=MARK[i % 3],
                color=SLOT[i % 3], lw=2.6, ms=9,
                label=f"RDF-Rust, {n} thread" + ("s" if n != 1 else ""))
    if vesin:
        ax.plot(sizes, [vesin[s] for s in sizes if s in vesin], color=INK, lw=2.0,
                ls="--", label="Vesin (1 core)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("atoms", color=INK, fontsize=FS_LABEL)
    ax.set_ylabel("time (ms)", color=INK, fontsize=FS_LABEL)
    ax.set_title("RDF-Rust: effect of thread count", color=INK, fontsize=FS_TITLE, pad=12)
    leg = ax.legend(fontsize=FS_LEGEND, frameon=False, labelcolor=INK)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out = os.path.join(HERE, "threads_time.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE); print(f"wrote {out}"); plt.close(fig)

    # ---- parallel speed-up, against the ideal ----
    base = dict(series[0][1])
    fig, ax = plt.subplots(figsize=(8.0, 6.0)); fig.patch.set_facecolor(SURFACE); style(ax)
    for i, (n, d) in enumerate(series):
        if n == 1:
            continue
        ax.plot(sizes, [base[s] / d[s] for s in sizes if s in d and s in base],
                marker=MARK[i % 3], color=SLOT[i % 3], lw=2.6, ms=9,
                label=f"{n} threads")
        ax.axhline(n, color=SLOT[i % 3], lw=1.2, ls=":")
        ax.text(sizes[0], n * 1.02, f"ideal {n}x", color=SLOT[i % 3], fontsize=FS_TICK,
                va="bottom")
    # From zero, with headroom above the highest ideal line so its label stays
    # inside the axes.
    top = max((n for n, _ in series if n != 1), default=2)
    ax.set_ylim(0, top * 1.15)
    ax.set_xscale("log")
    ax.set_xlabel("atoms", color=INK, fontsize=FS_LABEL)
    ax.set_ylabel("speed-up over 1 thread", color=INK, fontsize=FS_LABEL)
    ax.set_title("Parallel scaling", color=INK, fontsize=FS_TITLE, pad=12)
    leg = ax.legend(fontsize=FS_LEGEND, frameon=False, labelcolor=INK, loc="center right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out = os.path.join(HERE, "threads_speedup.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE); print(f"wrote {out}"); plt.close(fig)

    # ---- the numbers ----
    header = ["atoms"] + [f"RDF-Rust_{n}thread_ms" for n, _ in series] \
        + [f"speedup_{n}thread_vs_1" for n, _ in series if n != 1] + ["Vesin_1core_ms"]
    rows = []
    for s in sizes:
        row = [s] + [f"{d[s]:.4f}" if s in d else "" for _, d in series]
        row += [f"{base[s] / d[s]:.2f}" if s in d and s in base else ""
                for n, d in series if n != 1]
        row.append(f"{vesin[s]:.4f}" if s in vesin else "")
        rows.append(row)
    out = os.path.join(HERE, "threads_comparison.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
