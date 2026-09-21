"""Read the generated structures, run all three backends, plot speed and memory.

Each (structure, backend) measurement runs in its own subprocess: glibc keeps
freed pages mapped, so a second in-process measurement of the same backend
reads far lower than the first, and tracemalloc cannot see allocations made
inside Rust or C++ at all.

    python benchmarks/compare_methods.py            # measure, then plot
    python benchmarks/compare_methods.py --plot     # re-plot saved results
"""
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT_DIR = os.path.join(HERE, "structures")
RESULTS = os.path.join(HERE, "results_t1.json")   # rebound per --threads in main
INDEX = os.path.join(STRUCT_DIR, "index.json")
CUTOFF, BIN_SIZE = 10.0, 0.1
# 0 = every core. Set with --threads N; it only affects RDF-Rust, since the
# other two backends are single-threaded.
# RDF-Rust thread count; 1 by default so the comparison with the
# single-threaded Matminer and Vesin is like for like. --threads 4 for the
# parallel run.
N_THREADS = 1
SUFFIX = "_t1"

BACKENDS = ["matminer", "vesin", "rdfrust"]
LABEL = {"matminer": "Matminer", "vesin": "Vesin", "rdfrust": "RDF-Rust"}
# Validated categorical palette, slots 1-3; all-pairs separation checked.
COLOR = {"matminer": "#2a78d6", "vesin": "#eb6834", "rdfrust": "#1baf7a"}
MARKER = {"matminer": "o", "vesin": "s", "rdfrust": "D"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

WORKER = r'''
import json, os, sys, threading, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, %r)
import psutil, gc, ctypes

backend, path = sys.argv[1], sys.argv[2]
n_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 0
proc = psutil.Process()
from pymatgen.core import Structure as Pmg
s = Pmg.from_file(path)

if backend == "matminer":
    from matminer.featurizers.structure import PartialRadialDistributionFunction
    f = PartialRadialDistributionFunction(cutoff=%f, bin_size=%f); f.fit([s])
    run = lambda: f.compute_prdf(s)
elif backend == "vesin":
    from prdf_vesin import compute_prdf_vesin
    run = lambda: compute_prdf_vesin(s, %f, %f)
else:
    import rdfrust
    rr = rdfrust.from_pymatgen(s)
    # n_threads is left at its default of 0, i.e. every core rayon can see.
    run = lambda: rr.prdf(cutoff=%f, bin_size=%f, n_threads=n_threads)
    THREADS = n_threads or rdfrust.available_threads()

run_once = run  # warm up caches and lazy imports before measuring
try:
    small = Pmg.from_file(path)
except Exception:
    pass

gc.collect()
try: ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception: pass
peak = base = proc.memory_info().rss
stop = False
def sample():
    global peak
    while not stop:
        try: peak = max(peak, proc.memory_info().rss)
        except Exception: return
        time.sleep(0.001)
t = threading.Thread(target=sample, daemon=True); t.start()

best = float("inf")
reps = 3 if len(s) <= 1200 else 1
for _ in range(reps):
    t0 = time.perf_counter(); run(); best = min(best, time.perf_counter() - t0)
stop = True; t.join(timeout=1.0)
peak = max(peak, proc.memory_info().rss)
print(json.dumps({"backend": backend, "atoms": len(s), "seconds": best,
                  "memory_mb": (peak - base) / 2**20,
                  "threads": globals().get("THREADS", 1)}))
''' % (HERE, CUTOFF, BIN_SIZE, CUTOFF, BIN_SIZE, CUTOFF, BIN_SIZE)


def load_index():
    with open(INDEX) as fh:
        return json.load(fh)


def check_agreement(entry):
    """Max |difference| against matminer, and how many distances sit on a bin edge.

    Run in-process: this is about numbers, not timings, so contamination does
    not matter. The tie count explains any disagreement -- a distance sitting
    exactly on a bin edge is decided by its last few ulp, and the backends
    compute distances by different routes.
    """
    import numpy as np
    from pymatgen.core import Structure as Pmg
    from matminer.featurizers.structure import PartialRadialDistributionFunction
    from prdf_vesin import compute_prdf_vesin
    import rdfrust

    s = Pmg.from_file(os.path.join(STRUCT_DIR, entry["file"]))
    f = PartialRadialDistributionFunction(cutoff=CUTOFF, bin_size=BIN_SIZE)
    f.fit([s])
    _edges, mp = f.compute_prdf(s)
    _e2, vp = compute_prdf_vesin(s, CUTOFF, BIN_SIZE)
    _r, rp, _t = rdfrust.from_pymatgen(s).prdf(cutoff=CUTOFF, bin_size=BIN_SIZE)

    scale = max((float(np.max(np.abs(g))) for g in mp.values()), default=1.0) or 1.0
    out = {
        "vesin": max(float(np.max(np.abs(vp[k] - mp[k]))) for k in mp) / scale,
        "rdfrust": max(float(np.max(np.abs(np.asarray(rp[k]) - mp[k]))) for k in mp) / scale,
    }

    # Tie count, taken from vesin's list because it is the cheapest to get.
    from vesin import NeighborList
    nl = NeighborList(cutoff=CUTOFF, full_list=True)
    _i, _j, d = nl.compute(
        points=np.ascontiguousarray(s.cart_coords),
        box=np.ascontiguousarray(s.lattice.matrix),
        periodic=True, quantities="ijd",
    )
    edges = np.arange(0, CUTOFF + BIN_SIZE, BIN_SIZE)
    ties = int((np.min(np.abs(d[:, None] - edges[None, :]), axis=1) < 1e-10).sum())
    out["ties"] = ties
    out["pairs"] = int(len(d))
    return out


def measure_all(only=None):
    index = load_index()
    if only:
        index = [e for e in index if e["group"] == only]
        print(f"re-measuring group '{only}' only ({len(index)} structures)\n")
    rows = []
    for entry in index:
        name = entry["file"]
        path = os.path.join(STRUCT_DIR, name)
        for backend in BACKENDS:
            print(f"  {name:42} {backend:9} ...", end="", flush=True)
            try:
                out = subprocess.run(
                    [sys.executable, "-c", WORKER, backend, path, str(N_THREADS)],
                    capture_output=True, text=True, timeout=1800,
                )
                if out.returncode != 0:
                    print(" failed")
                    continue
                rec = json.loads(out.stdout.strip().splitlines()[-1])
                rec.update({k: entry[k] for k in
                            ("file", "group", "kind", "rattle", "elements", "density")})
                rows.append(rec)
                print(f" {rec['seconds'] * 1000:9.2f} ms  {rec['memory_mb']:8.1f} MB")
            except subprocess.TimeoutExpired:
                print(" timed out")
    print("\n  agreement check (relative to matminer):")
    agree = {}
    for entry in index:
        try:
            agree[entry["file"]] = check_agreement(entry)
            a = agree[entry["file"]]
            print(f"    {entry['file']:42} vesin {a['vesin']:.1e}  "
                  f"rdfrust {a['rdfrust']:.1e}  ties {a['ties']:>7,}/{a['pairs']:,}")
        except Exception as exc:
            print(f"    {entry['file']:42} failed: {exc}")
    if only and os.path.exists(RESULTS):
        # Keep the groups that were not re-measured.
        with open(RESULTS) as fh:
            prev = json.load(fh)
        rows = [r for r in prev.get("rows", []) if r.get("group") != only] + rows
        merged = dict(prev.get("agreement", {}))
        merged.update(agree)
        agree = merged
    payload = {"rows": rows, "agreement": agree}
    with open(RESULTS, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {RESULTS}")
    return payload


# Everything is set in black at a readable size; these figures end up in a
# README where they are scaled down.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 19, 16, 14, 16


# Figures that report agreement or bin-edge ties rather than timing do not
# depend on the thread count, so they keep a plain name.
UNTAGGED = {"md_ties.png", "md_ties.csv", "md_agreement.png", "md_agreement.csv"}


def tag(name):
    """Insert the thread suffix before the extension."""
    if name in UNTAGGED:
        return name
    stem, ext = os.path.splitext(name)
    return f"{stem}{SUFFIX}{ext}"


def write_csv(name, header, rows):
    """Save the numbers behind a figure, so the curves can be re-plotted or
    compared against another machine without re-measuring."""
    import csv
    out = os.path.join(HERE, tag(name))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {out}")


def style(ax):
    ax.set_facecolor(SURFACE)
    # Major lines only. A log axis draws eight minor lines per decade by
    # default, which turns the panel into graph paper and buries the data.
    ax.grid(True, which="major", color=GRID, lw=0.9)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b8b3")
    ax.tick_params(colors=INK, labelsize=FS_TICK)


#: Filled in from the measurements, so the figures state the hardware they
#: describe rather than leaving the reader to guess.
THREAD_NOTE = {"value": ""}


def finish(fig, ax, name, title, xlabel, ylabel, legend=True, ncol=3, note=None):
    ax.set_xlabel(xlabel, color=INK, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, color=INK, fontsize=FS_LABEL)
    ax.set_title(title, color=INK, fontsize=FS_TITLE, pad=12)
    if note:
        # tight_layout would crop a footnote drawn outside the axes, so room
        # is reserved for it first.
        fig.subplots_adjust(bottom=0.16)
        fig.text(0.5, 0.015, note, ha="center", color=INK, fontsize=12)
    if legend:
        leg = ax.legend(fontsize=FS_LEGEND, frameon=False, ncol=ncol,
                        labelcolor=INK, loc="best")
        if leg:
            for t in leg.get_texts():
                t.set_color(INK)
    fig.tight_layout(rect=[0, 0.06, 1, 1] if note else None)
    out = os.path.join(HERE, tag(name))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot(payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["rows"] if isinstance(payload, dict) else payload
    agree = payload.get("agreement", {}) if isinstance(payload, dict) else {}

    def pick(group, backend):
        return sorted([r for r in rows if r.get("group") == group and r["backend"] == backend],
                      key=lambda r: r["atoms"])

    _th = {r.get("threads", 1) for r in rows if r["backend"] == "rdfrust"}
    _n = max(_th) if _th else 1
    # Kept short: it has to fit the figure width. The single-threaded
    # comparison lives in BENCHMARKS.md, not on the axes.
    THREAD_NOTE["value"] = (
        f"RDF-Rust: {_n} cores   \u00b7   Matminer, Vesin: 1 core"
        if _n > 1 else "all backends single-threaded"
    )

    FIGSIZE = (8.0, 6.0)

    # ---------- scaling: one file per category ----------
    by = {b: pick("size", b) for b in BACKENDS}

    fig, ax = plt.subplots(figsize=FIGSIZE); fig.patch.set_facecolor(SURFACE); style(ax)
    for b in BACKENDS:
        d = by[b]
        if not d: continue
        ax.plot([r["atoms"] for r in d], [r["seconds"] * 1000 for r in d],
                marker=MARKER[b], color=COLOR[b], lw=2.6, ms=9, label=LABEL[b])
    ax.set_xscale("log"); ax.set_yscale("log")
    finish(fig, ax, "scaling_time.png", "Calculation time", "atoms", "time (ms)",
           ncol=1, note=THREAD_NOTE["value"])
    sizes = sorted({r["atoms"] for r in rows if r.get("group") == "size"})
    write_csv(
        "scaling_time.csv",
        ["atoms"] + [f"{LABEL[b]}_ms" for b in BACKENDS] + ["RDF-Rust_threads"],
        [[n] + [next((f"{r['seconds'] * 1000:.4f}" for r in by[b] if r["atoms"] == n), "")
                for b in BACKENDS]
         + [next((r.get("threads", 1) for r in by["rdfrust"] if r["atoms"] == n), "")]
         for n in sizes],
    )

    fig, ax = plt.subplots(figsize=FIGSIZE); fig.patch.set_facecolor(SURFACE); style(ax)
    for b in BACKENDS:
        d = by[b]
        if not d: continue
        ax.plot([r["atoms"] for r in d], [max(r["memory_mb"], 1e-2) for r in d],
                marker=MARKER[b], color=COLOR[b], lw=2.6, ms=9, label=LABEL[b])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1024, color=INK, lw=1.0, ls=":")
    if by["rdfrust"]:
        ax.text(by["rdfrust"][0]["atoms"], 1250, "1 GB", color=INK, fontsize=FS_TICK)
    finish(fig, ax, "scaling_memory.png", "Peak memory", "atoms", "peak memory (MB)",
           ncol=1, note=THREAD_NOTE["value"])
    write_csv(
        "scaling_memory.csv",
        ["atoms"] + [f"{LABEL[b]}_MB" for b in BACKENDS],
        [[n] + [next((f"{r['memory_mb']:.3f}" for r in by[b] if r["atoms"] == n), "")
                for b in BACKENDS] for n in sizes],
    )

    fig, ax = plt.subplots(figsize=FIGSIZE); fig.patch.set_facecolor(SURFACE); style(ax)
    mm = {r["atoms"]: r["seconds"] for r in by["matminer"]}
    for b in ("vesin", "rdfrust"):
        d = [r for r in by[b] if r["atoms"] in mm]
        if not d: continue
        ax.plot([r["atoms"] for r in d], [mm[r["atoms"]] / r["seconds"] for r in d],
                marker=MARKER[b], color=COLOR[b], lw=2.6, ms=9, label=LABEL[b])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1, color=INK, lw=1.0, ls=":")
    finish(fig, ax, "scaling_speedup.png", "Speed-up over Matminer",
           "atoms", "times faster than Matminer", ncol=1, note=THREAD_NOTE["value"])
    write_csv(
        "scaling_speedup.csv",
        ["atoms", "Vesin_x", "RDF-Rust_x"],
        [[n] + [next((f"{mm[n] / r['seconds']:.2f}" for r in by[b] if r["atoms"] == n), "")
                for b in ("vesin", "rdfrust")] for n in sizes if n in mm],
    )

    # ---------- by structure type: one file per category ----------
    var = [r for r in rows if r.get("group") == "variety"]
    kinds = sorted({r["kind"] for r in var})
    if kinds:
        labels = [k.replace("_", "\n") for k in kinds]
        x = np.arange(len(kinds)); w = 0.26
        for name, key, ylab, title in [
            ("types_time.png", "seconds", "time (ms)", "Calculation time by structure type"),
            ("types_memory.png", "memory_mb", "peak memory (MB)", "Peak memory by structure type"),
        ]:
            fig, ax = plt.subplots(figsize=(11.0, 6.4))
            fig.patch.set_facecolor(SURFACE); style(ax)
            for i, b in enumerate(BACKENDS):
                vals = []
                for k in kinds:
                    r = next((q for q in var if q["kind"] == k and q["backend"] == b), None)
                    v = (r[key] * 1000 if key == "seconds" else r[key]) if r else 0.0
                    vals.append(max(v, 1e-2))
                ax.bar(x + (i - 1) * w, vals, w * 0.9, label=LABEL[b],
                       color=COLOR[b], edgecolor=SURFACE, linewidth=1.6)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=FS_TICK, color=INK)
            counts = {k: next(r["atoms"] for r in var if r["kind"] == k) for k in kinds}
            nel = {k: len(next(r["elements"] for r in var if r["kind"] == k)) for k in kinds}
            ax.set_xlabel(
                "   ".join(f"{k.split('_')[0]}: {counts[k]:,} at., {nel[k]} el" for k in kinds),
                color=INK, fontsize=10.5, labelpad=12)
            ax.set_ylabel(ylab, color=INK, fontsize=FS_LABEL)
            ax.set_title(title, color=INK, fontsize=FS_TITLE, pad=12)
            leg = ax.legend(fontsize=FS_LEGEND, frameon=False, ncol=3, labelcolor=INK)
            for t in leg.get_texts():
                t.set_color(INK)
            fig.tight_layout()
            out = os.path.join(HERE, tag(name))
            fig.savefig(out, dpi=150, facecolor=SURFACE); print(f"wrote {out}")
            plt.close(fig)
            unit = "ms" if key == "seconds" else "MB"
            write_csv(
                name.replace(".png", ".csv"),
                ["structure_type", "atoms", "elements", "density_per_A3"]
                + [f"{LABEL[b]}_{unit}" for b in BACKENDS],
                [[k, counts[k], nel[k], f"{next(r['density'] for r in var if r['kind'] == k):.5f}"]
                 + [next((f"{(q[key] * 1000 if key == 'seconds' else q[key]):.4f}"
                          for q in var if q["kind"] == k and q["backend"] == b), "")
                    for b in BACKENDS]
                 for k in kinds],
            )

    # ---------- thermal disorder: one file per category ----------
    md = [r for r in rows if r.get("group") == "md"]
    if md and agree:
        # The md group carries its own sigma = 0 reference, so nothing is
        # spliced in from elsewhere.
        series = {}
        for kind in sorted({r["kind"] for r in md}):
            series[kind] = sorted(
                [r for r in md if r["kind"] == kind and r["backend"] == "rdfrust"],
                key=lambda r: r["rattle"])
        order = sorted(series)
        kcolor = {k: COLOR[b] for k, b in zip(order, ("vesin", "rdfrust"))}
        kmark = {k: m for k, m in zip(order, ("o", "s"))}

        fig, ax = plt.subplots(figsize=FIGSIZE); fig.patch.set_facecolor(SURFACE); style(ax)
        for kind, entries in series.items():
            ax.plot([r["rattle"] for r in entries],
                    [max(agree.get(r["file"], {}).get("ties", 0), 0.5) for r in entries],
                    marker=kmark[kind], color=kcolor[kind], lw=2.6, ms=9,
                    label=kind.split("_")[0])
        ax.set_yscale("log")
        # Zero cannot be drawn on a log axis, so it is floored at 0.5 and
        # labelled -- otherwise the flat line reads as "half a tie".
        ax.axhline(0.5, color=INK, lw=1.0, ls=":")
        ax.text(0.26, 0.56, "zero (floored for the log axis)", color=INK, fontsize=FS_TICK)
        finish(fig, ax, "md_ties.png", "Bin-edge ties vanish with thermal disorder",
               "RMS displacement (\u00c5)", "distances exactly on a bin edge", ncol=1)
        write_csv(
            "md_ties.csv",
            ["structure_type", "rms_displacement_A", "atoms", "pairs_in_range",
             "distances_on_a_bin_edge"],
            [[kind, r["rattle"], r["atoms"],
              agree.get(r["file"], {}).get("pairs", ""),
              agree.get(r["file"], {}).get("ties", "")]
             for kind, entries in series.items() for r in entries],
        )

        fig, ax = plt.subplots(figsize=FIGSIZE); fig.patch.set_facecolor(SURFACE); style(ax)
        for kind, entries in series.items():
            ax.plot([r["rattle"] for r in entries],
                    [max(agree.get(r["file"], {}).get("rdfrust", 0.0), 1e-17) for r in entries],
                    marker=kmark[kind], color=kcolor[kind], lw=2.6, ms=9,
                    label=kind.split("_")[0])
        ax.set_yscale("log")
        ax.axhline(1e-15, color=INK, lw=1.0, ls=":")
        ax.text(0.16, 2.6e-15, "double precision", color=INK, fontsize=FS_TICK)
        finish(fig, ax, "md_agreement.png",
               "With disorder the backends agree exactly",
               "RMS displacement (\u00c5)",
               "max relative |RDF-Rust \u2212 Matminer|", ncol=1)
        write_csv(
            "md_agreement.csv",
            ["structure_type", "rms_displacement_A",
             "Vesin_vs_Matminer_rel", "RDF-Rust_vs_Matminer_rel"],
            [[kind, r["rattle"],
              f"{agree.get(r['file'], {}).get('vesin', float('nan')):.3e}",
              f"{agree.get(r['file'], {}).get('rdfrust', float('nan')):.3e}"]
             for kind, entries in series.items() for r in entries],
        )

    # ---------- summary table ----------
    rr = pick("size", "rdfrust")
    fig, ax = plt.subplots(figsize=(13, 0.52 * max(len(rr), 1) + 1.6))
    fig.patch.set_facecolor(SURFACE); ax.axis("off")
    header = ["atoms", "Matminer", "Vesin", "RDF-Rust",
              "RDF-Rust\nvs Matminer", "RDF-Rust\nvs Vesin"]
    cells = []
    for r in rr:
        n = r["atoms"]
        m = next((x for x in pick("size", "matminer") if x["atoms"] == n), None)
        v = next((x for x in pick("size", "vesin") if x["atoms"] == n), None)
        row = [f"{n:,}"]
        for x in (m, v, r):
            row.append(f"{x['seconds']*1000:,.1f} ms / {x['memory_mb']:,.0f} MB" if x else "\u2014")
        row.append(f"{m['seconds']/r['seconds']:.0f}x" if m else "\u2014")
        row.append(f"{v['seconds']/r['seconds']:.1f}x" if v else "\u2014")
        cells.append(row)
    tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=INK)
        cell.set_facecolor("#f1f0ec" if row == 0 else SURFACE)
        if row == 0:
            cell.set_text_props(color=INK, fontweight="bold")
    ax.set_title(f"Time / peak memory  \u00b7  cutoff {CUTOFF} \u00c5, bin {BIN_SIZE} \u00c5",
                 color=INK, fontsize=FS_TITLE, pad=18)
    fig.tight_layout()
    out = os.path.join(HERE, tag("performance_table.png"))
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)

    # Every measurement in one long table, for anything the per-figure files
    # do not cover.
    write_csv(
        "all_measurements.csv",
        ["group", "kind", "file", "atoms", "elements", "density_per_A3",
         "rms_displacement_A", "backend", "time_ms", "peak_memory_MB"],
        [[r.get("group", ""), r.get("kind", ""), r.get("file", ""), r["atoms"],
          len(r.get("elements", [])), f"{r.get('density', 0):.5f}",
          r.get("rattle", 0), LABEL[r["backend"]], r.get("threads", 1),
          f"{r['seconds'] * 1000:.4f}", f"{r['memory_mb']:.3f}"]
         for r in sorted(rows, key=lambda q: (q.get("group", ""), q["atoms"], q["backend"]))],
    )


if __name__ == "__main__":
    only = None
    for i, a in enumerate(sys.argv):
        if a == "--only" and i + 1 < len(sys.argv):
            only = sys.argv[i + 1]
        if a == "--threads" and i + 1 < len(sys.argv):
            N_THREADS = int(sys.argv[i + 1])
            SUFFIX = f"_t{N_THREADS}" if N_THREADS else "_tall"
            RESULTS = os.path.join(HERE, f"results{SUFFIX}.json")
    print(f"RDF-Rust on {N_THREADS or 'all'} thread(s); outputs suffixed {SUFFIX}\n")
    if "--plot" in sys.argv:
        rows = json.load(open(RESULTS))
    else:
        if not os.path.isdir(STRUCT_DIR) or not os.listdir(STRUCT_DIR):
            print("no structures found; run benchmarks/make_structures.py first")
            sys.exit(1)
        rows = measure_all(only)
    plot(rows)
