"""Peak memory of each backend, measured in an isolated subprocess.

Peak RSS is sampled from a background thread so allocations made inside Rust
and C++ are counted too -- tracemalloc would only see the Python heap and would
report almost nothing for rdfrust and vesin.
"""
import json, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
CUT, BS = 10.0, 0.1

WORKER = r'''
import os, sys, json, threading, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, %r)
import psutil

backend, n = sys.argv[1], int(sys.argv[2])
proc = psutil.Process()

from pymatgen.core import Structure as PmgStructure, Lattice
s = PmgStructure(Lattice.cubic(3.905), ["Sr","Ti","O","O","O"],
                 [[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
s.make_supercell([n,n,n])

# Import and warm up the backend BEFORE the baseline, so the number reported is
# the cost of one calculation and not of loading the library.
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
    run = lambda: rr.prdf(cutoff=%f, bin_size=%f)

small = PmgStructure(Lattice.cubic(3.905), ["Sr","Ti","O","O","O"],
                     [[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
if backend == "matminer":
    g = PartialRadialDistributionFunction(cutoff=4.0, bin_size=0.5); g.fit([small]); g.compute_prdf(small)
elif backend == "vesin":
    compute_prdf_vesin(small, 4.0, 0.5)
else:
    rdfrust.from_pymatgen(small).prdf(cutoff=4.0, bin_size=0.5)

import gc; gc.collect()
peak = base = proc.memory_info().rss
stop = False
def sample():
    global peak
    while not stop:
        try: peak = max(peak, proc.memory_info().rss)
        except Exception: pass
        time.sleep(0.001)
t = threading.Thread(target=sample, daemon=True); t.start()
t0 = time.perf_counter()
run()
elapsed = time.perf_counter() - t0
stop = True; t.join(timeout=1.0)
peak = max(peak, proc.memory_info().rss)
print(json.dumps({"backend": backend, "atoms": len(s), "base_mb": base/2**20,
                  "peak_mb": peak/2**20, "delta_mb": (peak-base)/2**20,
                  "seconds": elapsed}))
''' % (HERE, CUT, BS, CUT, BS, CUT, BS)


def measure(backend, n):
    out = subprocess.run([sys.executable, "-c", WORKER, backend, str(n)],
                         capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        return None
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [4, 6, 8, 10, 12]
    print(f"Peak RSS above baseline for one PRDF, cutoff={CUT} A bin={BS} A")
    print("(isolated subprocess, sampled at 1 ms; library import excluded)\n")
    print(f"{'atoms':>7} {'pairs':>12} {'matminer':>12} {'vesin':>12} {'rdfrust':>12}"
          f" {'mm/rust':>9} {'vesin/rust':>11}")
    print("-" * 82)
    for n in sizes:
        row = {}
        for b in ["matminer", "vesin", "rdfrust"]:
            r = measure(b, n)
            row[b] = r["delta_mb"] if r else float("nan")
            atoms = r["atoms"] if r else 0
        # pairs in range, for context
        rho = 5 / 3.905 ** 3
        pairs = int(atoms * rho * 4 / 3 * 3.14159 * CUT ** 3)
        rr = row["rdfrust"]
        print(f"{atoms:>7} {pairs:>12,} {row['matminer']:>11.1f}M {row['vesin']:>11.1f}M "
              f"{rr:>11.1f}M {row['matminer']/rr if rr else 0:>8.1f}x "
              f"{row['vesin']/rr if rr else 0:>10.1f}x")


if __name__ == "__main__":
    main()
