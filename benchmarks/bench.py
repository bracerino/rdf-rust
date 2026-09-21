"""rdfrust vs matminer vs vesin+NumPy: same curves, and how much faster.

rdfrust uses rayon, so it is also timed with RAYON_NUM_THREADS=1 to separate
"better algorithm" from "more cores".
"""
import gc, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pymatgen.core import Structure as PmgStructure, Lattice
from matminer.featurizers.structure import PartialRadialDistributionFunction
from prdf_vesin import compute_prdf_vesin
import rdfrust

CUT, BS = 10.0, 0.1


def sto(n, a=3.905):
    s = PmgStructure(Lattice.cubic(a), ["Sr", "Ti", "O", "O", "O"],
                     [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    s.make_supercell([n, n, n])
    return s


def timeit(fn, reps):
    best = float("inf")
    for _ in range(reps):
        gc.collect()
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def agreement(pm, rr):
    """Max absolute difference between every backend, pair by pair."""
    f = PartialRadialDistributionFunction(cutoff=CUT, bin_size=BS)
    f.fit([pm])
    _, mp = f.compute_prdf(pm)
    _, vp = compute_prdf_vesin(pm, CUT, BS)
    _, rp, _ = rr.prdf(cutoff=CUT, bin_size=BS)
    d_v = max(np.max(np.abs(mp[k] - vp[k])) for k in mp)
    d_r = max(np.max(np.abs(mp[k] - np.asarray(rp[k]))) for k in mp)
    return d_v, d_r


def main():
    threads = os.environ.get("RAYON_NUM_THREADS", "all")
    print(f"cutoff={CUT} A  bin={BS} A   rayon threads: {threads}\n")
    print(f"{'atoms':>7} {'matminer':>10} {'vesin':>10} {'rdfrust':>10} "
          f"{'vs mm':>8} {'vs vesin':>9} {'max |diff| vs matminer':>24}")
    print("-" * 92)
    for n in [4, 5, 6, 8, 10, 12, 14]:
        pm = sto(n)
        N = len(pm)
        rr = rdfrust.from_pymatgen(pm)
        f = PartialRadialDistributionFunction(cutoff=CUT, bin_size=BS)
        f.fit([pm])
        reps = 5 if N <= 1200 else 2
        t_r = timeit(lambda: rr.prdf(cutoff=CUT, bin_size=BS), max(reps, 5))
        t_v = timeit(lambda: compute_prdf_vesin(pm, CUT, BS), max(reps, 5))
        t_m = timeit(lambda: f.compute_prdf(pm), reps)
        d_v, d_r = agreement(pm, rr)
        print(f"{N:>7} {t_m:>9.3f}s {t_v:>9.4f}s {t_r:>9.4f}s "
              f"{t_m/t_r:>7.1f}x {t_v/t_r:>8.1f}x   vesin {d_v:.1e} / rdfrust {d_r:.1e}")


if __name__ == "__main__":
    main()
