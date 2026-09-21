"""The partial-occupancy workflow, end to end.

matminer and vesin need an ordered structure, so a disordered CIF must first be
expanded into a random supercell -- that expansion is part of their cost, and
it is also an approximation. rdfrust weights by occupancy directly.
"""
import gc, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pymatgen.core import Structure as PmgStructure, Lattice, Composition
from matminer.featurizers.structure import PartialRadialDistributionFunction
from prdf_vesin import compute_prdf_vesin
import rdfrust

CUT, BS = 10.0, 0.1


def disordered(n, occ):
    sp = [Composition(occ), "Ti", "O", "O", "O"]
    s = PmgStructure(Lattice.cubic(3.905), sp,
                     [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    if n > 1:
        s.make_supercell([n, n, n])
    return s


def timeit(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        gc.collect()
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def expand_with_pymatgen(pm, scale):
    """The route the app takes today: supercell, then random site assignment."""
    import random
    from pymatgen.core import PeriodicSite
    sc = pm.copy()
    sc.make_supercell([scale, scale, scale])
    rng = random.Random(42)
    groups, ordered = {}, []
    for site in sc:
        if site.is_ordered:
            ordered.append(site)
        else:
            key = tuple(sorted((str(s), round(float(o), 4)) for s, o in site.species.items()))
            groups.setdefault(key, []).append(site)
    for key, sites in groups.items():
        n = len(sites)
        wanted = [o * n for _, o in key]
        counts = [int(w) for w in wanted]
        total = min(int(sum(wanted) + 0.5), n)
        for k in sorted(range(len(key)), key=lambda i: wanted[i] - counts[i], reverse=True)[:total - sum(counts)]:
            counts[k] += 1
        assign = []
        for (name, _), c in zip(key, counts):
            assign += [name] * c
        assign += [None] * (n - len(assign))
        rng.shuffle(assign)
        for site, name in zip(sites, assign):
            if name:
                ordered.append(PeriodicSite(name, site.frac_coords, site.lattice))
    return PmgStructure.from_sites(ordered)


def main():
    print(f"Disordered (Ba,Sr)TiO3, cutoff={CUT} A, bin={BS} A")
    print("matminer/vesin timings include the supercell expansion they require.\n")
    print(f"{'input':>18} {'expanded':>9} {'expand':>9} {'matminer':>10} {'vesin':>9} "
          f"{'rdfrust':>9} {'vs mm':>8} {'vs vesin':>9}")
    print("-" * 96)
    for n, occ, scale in [(1, {"Ba": .5, "Sr": .5}, 4),
                          (2, {"Ba": .5, "Sr": .5}, 3),
                          (1, {"Ba": 1/3, "Sr": 2/3}, 6),
                          (3, {"Ba": .5, "Sr": .5}, 2)]:
        pm = disordered(n, occ)
        rr = rdfrust.from_pymatgen(pm)
        t_exp = timeit(lambda: expand_with_pymatgen(pm, scale))
        ordered = expand_with_pymatgen(pm, scale)
        f = PartialRadialDistributionFunction(cutoff=CUT, bin_size=BS)
        f.fit([ordered])
        t_m = timeit(lambda: f.compute_prdf(expand_with_pymatgen(pm, scale)), 2)
        t_v = timeit(lambda: compute_prdf_vesin(expand_with_pymatgen(pm, scale), CUT, BS), 2)
        t_r = timeit(lambda: rr.prdf(cutoff=CUT, bin_size=BS), 5)
        label = "".join(f"{k}{v:.2f}" for k, v in occ.items())
        print(f"{label:>18} {len(ordered):>9} {t_exp:>8.3f}s {t_m:>9.3f}s {t_v:>8.3f}s "
              f"{t_r:>8.4f}s {t_m/t_r:>7.1f}x {t_v/t_r:>8.1f}x")


if __name__ == "__main__":
    main()
