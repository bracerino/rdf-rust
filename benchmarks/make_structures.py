"""Generate the benchmark structure set.

Three groups:

* **size series** - one prototype at nine sizes, for the scaling curves.
* **variety** - seven prototypes at a comparable size: close-packed metals, an
  open covalent network, a hexagonal cell, a two-element rocksalt, a perovskite
  and a five-element alloy. They differ in density, coordination number, cell
  shape and how many element pairs the histogram has to carry.
* **MD-like** - the same crystals with Gaussian displacements applied, the way
  a snapshot pulled out of a molecular-dynamics run looks. A perfect lattice is
  a degenerate case for an RDF benchmark: every distance repeats exactly, so
  the histogram is a set of spikes and a large share of the distances land
  precisely on a bin edge. Rattled positions remove that degeneracy and are far
  closer to what the code meets in practice.

All positions come from a seeded generator, so the set is reproducible.
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from pymatgen.core import Composition, Lattice, Structure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "structures")
META = os.path.join(OUT, "index.json")

SEED = 20260921
# RMS displacement per atom. 0.05-0.10 A is a typical thermal amplitude for a
# solid near room temperature; 0.20 A is a hot solid; 0.50 A is past melting
# and behaves like a dense liquid.
RATTLE_LEVELS = [0.05, 0.10, 0.20, 0.50]


# --- prototypes ------------------------------------------------------------

def fcc(symbol, a):
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def bcc(symbol, a):
    return Structure(Lattice.cubic(a), [symbol] * 2, [[0, 0, 0], [.5, .5, .5]])


def diamond(symbol, a):
    base = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
    coords = base + [[x + .25, y + .25, z + .25] for x, y, z in base]
    return Structure(Lattice.cubic(a), [symbol] * 8, coords)


def rocksalt(cation, anion, a):
    base = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
    coords = base + [[x + .5, y, z] for x, y, z in base]
    return Structure(Lattice.cubic(a), [cation] * 4 + [anion] * 4, coords)


def hcp(symbol, a, c):
    return Structure(Lattice.from_parameters(a, a, c, 90, 90, 120),
                     [symbol] * 2, [[1 / 3, 2 / 3, .25], [2 / 3, 1 / 3, .75]])


def perovskite(A, B, X, a):
    return Structure(Lattice.cubic(a), [A, B, X, X, X],
                     [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


def hea(elements, a, reps, rng):
    """Equiatomic solid solution: one FCC lattice, species drawn at random."""
    s = fcc(elements[0], a)
    s.make_supercell([reps, reps, reps])
    picks = rng.integers(0, len(elements), len(s))
    return Structure(s.lattice, [elements[k] for k in picks], s.frac_coords)


def rattle(structure, sigma, rng):
    """Gaussian displacement of every atom, as in an MD snapshot."""
    s = structure.copy()
    disp = rng.normal(0.0, sigma, (len(s), 3))
    cart = s.cart_coords + disp
    return Structure(s.lattice, [site.species for site in s], cart,
                     coords_are_cartesian=True)


def grow_to(structure, target):
    """Smallest cubic repetition reaching at least `target` atoms."""
    n = 1
    while len(structure) * n ** 3 < target:
        n += 1
    s = structure.copy()
    s.make_supercell([n, n, n])
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    index = []

    def write(struct, name, group, kind, rattle_sigma=0.0):
        path = os.path.join(OUT, name)
        struct.to(filename=path, fmt="cif")
        index.append({
            "file": name, "group": group, "kind": kind,
            "atoms": len(struct), "rattle": rattle_sigma,
            "elements": sorted({s.specie.symbol for s in struct}),
            "density": len(struct) / struct.volume,
        })

    # 1. size series -- one prototype, nine sizes.
    base = perovskite("Sr", "Ti", "O", 3.905)
    for n in [2, 3, 4, 5, 6, 8, 10, 12, 14]:
        s = base.copy()
        s.make_supercell([n, n, n])
        write(s, f"size_SrTiO3_{len(s):05d}.cif", "size", "perovskite")

    # 2. variety -- different prototypes at a comparable size (~2000 atoms).
    TARGET = 2000
    prototypes = [
        ("Cu_fcc", fcc("Cu", 3.615)),
        ("Fe_bcc", bcc("Fe", 2.867)),
        ("Si_diamond", diamond("Si", 5.431)),
        ("Ti_hcp", hcp("Ti", 2.951, 4.684)),
        ("NaCl_rocksalt", rocksalt("Na", "Cl", 5.640)),
        ("SrTiO3_perovskite", perovskite("Sr", "Ti", "O", 3.905)),
        ("CoCrFeMnNi_hea", hea(["Co", "Cr", "Fe", "Mn", "Ni"], 3.59, 8, rng)),
    ]
    variety = []
    for label, proto in prototypes:
        s = proto if len(proto) >= TARGET else grow_to(proto, TARGET)
        variety.append((label, s))
        write(s, f"type_{label}_{len(s):05d}.cif", "variety", label)

    # 3. MD-like snapshots.
    #
    # These deliberately use *round* lattice constants. A distance only lands
    # exactly on a bin edge when the spacing is commensurate with the bin
    # width: a = 3.600 A is exactly 36 bins of 0.1 A, whereas the true
    # a = 3.615 A of copper is 36.15 and never coincides. Ties are therefore
    # not a property of perfect crystals in general, only of perfect crystals
    # whose spacing divides the bin width -- which is easy to hit, since people
    # type round lattice constants and round bin sizes. The rattled members
    # show the degeneracy disappearing.
    md_protos = [
        ("Cu_fcc_a3.60", grow_to(fcc("Cu", 3.600), TARGET)),
        ("SrTiO3_a3.90", grow_to(perovskite("Sr", "Ti", "O", 3.900), TARGET)),
    ]
    for label, proto in md_protos:
        # sigma = 0 is the perfect, commensurate reference.
        for sigma in [0.0] + RATTLE_LEVELS:
            s = proto if sigma == 0.0 else rattle(proto, sigma, rng)
            write(s, f"md_{label}_s{sigma:.2f}_{len(s):05d}.cif",
                  "md", label, rattle_sigma=sigma)

    with open(META, "w") as fh:
        json.dump(index, fh, indent=2)

    print(f"wrote {len(index)} structures to {OUT}\n")
    for group in ("size", "variety", "md"):
        rows = [r for r in index if r["group"] == group]
        print(f"  [{group}]")
        for r in rows:
            extra = f"  rattle {r['rattle']:.2f} A" if r["rattle"] else ""
            print(f"    {r['file']:42} {r['atoms']:>6} atoms  "
                  f"{len(r['elements'])} el  rho={r['density']:.4f}/A^3{extra}")
    print(f"\n  index: {META}")


if __name__ == "__main__":
    main()
