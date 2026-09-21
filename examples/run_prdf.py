"""Minimal end-to-end RDF-Rust calculation.

    python examples/run_prdf.py examples/BaSrTiO3_sym.cif
"""
import sys

import numpy as np

import rdfrust

path = sys.argv[1] if len(sys.argv) > 1 else "examples/BaSrTiO3_sym.cif"
CUTOFF, BIN_SIZE, N_THREADS = 10.0, 0.1, 0      # 0 threads = use every core

# read() reports partial occupancies on stderr and stays quiet for an ordered
# structure. Use Structure.from_cif() if you want it silent.
s = rdfrust.read(path)
print(f"{path}: {s.n_sites} sites, {s.n_atoms:g} atoms, elements {s.elements}")

# Every element pair is computed; there is nothing to select.
r, partials, total = s.prdf(cutoff=CUTOFF, bin_size=BIN_SIZE, n_threads=N_THREADS)
print(f"{len(partials)} pairs on {len(r)} bins of {BIN_SIZE} A up to {CUTOFF} A")

# g(r) approaches the bulk number density at large r.
rho = s.n_atoms / s.volume
tail = total[r > 0.7 * CUTOFF]
print(f"number density {rho:.5f} /A^3; total RDF tail averages {tail.mean():.5f}")

for pair in sorted(partials):
    g = np.asarray(partials[pair])
    print(f"  g({pair[0]:>2}-{pair[1]:<2}) first peak at r = {r[int(np.argmax(g))]:.2f} A")

np.savetxt(
    "rdf.csv",
    np.column_stack([r, total] + [partials[p] for p in sorted(partials)]),
    delimiter=",",
    header="r," + "total," + ",".join(f"{a}-{b}" for a, b in sorted(partials)),
    comments="",
)
print("wrote rdf.csv")
