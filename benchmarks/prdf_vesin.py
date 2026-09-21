"""PRDF via vesin neighbour lists + NumPy histograms.

Reproduces matminer's PartialRadialDistributionFunction.compute_prdf exactly:
bins = arange(0, cutoff + bin_size, bin_size), each partial g_AB normalised by
the spherical shell volume and by N_A (the count of the *central* species).
"""
import numpy as np
from vesin import NeighborList


def compute_prdf_vesin(structure, cutoff, bin_size):
    pos = np.ascontiguousarray(structure.cart_coords, dtype=np.float64)
    cell = np.ascontiguousarray(structure.lattice.matrix, dtype=np.float64)

    # full_list=True lists every ordered pair (i,j) and (j,i), matching
    # pymatgen's get_all_neighbors, which reports neighbours from each centre.
    nl = NeighborList(cutoff=float(cutoff), full_list=True)
    i, j, d = nl.compute(points=pos, box=cell, periodic=True, quantities="ijd")

    symbols = np.array([site.specie.symbol for site in structure])
    elements = sorted(set(symbols.tolist()))
    code = {e: k for k, e in enumerate(elements)}
    z = np.array([code[s] for s in symbols], dtype=np.int64)
    n_el = len(elements)

    bins = np.arange(0, cutoff + bin_size, bin_size)
    n_bin = len(bins) - 1
    shell_volume = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    # searchsorted reproduces np.histogram's edge handling exactly, including
    # the closed right edge of the final bin.
    bin_idx = np.searchsorted(bins, d, side="right") - 1
    bin_idx[d == bins[-1]] = n_bin - 1
    keep = (bin_idx >= 0) & (bin_idx < n_bin)

    # One bincount over a flattened (centre, neighbour, bin) index is far
    # cheaper than a histogram pass per element pair.
    flat = (z[i[keep]] * n_el + z[j[keep]]) * n_bin + bin_idx[keep]
    hist = np.bincount(flat, minlength=n_el * n_el * n_bin).reshape(n_el, n_el, n_bin)

    n_alpha = np.bincount(z, minlength=n_el).astype(np.float64)
    rdf = hist / shell_volume[None, None, :] / n_alpha[:, None, None]

    prdf = {}
    for a in range(n_el):
        for b in range(n_el):
            prdf[(elements[a], elements[b])] = rdf[a, b]
    return bins[:-1], prdf
