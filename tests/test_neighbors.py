"""The cell list is the easiest thing to get subtly wrong, so it is checked
against matminer over cells that stress it: tiny cells where the cutoff reaches
many periodic images, strongly sheared triclinic cells, and long thin cells.
"""
import warnings, numpy as np, pytest
warnings.filterwarnings("ignore")
import rdfrust
from pymatgen.core import Structure as PmgStructure, Lattice
from matminer.featurizers.structure import PartialRadialDistributionFunction

def matminer_reference(pm, cutoff, bin_size):
    """matminer's PRDF, with its normalisation corrected where it is wrong.

    matminer divides by ``to_reduced_dict[A] * num_sites``. For every ordinary
    composition that is exactly N_A, but for a cell of a single diatomic
    element pymatgen reduces the formula to its molecular form (O -> O2), so
    the divisor comes out twice too large. The reference is rescaled to the
    true atom count so the comparison tests the RDF and not that quirk.
    """
    f = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
    f.fit([pm])
    mb, mp = f.compute_prdf(pm)
    reduced = pm.composition.fractional_composition.to_reduced_dict
    out = {}
    for key, g in mp.items():
        true_na = sum(1 for s in pm if s.specie.symbol == key[0])
        mm_na = reduced[key[0]] * pm.num_sites
        out[key] = g * (mm_na / true_na)
    return mb, out


def edge_tie_count(pm, cutoff, bin_size):
    """Distances sitting exactly on a bin edge, where the bin is a coin flip."""
    ds = np.array([nb[1] for site, nl in zip(pm.sites, pm.get_all_neighbors(cutoff))
                   for nb in nl])
    if ds.size == 0:
        return 0
    edges = np.arange(0, cutoff + bin_size, bin_size)
    return int((np.min(np.abs(ds[:, None] - edges[None, :]), axis=1) < 1e-10).sum())


def check(pm, cutoff, bin_size=0.1, tol=1e-9):
    mb, ref = matminer_reference(pm, cutoff, bin_size)
    r, rp, _ = rdfrust.from_pymatgen(pm).prdf(cutoff=cutoff, bin_size=bin_size)
    assert np.allclose(r, mb), "bin edges differ"

    # The neighbour counts must match exactly whatever the binning does.
    shell = 4 / 3 * np.pi * ((mb + bin_size) ** 3 - mb ** 3)
    for key in ref:
        na = sum(1 for s in pm if s.specie.symbol == key[0])
        n_ref = np.sum(ref[key] * shell * na)
        n_got = np.sum(np.asarray(rp[key]) * shell * na)
        assert abs(n_ref - n_got) < 1e-6, (
            f"{key}: {n_ref:.4f} pairs vs {n_got:.4f} -- the neighbour search disagrees")

    worst = max(np.max(np.abs(np.asarray(rp[k]) - ref[k])) for k in ref)
    ties = edge_tie_count(pm, cutoff, bin_size)
    if ties == 0:
        assert worst < tol, f"max diff {worst:.3e} with no edge ties to explain it"
    else:
        # Distances that land on a bin edge are decided by the last few ulp of
        # a distance, and rdfrust and pymatgen compute distances by different
        # routes. Counts are conserved (asserted above); only the split between
        # two adjacent bins can move.
        n_diff = sum(int(np.sum(np.abs(np.asarray(rp[k]) - ref[k]) > 1e-12)) for k in ref)
        assert n_diff <= 2 * ties, f"{n_diff} bins differ, more than {ties} edge ties explain"
    return worst

@pytest.mark.parametrize("cutoff", [2.0, 5.0, 10.0, 15.0, 20.0])
def test_tiny_cell_many_images(cutoff):
    """One atom in a 2.9 A cell: at 20 A the cutoff spans ~7 images each way."""
    pm = PmgStructure(Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]])
    check(pm, cutoff)

@pytest.mark.parametrize("cutoff", [4.0, 8.0, 12.0])
def test_triclinic(cutoff):
    lat = Lattice.from_parameters(5.1, 6.3, 7.2, 81.0, 97.0, 103.0)
    pm = PmgStructure(lat, ["Fe", "Ni", "O", "O"],
                      [[0, 0, 0], [.5, .31, .22], [.17, .63, .41], [.77, .12, .88]])
    check(pm, cutoff)

@pytest.mark.parametrize("cutoff", [4.0, 9.0])
def test_extremely_sheared(cutoff):
    """gamma = 150 deg: the cell is far thinner than its edge lengths suggest."""
    lat = Lattice.from_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 150.0)
    pm = PmgStructure(lat, ["Si", "O"], [[0, 0, 0], [.4, .25, .6]])
    check(pm, cutoff)

@pytest.mark.parametrize("cutoff", [3.0, 7.0])
def test_long_thin_cell(cutoff):
    """A slab-like cell: 40 A along a, 3 A along b and c."""
    lat = Lattice.from_parameters(40.0, 3.0, 3.1, 90.0, 90.0, 90.0)
    pm = PmgStructure(lat, ["C", "C", "N"], [[0, 0, 0], [.33, .5, .5], [.66, .2, .8]])
    check(pm, cutoff)

def test_hexagonal():
    lat = Lattice.from_parameters(2.95, 2.95, 4.68, 90.0, 90.0, 120.0)
    pm = PmgStructure(lat, ["Ti", "Ti"], [[1/3, 2/3, .25], [2/3, 1/3, .75]])
    check(pm, 12.0)

@pytest.mark.parametrize("bin_size", [0.01, 0.05, 0.25, 1.0])
def test_bin_sizes(bin_size):
    pm = PmgStructure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"],
                      [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    pm.make_supercell([2, 2, 2])
    check(pm, 8.0, bin_size)

def test_random_structures_against_matminer():
    """Random cells and occupancies, to catch anything the fixed cases miss."""
    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(12):
        a, b, c = rng.uniform(3.0, 9.0, 3)
        al, be, ga = rng.uniform(70.0, 110.0, 3)
        lat = Lattice.from_parameters(a, b, c, al, be, ga)
        n = int(rng.integers(2, 7))
        els = list(rng.choice(["Fe", "Ni", "O", "Si"], n))
        pm = PmgStructure(lat, els, rng.random((n, 3)))
        worst = max(worst, check(pm, float(rng.uniform(5.0, 12.0))))
    assert worst < 1e-9


def test_matminer_normalisation_quirk_is_matminers():
    """Pin down the one case where rdfrust and matminer genuinely disagree.

    matminer normalises g_AB by ``to_reduced_dict[A] * num_sites``. pymatgen
    reduces a pure-oxygen formula to its molecular form O2, so that product is
    2*N_O rather than N_O and matminer's g_OO comes out half its proper size.
    rdfrust counts atoms directly and is the one that is right; the factor is
    asserted here so a future change to either side is noticed.
    """
    lat = Lattice.from_parameters(4.0646, 8.2659, 8.2823, 98.3814, 107.3372, 109.1999)
    pm = PmgStructure(lat, ["O", "O"],
                      [[.75757, .616619, .114889], [.315988, .067734, .886871]])
    cutoff, bs = 5.1545, 0.1
    f = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bs)
    f.fit([pm])
    mb, mp = f.compute_prdf(pm)
    _, rp, _ = rdfrust.from_pymatgen(pm).prdf(cutoff=cutoff, bin_size=bs)

    shell = 4 / 3 * np.pi * ((mb + bs) ** 3 - mb ** 3)
    # Ten ordered pairs are present: five neighbours seen from each of two atoms.
    n_pairs = np.sum(np.asarray(rp[("O", "O")]) * shell * 2)
    assert abs(n_pairs - 10.0) < 1e-6, n_pairs
    assert np.allclose(np.asarray(rp[("O", "O")]), 2.0 * mp[("O", "O")], rtol=1e-9)

    # A single-element cell of a non-diatomic element is unaffected.
    fe = PmgStructure(Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]])
    check(fe, 6.0)
