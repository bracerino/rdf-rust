import os, warnings, numpy as np, pytest
warnings.filterwarnings("ignore")
import rdfrust
from pymatgen.io.cif import CifParser

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")

def pmg(path):
    return CifParser(path).parse_structures(primitive=False)[0]

def comp_of(pmg_struct):
    d = {}
    for site in pmg_struct:
        for sp, amt in site.species.items():
            sym = getattr(sp, "symbol", None) or sp.element.symbol
            d[sym] = d.get(sym, 0.0) + float(amt)
    return d

@pytest.mark.parametrize("name", ["bsto.cif", "bsto_ox.cif", "BaSrTiO3_sym.cif"])
def test_cif_composition_matches_pymatgen(name):
    path = os.path.join(EX, name)
    ours = rdfrust.Structure.from_cif(path)
    theirs = pmg(path)
    a, b = ours.composition, comp_of(theirs)
    assert set(a) == set(b), (a, b)
    # compare per formula unit, since we may keep a different cell setting
    na, nb = ours.n_atoms, sum(b.values())
    for el in a:
        assert abs(a[el]/na - b[el]/nb) < 1e-6, (el, a, b)

def test_partial_occupancy_preserved():
    s = rdfrust.Structure.from_cif(os.path.join(EX, "bsto.cif"))
    assert not s.is_ordered
    shared = [sp for sp in s.species if len(sp) > 1]
    assert len(shared) == 4, shared
    for sp in shared:
        assert set(sp) == {"Ba", "Sr"}
        assert abs(sum(sp.values()) - 1.0) < 1e-9

def test_symmetry_expansion_and_uncertainties():
    s = rdfrust.Structure.from_cif(os.path.join(EX, "BaSrTiO3_sym.cif"))
    # Pm-3m perovskite: 1 A site + 1 B site + 3 O
    assert s.n_sites == 5, s.n_sites
    a_site = [sp for sp in s.species if "Ba" in sp][0]
    assert abs(a_site["Ba"] - 0.48) < 1e-9   # standard uncertainty stripped
    assert abs(a_site["Sr"] - 0.52) < 1e-9
    assert abs(s.composition["O"] - 3.0) < 1e-9

def test_oxidation_states_stripped():
    s = rdfrust.Structure.from_cif(os.path.join(EX, "bsto_ox.cif"))
    assert set(s.elements) == {"Ba", "O", "Sr", "Ti"}

def test_ordered_cif_rdf_matches_pymatgen_path():
    from matminer.featurizers.structure import PartialRadialDistributionFunction
    path = os.path.join(EX, "SrTiO3.cif")
    if not os.path.exists(path):
        pytest.skip("SrTiO3.cif not available")
    ours = rdfrust.Structure.from_cif(path).make_supercell(3,3,3)
    theirs = pmg(path); theirs.make_supercell([3,3,3])
    f = PartialRadialDistributionFunction(cutoff=8.0, bin_size=0.1); f.fit([theirs])
    mb, mp = f.compute_prdf(theirs)
    r, partials, _ = ours.prdf(cutoff=8.0, bin_size=0.1)
    for key, ref in mp.items():
        assert np.max(np.abs(partials[key] - ref)) < 1e-9, key

def test_missing_file_and_garbage():
    with pytest.raises(IOError):
        rdfrust.Structure.from_cif("/nonexistent/nope.cif")
    with pytest.raises(ValueError):
        rdfrust.Structure.from_cif_str("data_x\n_cell_length_a 4.0\n")
