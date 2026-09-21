import warnings, numpy as np, pytest
warnings.filterwarnings("ignore")
import rdfrust
from pymatgen.core import Structure as PmgStructure, Lattice, Composition

def sto(n=1, a=3.905):
    s = PmgStructure(Lattice.cubic(a), ["Sr","Ti","O","O","O"],
                     [[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
    if n > 1: s.make_supercell([n,n,n])
    return s

def test_roundtrip_pymatgen():
    s = rdfrust.from_pymatgen(sto(2))
    assert s.n_sites == 40
    assert s.is_ordered
    assert abs(s.n_atoms - 40) < 1e-9
    assert s.elements == ["O","Sr","Ti"]
    assert abs(s.composition["O"] - 24) < 1e-9

def test_matches_matminer_ordered():
    from matminer.featurizers.structure import PartialRadialDistributionFunction
    pm = sto(3)
    f = PartialRadialDistributionFunction(cutoff=8.0, bin_size=0.1); f.fit([pm])
    mb, mp = f.compute_prdf(pm)
    r, partials, total = rdfrust.from_pymatgen(pm).prdf(cutoff=8.0, bin_size=0.1)
    assert np.allclose(r, mb)
    for key, ref in mp.items():
        got = partials[key]
        assert np.max(np.abs(got - ref)) < 1e-9, f"{key} differs"

def test_total_rdf_asymptote():
    pm = sto(6)
    s = rdfrust.from_pymatgen(pm)
    r, total = s.total_rdf(cutoff=12.0, bin_size=0.1)
    rho = pm.num_sites / pm.volume
    tail = total[(r > 8) & (r < 12)]
    assert abs(tail.mean() / rho - 1.0) < 0.05

def test_disordered_is_supported_natively():
    pm = PmgStructure(Lattice.cubic(3.905),
                      [Composition({"Ba":0.5,"Sr":0.5}),"Ti","O","O","O"],
                      [[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
    s = rdfrust.from_pymatgen(pm)
    assert not s.is_ordered
    assert abs(s.composition["Ba"] - 0.5) < 1e-9
    r, partials, total = s.prdf(cutoff=8.0, bin_size=0.1)
    assert ("Ba","Sr") in partials
    assert np.all(np.isfinite(total))

def test_expansion_preserves_composition():
    pm = PmgStructure(Lattice.cubic(3.905),
                      [Composition({"Ba":0.5,"Sr":0.5}),"Ti","O","O","O"],
                      [[0,0,0],[.5,.5,.5],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
    s = rdfrust.from_pymatgen(pm)
    o, info = s.expand_occupancies(max_sites=4096, seed=42)
    assert o.is_ordered
    n = o.n_sites
    assert abs(o.composition["Ba"]/n - 0.1) < 1e-6
    assert abs(o.composition["Sr"]/n - 0.1) < 1e-6
    assert info["warnings"] == []

def test_bad_input_rejected():
    with pytest.raises(ValueError):
        rdfrust.Structure([[1,0,0],[0,1,0],[0,0,1]], [[0,0,0]], [{"Fe":1.5}])
    s = rdfrust.from_pymatgen(sto(1))
    with pytest.raises(ValueError):
        s.prdf(cutoff=-1.0, bin_size=0.1)
    with pytest.raises(ValueError):
        s.prdf(cutoff=5.0, bin_size=10.0)
