"""POSCAR and extended-XYZ reading, checked against the CIF route."""
import os, tempfile, warnings
import numpy as np
import pytest
warnings.filterwarnings("ignore")
import rdfrust
from pymatgen.core import Lattice, Structure as PmgStructure
from pymatgen.io.vasp import Poscar


CUT, BS = 8.0, 0.1


def assert_same_rdf(ref, got, pmg):
    """Compare two readings of one crystal.

    Writing a cell as a CIF and as a POSCAR does not preserve the *orientation*
    of the lattice matrix: the CIF writer rebuilds it from a/b/c/alpha/beta/gamma
    with a along x, while POSCAR keeps the matrix as given. The crystal is the
    same, but distances come out differing in the last few ulp, which is enough
    to move a count across a bin edge when a distance sits exactly on one. So
    the pair counts are required to match exactly and the per-bin comparison
    allows only as many differing bins as there are such ties.
    """
    r1, p1, t1 = ref.prdf(cutoff=CUT, bin_size=BS)
    r2, p2, t2 = got.prdf(cutoff=CUT, bin_size=BS)
    assert np.allclose(r1, r2), "bin edges differ"
    assert set(p1) == set(p2), "element pairs differ"

    edges = np.arange(0, CUT + BS, BS)
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    for key in p1:
        n_a = ref.composition[key[0]]
        c1 = float(np.sum(np.asarray(p1[key]) * shell * n_a))
        c2 = float(np.sum(np.asarray(p2[key]) * shell * n_a))
        assert abs(c1 - c2) < 1e-6, f"{key}: {c1:.4f} vs {c2:.4f} pairs"

    ds = np.array([nb[1] for site, nl in zip(pmg.sites, pmg.get_all_neighbors(CUT))
                   for nb in nl])
    ties = int((np.min(np.abs(ds[:, None] - edges[None, :]), axis=1) < 1e-9).sum())
    n_diff = sum(int(np.sum(np.abs(np.asarray(p1[k]) - np.asarray(p2[k])) > 1e-12))
                 for k in p1)
    if ties == 0:
        assert n_diff == 0, f"{n_diff} bins differ with no edge ties to explain it"
    else:
        assert n_diff <= 2 * ties, f"{n_diff} bins differ, more than {ties} ties explain"
    # Summing g itself would be wrong here: moving one count between bins of
    # different shell volume changes sum(g) while conserving the pair count.
    # So the total RDF is converted back to a count before comparing.
    n_tot = ref.n_atoms
    tot1 = float(np.sum(np.asarray(t1) * shell * n_tot))
    tot2 = float(np.sum(np.asarray(t2) * shell * n_tot))
    assert abs(tot1 - tot2) < 1e-6, f"total RDF holds {tot1:.4f} vs {tot2:.4f} pairs"


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    d = tmp_path_factory.mktemp("fmt")
    s = PmgStructure(Lattice.from_parameters(4.1, 5.2, 6.3, 88.0, 95.0, 101.0),
                     ["Sr", "Ti", "O", "O", "O"],
                     [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    s.make_supercell([2, 2, 2])
    s.to(filename=str(d / "ref.cif"), fmt="cif")
    Poscar(s).write_file(str(d / "POSCAR"))
    s.to(filename=str(d / "ref.vasp"), fmt="poscar")
    m = s.lattice.matrix
    with open(d / "ref.xyz", "w") as f:
        f.write(f"{len(s)}\n")
        f.write('Lattice="' + " ".join(f"{x:.12f}" for r in m for x in r)
                + '" Properties=species:S:1:pos:R:3\n')
        for site in s:
            c = site.coords
            f.write(f"{site.specie.symbol} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n")
    return d, s


@pytest.mark.parametrize("name", ["POSCAR", "ref.vasp", "ref.xyz"])
def test_matches_cif_route(written, name):
    d, pmg = written
    ref = rdfrust.read(str(d / "ref.cif"), verbose=False)
    got = rdfrust.read(str(d / name), verbose=False)
    assert got.n_sites == ref.n_sites == len(pmg)
    assert got.elements == ref.elements
    assert got.is_ordered
    assert_same_rdf(ref, got, pmg)


def test_cartesian_poscar(written):
    d, pmg = written
    ref = rdfrust.read(str(d / "ref.cif"), verbose=False)
    p = Poscar(pmg)
    text = p.get_string() if hasattr(p, "get_string") else str(p)
    lines = text.splitlines()
    # Rewrite the same structure in Cartesian coordinates.
    idx = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("direct"))
    out = lines[:idx] + ["Cartesian"]
    for site in pmg:
        c = site.coords
        out.append(f"  {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}")
    cart = d / "cart.vasp"
    cart.write_text("\n".join(out) + "\n")
    got = rdfrust.read(str(cart), verbose=False)
    assert_same_rdf(ref, got, pmg)


def test_xyz_without_lattice_is_rejected(tmp_path):
    p = tmp_path / "plain.xyz"
    p.write_text("2\njust a comment\nFe 0.0 0.0 0.0\nFe 1.0 1.0 1.0\n")
    with pytest.raises(ValueError, match="Lattice"):
        rdfrust.read(str(p), verbose=False)


def test_unknown_extension_message(tmp_path):
    p = tmp_path / "thing.bogus"
    p.write_text("nonsense\n")
    with pytest.raises(ValueError, match="CIF, POSCAR"):
        rdfrust.read(str(p), verbose=False)


def test_format_can_be_forced(written):
    d, _ = written
    got = rdfrust.read(str(d / "ref.vasp"), fmt="poscar", verbose=False)
    assert got.n_sites == 40


def test_from_str_dispatch(written):
    d, _ = written
    text = (d / "ref.cif").read_text()
    s = rdfrust.Structure.from_str(text, "cif")
    assert s.n_sites == 40
    with pytest.raises(ValueError, match="unknown format"):
        rdfrust.Structure.from_str(text, "nope")


def test_disorder_reporting():
    here = os.path.dirname(os.path.abspath(__file__))
    cif = os.path.join(here, "..", "examples", "BaSrTiO3_sym.cif")
    s = rdfrust.read(cif, verbose=False)
    assert not s.is_ordered
    note = rdfrust.describe_disorder(s, name="x.cif")
    assert "partially occupied" in note and "Ba" in note and "Sr" in note
    summary = s.disorder_summary()
    assert len(summary) == 1
    assert set(summary[0]["occupancies"]) == {"Ba", "Sr"}
    # an ordered structure must report nothing at all
    ordered = rdfrust.read(os.path.join(here, "..", "examples", "SrTiO3.cif"),
                           verbose=False)
    assert ordered.disorder_summary() == []
    assert rdfrust.describe_disorder(ordered) == ""


def test_thread_count_does_not_change_the_answer(written):
    d, _ = written
    s = rdfrust.read(str(d / "ref.cif"), verbose=False)
    base = None
    for nt in (0, 1, 2, 4):
        _r, p, t = s.prdf(cutoff=8.0, bin_size=0.1, n_threads=nt)
        cur = (np.asarray(t), {k: np.asarray(v) for k, v in p.items()})
        if base is None:
            base = cur
        else:
            assert np.array_equal(base[0], cur[0])
            for k in base[1]:
                assert np.array_equal(base[1][k], cur[1][k])
