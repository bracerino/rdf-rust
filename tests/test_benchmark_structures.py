"""The benchmark set must stay reproducible and physically sane."""
import json, os, sys, warnings
import numpy as np
import pytest
warnings.filterwarnings("ignore")
import rdfrust

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "benchmarks")
STRUCT = os.path.join(BENCH, "structures")
sys.path.insert(0, BENCH)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(STRUCT, "index.json")),
    reason="run benchmarks/make_structures.py first",
)


@pytest.fixture(scope="module")
def index():
    with open(os.path.join(STRUCT, "index.json")) as fh:
        return json.load(fh)


def test_every_listed_file_reads_back(index):
    for entry in index:
        s = rdfrust.read(os.path.join(STRUCT, entry["file"]), verbose=False)
        assert s.n_sites == entry["atoms"], entry["file"]
        assert sorted(s.elements) == entry["elements"], entry["file"]
        assert abs(s.n_atoms / s.volume - entry["density"]) < 1e-6, entry["file"]


def test_groups_are_populated(index):
    groups = {}
    for e in index:
        groups.setdefault(e["group"], []).append(e)
    assert len(groups["size"]) >= 5
    assert len(groups["variety"]) >= 5
    assert len(groups["md"]) >= 4
    # the variety set must actually vary
    densities = {round(e["density"], 3) for e in groups["variety"]}
    assert len(densities) >= 3, "structure types are too similar to be a useful set"
    assert max(len(e["elements"]) for e in groups["variety"]) >= 5


def test_generator_is_reproducible(tmp_path, index, monkeypatch):
    """Re-running the generator must reproduce the same coordinates."""
    import make_structures as ms
    monkeypatch.setattr(ms, "OUT", str(tmp_path))
    monkeypatch.setattr(ms, "META", str(tmp_path / "index.json"))
    ms.main()
    with open(tmp_path / "index.json") as fh:
        again = json.load(fh)
    assert [e["file"] for e in again] == [e["file"] for e in index]
    for e in again:
        a = rdfrust.read(os.path.join(STRUCT, e["file"]), verbose=False)
        b = rdfrust.read(str(tmp_path / e["file"]), verbose=False)
        assert np.allclose(np.array(a.frac_coords), np.array(b.frac_coords), atol=1e-9), e["file"]


def test_rattling_actually_displaces_atoms(index):
    """Each rattle level must broaden the first peak more than the last."""
    md = sorted([e for e in index if e["group"] == "md" and "Cu" in e["kind"]],
                key=lambda e: e["rattle"])
    perfect = next(e for e in index if e["group"] == "variety" and e["kind"] == "Cu_fcc")
    widths = []
    for entry in [perfect] + md:
        s = rdfrust.read(os.path.join(STRUCT, entry["file"]), verbose=False)
        r, _p, total = s.prdf(cutoff=6.0, bin_size=0.05)
        total = np.asarray(total)
        # width of the first-neighbour peak, as the count of bins above a tenth
        # of its height
        peak = int(np.argmax(total))
        thresh = total[peak] * 0.1
        lo = peak
        while lo > 0 and total[lo] > thresh:
            lo -= 1
        hi = peak
        while hi < len(total) - 1 and total[hi] > thresh:
            hi += 1
        widths.append(hi - lo)
    assert widths == sorted(widths), f"peak widths not monotonic in rattle: {widths}"
    assert widths[-1] > widths[0], "heaviest rattling did not broaden the peak"


def test_thermal_disorder_removes_bin_edge_ties(index):
    """The point of the MD set: no exact ties, so backends agree exactly."""
    from vesin import NeighborList
    from pymatgen.core import Structure as Pmg

    def ties(path, cutoff=10.0, bin_size=0.1):
        s = Pmg.from_file(path)
        nl = NeighborList(cutoff=cutoff, full_list=True)
        _i, _j, d = nl.compute(points=np.ascontiguousarray(s.cart_coords),
                               box=np.ascontiguousarray(s.lattice.matrix),
                               periodic=True, quantities="ijd")
        edges = np.arange(0, cutoff + bin_size, bin_size)
        return int((np.min(np.abs(d[:, None] - edges[None, :]), axis=1) < 1e-10).sum())

    md = [e for e in index if e["group"] == "md" and "Cu" in e["kind"]]
    commensurate = min(md, key=lambda e: e["rattle"])      # sigma = 0, a = 3.600 A
    hot = max(md, key=lambda e: e["rattle"])
    assert commensurate["rattle"] == 0.0

    n_perfect = ties(os.path.join(STRUCT, commensurate["file"]))
    n_hot = ties(os.path.join(STRUCT, hot["file"]))
    assert n_perfect > 1000, (
        f"a = 3.600 A is exactly 36 bins of 0.1 A, so this cell should be full of "
        f"bin-edge ties; got {n_perfect}")
    assert n_hot == 0, f"a rattled structure should have no exact ties, got {n_hot}"

    # And the converse, which is the part that is easy to get wrong: a perfect
    # lattice whose spacing is NOT a multiple of the bin width has no ties at
    # all. Ties come from commensurability, not from crystallinity.
    real_cu = next(e for e in index if e["group"] == "variety" and e["kind"] == "Cu_fcc")
    assert ties(os.path.join(STRUCT, real_cu["file"])) == 0, (
        "copper at its true a = 3.615 A gives 36.15 bins and should produce no ties")
