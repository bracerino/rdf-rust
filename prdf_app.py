import warnings
import io
from fractions import Fraction
from math import gcd
from functools import reduce
import streamlit as st

st.set_page_config(page_title="XRDlicious – (P)RDF Calculator", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    /* Just enough room for the sidebar button, which is fixed to the top-left
       corner and would otherwise sit across the title. */
    .block-container { padding-top: 2.2rem; }
    /* Streamlit keeps its own top padding on the first block and on headings
       as well; it is removed so the gap above the title is only the one set
       above. */
    .block-container > div:first-child { padding-top: 0 !important; }
    .block-container [data-testid="stHeading"]:first-child,
    .block-container [data-testid="stMarkdownContainer"]:first-child > h3:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    /* The top bar is hidden to keep the page compact, but the button that
       re-opens a collapsed sidebar lives inside it — without the overrides
       below it disappears with the header and the sidebar cannot be brought
       back. Visibility is inherited, so the controls are switched back on
       explicitly. */
    header     {visibility: hidden;}
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"] * {
        visibility: visible !important;
    }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {display: block !important;}

    /* Same open/close control as in the main app: a solid dark-red button
       with a white icon. */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] button {
        background-color: #8b0000 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
        transition: background-color 0.15s ease, transform 0.15s ease;
    }
    /* Only the collapsed-state button is always on screen; the one inside the
       sidebar keeps Streamlit's own hover reveal, so its opacity is left
       alone. */
    [data-testid="stExpandSidebarButton"] {opacity: 1 !important;}
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #b30000 !important;
        transform: scale(1.06);
    }
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapseButton"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stExpandSidebarButton"] {
        min-height: 34px !important;
        min-width: 34px !important;
        padding: 0 6px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Tab styling. Streamlit 1.60+ builds tabs with React Aria: each tab is
# [data-testid="stTab"] inside a [role="tablist"], and its state is exposed as
# [data-selected] / [data-hovered]. Earlier releases used BaseWeb
# ([data-baseweb="tab-list"] > button, aria-selected). Both are targeted, so the
# look holds on either side of that change.
css = """
<style>
.stTabs [role="tablist"],
.stTabs [data-baseweb="tab-list"] {
    gap: 20px !important;
}

.stTabs [data-testid="stTab"],
.stTabs [data-baseweb="tab-list"] button {
    background-color: #f0f4ff !important;
    border-radius: 12px !important;
    padding: 8px 16px !important;
    /* The new tabs have a fixed height; let the padding set it instead. */
    height: auto !important;
    transition: all 0.3s ease !important;
    border: none !important;
    color: #1e3a8a !important;
}

.stTabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.15rem !important;
    color: #1e3a8a !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

.stTabs [data-testid="stTab"][data-hovered],
.stTabs [data-testid="stTab"]:hover,
.stTabs [data-baseweb="tab-list"] button:hover {
    background-color: #dbe5ff !important;
    cursor: pointer;
}

.stTabs [data-testid="stTab"][data-selected],
.stTabs [data-testid="stTab"][aria-selected="true"],
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #e0e7ff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
}

.stTabs [data-baseweb="tab-list"] button:focus {
    outline: none !important;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)
import os, tempfile, zipfile
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from matminer.featurizers.structure import PartialRadialDistributionFunction

MAX_SUPERCELL_ATOMS = 750
MAX_SPECIES = 6


# ---------------------------------------------------------------------------
# PRDF backends
#
# Three implementations of the same quantity. matminer is the reference;
# vesin+NumPy and rdfrust are faster and are kept bin-for-bin compatible with
# it so their curves can be overlaid rather than merely compared.
# ---------------------------------------------------------------------------

BACKEND_MATMINER = "Matminer"
BACKEND_VESIN = "Vesin"
BACKEND_RDFRUST = "RDF-Rust"
# RDF-Rust leads: it is the fastest, the lightest on memory, and the only one
# that handles partial occupancies without a supercell, so it is the default.
BACKEND_ORDER = [BACKEND_RDFRUST, BACKEND_VESIN, BACKEND_MATMINER]
BACKEND_COLORS = {
    BACKEND_MATMINER: "#2a78d6",
    BACKEND_VESIN: "#eb6834",
    BACKEND_RDFRUST: "#1baf7a",
}
BACKEND_DASH = {BACKEND_MATMINER: "solid", BACKEND_VESIN: "dash", BACKEND_RDFRUST: "dot"}

# Per-backend ceilings on structure size, set by peak memory -- which is what
# actually runs out first. Measured on SrTiO3 supercells at a 10 A cutoff:
# Matminer holds every pair distance in a Python list (~660 bytes per pair) and
# Vesin in NumPy arrays (~97 bytes per pair), so both grow with the structure
# and reach ~175 MB at their caps. RDF-Rust streams pairs straight into
# histograms and stays near 2 MB at any size, so its cap is about upload and
# plotting rather than the calculation.
#
#   Matminer      750 atoms   ~175 MB
#   Vesin       5,000 atoms   ~176 MB    (same memory budget, 6.7x the atoms)
#   RDF-Rust   50,000 atoms     ~3 MB
BACKEND_MAX_ATOMS = {
    BACKEND_MATMINER: 750,
    BACKEND_VESIN: 5_000,
    BACKEND_RDFRUST: 50_000,
}
MAX_ATOMS = max(BACKEND_MAX_ATOMS.values())

# How many random fillings of a disordered cell each backend can reasonably be
# asked for. Realisations are independent calculations, so the cost is simply N
# times one run: at a few ms each RDF-Rust manages 500 in about a second, while
# Matminer at its 750-atom cap needs ~0.7 s per realisation and 20 is already a
# quarter of a minute.
BACKEND_MAX_REALISATIONS = {
    BACKEND_MATMINER: 20,
    BACKEND_VESIN: 100,
    BACKEND_RDFRUST: 500,
}

# Whenever more than one realisation is averaged, the Vacancy Supercells tab
# draws the individual fillings behind their average. It is worth showing even
# for two: the fewer are averaged, the noisier the mean, so that is exactly
# when seeing the scatter matters most.
REALISATION_OVERLAY_MIN = 1

try:
    from vesin import NeighborList as _VesinNeighborList
    _VESIN_OK, _VESIN_WHY = True, ""
except Exception as _exc:
    _VESIN_OK, _VESIN_WHY = False, str(_exc)
try:
    import rdfrust as _rdfrust
    _RDFRUST_OK, _RDFRUST_WHY = True, ""
except Exception as _exc:
    _RDFRUST_OK, _RDFRUST_WHY = False, str(_exc)

BACKEND_AVAILABLE = {
    BACKEND_MATMINER: (True, ""),
    BACKEND_VESIN: (_VESIN_OK, _VESIN_WHY),
    BACKEND_RDFRUST: (_RDFRUST_OK, _RDFRUST_WHY),
}


def _bin_edges(cut, bs):
    """The bin edges every backend shares, matching numpy.arange."""
    return np.arange(0, cut + bs, bs)


def _measure(fn):
    """Run `fn`, returning (result, seconds, peak MB of extra memory).

    Peak RSS is sampled from a thread, so allocations made inside Rust and C++
    are counted -- tracemalloc only sees the Python heap and would report
    nothing at all for rdfrust. malloc_trim is called first because glibc hangs
    on to freed pages, which otherwise makes the second measurement of a
    backend look far smaller than the first.
    """
    import threading
    import time as _time

    proc = trim = None
    try:
        import ctypes
        import psutil

        proc = psutil.Process()
        _libc = ctypes.CDLL("libc.so.6")
        trim = getattr(_libc, "malloc_trim", None)
    except Exception:
        proc = None

    if proc is None:
        t0 = _time.perf_counter()
        return fn(), _time.perf_counter() - t0, float("nan")

    import gc

    gc.collect()
    if trim is not None:
        try:
            trim(0)
        except Exception:
            pass
    base = peak = proc.memory_info().rss
    stop = False

    def _sample():
        nonlocal peak
        while not stop:
            try:
                peak = max(peak, proc.memory_info().rss)
            except Exception:
                return
            _time.sleep(0.001)

    watcher = threading.Thread(target=_sample, daemon=True)
    watcher.start()
    t0 = _time.perf_counter()
    try:
        out = fn()
    finally:
        elapsed = _time.perf_counter() - t0
        stop = True
        watcher.join(timeout=1.0)
    try:
        peak = max(peak, proc.memory_info().rss)
    except Exception:
        pass
    return out, elapsed, max(peak - base, 0) / 2 ** 20


def _assemble(pairs_g, centers, struct, r_min):
    """Shared post-processing: drop bins below r_min and build the total RDF.

    `pairs_g` maps (A, B) with A <= B to g_AB on `centers`.
    """
    prdf_dict, dist_dict, global_rdf = {}, {}, {}
    comp_frac = struct.composition.fractional_composition.get_el_amt_dict()
    keep = centers >= r_min
    for pair, g in pairs_g.items():
        g = np.asarray(g, dtype=float)[keep]
        prdf_dict[pair] = g
        dist_dict[pair] = list(centers[keep])
        c_a = comp_frac.get(pair[0], 0.0)
        weight = c_a if pair[0] == pair[1] else 2.0 * c_a
        for bc, val in zip(centers[keep], g):
            global_rdf[bc] = global_rdf.get(bc, 0.0) + weight * val
    return {"prdf_dict": prdf_dict, "dist_dict": dist_dict, "global_rdf": global_rdf}


def _prdf_matminer(struct, cut, bs, r_min):
    featurizer = PartialRadialDistributionFunction(cutoff=cut, bin_size=bs)
    featurizer.fit([struct])
    _edges, prdf = featurizer.compute_prdf(struct)
    elements = sorted({s.specie.symbol for s in struct})
    edges = _bin_edges(cut, bs)
    centers = (edges[:-1] + edges[1:]) / 2.0
    # compute_prdf returns every ordered pair; the app plots A <= B, which is
    # what matminer's own featurize() emits.
    pairs_g = {}
    for a_i, a in enumerate(elements):
        for b in elements[a_i:]:
            if (a, b) in prdf:
                pairs_g[(a, b)] = prdf[(a, b)]
    return _assemble(pairs_g, centers, struct, r_min)


def _prdf_vesin(struct, cut, bs, r_min):
    pos = np.ascontiguousarray(struct.cart_coords, dtype=np.float64)
    cell = np.ascontiguousarray(struct.lattice.matrix, dtype=np.float64)
    nl = _VesinNeighborList(cutoff=float(cut), full_list=True)
    i, j, d = nl.compute(points=pos, box=cell, periodic=True, quantities="ijd")

    symbols = np.array([s.specie.symbol for s in struct])
    elements = sorted(set(symbols.tolist()))
    code = {e: k for k, e in enumerate(elements)}
    z = np.array([code[s] for s in symbols], dtype=np.int64)
    n_el = len(elements)

    edges = _bin_edges(cut, bs)
    n_bin = len(edges) - 1
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    # searchsorted reproduces numpy.histogram's edge handling exactly.
    bin_idx = np.searchsorted(edges, d, side="right") - 1
    bin_idx[d == edges[-1]] = n_bin - 1
    keep = (bin_idx >= 0) & (bin_idx < n_bin)
    flat = (z[i[keep]] * n_el + z[j[keep]]) * n_bin + bin_idx[keep]
    hist = np.bincount(flat, minlength=n_el * n_el * n_bin).reshape(n_el, n_el, n_bin)
    n_alpha = np.bincount(z, minlength=n_el).astype(float)
    rdf = hist / shell[None, None, :] / n_alpha[:, None, None]

    centers = (edges[:-1] + edges[1:]) / 2.0
    pairs_g = {}
    for a in range(n_el):
        for b in range(a, n_el):
            pairs_g[(elements[a], elements[b])] = rdf[a, b]
    return _assemble(pairs_g, centers, struct, r_min)


def _prdf_rdfrust(struct, cut, bs, r_min):
    rr = _rdfrust.from_pymatgen(struct)
    _r, partials, _total = rr.prdf(cutoff=cut, bin_size=bs)
    elements = sorted({s.specie.symbol for s in struct})
    edges = _bin_edges(cut, bs)
    centers = (edges[:-1] + edges[1:]) / 2.0
    pairs_g = {}
    for a_i, a in enumerate(elements):
        for b in elements[a_i:]:
            if (a, b) in partials:
                pairs_g[(a, b)] = np.asarray(partials[(a, b)])
    return _assemble(pairs_g, centers, struct, r_min)


def _average_realisations(outs):
    """Mean over random fillings of the same disordered structure.

    `outs` is one `_assemble` result per realisation. The mean is taken per
    pair and the standard deviation kept alongside it, so the plots can show
    how much the answer depends on which arrangement was drawn. A pair absent
    from some realisation (an element that could not be placed at all) is
    averaged over the realisations that do contain it.
    """
    if len(outs) == 1:
        return {**outs[0], "n_realisations": 1, "spread": {}, "per_realisation": []}
    pairs = set()
    for o in outs:
        pairs.update(o["prdf_dict"])
    prdf_dict, spread, dist_dict = {}, {}, {}
    for pair in pairs:
        stack = [np.asarray(o["prdf_dict"][pair]) for o in outs if pair in o["prdf_dict"]]
        prdf_dict[pair] = np.mean(stack, axis=0)
        # Sample standard deviation per bin, across realisations. ddof=1 is the
        # unbiased estimator: the realisations are a sample drawn from the
        # ensemble, not the whole of it. With the default ddof=0 a pair of
        # realisations would report a spread 29% too small.
        spread[pair] = (
            np.std(stack, axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(stack[0])
        )
        dist_dict[pair] = next(o["dist_dict"][pair] for o in outs if pair in o["dist_dict"])
    global_rdf = {}
    for bc in outs[0]["global_rdf"]:
        vals = [o["global_rdf"][bc] for o in outs if bc in o["global_rdf"]]
        global_rdf[bc] = float(np.mean(vals))
    return {
        "prdf_dict": prdf_dict,
        "dist_dict": dist_dict,
        "global_rdf": global_rdf,
        "spread": spread,
        # The individual curves are kept so the Vacancy Supercells tab can show
        # how far single fillings stray from their mean. A handful of arrays per
        # pair costs a few hundred kB at most.
        "per_realisation": [
            {pair: np.asarray(o["prdf_dict"][pair]) for pair in o["prdf_dict"]} for o in outs
        ],
        "n_realisations": len(outs),
    }


PRDF_BACKENDS = {
    BACKEND_MATMINER: _prdf_matminer,
    BACKEND_VESIN: _prdf_vesin,
    BACKEND_RDFRUST: _prdf_rdfrust,
}


def rgb_to_hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


COLORS = [rgb_to_hex(c) for c in plt.cm.tab10.colors]
FONT = dict(size=22, color="black")


def _cif_site_occupancies(cif_text: str):
    """Occupancy values listed in the _atom_site loop of a CIF."""
    occs = []
    lines = cif_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        headers = []
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip().split()[0].lower())
            j += 1
        if "_atom_site_occupancy" in headers:
            col = headers.index("_atom_site_occupancy")
            while j < len(lines):
                row = lines[j].strip()
                if not row or row.startswith(("_", "#", "loop_", "data_")):
                    break
                fields = row.split()
                if len(fields) > col:
                    try:
                        # Values may carry a standard uncertainty, e.g. "0.83(2)".
                        occs.append(float(fields[col].split("(")[0]))
                    except ValueError:
                        pass
                j += 1
        i = j
    return occs


def _cif_has_partial_occupancy(cif_text: str) -> bool:
    return any(abs(o - 1.0) > 0.01 for o in _cif_site_occupancies(cif_text))


def load_structure_from_file(uploaded_file) -> Structure:
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    if suffix == "cif":
        cif_text = raw.decode("utf-8", errors="replace")
        struct = None
        try:
            parser = CifParser.from_str(cif_text)
            structs = parser.parse_structures(primitive=False)
            if structs:
                struct = structs[0]
        except Exception:
            pass
        if struct is None:
            # CIFs pymatgen rejects (e.g. every atom of the cell listed
            # together with the symmetry operations of its space group) are
            # repaired and reported instead of falling straight through to ASE.
            # An empty result without an exception is treated the same way.
            try:
                from cif_repair import parse_cif_content, record_cif_repair
                struct, notes = parse_cif_content(cif_text)
                record_cif_repair(uploaded_file.name, notes)
            except Exception:
                struct = None
        if struct is not None:
            return struct
        # ASE has no concept of site occupancy: it keeps one atom per position
        # and drops the rest, so a disordered CIF would come back looking
        # perfectly ordered with a whole element missing and no warning
        # anywhere. Rejecting the file is the only honest option.
        if _cif_has_partial_occupancy(cif_text):
            raise ValueError(
                "this CIF lists **partial occupancies** but could not be read by "
                "pymatgen, and the fallback reader (ASE) ignores occupancies — it "
                "would silently drop atoms and change the composition. Please "
                "correct the CIF (check that the occupancies of each shared site "
                "sum to at most 1) and upload it again."
            )
    fmt_map = {
        "poscar": "vasp",
        "vasp": "vasp",
        "lmp": "lammps-data",
        "data": "lammps-data",
        "xsf": "xsf",
        "xyz": "extxyz",
        "cfg": "cfg",
        "pw": "espresso-in",
    }
    ase_fmt = fmt_map.get(suffix, None)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        atoms = ase_read(tmp_path, format=ase_fmt) if ase_fmt else ase_read(tmp_path)
        return AseAtomsAdaptor.get_structure(atoms)
    finally:
        os.remove(tmp_path)


def validate_structure(struct: Structure):
    n_atoms = len(struct)
    species = set()
    for site in struct:
        if site.is_ordered:
            species.add(site.specie.symbol)
        else:
            for sp in site.species:
                species.add(sp.symbol)
    n_sp = len(species)
    if n_atoms > MAX_ATOMS:
        return (
            False,
            f"Structure has **{n_atoms} atoms**, which exceeds the largest limit any backend "
            f"supports (**{MAX_ATOMS:,}**, rdfrust). Please use a smaller supercell or a "
            f"primitive cell.",
        )
    if n_sp > MAX_SPECIES:
        return (
            False,
            f"Structure has **{n_sp} element types**, which exceeds the limit of **{MAX_SPECIES}**. Please use a structure with fewer species.",
        )
    return (True, "")


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _supercell_scale(struct: Structure):
    """Supercell scale that fits the occupancies, and the ideal one.

    The two differ when the cell the occupancies really need would exceed
    MAX_SUPERCELL_ATOMS; the caller warns about the approximation in that case.
    """
    denoms = []
    for site in struct:
        if not site.is_ordered:
            for _sp, occ in site.species.items():
                f = float(occ)
                if 0.0 < f < 1.0:
                    denoms.append(Fraction(f).limit_denominator(20).denominator)
    if not denoms:
        return (1, 1)
    ideal = reduce(_lcm, denoms, 1)
    n = len(struct)
    # Every multiple of the ideal denominator reproduces the occupancies just
    # as exactly, and a larger cell is strictly better: the random filling
    # scatters less, and once the cell edge exceeds the PRDF cutoff each atom
    # stops seeing its own periodic images. Those images are the same atom and
    # so are perfectly correlated, which biases every realisation identically --
    # averaging cannot remove it, only a bigger cell can. So take the largest
    # multiple that fits, not the smallest.
    largest = 0
    k = 1
    while True:
        s = ideal * k
        if s**3 * n > MAX_SUPERCELL_ATOMS:
            break
        largest = s
        k += 1
    if largest:
        return (largest, ideal)
    # Not even the smallest exact cell fits; fall back to the biggest
    # approximate one, and the caller warns about the deviation.
    for s in range(ideal - 1, 0, -1):
        if s**3 * n <= MAX_SUPERCELL_ATOMS:
            return (s, ideal)
    return (1, ideal)


def _clean_occupancies(site):
    """Occupancies of a disordered site, with vacancy placeholders removed.

    Returns None when nothing real is left on the site.
    """
    from pymatgen.core import DummySpecies

    real = {
        sp: float(occ)
        for sp, occ in site.species.items()
        if not isinstance(sp, DummySpecies) and str(sp).lower() not in ("x0+", "x")
    }
    snapped = {sp: round(occ, 4) for sp, occ in real.items()}
    snapped = {sp: occ for sp, occ in snapped.items() if occ > 0.0001}
    if not snapped:
        return None
    total = sum(snapped.values())
    if total > 1.0 + 0.001:
        snapped = {sp: occ / total for sp, occ in snapped.items()}
    return snapped


def _composition_key(occupancies):
    """Key that groups sites sharing the same disordered composition."""
    return tuple(sorted(((str(sp), round(float(occ), 4)) for sp, occ in occupancies.items())))


def _round_supercell_occupancies(supercell: Structure) -> Structure:
    from pymatgen.core import PeriodicSite
    from pymatgen.core.composition import Composition

    new_sites = []
    for site in supercell:
        if site.is_ordered:
            new_sites.append(site)
            continue
        snapped = _clean_occupancies(site)
        if not snapped:
            continue
        new_sites.append(
            PeriodicSite(
                Composition(snapped), site.frac_coords, site.lattice, properties=site.properties
            )
        )
    return Structure.from_sites(new_sites)


def _proportional_order_fallback(supercell: Structure, seed: int = 42):
    """Fill each disordered sublattice with atoms in proportion to its occupancies.

    Returns the ordered structure and, per composition group, how many atoms of
    each species were actually placed -- the occupancy table reports those
    numbers rather than re-deriving them, because sites that share a
    composition are filled as one pool and not copy by copy.
    """
    import random
    from pymatgen.core import PeriodicSite

    rng = random.Random(seed)
    ordered_sites = []
    disordered_by_comp = defaultdict(list)
    for site in supercell:
        if site.is_ordered:
            ordered_sites.append(site)
        else:
            disordered_by_comp[_composition_key(site.species)].append(site)
    extra_sites = []
    realized = {}
    for comp_key, sites in disordered_by_comp.items():
        n = len(sites)
        wanted = [float(occ) * n for _, occ in comp_key]
        # Largest-remainder apportionment. Rounding each species on its own
        # over- or under-fills the sublattice and, once the list is truncated,
        # always at the expense of the last species; handing the leftover sites
        # to the largest fractional parts spreads the error fairly instead.
        # int(x + 0.5) is used rather than round(), whose banker's rounding
        # turns a half-occupied single site into zero atoms.
        counts = [int(w) for w in wanted]
        total_atoms = min(int(sum(wanted) + 0.5), n)
        leftover = total_atoms - sum(counts)
        order = sorted(range(len(comp_key)), key=lambda k: wanted[k] - counts[k], reverse=True)
        for k in order[: max(leftover, 0)]:
            counts[k] += 1
        assignment = []
        for (sp_name, _occ), cnt in zip(comp_key, counts):
            assignment.extend([sp_name] * cnt)
        assignment += ["__vacancy__"] * (n - len(assignment))
        rng.shuffle(assignment)
        for site, sp_name in zip(sites, assignment):
            if sp_name == "__vacancy__":
                continue
            extra_sites.append(
                PeriodicSite(sp_name, site.frac_coords, site.lattice, properties=site.properties)
            )
        realized[comp_key] = {
            "n_sites": n,
            "counts": {sp_name: cnt for (sp_name, _occ), cnt in zip(comp_key, counts)},
        }
    all_sites = ordered_sites + extra_sites
    if not all_sites:
        raise ValueError("No sites remain after ordering -- check occupancy values.")
    return (Structure.from_sites(all_sites), realized)


def make_ordered_supercell(struct: Structure, seed: int = 42):
    scale, ideal_scale = _supercell_scale(struct)
    warnings_out = []
    supercell = struct.copy()
    supercell.make_supercell([scale, scale, scale])
    n_super = len(supercell)
    if n_super > MAX_SUPERCELL_ATOMS:
        raise ValueError(
            f"The smallest possible supercell ({scale}×{scale}×{scale}) already contains **{n_super} sites**, which exceeds the supercell limit of **{MAX_SUPERCELL_ATOMS} atoms**. Please provide a smaller primitive cell, or pre-order the structure externally and upload it directly."
        )
    supercell_clean = _round_supercell_occupancies(supercell)
    try:
        ordered, realized = _proportional_order_fallback(supercell_clean, seed=seed)
    except Exception as exc:
        raise ValueError(f"Could not build ordered supercell: {exc}") from exc
    n_ordered = len(ordered)
    n_sites = len(supercell_clean)
    n_vac = n_sites - n_ordered
    n_copies = scale**3
    # Sites sharing a composition are filled as one pool, so the occupancy
    # table is reported per group rather than per site -- a per-site figure
    # would claim a shortfall the structure does not actually have.
    groups = {}
    for i, site in enumerate(struct):
        if site.is_ordered:
            continue
        occs = _clean_occupancies(site)
        if not occs:
            continue
        groups.setdefault(_composition_key(occs), []).append((i, site.frac_coords))
    occ_rows = []
    dropped = set()
    for comp_key, members in groups.items():
        info = realized.get(comp_key)
        group_sites = info["n_sites"] if info else len(members) * n_copies
        idx_list = ", ".join(str(i) for i, _fc in members)
        fc = members[0][1]
        coords = f"({fc[0]:.4f}, {fc[1]:.4f}, {fc[2]:.4f})"
        if len(members) > 1:
            coords += f" +{len(members) - 1} more"
        for sp_name, target in comp_key:
            assigned = info["counts"].get(sp_name, 0) if info else 0
            actual = assigned / group_sites if group_sites > 0 else 0.0
            if assigned == 0 and target > 0:
                dropped.add(sp_name)
            occ_rows.append(
                {
                    "site_idx": idx_list,
                    "frac": coords,
                    "element": sp_name,
                    "target_occ": target,
                    "n_copies": n_copies,
                    "n_sites": group_sites,
                    "n_assigned": assigned,
                    "actual_occ": actual,
                }
            )
    if dropped:
        warnings_out.append(
            f"**No atoms could be placed for {', '.join(sorted(dropped))}** — the occupancy is too "
            f"small to be represented by the {scale}×{scale}×{scale} supercell. These elements are "
            "missing from the structure and will not appear in any PRDF pair. Compile the app "
            "locally and raise MAX_SUPERCELL_ATOMS in prdf_app.py to resolve them."
        )
    # Pooling every site of a sublattice often reproduces the occupancies
    # exactly even when the scale had to be capped, so the warning follows the
    # error the table actually shows rather than the scale alone.
    worst = max((abs(r["actual_occ"] - r["target_occ"]) for r in occ_rows), default=0.0)
    if scale < ideal_scale and worst >= 0.02:
        warnings_out.append(
            f"The occupancies would need a {ideal_scale}×{ideal_scale}×{ideal_scale} supercell to be "
            f"reproduced exactly, but that exceeds the {MAX_SUPERCELL_ATOMS}-site cap, so "
            f"{scale}×{scale}×{scale} was used and the occupancies are approximated — check the "
            "Delta column below."
        )
    msg = f"Disordered structure converted to a **{scale}×{scale}×{scale} vacancy supercell** ({n_ordered} atoms + {n_vac} vacant sites out of {n_sites} total). Atom counts approximate the CIF occupancies; vacant sites are assigned randomly (seed=42). Download this structure from the **🧩 Vacancy Supercells** tab."
    return (ordered, scale, msg, warnings_out, occ_rows)


def load_experimental_data(uploaded_file):
    content = uploaded_file.read().decode("utf-8", errors="replace")
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(io.StringIO(content), sep=None, engine="python", comment="#", header=None)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns.")
        df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 5:
            raise ValueError("Fewer than 5 valid numeric rows found.")
        return (df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float))
    except Exception as exc:
        raise ValueError(f"Could not parse experimental data file: {exc}") from exc


def _norm(y):
    y = np.asarray(y, dtype=float)
    m = float(np.max(y))
    return y / m if m > 0 else y


def smooth_gaussian(y, sigma=1.5):
    return gaussian_filter1d(y, sigma=sigma)


def smooth_savgol(y, window=11, polyorder=3):
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    polyorder = min(polyorder, window - 1)
    return savgol_filter(y, window, polyorder)


def smooth_spline(x, y, n_pts=300):
    xs = np.linspace(x[0], x[-1], n_pts)
    return (xs, np.maximum(0, make_interp_spline(x, y, k=3)(xs)))


st.markdown(
    "\n    <h3 style='color:#8b0000; margin-top:0; padding-top:0;'>\n        <strong>XRDlicious</strong> – <em>(P)RDF Calculator</em>\n    </h3>\n    <hr style=\"border:none;height:5px;background-color:#8b0000;\n               border-radius:6px;margin:0 0 8px 0;\">\n",
    unsafe_allow_html=True,
)
st.info(
    f"Upload one or more crystal structure files in the **sidebar** and press **▶️ Calculate RDF**. Each structure is processed **individually**. Use the **🔀 Comparison** tab to overlay PRDFs from different structures.  \n⚠️ Limits: **max {MAX_SPECIES} element types**, and an atom cap that depends on the backend — **{BACKEND_MAX_ATOMS[BACKEND_RDFRUST]:,}** for RDF-Rust, **{BACKEND_MAX_ATOMS[BACKEND_VESIN]:,}** for Vesin, **{BACKEND_MAX_ATOMS[BACKEND_MATMINER]:,}** for Matminer. To increase this limit or to have a stable personal version, please compile the app locally from **[GitHub](https://github.com/bracerino/rdf-rust)**, open file prdf_app.py, and change the values in BACKEND_MAX_ATOMS. Disordered CIFs are auto-converted to vacancy supercells capped at **{MAX_SUPERCELL_ATOMS} sites** (vacancies placed randomly).  \n⚙️ Pick the **calculation method** below — *RDF-Rust*, *Vesin*, *Matminer*, or **all three at once** to see their speed, memory and curves side by side."
)
# Filled in further down, once the calculation method is known, so that only
# the backends actually in use are listed.
_cite_slot = st.container()

st.sidebar.info(
    "🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. Spot a bug or have a feature idea? Let us know at: **lebedmi2@cvut.cz**. To compile the full app locally, visit our **[GitHub page](https://github.com/bracerino/xrdlicious)**. If you use this tool, please cite the **[article in IUCr](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)**. ❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**"
)
st.sidebar.header("📁 Upload Structure Files")
uploaded_files = st.sidebar.file_uploader(
    "CIF, POSCAR, LMP, XSF, XYZ (with cell), CFG, PW …",
    type=["cif", "xyz", "vasp", "poscar", "lmp", "data", "xsf", "pw", "cfg"],
    accept_multiple_files=True,
    key="prdf_uploader",
)
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Experimental Data")
st.sidebar.caption(
    "Two-column files (r, intensity) in any text format – CSV, TSV, space-separated, etc. Comment lines starting with **#** are skipped."
)
uploaded_exp_files = st.sidebar.file_uploader(
    "Select experimental data file(s)",
    type=["csv", "txt", "dat", "xy", "gr", "xye"],
    accept_multiple_files=True,
    key="exp_uploader",
)
if "prdf_structures" not in st.session_state:
    st.session_state.prdf_structures = {}
if "prdf_rejected" not in st.session_state:
    st.session_state.prdf_rejected = {}
if "ordered_structures" not in st.session_state:
    st.session_state.ordered_structures = {}
if "disorder_messages" not in st.session_state:
    st.session_state.disorder_messages = {}
if "disorder_occ_info" not in st.session_state:
    st.session_state.disorder_occ_info = {}
if "disorder_sources" not in st.session_state:
    # The structure as uploaded, occupancies intact. Kept so that extra random
    # realisations can be generated at calculation time without re-reading the
    # file.
    st.session_state.disorder_sources = {}
if "prdf_cif_ready" not in st.session_state:
    st.session_state.prdf_cif_ready = {}
if "prdf_experimental" not in st.session_state:
    st.session_state.prdf_experimental = {}
active_names = {f.name for f in uploaded_files} if uploaded_files else set()
if uploaded_files:
    for f in uploaded_files:
        if (
            f.name not in st.session_state.prdf_structures
            and f.name not in st.session_state.prdf_rejected
        ):
            try:
                struct = load_structure_from_file(f)
                if not struct.is_ordered:
                    with st.spinner(f"⚙️ Ordering disordered structure: {f.name} …"):
                        try:
                            ordered, scale, msg, order_warns, occ_rows = make_ordered_supercell(
                                struct
                            )
                        except Exception as order_exc:
                            st.session_state.prdf_rejected[f.name] = str(order_exc)
                            st.sidebar.error(f"❌ {f.name} rejected")
                            continue
                    st.session_state.ordered_structures[f.name] = ordered
                    st.session_state.disorder_sources[f.name] = struct
                    st.session_state.disorder_occ_info[f.name] = occ_rows
                    st.session_state.prdf_cif_ready[f.name] = False
                    full_msg = msg
                    if order_warns:
                        full_msg += "  \n⚠️ " + "  \n⚠️ ".join(order_warns)
                    st.session_state.disorder_messages[f.name] = full_msg
                    struct = ordered
                ok, reason = validate_structure(struct)
                if ok:
                    st.session_state.prdf_structures[f.name] = struct
                    st.sidebar.success(f"✅ {f.name}")
                else:
                    st.session_state.prdf_rejected[f.name] = reason
                    st.sidebar.error(f"❌ {f.name} rejected")
            except Exception as e:
                st.session_state.prdf_rejected[f.name] = str(e)
                st.sidebar.error(f"❌ {f.name}: {e}")
for name in list(st.session_state.prdf_structures.keys()):
    if name not in active_names:
        del st.session_state.prdf_structures[name]
        st.session_state.ordered_structures.pop(name, None)
        st.session_state.disorder_messages.pop(name, None)
        st.session_state.disorder_occ_info.pop(name, None)
        st.session_state.disorder_sources.pop(name, None)
        st.session_state.prdf_cif_ready.pop(name, None)
        if "prdf_results" in st.session_state:
            st.session_state.prdf_results.pop(name, None)
        st.session_state.prdf_download_ready = False
for name in list(st.session_state.prdf_rejected.keys()):
    if name not in active_names:
        del st.session_state.prdf_rejected[name]
# Subtle note about CIF files that had to be repaired while reading.
for _name, _notes in (st.session_state.get("cif_repair_notes") or {}).items():
    if _name in active_names:
        for _note in _notes:
            st.sidebar.caption(f"ℹ️ **{_name}**: {_note}")
active_exp_names = {f.name for f in uploaded_exp_files} if uploaded_exp_files else set()
if uploaded_exp_files:
    for ef in uploaded_exp_files:
        if ef.name not in st.session_state.prdf_experimental:
            try:
                xd, yd = load_experimental_data(ef)
                st.session_state.prdf_experimental[ef.name] = (xd, yd)
                st.sidebar.success(f"✅ Exp: {ef.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Exp {ef.name}: {e}")
for name in list(st.session_state.prdf_experimental.keys()):
    if name not in active_exp_names:
        del st.session_state.prdf_experimental[name]
structures: dict = st.session_state.prdf_structures
for name, reason in st.session_state.prdf_rejected.items():
    st.error(f"**{name}** was not loaded: {reason}")
for fname, msg in st.session_state.disorder_messages.items():
    if fname in structures:
        st.info(f"🔀 **{fname}**: {msg}")
if structures:
    st.sidebar.markdown(f"**{len(structures)} valid structure(s):**")
    for sname, s in structures.items():
        sp_set = set()
        for site in s:
            if site.is_ordered:
                sp_set.add(site.specie.symbol)
            else:
                for sp in site.species:
                    sp_set.add(sp.symbol)
        st.sidebar.caption(f"• {sname}  ({len(s)} atoms, {len(sp_set)} species)")
st.subheader(
    "⚙️ Settings",
    help="PRDF describes atom-pair distance distributions, providing insight into local environments and structural disorder. Values are unitless relative intensities. Peaks = preferred bonding distances; peak width = disorder.",
)
col_rmin, col_cut, col_bin = st.columns(3)
r_min = col_rmin.number_input(
    "Min cutoff (Å)", min_value=0.0, max_value=14.9, value=0.0, step=0.1, format="%.2f",
    help="Distances below this value are excluded from every PRDF/RDF trace.",
)
cutoff = col_cut.number_input(
    "Cutoff (Å)", min_value=1.0, max_value=15.0, value=10.0, step=0.5, format="%.1f"
)
bin_size = col_bin.number_input(
    "Bin size (Å)", min_value=0.005, max_value=2.0, value=0.1, step=0.005, format="%.3f"
)
if r_min >= cutoff:
    st.warning("Min cutoff must be smaller than Cutoff — Min cutoff will be ignored.")
    r_min = 0.0
st.markdown("#### Plot options")
col_ps, col_ls = st.columns(2)
plot_style = col_ps.radio(
    "Plot style", ["Smooth Curve", "Raw Data Points", "Bars (Histogram)"], index=0
)
line_style = col_ls.radio("Marker style", ["Lines Only", "Lines + Markers"], index=0)
sigma, sg_win, sg_ord, spline_pts, bar_width_factor = (1.5, 11, 3, 300, 0.8)
smoothing_method = "Gaussian"
# Smoothing choice, its parameters, and the calculation method share one row.
col_sm, col_sp, col_calc = st.columns([2, 2, 3])
if plot_style == "Smooth Curve":
    smoothing_method = col_sm.radio(
        "Smoothing method", ["Gaussian", "Savitzky-Golay", "Cubic Spline"], index=0, horizontal=True
    )
    if smoothing_method == "Gaussian":
        sigma = col_sp.slider("Gaussian σ", 0.5, 5.0, 1.5, 0.1)
    elif smoothing_method == "Savitzky-Golay":
        sg_win = col_sp.slider("Window length (odd)", 5, 21, 11, 2)
        sg_ord = col_sp.slider("Polynomial order", 2, 5, 3, 1)
    else:
        spline_pts = col_sp.slider("Interpolation points", 100, 600, 300, 50)
elif plot_style == "Bars (Histogram)":
    bar_width_factor = col_sm.slider("Bar width factor", 0.1, 2.0, 0.8, 0.1)

with col_calc:
    st.markdown("#### Calculation method")
    _unavailable = [b for b in BACKEND_ORDER if not BACKEND_AVAILABLE[b][0]]
    # A backend is offered only if it can handle every loaded structure. The caps
    # differ by an order of magnitude, so a large upload silently narrows the
    # choice rather than failing mid-calculation.
    _largest = max((len(s) for s in structures.values()), default=0)
    _too_big = {b: _largest > BACKEND_MAX_ATOMS[b] for b in BACKEND_ORDER}
    _usable = [b for b in BACKEND_ORDER if BACKEND_AVAILABLE[b][0] and not _too_big[b]]
    _label_of = {b: b for b in BACKEND_ORDER}
    with st.expander("Size limits per backend", expanded=False):
        st.markdown(
            "\n".join(
                f"- **{b}** — up to **{BACKEND_MAX_ATOMS[b]:,} atoms**"
                + ("  ← too small for this upload" if _too_big[b] else "")
                for b in BACKEND_ORDER
            )
            + "\n\nThe caps are set by peak memory."
        )
    if _largest and any(_too_big.values()):
        _blocked = [b for b in BACKEND_ORDER if _too_big[b] and BACKEND_AVAILABLE[b][0]]
        st.info(
            f"Largest uploaded structure has **{_largest:,} atoms**, so "
            + ", ".join(f"**{b}** (max {BACKEND_MAX_ATOMS[b]:,})" for b in _blocked)
            + (" is" if len(_blocked) == 1 else " are")
            + " not offered below."
        )
    _method_options = [_label_of[b] for b in _usable]
    if len(_usable) > 1:
        _method_options.append(
            "All three (compare)" if len(_usable) == 3 else f"All {len(_usable)} (compare)"
        )
    if not _method_options:
        st.error(
            "No backend can handle this structure. Install rdfrust for the largest limit, or "
            "reduce the structure size."
        )
        st.stop()
    _method_choice = st.radio(
        "Backend",
        _method_options,
        index=0,
        horizontal=True,
        key="prdf_backend_choice",
        help=(
            "All three compute the same quantity on the same bins.\n\n"
            "• **RDF-Rust** – Rust backend, the default. Streams pairs into histograms, so "
            "its memory stays near 2 MB at any size, and it is the only one that handles "
            "partial occupancies without building a supercell.\n\n"
            "• **Vesin** – C++ neighbour list with vectorised NumPy histograms.\n\n"
            "• **Matminer** – the reference implementation.\n\n"
            "• **All three** – runs each one, reports time and memory for each, and overlays "
            "the curves so you can see how closely they agree."
        ),
    )
    _choice_map = {_label_of[b]: [b] for b in _usable}
    if len(_usable) > 1:
        _choice_map[_method_options[-1]] = list(_usable)
    selected_backends = [b for b in _choice_map[_method_choice] if BACKEND_AVAILABLE[b][0]]
    _missing = [
        b
        for b in BACKEND_ORDER
        if not BACKEND_AVAILABLE[b][0] and _method_choice == _method_options[-1]
    ]
    if _missing:
        _how = {
            BACKEND_VESIN: "pip install vesin",
            BACKEND_RDFRUST: "pip install rdf-rust",
        }
        for _b in _missing:
            st.warning(
                f"**{_b}** is not installed, so it will be skipped. Install it with "
                f"`{_how.get(_b, 'pip install ' + _b)}`."
            )
    if not selected_backends:
        st.error(
            "No calculation backend is available. Install at least one — RDF-Rust is "
            "recommended, since it has the highest size limit and the lowest memory use."
        )
        st.stop()
    compare_mode = len(selected_backends) > 1
    if compare_mode:
        st.caption(
            f"Comparison mode: every structure is computed {len(selected_backends)} times "
            "— once per backend. Timings and peak memory are measured separately and shown "
            "in the **⚖️ Method Comparison** tab."
        )

CITATIONS = {
    "xrdlicious": (
        "**XRDlicious** — this platform:  \n"
        "  [LEBEDA, Miroslav, et al. *XRDlicious: an interactive web-based platform for "
        "online calculation of diffraction patterns and radial distribution functions from "
        "crystal structures.* Journal of Applied Crystallography, 2025, 58.5.]"
        "(https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)"
    ),
    BACKEND_MATMINER: (
        "**Matminer** — used for this calculation:  \n"
        "  [WARD, Logan, et al. *Matminer: An open source toolkit for materials data "
        "mining.* Computational Materials Science, 2018, 152: 60–69.]"
        "(https://www.sciencedirect.com/science/article/pii/S0927025618303252)"
    ),
    BACKEND_VESIN: (
        "**Vesin** — used for this calculation:  \n"
        "  [BIGI, Filippo; ABBOTT, Joseph W., et al. *metatensor and metatomic: "
        "Foundational libraries for interoperable atomistic machine learning.* "
        "The Journal of Chemical Physics, 2026, 164.6: 064113.]"
        "(https://doi.org/10.1063/5.0304911)"
    ),
    BACKEND_RDFRUST: (
        "**RDF-Rust** — used for this calculation; it is part of the XRD-Rust project:  \n"
        "  [LEBEDA, Miroslav, et al. *Rust-accelerated powder X-ray diffraction simulation "
        "for high-throughput and machine-learning-driven materials science.* "
        "Journal of Applied Crystallography, 2026, 59.4: 1344–1350.]"
        "(https://journals.iucr.org/j/issues/2026/04/00/hat5023/index.html)"
    ),
}
with _cite_slot:
    with st.expander("📖 How to **Cite**", expanded=False):
        st.markdown("### How to cite")
        st.markdown(
            "Please cite **XRDlicious** together with whichever calculation backend you "
            "used — the list below follows the method selected above."
        )
        _entries = [CITATIONS["xrdlicious"]] + [
            CITATIONS[b] for b in BACKEND_ORDER if b in selected_backends
        ]
        st.markdown("\n\n".join(f"- {e}" for e in _entries))
        _unused = [b for b in BACKEND_ORDER if b not in selected_backends]
        if _unused:
            st.caption(
                "Not currently selected, so not listed: "
                + ", ".join(_unused)
                + ". Switch the calculation method above and this list updates."
            )

normalize_to_max = st.checkbox(
    "📐 Normalize each trace to its maximum (max = 1)",
    value=False,
    help="When enabled, every calculated PRDF/RDF trace and every experimental data set is independently divided by its own maximum value, so all profiles share the same [0, 1] y-scale for easy shape comparison.",
)

_has_disorder = bool(st.session_state.get("disorder_sources"))
n_realisations = 1
if _has_disorder:
    st.markdown("#### Disorder sampling")
    # In comparison mode every backend runs, so the slowest one sets the cap.
    _max_real = min((BACKEND_MAX_REALISATIONS[b] for b in selected_backends), default=1)
    # Streamlit rejects a stored value above max_value, which happens when the
    # method is switched after a large count was set.
    if int(st.session_state.get("prdf_n_realisations", 1) or 1) > _max_real:
        st.session_state["prdf_n_realisations"] = _max_real
    n_realisations = int(
        st.number_input(
            "Random realisations to average",
            min_value=1,
            max_value=_max_real,
            value=1,
            step=1,
            key="prdf_n_realisations",
            help=(
                "A vacancy supercell fills the shared sites at random, so one of them carries "
                "the scatter of that particular arrangement. Averaging several independent "
                "fillings reduces that scatter as 1/sqrt(N).\n\n"
                "It only helps once the supercell edge exceeds the PRDF cutoff. Below that "
                "every atom sees its own periodic images -- the same atom, so perfectly "
                "correlated -- which biases every realisation identically and no amount of "
                "averaging removes it. The supercell is therefore already made as large as "
                "the site cap allows.\n\n"
                "The ceiling depends on the calculation method, because each "
                "realisation is a full extra calculation: "
                + ", ".join(
                    f"{b} up to {BACKEND_MAX_REALISATIONS[b]}" for b in BACKEND_ORDER
                )
                + "."
            ),
        )
    )
    if _max_real < max(BACKEND_MAX_REALISATIONS.values()):
        _slowest = min(selected_backends, key=lambda b: BACKEND_MAX_REALISATIONS[b])
        st.caption(
            f"Capped at **{_max_real}** by {_slowest}. Select **RDF-Rust** on its own to "
            f"average up to **{BACKEND_MAX_REALISATIONS[BACKEND_RDFRUST]}** realisations."
        )
    if n_realisations > 1:
        st.caption(
            f"Each disordered structure is filled **{n_realisations} times** with different "
            "random arrangements and the PRDFs averaged. The shaded band on the plots is the "
            "spread between realisations. Ordered structures are unaffected."
        )



def apply_smoothing(x_arr, y_arr):
    x_arr, y_arr = (np.array(x_arr), np.array(y_arr))
    if plot_style != "Smooth Curve":
        return (x_arr, y_arr)
    if smoothing_method == "Gaussian":
        return (x_arr, smooth_gaussian(y_arr, sigma))
    elif smoothing_method == "Savitzky-Golay":
        return (x_arr, smooth_savgol(y_arr, sg_win, sg_ord))
    else:
        return smooth_spline(x_arr, y_arr, spline_pts)


def _hex_to_rgba(color, alpha):
    c = color.lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def add_trace(fig, x, y, name, color, dash="solid", spread=None):
    if normalize_to_max:
        # The band has to be scaled by the same factor as the curve, so it is
        # divided by the *curve's* maximum rather than its own.
        peak = float(np.max(y)) if len(y) and float(np.max(y)) > 0 else 1.0
        if spread is not None:
            spread = np.asarray(spread) / peak
        y = _norm(y)
    if spread is not None and np.any(np.asarray(spread) > 0):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        sd = np.asarray(spread)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_arr, x_arr[::-1]]),
                y=np.concatenate([y_arr + sd, np.maximum(y_arr - sd, 0)[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.18),
                line=dict(width=0),
                hoverinfo="skip",
                name=f"{name} (spread)",
                showlegend=True,
            )
        )
    mode = "lines" if line_style == "Lines Only" else "lines+markers"
    if plot_style == "Bars (Histogram)":
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=name,
                marker=dict(color=color, line=dict(width=0)),
                width=bin_size * bar_width_factor,
                opacity=0.75,
            )
        )
    else:
        xp, yp = apply_smoothing(x, y)
        if plot_style == "Smooth Curve":
            x_arr_raw, y_arr_raw = (np.array(x), np.array(y))
            x_stems, y_stems = ([], [])
            for xi, yi in zip(x_arr_raw, y_arr_raw):
                x_stems.extend([xi, xi, None])
                y_stems.extend([0, yi, None])
            fig.add_trace(
                go.Scatter(
                    x=x_stems,
                    y=y_stems,
                    mode="lines",
                    name=f"{name} (raw)",
                    line=dict(color=color, width=1),
                    opacity=0.35,
                    showlegend=True,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xp,
                y=yp,
                mode=mode,
                name=name,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=7) if "markers" in mode else dict(),
            )
        )


def add_experimental_traces(fig, selected_exp_names, color_offset=0):
    dash_styles_exp = ["dot", "dashdot", "longdash", "longdashdot"]
    for ei, ename in enumerate(selected_exp_names):
        xd, yd = st.session_state.prdf_experimental[ename]
        if normalize_to_max:
            yd = _norm(yd)
        color_idx = (color_offset + ei) % len(COLORS)
        dash_exp = dash_styles_exp[ei % len(dash_styles_exp)]
        short = ename if len(ename) <= 35 else ename[:32] + "…"
        fig.add_trace(
            go.Scatter(
                x=xd,
                y=yd,
                mode="lines+markers" if line_style == "Lines + Markers" else "lines",
                name=f"Exp: {short}",
                line=dict(color=COLORS[color_idx], width=2, dash=dash_exp),
                marker=(
                    dict(symbol="circle-open", size=6)
                    if line_style == "Lines + Markers"
                    else dict()
                ),
            )
        )


def make_layout(title, barmode=None):
    ylabel = "Pair correlation function g(r)" if not normalize_to_max else "Normalized intensity"
    d = dict(
        title=dict(text=title, font=FONT),
        xaxis=dict(title=dict(text="Distance (Å)", font=FONT), tickfont=FONT),
        yaxis=dict(title=dict(text=ylabel, font=FONT), tickfont=FONT, range=[0, None]),
        hovermode="x",
        font=FONT,
        hoverlabel=dict(font=FONT),
        legend=dict(
            orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, font=dict(size=18)
        ),
    )
    if barmode:
        d["barmode"] = barmode
    return d


if "prdf_results" not in st.session_state:
    st.session_state.prdf_results = {}
if "prdf_do_calc" not in st.session_state:
    st.session_state.prdf_do_calc = False
if "prdf_download_ready" not in st.session_state:
    st.session_state.prdf_download_ready = False
if "comp_mode" not in st.session_state:
    st.session_state.comp_mode = "One pair across structures"


def trigger_calculation():
    st.session_state.prdf_do_calc = True
    st.session_state.prdf_results = {}
    st.session_state.prdf_download_ready = False


def prepare_downloads():
    st.session_state.prdf_download_ready = True


st.button(
    "▶️  Calculate RDF", on_click=trigger_calculation, type="primary", disabled=len(structures) == 0
)
if not structures:
    st.warning("⬆️ Please upload at least one valid structure file in the sidebar.")
_calc_status_slot = st.empty()
if st.session_state.prdf_do_calc and structures:
    struct_items = list(structures.items())
    with _calc_status_slot.container():
        progress_bar = st.progress(0, text="Starting …")
        calc_errors = []
        for s_idx, (fname, mg_struct) in enumerate(struct_items):
            progress_bar.progress(s_idx / len(struct_items), text=f"Processing {fname} …")
            # For a disordered upload, extra realisations differ only in how the
            # shared sites were filled; realisation 1 is the structure already
            # built at upload time, so it is reused rather than rebuilt.
            realisations = [mg_struct]
            source = st.session_state.disorder_sources.get(fname)
            if n_realisations > 1 and source is not None:
                for k in range(1, n_realisations):
                    try:
                        extra, _sc, _msg, _w, _rows = make_ordered_supercell(source, seed=42 + k)
                        realisations.append(extra)
                    except Exception as e:
                        calc_errors.append((f"{fname} [realisation {k + 1}]", str(e)))
            n_real = len(realisations)

            per_backend, perf = {}, {}
            for b_name in selected_backends:
                # Guard again here: the selection was made against the largest
                # structure, but a file can be swapped after the radio is set.
                if len(mg_struct) > BACKEND_MAX_ATOMS[b_name]:
                    calc_errors.append((
                        f"{fname} [{b_name}]",
                        f"{len(mg_struct):,} atoms exceeds this backend's limit of "
                        f"{BACKEND_MAX_ATOMS[b_name]:,}",
                    ))
                    continue
                progress_bar.progress(
                    s_idx / len(struct_items), text=f"Processing {fname} – {b_name} …"
                )
                try:
                    def _run(bn=b_name, reals=realisations):
                        return [
                            PRDF_BACKENDS[bn](s, cutoff, bin_size, r_min) for s in reals
                        ]

                    outs, secs, mem = _measure(_run)
                    per_backend[b_name] = _average_realisations(outs)
                    perf[b_name] = {
                        "seconds": secs,
                        "memory_mb": mem,
                        "atoms": len(mg_struct),
                        "realisations": n_real,
                    }
                except Exception as e:
                    calc_errors.append((f"{fname} [{b_name}]", str(e)))
            if per_backend:
                # The first selected backend drives the existing tabs; the rest
                # ride along for the comparison view.
                primary = next(b for b in BACKEND_ORDER if b in per_backend)
                st.session_state.prdf_results[fname] = {
                    **per_backend[primary],
                    "backends": per_backend,
                    "perf": perf,
                    "primary": primary,
                }
        progress_bar.progress(1.0, text="Done!")
    st.session_state.prdf_do_calc = False
    _calc_status_slot.empty()
    if calc_errors:
        for fname, msg in calc_errors:
            st.toast(f"Error processing {fname}: {msg}", icon="❌")
    else:
        n = len(st.session_state.prdf_results)
        st.toast(f"Calculated PRDF for {n} structure(s).", icon="✅")
results: dict = st.session_state.prdf_results
if results:
    st.divider()
    all_pairs = sorted(
        {pair for r in results.values() for pair in r["prdf_dict"]}, key=lambda p: (p[0], p[1])
    )
    has_ordered = bool(st.session_state.ordered_structures)
    _any_multi = any(len(r.get("backends", {})) > 1 for r in results.values())
    _tab_labels = ["📊 Individual PRDFs", "🔀 Comparison", "📈 Total RDF", "💾 Download"]
    if _any_multi:
        _tab_labels.append("⚖️ Method Comparison")
    if has_ordered:
        _tab_labels.append("🧩 Vacancy Supercells")
    _tabs = st.tabs(_tab_labels)
    tab_indiv = _tabs[0]
    tab_comp = _tabs[1]
    tab_total = _tabs[2]
    tab_dl = _tabs[3]
    _next = 4
    tab_methods = None
    if _any_multi:
        tab_methods = _tabs[_next]
        _next += 1
    tab_ord = _tabs[_next] if has_ordered else None
    with tab_indiv:
        st.markdown("### Per-structure (P)RDF plots")
        fname_list = list(results.keys())
        selected_struct = (
            fname_list[0]
            if len(fname_list) == 1
            else st.selectbox("Select structure:", fname_list, key="indiv_select")
        )
        if selected_struct in st.session_state.disorder_messages:
            st.info(f"ℹ️ {st.session_state.disorder_messages[selected_struct]}")
        res = results[selected_struct]
        pairs = list(res["prdf_dict"].keys())
        st.markdown(f"**{selected_struct}** — {len(pairs)} element pair(s)")
        layout_mode = st.radio(
            "Layout",
            ["Separate plot per pair", "All pairs in one plot"],
            horizontal=True,
            key="indiv_layout",
        )
        _spread = res.get("spread") or {}
        _n_real = res.get("n_realisations", 1)
        if _n_real > 1:
            st.caption(
                f"Averaged over **{_n_real} random fillings** of the disordered sites; the "
                "shaded band is one standard deviation between them."
            )
        if layout_mode == "All pairs in one plot":
            fig = go.Figure()
            for idx, pair in enumerate(pairs):
                add_trace(
                    fig,
                    res["dist_dict"][pair],
                    res["prdf_dict"][pair],
                    f"{pair[0]}–{pair[1]}",
                    COLORS[idx % len(COLORS)],
                    spread=_spread.get(pair),
                )
            fig.update_layout(
                **make_layout(
                    f"PRDF – {selected_struct}: all pairs",
                    barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                )
            )
            st.plotly_chart(fig, width="stretch")
        else:
            for idx, pair in enumerate(pairs):
                fig = go.Figure()
                add_trace(
                    fig,
                    res["dist_dict"][pair],
                    res["prdf_dict"][pair],
                    f"{pair[0]}–{pair[1]}",
                    COLORS[idx % len(COLORS)],
                    spread=_spread.get(pair),
                )
                fig.update_layout(**make_layout(f"PRDF: {pair[0]}–{pair[1]}  |  {selected_struct}"))
                st.plotly_chart(fig, width="stretch")
    with tab_comp:
        st.markdown("### Compare PRDFs across structures")
        if len(results) < 1:
            st.info("Upload and calculate at least **1 structure** to use the comparison view.")
        else:
            comp_mode = st.radio(
                "Comparison mode",
                ["One pair across structures", "Multiple pairs in one plot"],
                horizontal=True,
                key="comp_mode",
                help=(
                    "• **One pair across structures**: pick a single element pair and "
                    "overlay it from every selected structure.\n\n"
                    "• **Multiple pairs in one plot**: pick one structure and overlay "
                    "any number of element pairs from it."
                ),
            )
            if comp_mode == "One pair across structures":
                if len(results) < 2:
                    st.info("Upload and calculate at least **2 structures** to use this mode.")
                else:
                    col_pair, col_structs = st.columns([1, 2])
                    with col_pair:
                        pair_labels = [f"{p[0]}–{p[1]}" for p in all_pairs]
                        chosen_label = st.selectbox("Element pair:", pair_labels, key="comp_pair")
                        chosen_pair = all_pairs[pair_labels.index(chosen_label)]
                    with col_structs:
                        all_names = list(results.keys())
                        chosen_structs = st.multiselect(
                            "Structures to overlay:", all_names, default=all_names, key="comp_structs"
                        )
                    if not chosen_structs:
                        st.warning("Select at least one structure.")
                    else:
                        fig_comp = go.Figure()
                        skipped = []
                        for s_idx, sname in enumerate(chosen_structs):
                            r = results[sname]
                            if chosen_pair not in r["prdf_dict"]:
                                skipped.append(sname)
                                continue
                            short = sname if len(sname) <= 35 else sname[:32] + "…"
                            add_trace(
                                fig_comp,
                                r["dist_dict"][chosen_pair],
                                r["prdf_dict"][chosen_pair],
                                short,
                                COLORS[s_idx % len(COLORS)],
                            )
                        if skipped:
                            st.warning(
                                f"Pair **{chosen_label}** not present in: "
                                + ", ".join((f"*{s}*" for s in skipped))
                                + " — skipped."
                            )
                        if st.session_state.prdf_experimental:
                            add_experimental_traces(
                                fig_comp,
                                list(st.session_state.prdf_experimental.keys()),
                                color_offset=len(chosen_structs),
                            )
                        fig_comp.update_layout(
                            **make_layout(
                                f"PRDF comparison: {chosen_label}",
                                barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                            )
                        )
                        st.plotly_chart(fig_comp, width="stretch")
                        st.markdown("#### Total RDF comparison")
                        fig_tot = go.Figure()
                        for s_idx, sname in enumerate(chosen_structs):
                            r = results[sname]
                            bins = sorted(r["global_rdf"].keys())
                            vals = [r["global_rdf"][b] for b in bins]
                            short = sname if len(sname) <= 35 else sname[:32] + "…"
                            add_trace(
                                fig_tot,
                                bins,
                                vals,
                                short,
                                COLORS[s_idx % len(COLORS)],
                            )
                        if st.session_state.prdf_experimental:
                            add_experimental_traces(
                                fig_tot,
                                list(st.session_state.prdf_experimental.keys()),
                                color_offset=len(chosen_structs),
                            )
                        fig_tot.update_layout(
                            **make_layout(
                                "Total RDF comparison",
                                barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                            )
                        )
                        st.plotly_chart(fig_tot, width="stretch")
            else:
                combo_options = []
                combo_lookup = {}
                for sname in results.keys():
                    short = sname if len(sname) <= 30 else sname[:27] + "…"
                    for pair in sorted(
                        results[sname]["prdf_dict"].keys(),
                        key=lambda p: (p[0], p[1]),
                    ):
                        label = f"{short} | {pair[0]}–{pair[1]}"
                        combo_options.append(label)
                        combo_lookup[label] = (sname, pair)
                if not combo_options:
                    st.info("No PRDF data available to plot.")
                else:
                    chosen_combos = st.multiselect(
                        "Structure–pair combinations to overlay:",
                        combo_options,
                        default=combo_options,
                        key="comp_multi_combos",
                        help=(
                            "Each entry is a (structure, element pair) pair. "
                            "Pick any number to overlay them in one plot."
                        ),
                    )
                    if not chosen_combos:
                        st.warning("Select at least one structure–pair combination.")
                    else:
                        fig_comp = go.Figure()
                        for c_idx, lbl in enumerate(chosen_combos):
                            sname, pair = combo_lookup[lbl]
                            r = results[sname]
                            add_trace(
                                fig_comp,
                                r["dist_dict"][pair],
                                r["prdf_dict"][pair],
                                lbl,
                                COLORS[c_idx % len(COLORS)],
                            )
                        if st.session_state.prdf_experimental:
                            add_experimental_traces(
                                fig_comp,
                                list(st.session_state.prdf_experimental.keys()),
                                color_offset=len(chosen_combos),
                            )
                        fig_comp.update_layout(
                            **make_layout(
                                "PRDF overlay – multiple structures & pairs",
                                barmode="overlay" if plot_style == "Bars (Histogram)" else None,
                            )
                        )
                        st.plotly_chart(fig_comp, width="stretch")
    with tab_total:
        st.markdown("### Total RDF – individual structures")
        for s_idx, (fname, r) in enumerate(results.items()):
            bins = sorted(r["global_rdf"].keys())
            vals = [r["global_rdf"][b] for b in bins]
            fig_g = go.Figure()
            add_trace(fig_g, bins, vals, fname, COLORS[s_idx % len(COLORS)])
            if st.session_state.prdf_experimental:
                add_experimental_traces(
                    fig_g, list(st.session_state.prdf_experimental.keys()), color_offset=s_idx + 1
                )
            fig_g.update_layout(**make_layout(f"Total RDF – {fname}"))
            st.plotly_chart(fig_g, width="stretch")
    with tab_dl:
        st.markdown("### Download results as CSV")
        st.button("Prepare CSV files", on_click=prepare_downloads, type="secondary")
        if st.session_state.prdf_download_ready:
            for fname, r in results.items():
                safe = fname.rsplit(".", 1)[0].replace(" ", "_")
                st.markdown(f"#### {fname}")
                for pair, intensities in r["prdf_dict"].items():
                    pair_label = f"{pair[0]}_{pair[1]}"
                    df = pd.DataFrame(
                        {"Distance_Ang": r["dist_dict"][pair], "PRDF_Intensity": intensities}
                    )
                    st.download_button(
                        label=f"⬇️  {pair[0]}–{pair[1]} PRDF",
                        data=df.to_csv(index=False).encode(),
                        file_name=f"{safe}_PRDF_{pair_label}.csv",
                        mime="text/csv",
                        key=f"dl_{safe}_{pair_label}",
                    )
                bins = sorted(r["global_rdf"].keys())
                df_g = pd.DataFrame(
                    {"Distance_Ang": bins, "Total_RDF": [r["global_rdf"][b] for b in bins]}
                )
                st.download_button(
                    label="⬇️  Total RDF",
                    data=df_g.to_csv(index=False).encode(),
                    file_name=f"{safe}_Total_RDF.csv",
                    mime="text/csv",
                    key=f"dl_total_{safe}",
                )
    if tab_methods is not None:
        with tab_methods:
            st.markdown("### Method comparison — speed, memory, and agreement")
            st.info(
                "Each selected backend computed the **same** PRDF on the **same** bins. "
                "Timings are wall clock for the calculation only (parsing and plotting "
                "excluded). Memory is the peak increase in resident set size during the "
                "call, so allocations inside Rust and C++ are included."
            )
            _multi = {f: r for f, r in results.items() if len(r.get("backends", {})) > 1}
            if not _multi:
                st.warning("Only one backend was run — nothing to compare.")
            else:
                m_fnames = list(_multi.keys())
                m_sel = (
                    m_fnames[0]
                    if len(m_fnames) == 1
                    else st.selectbox("Structure:", m_fnames, key="methods_select")
                )
                m_res = _multi[m_sel]
                perf = m_res.get("perf", {})
                backends_here = [b for b in BACKEND_ORDER if b in m_res["backends"]]

                st.markdown("#### Performance")
                rows = []
                fastest = min(
                    (perf[b]["seconds"] for b in backends_here if b in perf), default=None
                )
                for b in backends_here:
                    p = perf.get(b)
                    if not p:
                        continue
                    mem = p["memory_mb"]
                    rows.append(
                        {
                            "Backend": b,
                            "Atoms": p["atoms"],
                            "Time (s)": round(p["seconds"], 4),
                            "Relative speed": (
                                "fastest"
                                if fastest and abs(p["seconds"] - fastest) < 1e-12
                                else f"{p['seconds'] / fastest:.1f}× slower"
                                if fastest
                                else "—"
                            ),
                            "Peak memory (MB)": "n/a" if mem != mem else round(mem, 1),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                cols = st.columns(len(rows)) if rows else []
                for c, row in zip(cols, rows):
                    c.metric(
                        row["Backend"],
                        f"{row['Time (s)']:.4f} s",
                        (
                            None
                            if row["Peak memory (MB)"] == "n/a"
                            else f"{row['Peak memory (MB)']} MB peak"
                        ),
                        delta_color="off",
                    )

                st.markdown("#### Do the curves agree?")
                ref_name = BACKEND_MATMINER if BACKEND_MATMINER in backends_here else backends_here[0]
                ref = m_res["backends"][ref_name]
                agree_rows = []
                for b in backends_here:
                    if b == ref_name:
                        continue
                    other = m_res["backends"][b]
                    worst, worst_pair = 0.0, None
                    for pair, g in ref["prdf_dict"].items():
                        if pair not in other["prdf_dict"]:
                            continue
                        d = float(np.max(np.abs(np.asarray(other["prdf_dict"][pair]) - g)))
                        if d > worst:
                            worst, worst_pair = d, pair
                    scale = max(
                        (float(np.max(np.abs(g))) for g in ref["prdf_dict"].values()), default=1.0
                    )
                    agree_rows.append(
                        {
                            "Backend": b,
                            f"Max |difference| vs {ref_name}": f"{worst:.3e}",
                            "Relative to largest peak": f"{worst / scale:.2e}" if scale else "—",
                            "Worst pair": f"{worst_pair[0]}–{worst_pair[1]}" if worst_pair else "—",
                        }
                    )
                if agree_rows:
                    st.dataframe(pd.DataFrame(agree_rows), width="stretch", hide_index=True)
                    st.caption(
                        "A difference around 1e-15 relative is double-precision noise and means "
                        "the curves are identical. Larger values are almost always distances "
                        "sitting exactly on a bin edge, where a few ulp decide which of two "
                        "neighbouring bins a count lands in — the total number of pairs is "
                        "unchanged."
                    )

                st.markdown("#### Curves, all methods overlaid")
                m_pairs = sorted(ref["prdf_dict"].keys())
                m_layout = st.radio(
                    "Layout",
                    ["Separate plot per pair", "All pairs in one plot"],
                    horizontal=True,
                    key="methods_layout",
                )
                # Widths are fixed per backend rather than by position: the
                # reference is drawn thick and pale underneath, and the faster
                # backends sit thin on top, so agreement shows as a thin line
                # nested inside a thick one whatever the selection order.
                _widths = {BACKEND_MATMINER: 7.0, BACKEND_VESIN: 3.2, BACKEND_RDFRUST: 1.6}

                def _overlay(fig, pair, show_legend, label_pair=False):
                    for b in backends_here:
                        data = m_res["backends"][b]
                        if pair not in data["prdf_dict"]:
                            continue
                        y = data["prdf_dict"][pair]
                        if normalize_to_max:
                            y = _norm(y)
                        name = f"{pair[0]}–{pair[1]} · {b}" if label_pair else b
                        fig.add_trace(
                            go.Scatter(
                                x=data["dist_dict"][pair],
                                y=y,
                                mode="lines",
                                name=name,
                                legendgroup=b,
                                showlegend=show_legend,
                                line=dict(
                                    color=BACKEND_COLORS[b],
                                    width=_widths.get(b, 2),
                                    dash=BACKEND_DASH[b],
                                ),
                                opacity=0.45 if _widths.get(b, 2) > 5 else 1.0,
                            )
                        )

                if m_layout == "All pairs in one plot":
                    fig_m = go.Figure()
                    for idx, pair in enumerate(m_pairs):
                        _overlay(fig_m, pair, show_legend=(idx == 0), label_pair=True)
                    fig_m.update_layout(**make_layout(f"All pairs, all methods — {m_sel}"))
                    st.plotly_chart(fig_m, width="stretch")
                else:
                    for pair in m_pairs:
                        fig_m = go.Figure()
                        _overlay(fig_m, pair, show_legend=True)
                        fig_m.update_layout(
                            **make_layout(f"{pair[0]}–{pair[1]} — all methods — {m_sel}")
                        )
                        st.plotly_chart(fig_m, width="stretch")

                st.markdown("#### Total RDF, all methods overlaid")
                fig_t = go.Figure()
                for b in backends_here:
                    g = m_res["backends"][b]["global_rdf"]
                    bins = sorted(g.keys())
                    vals = [g[k] for k in bins]
                    if normalize_to_max:
                        vals = _norm(vals)
                    fig_t.add_trace(
                        go.Scatter(
                            x=bins,
                            y=vals,
                            mode="lines",
                            name=b,
                            line=dict(
                                color=BACKEND_COLORS[b],
                                width=_widths.get(b, 2),
                                dash=BACKEND_DASH[b],
                            ),
                            opacity=0.45 if _widths.get(b, 2) > 5 else 1.0,
                        )
                    )
                fig_t.update_layout(**make_layout(f"Total RDF, all methods — {m_sel}"))
                st.plotly_chart(fig_t, width="stretch")

    if has_ordered and tab_ord is not None:
        with tab_ord:
            st.markdown("### Vacancy supercells (random site assignment)")
            st.info(
                f"These supercells were generated from disordered CIF files. The number of atoms per species approximates the CIF occupancies; vacant sites are chosen **randomly** (fixed seed for reproducibility). This is **not** a crystallographically ordered structure — it is a single randomised realisation of the disordered occupancy. Atom cap: **{MAX_SUPERCELL_ATOMS} sites**."
            )
            for fname, ordered_struct in st.session_state.ordered_structures.items():
                if fname not in results:
                    continue
                safe = fname.rsplit(".", 1)[0].replace(" ", "_")
                st.markdown(f"#### {fname}")
                if fname in st.session_state.disorder_messages:
                    st.success(st.session_state.disorder_messages[fname])
                occ_rows = st.session_state.disorder_occ_info.get(fname, [])
                # Needed further down for the archive README, and occ_rows can
                # legitimately be empty, so it is resolved before the branch.
                _cbrt = round(occ_rows[0]["n_copies"] ** (1 / 3)) if occ_rows else 1
                if occ_rows:
                    scale_used = occ_rows[0]["n_copies"]
                    cbrt = round(scale_used ** (1 / 3))
                    st.markdown("**Occupancy approximation — CIF vs. vacancy supercell:**")
                    table_md = "| Site(s) | Fractional coords | Element | CIF occupancy | Sites in supercell | Atoms assigned | Actual occupancy | Delta |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
                    for row in occ_rows:
                        delta = row["actual_occ"] - row["target_occ"]
                        sign = "+" if delta >= 0 else ""
                        flag = "OK" if abs(delta) < 0.02 else "~" if abs(delta) < 0.05 else "(!)"
                        table_md += f"| {row['site_idx']} | {row['frac']} | **{row['element']}** | {row['target_occ']:.4f} | {row['n_sites']} | {row['n_assigned']} | {row['actual_occ']:.4f} | {sign}{delta:.4f} {flag} |\n"
                    st.markdown(table_md)
                    st.caption(
                        f"Supercell scale: {cbrt}x{cbrt}x{cbrt} ({scale_used} copies of each original site). Atom cap: {MAX_SUPERCELL_ATOMS}. OK = |d| < 0.02  ~= |d| < 0.05  (!) = |d| >= 0.05. Original sites that share the same disordered composition are filled as a single pool, so one row covers all of them and \"Sites in supercell\" counts the whole group."
                    )
                _n_used = results[fname].get("n_realisations", 1)
                _per_real = results[fname].get("per_realisation") or []
                if _n_used > REALISATION_OVERLAY_MIN and _per_real:
                    st.markdown("**Individual realisations vs. their average**")
                    st.caption(
                        f"Each of the {_n_used} random fillings is drawn thin and pale; their "
                        "average is the bold line. The scatter between them is the uncertainty "
                        "a single vacancy supercell carries, and it shrinks as 1/√N with the "
                        "number averaged."
                    )
                    _rp = results[fname]["prdf_dict"]
                    _rd = results[fname]["dist_dict"]
                    _varying = sorted(
                        _rp.keys(),
                        key=lambda pr: -float(
                            np.max(np.asarray(results[fname].get("spread", {}).get(pr, [0.0])))
                        ),
                    )
                    _pair_lbl = [f"{a}–{b}" for a, b in _varying]
                    _chosen = st.selectbox(
                        "Element pair",
                        _pair_lbl,
                        key=f"real_pair_{safe}",
                        help="Pairs are ordered by how much they vary between realisations, "
                        "so the most arrangement-sensitive one comes first.",
                    )
                    _pair = _varying[_pair_lbl.index(_chosen)]
                    # Drawing 500 traces would bog the browser down without
                    # showing anything extra, so a spread-out sample is plotted
                    # while the average still uses every realisation.
                    _MAX_TRACES = 40
                    if len(_per_real) > _MAX_TRACES:
                        _idx = np.unique(
                            np.linspace(0, len(_per_real) - 1, _MAX_TRACES).astype(int)
                        )
                        st.caption(
                            f"Showing {len(_idx)} of the {len(_per_real)} realisations to keep "
                            "the plot responsive; the average and the statistics below use all "
                            f"{len(_per_real)}."
                        )
                    else:
                        _idx = np.arange(len(_per_real))
                    _fig_r = go.Figure()
                    for _i in _idx:
                        _real = _per_real[int(_i)]
                        if _pair not in _real:
                            continue
                        _fig_r.add_trace(
                            go.Scatter(
                                x=_rd[_pair],
                                y=_real[_pair],
                                mode="lines",
                                name=f"realisation {int(_i) + 1}",
                                line=dict(color="#8a8a85", width=1),
                                opacity=0.55,
                                legendgroup="reals",
                                showlegend=(int(_i) == int(_idx[0])),
                                hovertemplate="realisation %d<br>r=%%{x:.2f} Å<br>g=%%{y:.4f}<extra></extra>"
                                % (int(_i) + 1),
                            )
                        )
                    _fig_r.add_trace(
                        go.Scatter(
                            x=_rd[_pair],
                            y=_rp[_pair],
                            mode="lines",
                            name=f"average of {_n_used}",
                            line=dict(color="#8b0000", width=3),
                        )
                    )
                    _fig_r.update_layout(
                        **make_layout(f"{_pair[0]}–{_pair[1]} — {_n_used} realisations vs. average")
                    )
                    st.plotly_chart(_fig_r, width="stretch")
                    _sp = np.asarray(results[fname].get("spread", {}).get(_pair, [0.0]))
                    _mean = np.asarray(_rp[_pair])
                    _m = _mean > 0.02 * _mean.max() if _mean.max() > 0 else _mean > 0
                    if _m.any():
                        _rel = float(np.mean(_sp[_m] / _mean[_m])) * 100
                        st.caption(
                            f"**Spread** is the per-bin sample standard deviation "
                            f"(`numpy.std(..., ddof=1)`) of g(r) across the {_n_used} "
                            f"realisations, and the shaded band is mean ± 1 SD. For "
                            f"{_chosen} it averages **{_rel:.1f}%** of the signal — that is "
                            f"how far a *single* random filling typically sits from the mean. "
                            f"The mean of {_n_used} is more precise than that by √{_n_used}, "
                            f"so its own standard error is about "
                            f"**{_rel / np.sqrt(_n_used):.2f}%**."
                        )
                    st.markdown("---")
                if _n_used > 1:
                    st.info(
                        f"The PRDF shown elsewhere is the average of **{_n_used} random "
                        "fillings** of this supercell, so no single structure reproduces it. "
                        "Every realisation is offered below; they share the same cell and "
                        "composition and differ only in which site got which species "
                        "(realisation *k* uses seed 42+*k*-1)."
                    )
                _source = st.session_state.disorder_sources.get(fname)
                _variants = [("", ordered_struct)]
                if _n_used > 1 and _source is not None:
                    for _k in range(1, _n_used):
                        try:
                            _extra, *_ = make_ordered_supercell(_source, seed=42 + _k)
                            _variants.append((f"_realisation{_k + 1}", _extra))
                        except Exception:
                            pass
                st.markdown("**Download vacancy supercell:**")
                if len(_variants) == 1:
                    try:
                        st.download_button(
                            label="⬇️  Download vacancy supercell (CIF)",
                            data=str(CifWriter(_variants[0][1])).encode("utf-8"),
                            file_name=f"{safe}_vacancy_supercell.cif",
                            mime="chemical/x-cif",
                            key=f"dl_ordered_tab_{safe}",
                            type="primary",
                        )
                    except Exception as e:
                        st.warning(f"Could not export CIF for {fname}: {e}")
                else:
                    # One archive rather than N buttons. A README goes in with
                    # the CIFs so the files still make sense once they are
                    # unpacked somewhere else.
                    _buf = io.BytesIO()
                    _failed = []
                    try:
                        with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                            _names = []
                            for _k, (_suffix, _struct_v) in enumerate(_variants, start=1):
                                _member = f"{safe}_realisation{_k:02d}.cif"
                                try:
                                    _zf.writestr(_member, str(CifWriter(_struct_v)))
                                    _names.append((_member, _k, len(_struct_v)))
                                except Exception as e:
                                    _failed.append((_k, str(e)))
                            _readme = [
                                f"{fname} — vacancy supercells",
                                "",
                                f"{len(_names)} independent random fillings of the same "
                                f"{_cbrt}x{_cbrt}x{_cbrt} supercell.",
                                "They share one cell and one composition and differ only in "
                                "which site received which species.",
                                f"Realisation k was generated with seed {42} + k - 1, so the "
                                "set is reproducible.",
                                "",
                                "The PRDF shown in the app is the average over all of them; "
                                "no single file here reproduces it.",
                                "",
                                "file                                   realisation   atoms",
                            ]
                            for _member, _k, _n in _names:
                                _readme.append(f"{_member:38} {_k:>11}   {_n:>5}")
                            _zf.writestr("README.txt", "\n".join(_readme) + "\n")
                        st.download_button(
                            label=f"⬇️  Download all {len(_variants)} realisations (ZIP)",
                            data=_buf.getvalue(),
                            file_name=f"{safe}_vacancy_supercells_{len(_variants)}realisations.zip",
                            mime="application/zip",
                            key=f"dl_ordered_zip_{safe}",
                            type="primary",
                        )
                        st.caption(
                            f"{len(_variants)} CIF files plus a README describing how they "
                            "were generated."
                        )
                    except Exception as e:
                        st.warning(f"Could not build the archive for {fname}: {e}")
                    for _k, _msg in _failed:
                        st.warning(f"Realisation {_k} could not be exported: {_msg}")
                st.markdown("---")
                st.markdown("---")
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<hr style="border:none;height:5px;background-color:#8b0000;\n                 border-radius:6px;margin:0 0 12px 0;">',
    unsafe_allow_html=True,
)

st.markdown(
    "\n### Acknowledgments\n\nThis module uses several open-source tools. We gratefully acknowledge their authors:\n\n- **[RDF-Rust](https://github.com/bracerino/rdf-rust)**\n- **[Vesin](https://github.com/Luthaf/vesin)**\n- **[Matminer](https://github.com/hackingmaterials/matminer)**\n- **[Pymatgen](https://github.com/materialsproject/pymatgen)**\n- **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)**\n- **[Plotly](https://plotly.com)**\n- **[SciPy](https://scipy.org)**\n\n**XRDlicious (P)RDF module** is open-source and released under the\n[MIT License](https://github.com/bracerino/rdf-rust/blob/main/LICENSE).\n"
)
