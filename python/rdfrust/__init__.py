"""Fast partial and total radial distribution functions, with a Rust backend.

Partial occupancies are supported natively: a site may carry several species,
and the RDF is then the ensemble average over an ideally mixed lattice, with no
supercell approximation. Use :meth:`Structure.expand_occupancies` when an
explicit ordered structure is wanted instead.

Typical use::

    import rdfrust

    s = rdfrust.read("BaSrTiO3.cif")          # reports disorder on the console
    r, partials, total = s.prdf(cutoff=10.0, bin_size=0.1, n_threads=0)

    partials[("Ba", "O")]     # numpy array, g_BaO(r)
    total                     # number-weighted total RDF

`r` holds the left edge of each bin and matches
``numpy.arange(0, cutoff + bin_size, bin_size)[:-1]``.
"""

import os
import sys

from ._rdfrust import Structure, __version__, read_xyz_trajectory
from .trajectory import TrajectoryRDF, read_trajectory, trajectory_prdf

__all__ = [
    "Structure",
    "read_trajectory",
    "trajectory_prdf",
    "TrajectoryRDF",
    "read",
    "from_pymatgen",
    "describe_disorder",
    "NATIVE_FORMATS",
    "available_threads",
    "__version__",
]


def available_threads():
    """Worker threads used when ``n_threads=0`` (i.e. every core rayon sees)."""
    return Structure.available_threads()


def describe_disorder(structure, name=None, max_sites=12):
    """Human-readable account of a structure's partial occupancies.

    Returns an empty string for a fully ordered structure, so the caller can
    print it unconditionally and stay quiet when there is nothing to say.
    """
    summary = structure.disorder_summary()
    label = f"{name}: " if name else ""
    if not summary:
        return ""
    lines = [
        f"{label}{len(summary)} of {structure.n_sites} sites are partially occupied "
        f"({structure.n_atoms:g} atoms in a {structure.n_sites}-site cell)."
    ]
    for entry in summary[:max_sites]:
        fx, fy, fz = entry["frac_coords"]
        parts = ", ".join(
            f"{el} {occ:.4g}" for el, occ in sorted(entry["occupancies"].items())
        )
        vac = entry["vacancy"]
        if vac > 1e-6:
            parts += f", vacancy {vac:.4g}"
        lines.append(f"  site {entry['site_index']:>4}  ({fx:.4f}, {fy:.4f}, {fz:.4f})  {parts}")
    if len(summary) > max_sites:
        lines.append(f"  ... and {len(summary) - max_sites} more")
    lines.append(
        "  The RDF is weighted by these occupancies, which is the exact ensemble average "
        "over an ideally mixed lattice -- no supercell is built and no random seed is used."
    )
    return "\n".join(lines)


#: Extensions recognised natively, mapped to the parser they use.
NATIVE_FORMATS = {
    ".cif": "cif",
    ".vasp": "poscar",
    ".poscar": "poscar",
    ".contcar": "poscar",
    ".xyz": "xyz",
    ".extxyz": "xyz",
}


def _guess_format(path):
    """Format name for a path, or None if nothing native fits.

    POSCAR and CONTCAR are usually named without an extension at all, so the
    bare filename is checked as well.
    """
    base = os.path.basename(path)
    stem = base.upper().split(".")[0]
    if stem in ("POSCAR", "CONTCAR"):
        return "poscar"
    ext = os.path.splitext(base)[1].lower()
    return NATIVE_FORMATS.get(ext)


def read(path, symprec=1e-4, verbose=True, stream=None, fmt=None):
    """Read a structure file and report any partial occupancies.

    Formats
    -------
    Read natively: **CIF** (the only one that carries partial occupancies),
    **VASP POSCAR/CONTCAR**, and **extended XYZ** (which needs a
    ``Lattice="..."`` field, since an RDF requires a cell).

    Anything else is handed to pymatgen, then ASE, if either is installed —
    so LAMMPS data files, Quantum ESPRESSO input, XSF and the rest work when
    you have those packages, and raise a clear error when you do not.

    Parameters
    ----------
    path : str
        Path to the structure file.
    symprec : float
        Tolerance for merging symmetry images onto one site (CIF only).
    verbose : bool
        Print a note to `stream` when the file turns out to be disordered.
        Nothing is printed for an ordered structure.
    stream : file-like, optional
        Where the note goes; defaults to stderr so it does not contaminate
        piped data on stdout.
    fmt : str, optional
        Force a format ("cif", "poscar", "xyz") instead of guessing.
    """
    chosen = fmt or _guess_format(path)
    if chosen == "cif":
        structure = Structure.from_cif(path, symprec)
    elif chosen == "poscar":
        structure = Structure.from_poscar(path)
    elif chosen == "xyz":
        structure = Structure.from_xyz(path)
    else:
        structure = _read_via_fallback(path)
    if verbose:
        note = describe_disorder(structure, name=path)
        if note:
            print(note, file=stream if stream is not None else sys.stderr)
    return structure


def _read_via_fallback(path):
    """Last resort for formats rdfrust does not parse itself."""
    errors = []
    try:
        from pymatgen.core import Structure as _Pmg

        return from_pymatgen(_Pmg.from_file(path))
    except ImportError:
        errors.append("pymatgen is not installed")
    except Exception as exc:
        errors.append(f"pymatgen could not read it: {exc}")
    try:
        from ase.io import read as _ase_read
        from pymatgen.io.ase import AseAtomsAdaptor

        return from_pymatgen(AseAtomsAdaptor.get_structure(_ase_read(path)))
    except ImportError:
        errors.append("ASE is not installed")
    except Exception as exc:
        errors.append(f"ASE could not read it: {exc}")
    raise ValueError(
        f"cannot read {path!r}. rdfrust reads CIF, POSCAR/CONTCAR and extended XYZ "
        f"natively; anything else needs pymatgen or ASE installed. Tried: "
        + "; ".join(errors)
    )


def from_pymatgen(structure):
    """Convert a pymatgen Structure, keeping any partial occupancies."""
    matrix = [list(map(float, row)) for row in structure.lattice.matrix]
    frac, species = [], []
    for site in structure:
        frac.append([float(x) for x in site.frac_coords])
        occ = {}
        for sp, amount in site.species.items():
            symbol = getattr(sp, "symbol", None) or getattr(
                getattr(sp, "element", None), "symbol", None
            )
            if symbol is None:
                raise ValueError(f"cannot determine the element of species {sp!r}")
            occ[symbol] = occ.get(symbol, 0.0) + float(amount)
        if not occ:
            raise ValueError("encountered a site with no species")
        species.append(occ)
    return Structure(matrix, frac, species)
