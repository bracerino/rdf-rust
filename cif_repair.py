"""CIF repair used by prdf_app.py.

Copied from XRDlicious's helpers.py so the (P)RDF app runs on its own. Some
CIF files list *all* atoms of the unit cell but keep the symmetry operations
of the original space group. Applying those operations puts several copies of
the same atom on one position, pymatgen adds their occupancies up and rejects
the file. Other programs ignore the redundant operations; this does the same,
but only after checking that the atom list really is complete.
"""
import warnings

import numpy as np
import streamlit as st


def _clean_cif_value(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value.strip()

import os
from pymatgen.io.cif import CifFile, CifParser as _CifParser, str2float
from pymatgen.core.operations import SymmOp
from pymatgen.util.coord import find_in_coord_list_pbc

CIF_REPAIR_NOTES_KEY = "cif_repair_notes"

_CIF_SYMOP_LOOP_TAGS = (
    "_space_group_symop_operation_xyz",
    "_symmetry_equiv_pos_as_xyz",
    "_space_group_symop_id",
    "_symmetry_equiv_pos_site_id",
)
_CIF_SG_DROP_TAGS = (
    "_symmetry_space_group_name_hall",
    "_space_group_name_hall",
    "_symmetry_cell_setting",
)


def _cif_as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _cif_symmetry_is_redundant(cif_content, tol=1e-3):
    """True when the CIF atom list already contains every symmetry image.

    In that case the symmetry operations only duplicate atoms that are written
    in the file anyway, and can safely be dropped. Returns False whenever the
    operations still generate new atoms (a normal CIF holding only the
    asymmetric unit), so that such files are never silently truncated.
    """
    try:
        blocks = list(CifFile.from_str(cif_content).data.values())
    except Exception:
        return False

    for block in blocks:
        data = block.data
        ops_raw = None
        for tag in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
            if data.get(tag):
                ops_raw = _cif_as_list(data[tag])
                break
        if not ops_raw or len(ops_raw) < 2:
            continue
        try:
            ops = [SymmOp.from_xyz_str(_clean_cif_value(op)) for op in ops_raw]
        except Exception:
            continue

        try:
            coords = np.array([
                [str2float(x), str2float(y), str2float(z)]
                for x, y, z in zip(_cif_as_list(data["_atom_site_fract_x"]),
                                   _cif_as_list(data["_atom_site_fract_y"]),
                                   _cif_as_list(data["_atom_site_fract_z"]))
            ], dtype=float)
        except Exception:
            continue
        if len(coords) < 2:
            continue
        # The check below is quadratic in the number of sites; huge files are
        # left alone rather than spending seconds on them.
        if len(coords) * len(ops) > 50000:
            continue

        symbols = _cif_as_list(data.get("_atom_site_type_symbol")) or \
            _cif_as_list(data.get("_atom_site_label"))
        if len(symbols) != len(coords):
            symbols = [""] * len(coords)

        overlapping = False
        for idx, coord in enumerate(coords):
            for op in ops:
                image = op.operate(coord)
                matches = find_in_coord_list_pbc(coords, image, atol=tol)
                # The image of a listed atom must again be a listed atom of the
                # same element, otherwise the atom list is not complete.
                matches = [m for m in matches if symbols[m] == symbols[idx]]
                if not len(matches):
                    return False
                if any(int(m) != idx for m in matches):
                    overlapping = True
        if overlapping:
            return True
    return False


def _cif_content_without_symmetry(cif_content):
    """Copy of the CIF text with the symmetry operations reduced to P1."""
    lines = cif_content.splitlines()
    out, idx = [], 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        low = stripped.lower()

        if low.startswith("loop_"):
            header_end = idx + 1
            headers = []
            while header_end < len(lines) and lines[header_end].strip().startswith("_"):
                headers.append(lines[header_end].strip().split()[0].lower())
                header_end += 1
            if any(h in _CIF_SYMOP_LOOP_TAGS for h in headers):
                data_end = header_end
                while data_end < len(lines):
                    row = lines[data_end].strip()
                    if not row or row.startswith("_") or \
                            row.lower().startswith(("loop_", "data_")):
                        break
                    data_end += 1
                out.append("loop_")
                out.append("_space_group_symop_operation_xyz")
                out.append("   'x, y, z'")
                idx = data_end
                continue

        tag = low.split()[0] if stripped.startswith("_") else ""
        if tag in ("_symmetry_space_group_name_h-m", "_space_group_name_h-m",
                   "_space_group_name_h-m_alt"):
            out.append("_symmetry_space_group_name_H-M    'P 1'")
            idx += 1
            continue
        if tag in ("_symmetry_int_tables_number", "_space_group_it_number"):
            out.append("_symmetry_Int_Tables_number       1")
            idx += 1
            continue
        if tag in _CIF_SG_DROP_TAGS:
            idx += 1
            continue

        out.append(line)
        idx += 1
    return "\n".join(out) + "\n"


def parse_cif_content(cif_content, primitive=False):
    """Parse CIF text, repairing files pymatgen rejects. Returns (structure, notes).

    'notes' is a list of short messages describing what had to be fixed; it is
    empty for a file that parses as written.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = _CifParser.from_str(cif_content).parse_structures(
                primitive=primitive)
        if structures:
            return structures[0], []
        first_error = "no structure could be read from the file"
    except Exception as exc:
        first_error = str(exc)

    if _cif_symmetry_is_redundant(cif_content):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = _CifParser.from_str(
                _cif_content_without_symmetry(cif_content)
            ).parse_structures(primitive=primitive)
        if structures:
            note = (
                f"all {len(structures[0])} atoms of the cell are listed "
                "together with the space-group operations, which duplicated "
                "them; the operations were ignored."
            )
            return structures[0], [note]

    raise ValueError(first_error)


def record_cif_repair(filename, notes):
    """Remember what had to be fixed in an uploaded CIF file."""
    if not notes:
        return
    name = os.path.basename(str(filename))
    try:
        st.session_state.setdefault(CIF_REPAIR_NOTES_KEY, {})[name] = list(notes)
    except Exception:
        pass


