//! A CIF reader that keeps site occupancies instead of flattening them.

use crate::lattice::Lattice;
use crate::structure::{Site, Structure};
use crate::symmetry::{parse_symmop, SymmOp};
use std::collections::HashMap;

/// Strip a CIF standard uncertainty and parse: "3.9050(4)" -> 3.9050.
pub fn parse_cif_number(s: &str) -> Option<f64> {
    let s = s.trim().trim_matches('\'').trim_matches('"');
    if s.is_empty() || s == "?" || s == "." {
        return None;
    }
    let cut = s.find('(').unwrap_or(s.len());
    s[..cut].trim().parse::<f64>().ok()
}

/// Split a CIF line into tokens, honouring quotes.
fn tokenize(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for ch in line.chars() {
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(ch);
                }
            }
            None => {
                if ch == '\'' || ch == '"' {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    quote = Some(ch);
                } else if ch.is_whitespace() {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                } else if ch == '#' && cur.is_empty() {
                    break; // trailing comment
                } else {
                    cur.push(ch);
                }
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Element symbol out of a CIF type symbol or site label: "Ba2+" -> "Ba",
/// "O1" -> "O", "Fe3+" -> "Fe".
pub fn element_from_label(s: &str) -> Option<String> {
    let s = s.trim();
    let mut chars = s.chars();
    let first = chars.next()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    let mut sym = String::new();
    sym.push(first.to_ascii_uppercase());
    if let Some(second) = chars.next() {
        if second.is_ascii_lowercase() {
            // Two-letter symbols only; "Ba1" must not become "Ba1".
            sym.push(second);
        }
    }
    Some(sym)
}

#[derive(Default)]
struct Block {
    values: HashMap<String, String>,
    loops: Vec<(Vec<String>, Vec<Vec<String>>)>,
}

fn parse_blocks(text: &str) -> Block {
    let mut block = Block::default();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i];
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }
        // Multi-line semicolon text block: keep it out of the token stream.
        if trimmed.starts_with(';') {
            i += 1;
            while i < lines.len() && !lines[i].trim_start().starts_with(';') {
                i += 1;
            }
            i += 1;
            continue;
        }
        if trimmed.eq_ignore_ascii_case("loop_") {
            let mut headers = Vec::new();
            i += 1;
            while i < lines.len() {
                let t = lines[i].trim();
                if t.starts_with('_') {
                    headers.push(t.split_whitespace().next().unwrap().to_lowercase());
                    i += 1;
                } else if t.is_empty() || t.starts_with('#') {
                    i += 1;
                } else {
                    break;
                }
            }
            let mut rows: Vec<Vec<String>> = Vec::new();
            let mut pending: Vec<String> = Vec::new();
            while i < lines.len() {
                let t = lines[i].trim();
                if t.is_empty() || t.starts_with('#') {
                    i += 1;
                    continue;
                }
                if t.starts_with('_') || t.starts_with("loop_") || t.starts_with("data_") {
                    break;
                }
                if t.starts_with(';') {
                    // A value spanning lines; treat the whole block as one token.
                    let mut buf = String::new();
                    i += 1;
                    while i < lines.len() && !lines[i].trim_start().starts_with(';') {
                        buf.push_str(lines[i].trim());
                        i += 1;
                    }
                    i += 1;
                    pending.push(buf);
                } else {
                    pending.extend(tokenize(t));
                    i += 1;
                }
                // A loop row may be wrapped across lines; emit once full.
                while pending.len() >= headers.len() && !headers.is_empty() {
                    rows.push(pending.drain(..headers.len()).collect());
                }
            }
            if !headers.is_empty() {
                block.loops.push((headers, rows));
            }
            continue;
        }
        if trimmed.starts_with('_') {
            let toks = tokenize(trimmed);
            if toks.len() >= 2 {
                block
                    .values
                    .insert(toks[0].to_lowercase(), toks[1..].join(" "));
            } else if toks.len() == 1 {
                // Value sits on the following line(s).
                let key = toks[0].to_lowercase();
                i += 1;
                while i < lines.len() && lines[i].trim().is_empty() {
                    i += 1;
                }
                if i < lines.len() {
                    let t = lines[i].trim();
                    if t.starts_with(';') {
                        let mut buf = String::new();
                        i += 1;
                        while i < lines.len() && !lines[i].trim_start().starts_with(';') {
                            buf.push_str(lines[i].trim());
                            i += 1;
                        }
                        block.values.insert(key, buf);
                    } else {
                        block.values.insert(key, tokenize(t).join(" "));
                    }
                }
            }
            i += 1;
            continue;
        }
        i += 1;
    }
    block
}

fn find_loop<'a>(
    block: &'a Block,
    needed: &[&str],
) -> Option<(&'a Vec<String>, &'a Vec<Vec<String>>)> {
    block
        .loops
        .iter()
        .find(|(h, _)| needed.iter().any(|n| h.iter().any(|x| x == n)))
        .map(|(h, r)| (h, r))
}

/// Read a CIF, expand its symmetry, and keep every site occupancy.
pub fn parse_cif(text: &str, symprec: f64) -> Result<Structure, String> {
    let block = parse_blocks(text);
    let get = |k: &str| block.values.get(k).and_then(|v| parse_cif_number(v));

    let a = get("_cell_length_a").ok_or("missing _cell_length_a")?;
    let b = get("_cell_length_b").ok_or("missing _cell_length_b")?;
    let c = get("_cell_length_c").ok_or("missing _cell_length_c")?;
    let alpha = get("_cell_angle_alpha").unwrap_or(90.0);
    let beta = get("_cell_angle_beta").unwrap_or(90.0);
    let gamma = get("_cell_angle_gamma").unwrap_or(90.0);
    let lattice = Lattice::from_parameters(a, b, c, alpha, beta, gamma)?;

    // Symmetry operations, if the file lists them.
    let mut ops: Vec<SymmOp> = Vec::new();
    if let Some((headers, rows)) = find_loop(
        &block,
        &[
            "_symmetry_equiv_pos_as_xyz",
            "_space_group_symop_operation_xyz",
        ],
    ) {
        let col = headers
            .iter()
            .position(|h| {
                h == "_symmetry_equiv_pos_as_xyz" || h == "_space_group_symop_operation_xyz"
            })
            .unwrap();
        for row in rows {
            if let Some(cell) = row.get(col) {
                // Some files number their operations: "1 x,y,z".
                let cleaned = cell.trim();
                if cleaned.is_empty() {
                    continue;
                }
                match parse_symmop(cleaned) {
                    Ok(op) => ops.push(op),
                    Err(_) => {
                        if let Some((_, rest)) = cleaned.split_once(' ') {
                            if let Ok(op) = parse_symmop(rest) {
                                ops.push(op);
                            }
                        }
                    }
                }
            }
        }
    }
    if ops.is_empty() {
        ops.push(SymmOp::identity());
    }

    // Atom sites.
    let (headers, rows) = find_loop(&block, &["_atom_site_fract_x"])
        .ok_or("no _atom_site loop with fractional coordinates found")?;
    let col = |name: &str| headers.iter().position(|h| h == name);
    let cx = col("_atom_site_fract_x").ok_or("missing _atom_site_fract_x")?;
    let cy = col("_atom_site_fract_y").ok_or("missing _atom_site_fract_y")?;
    let cz = col("_atom_site_fract_z").ok_or("missing _atom_site_fract_z")?;
    let c_occ = col("_atom_site_occupancy");
    let c_sym = col("_atom_site_type_symbol");
    let c_lab = col("_atom_site_label");

    // Each asymmetric-unit entry, expanded over the symmetry operations.
    struct Raw {
        element: String,
        occ: f64,
        frac: [f64; 3],
    }
    let mut raws: Vec<Raw> = Vec::new();
    for row in rows {
        let (fx, fy, fz) = match (
            row.get(cx).and_then(|s| parse_cif_number(s)),
            row.get(cy).and_then(|s| parse_cif_number(s)),
            row.get(cz).and_then(|s| parse_cif_number(s)),
        ) {
            (Some(x), Some(y), Some(z)) => (x, y, z),
            _ => continue,
        };
        let element = c_sym
            .and_then(|i| row.get(i))
            .and_then(|s| element_from_label(s))
            .or_else(|| c_lab.and_then(|i| row.get(i)).and_then(|s| element_from_label(s)))
            .ok_or("could not determine the element of an atom site")?;
        let occ = c_occ
            .and_then(|i| row.get(i))
            .and_then(|s| parse_cif_number(s))
            .unwrap_or(1.0);
        if occ <= 0.0 {
            continue;
        }
        for op in &ops {
            let f = op.apply([fx, fy, fz]);
            raws.push(Raw {
                element: element.clone(),
                occ,
                frac: [
                    f[0] - f[0].floor(),
                    f[1] - f[1].floor(),
                    f[2] - f[2].floor(),
                ],
            });
        }
    }
    if raws.is_empty() {
        return Err("CIF contains no usable atom sites".to_string());
    }

    // Merge images that symmetry produced more than once, and gather the
    // species that share one crystallographic position into a single site.
    let mut sites: Vec<Site> = Vec::new();
    'outer: for raw in raws {
        for site in sites.iter_mut() {
            if frac_close(site.frac, raw.frac, symprec) {
                match site.species.iter_mut().find(|(e, _)| *e == raw.element) {
                    // The same element landing on the same spot is a duplicate
                    // image, not extra occupancy, so the value is not summed.
                    Some((_, o)) => *o = o.max(raw.occ),
                    None => site.species.push((raw.element.clone(), raw.occ)),
                }
                continue 'outer;
            }
        }
        sites.push(Site {
            frac: raw.frac,
            species: vec![(raw.element.clone(), raw.occ)],
        });
    }

    Structure::new(lattice, sites)
}

#[inline]
fn frac_close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|i| {
        let mut d = (a[i] - b[i]).abs();
        if d > 0.5 {
            d = 1.0 - d;
        }
        d < tol
    })
}
