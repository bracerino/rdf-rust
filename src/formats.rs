//! Readers for the common non-CIF structure formats.
//!
//! Neither format can express fractional site occupancy, so anything read here
//! is fully ordered by construction. CIF remains the route for disorder.

use crate::lattice::Lattice;
use crate::structure::{Site, Structure};

fn parse_f64(tok: &str) -> Result<f64, String> {
    tok.parse::<f64>()
        .map_err(|_| format!("expected a number, found '{}'", tok))
}

/// VASP POSCAR / CONTCAR.
pub fn parse_poscar(text: &str) -> Result<Structure, String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 8 {
        return Err("POSCAR is too short to contain a structure".to_string());
    }
    let scale = parse_f64(lines[1].split_whitespace().next().unwrap_or(""))?;
    let mut m = [[0.0f64; 3]; 3];
    for i in 0..3 {
        let toks: Vec<&str> = lines[2 + i].split_whitespace().collect();
        if toks.len() < 3 {
            return Err(format!("lattice vector {} is incomplete", i + 1));
        }
        for j in 0..3 {
            m[i][j] = parse_f64(toks[j])?;
        }
    }
    // A negative scale is a target cell volume rather than a factor.
    let factor = if scale < 0.0 {
        let vol = Lattice::new(m)?.volume();
        (-scale / vol).cbrt()
    } else {
        scale
    };
    for row in m.iter_mut() {
        for v in row.iter_mut() {
            *v *= factor;
        }
    }
    let lattice = Lattice::new(m)?;

    // VASP 5 names the species on line 6; VASP 4 omits it and the names have
    // to come from the comment line instead.
    let line5: Vec<&str> = lines[5].split_whitespace().collect();
    let has_symbols = !line5.is_empty() && line5[0].parse::<usize>().is_err();
    let (symbols, counts_line, mut idx) = if has_symbols {
        (line5.clone(), lines[6], 7)
    } else {
        (
            lines[0].split_whitespace().collect::<Vec<&str>>(),
            lines[5],
            6,
        )
    };
    let counts: Vec<usize> = counts_line
        .split_whitespace()
        .map(|t| t.parse::<usize>().map_err(|_| format!("bad atom count '{}'", t)))
        .collect::<Result<_, _>>()?;
    if symbols.len() < counts.len() {
        return Err(
            "POSCAR does not name one species per atom count (VASP 4 files need the \
             species on the comment line)"
                .to_string(),
        );
    }

    // Optional selective-dynamics line sits before the coordinate mode.
    let mut mode_line = *lines.get(idx).ok_or("POSCAR ends before the coordinates")?;
    if mode_line
        .trim_start()
        .chars()
        .next()
        .map(|c| c == 'S' || c == 's')
        .unwrap_or(false)
    {
        idx += 1;
        mode_line = *lines.get(idx).ok_or("POSCAR ends before the coordinates")?;
    }
    let first = mode_line.trim_start().chars().next().unwrap_or('d');
    let cartesian = matches!(first, 'C' | 'c' | 'K' | 'k');
    idx += 1;

    let total: usize = counts.iter().sum();
    let mut sites = Vec::with_capacity(total);
    let mut which = 0usize;
    let mut left = *counts.first().unwrap_or(&0);
    for n in 0..total {
        while left == 0 {
            which += 1;
            if which >= counts.len() {
                return Err("more coordinates than atom counts".to_string());
            }
            left = counts[which];
        }
        let line = lines
            .get(idx + n)
            .ok_or_else(|| format!("POSCAR ends after {} of {} coordinates", n, total))?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 3 {
            return Err(format!("coordinate line {} is incomplete", n + 1));
        }
        let raw = [parse_f64(toks[0])?, parse_f64(toks[1])?, parse_f64(toks[2])?];
        let frac = if cartesian {
            lattice.to_fractional([raw[0] * factor, raw[1] * factor, raw[2] * factor])
        } else {
            raw
        };
        sites.push(Site {
            frac,
            species: vec![(clean_symbol(symbols[which]), 1.0)],
        });
        left -= 1;
    }
    Structure::new(lattice, sites)
}

/// Extended XYZ, which carries the cell in a `Lattice="..."` comment field.
pub fn parse_extxyz(text: &str) -> Result<Structure, String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 2 {
        return Err("XYZ file is too short".to_string());
    }
    let n: usize = lines[0]
        .trim()
        .parse()
        .map_err(|_| "first line of an XYZ file must be the atom count".to_string())?;
    let comment = lines[1];
    let lat_start = comment
        .find("Lattice=\"")
        .or_else(|| comment.find("lattice=\""))
        .ok_or(
            "this XYZ file has no Lattice=\"...\" field, so it has no cell and no \
             periodic RDF can be computed",
        )?;
    let rest = &comment[lat_start + 9..];
    let end = rest.find('"').ok_or("unterminated Lattice=\"...\" field")?;
    let vals: Vec<f64> = rest[..end]
        .split_whitespace()
        .map(parse_f64)
        .collect::<Result<_, _>>()?;
    if vals.len() != 9 {
        return Err(format!(
            "Lattice=\"...\" must hold 9 numbers, found {}",
            vals.len()
        ));
    }
    let m = [
        [vals[0], vals[1], vals[2]],
        [vals[3], vals[4], vals[5]],
        [vals[6], vals[7], vals[8]],
    ];
    let lattice = Lattice::new(m)?;
    let mut sites = Vec::with_capacity(n);
    for i in 0..n {
        let line = lines
            .get(2 + i)
            .ok_or_else(|| format!("XYZ ends after {} of {} atoms", i, n))?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(format!("atom line {} is incomplete", i + 1));
        }
        let cart = [parse_f64(toks[1])?, parse_f64(toks[2])?, parse_f64(toks[3])?];
        sites.push(Site {
            frac: lattice.to_fractional(cart),
            species: vec![(clean_symbol(toks[0]), 1.0)],
        });
    }
    Structure::new(lattice, sites)
}

/// "Fe", "Fe3+", "Fe1" -> "Fe".
fn clean_symbol(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == 0 && c.is_ascii_alphabetic() {
            out.push(c.to_ascii_uppercase());
        } else if i == 1 && c.is_ascii_lowercase() {
            out.push(c);
        } else {
            break;
        }
    }
    if out.is_empty() {
        s.to_string()
    } else {
        out
    }
}
