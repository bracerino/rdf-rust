//! Multi-frame XYZ trajectories, as written by MD codes.
//!
//! The file is streamed line by line and only the frames that are kept are
//! parsed; the atom lines of skipped frames are read past without converting a
//! single number, so taking every 10th frame of a long run costs little more
//! than reading the file.

use crate::lattice::Lattice;
use crate::structure::{Site, Structure};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct Trajectory {
    pub frames: Vec<Structure>,
    /// Index of each kept frame in the file, counting from 0.
    pub indices: Vec<usize>,
    /// Frames present in the file (up to `stop`, if one was given).
    pub n_read: usize,
}

fn parse_f64(tok: &str) -> Result<f64, String> {
    tok.parse::<f64>()
        .map_err(|_| format!("expected a number, found '{}'", tok))
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

/// Value of `key=...` in an extended-XYZ comment line, with quotes removed.
fn comment_value<'a>(comment: &'a str, key: &str) -> Option<&'a str> {
    let lower = comment.to_ascii_lowercase();
    let pat = format!("{}=", key.to_ascii_lowercase());
    let mut from = 0;
    while let Some(pos) = lower[from..].find(&pat) {
        let at = from + pos;
        // Must start a token, so "Lattice=" does not match inside "superLattice=".
        if at == 0 || comment.as_bytes()[at - 1].is_ascii_whitespace() {
            let rest = &comment[at + pat.len()..];
            return Some(if let Some(stripped) = rest.strip_prefix('"') {
                &stripped[..stripped.find('"').unwrap_or(stripped.len())]
            } else {
                &rest[..rest.find(char::is_whitespace).unwrap_or(rest.len())]
            });
        }
        from = at + pat.len();
    }
    None
}

/// The cell given in the comment line, if there is one.
fn lattice_from_comment(comment: &str) -> Result<Option<Lattice>, String> {
    let Some(v) = comment_value(comment, "Lattice") else {
        return Ok(None);
    };
    let vals: Vec<f64> = v.split_whitespace().map(parse_f64).collect::<Result<_, _>>()?;
    if vals.len() != 9 {
        return Err(format!(
            "Lattice=\"...\" must hold 9 numbers, found {}",
            vals.len()
        ));
    }
    Lattice::new([
        [vals[0], vals[1], vals[2]],
        [vals[3], vals[4], vals[5]],
        [vals[6], vals[7], vals[8]],
    ])
    .map(Some)
}

/// Columns holding the element and the first coordinate.
///
/// Read from `Properties=species:S:1:pos:R:3:...`; a plain XYZ file (no
/// Properties) is element, x, y, z.
fn columns_from_comment(comment: &str) -> Result<(usize, usize), String> {
    let Some(props) = comment_value(comment, "Properties") else {
        return Ok((0, 1));
    };
    let parts: Vec<&str> = props.split(':').collect();
    if parts.len() % 3 != 0 {
        return Err(format!("cannot read Properties={}", props));
    }
    let (mut species, mut pos, mut col) = (None, None, 0usize);
    for chunk in parts.chunks(3) {
        let name = chunk[0].to_ascii_lowercase();
        let count: usize = chunk[2]
            .parse()
            .map_err(|_| format!("bad column count in Properties={}", props))?;
        match name.as_str() {
            "species" | "element" | "elements" | "symbols" => species = Some(col),
            "pos" | "positions" => pos = Some(col),
            _ => {}
        }
        col += count;
    }
    match (species, pos) {
        (Some(s), Some(p)) => Ok((s, p)),
        _ => Err(format!(
            "Properties={} names no species or pos column",
            props
        )),
    }
}

fn build_frame(
    comment: &str,
    atom_lines: &[String],
    cell: Option<&[[f64; 3]; 3]>,
    index: usize,
) -> Result<Structure, String> {
    let lattice = match lattice_from_comment(comment)? {
        Some(l) => l,
        None => match cell {
            Some(m) => Lattice::new(*m)?,
            None => {
                return Err(format!(
                    "frame {} has no Lattice=\"...\" and no cell was given; an RDF of a \
                     periodic system needs the cell, so pass cell=[[a],[b],[c]]",
                    index
                ))
            }
        },
    };
    let (c_sp, c_pos) = columns_from_comment(comment)?;
    let need = c_sp.max(c_pos + 2) + 1;
    let mut sites = Vec::with_capacity(atom_lines.len());
    for (k, line) in atom_lines.iter().enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < need {
            return Err(format!("frame {}, atom line {} is incomplete", index, k + 1));
        }
        let cart = [
            parse_f64(toks[c_pos])?,
            parse_f64(toks[c_pos + 1])?,
            parse_f64(toks[c_pos + 2])?,
        ];
        sites.push(Site {
            frac: lattice.to_fractional(cart),
            species: vec![(clean_symbol(toks[c_sp]), 1.0)],
        });
    }
    Structure::new(lattice, sites)
}

/// Read the frames `start, start + every, start + 2*every, ...` before `stop`.
pub fn read_xyz_trajectory(
    path: &str,
    every: usize,
    start: usize,
    stop: Option<usize>,
    cell: Option<[[f64; 3]; 3]>,
) -> Result<Trajectory, String> {
    if every == 0 {
        return Err("every must be at least 1".to_string());
    }
    let file = File::open(path).map_err(|e| format!("could not open {}: {}", path, e))?;
    let mut lines = BufReader::new(file).lines();
    let mut next = |what: &str| -> Result<Option<String>, String> {
        match lines.next() {
            None => Ok(None),
            Some(Ok(l)) => Ok(Some(l)),
            Some(Err(e)) => Err(format!("error reading {}: {}", what, e)),
        }
    };

    let mut traj = Trajectory { frames: Vec::new(), indices: Vec::new(), n_read: 0 };
    let mut index = 0usize;
    loop {
        if let Some(s) = stop {
            if index >= s {
                break;
            }
        }
        // Atom count; blank lines between frames are tolerated.
        let count_line = loop {
            match next("the atom count")? {
                None => return finish(traj),
                Some(l) if l.trim().is_empty() => continue,
                Some(l) => break l,
            }
        };
        let n: usize = count_line.trim().parse().map_err(|_| {
            format!(
                "frame {}: expected an atom count, found '{}'",
                index,
                count_line.trim()
            )
        })?;
        let comment = next("the comment line")?
            .ok_or_else(|| format!("frame {} ends before its comment line", index))?;
        let keep = index >= start && (index - start) % every == 0;
        let mut atom_lines = Vec::with_capacity(if keep { n } else { 0 });
        for k in 0..n {
            let line = next("an atom line")?.ok_or_else(|| {
                format!("frame {} ends after {} of {} atoms", index, k, n)
            })?;
            if keep {
                atom_lines.push(line);
            }
        }
        if keep {
            traj.frames
                .push(build_frame(&comment, &atom_lines, cell.as_ref(), index)?);
            traj.indices.push(index);
        }
        index += 1;
        traj.n_read = index;
    }
    finish(traj)
}

fn finish(traj: Trajectory) -> Result<Trajectory, String> {
    if traj.n_read == 0 {
        return Err("no frames found in the file".to_string());
    }
    if traj.frames.is_empty() {
        return Err(format!(
            "the file has {} frame(s) but none matched start/every/stop",
            traj.n_read
        ));
    }
    Ok(traj)
}
