//! Symmetry operations written the way CIF files spell them, e.g. "-y, x-y, z+1/3".

/// An affine operation on fractional coordinates: `rot * f + trans`.
#[derive(Clone, Copy, Debug)]
pub struct SymmOp {
    pub rot: [[f64; 3]; 3],
    pub trans: [f64; 3],
}

impl SymmOp {
    pub fn identity() -> Self {
        SymmOp {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            trans: [0.0; 3],
        }
    }

    #[inline]
    pub fn apply(&self, f: [f64; 3]) -> [f64; 3] {
        let r = &self.rot;
        [
            r[0][0] * f[0] + r[0][1] * f[1] + r[0][2] * f[2] + self.trans[0],
            r[1][0] * f[0] + r[1][1] * f[1] + r[1][2] * f[2] + self.trans[1],
            r[2][0] * f[0] + r[2][1] * f[1] + r[2][2] * f[2] + self.trans[2],
        ]
    }
}

/// Parse one component, e.g. "-x+1/2" or "1/2-x" or "2/3+y".
fn parse_component(s: &str, row: &mut [f64; 3], trans: &mut f64) -> Result<(), String> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if s.is_empty() {
        return Err("empty symmetry component".to_string());
    }
    // Split into signed terms without losing the sign, so "1/2-x" splits as
    // ["+1/2", "-x"].
    let mut terms: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, ch) in s.chars().enumerate() {
        if (ch == '+' || ch == '-') && i > 0 {
            terms.push(std::mem::take(&mut cur));
        }
        cur.push(ch);
    }
    terms.push(cur);

    for term in terms {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        let (sign, body) = match term.as_bytes()[0] {
            b'+' => (1.0, &term[1..]),
            b'-' => (-1.0, &term[1..]),
            _ => (1.0, term),
        };
        if body.is_empty() {
            return Err(format!("dangling sign in '{}'", s));
        }
        // Does this term carry a variable?
        let axis = body.chars().position(|c| matches!(c, 'x' | 'y' | 'z' | 'X' | 'Y' | 'Z'));
        match axis {
            Some(pos) => {
                let idx = match body.as_bytes()[pos] {
                    b'x' | b'X' => 0,
                    b'y' | b'Y' => 1,
                    _ => 2,
                };
                // A coefficient may precede the variable, as in "2x" or "0.5x".
                let coeff_str: String = body[..pos].to_string();
                let rest: String = body[pos + 1..].to_string();
                let mut coeff = if coeff_str.is_empty() {
                    1.0
                } else {
                    parse_number(coeff_str.trim_end_matches('*'))?
                };
                if !rest.is_empty() {
                    // Something like "x/2".
                    let r = rest.trim_start_matches('/');
                    if rest.starts_with('/') {
                        coeff /= parse_number(r)?;
                    } else {
                        return Err(format!("cannot read symmetry term '{}'", body));
                    }
                }
                row[idx] += sign * coeff;
            }
            None => *trans += sign * parse_number(body)?,
        }
    }
    Ok(())
}

fn parse_number(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.trim().parse().map_err(|_| format!("bad number '{}'", s))?;
        let d: f64 = den.trim().parse().map_err(|_| format!("bad number '{}'", s))?;
        if d == 0.0 {
            return Err(format!("division by zero in '{}'", s));
        }
        Ok(n / d)
    } else {
        s.parse().map_err(|_| format!("bad number '{}'", s))
    }
}

/// Parse a full operation such as "-x, y+1/2, -z".
pub fn parse_symmop(text: &str) -> Result<SymmOp, String> {
    let text = text.trim().trim_matches('\'').trim_matches('"').trim();
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("expected 3 components in symmetry op '{}'", text));
    }
    let mut op = SymmOp {
        rot: [[0.0; 3]; 3],
        trans: [0.0; 3],
    };
    for (i, p) in parts.iter().enumerate() {
        let mut row = [0.0; 3];
        let mut t = 0.0;
        parse_component(p, &mut row, &mut t)?;
        op.rot[i] = row;
        op.trans[i] = t;
    }
    Ok(op)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12)
    }

    #[test]
    fn identity_op() {
        let op = parse_symmop("x, y, z").unwrap();
        assert!(close(op.apply([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3]));
    }

    #[test]
    fn translation_before_variable() {
        let op = parse_symmop("1/2-x, -y, z+1/2").unwrap();
        assert!(close(op.apply([0.1, 0.2, 0.3]), [0.4, -0.2, 0.8]));
    }

    #[test]
    fn hexagonal_op() {
        let op = parse_symmop("-y, x-y, z+1/3").unwrap();
        assert!(close(op.apply([0.1, 0.2, 0.3]), [-0.2, -0.1, 0.3 + 1.0 / 3.0]));
    }

    #[test]
    fn quoted_and_spaced() {
        let op = parse_symmop("'-x + 1/2 , y , -z'").unwrap();
        assert!(close(op.apply([0.1, 0.2, 0.3]), [0.4, 0.2, -0.3]));
    }
}
