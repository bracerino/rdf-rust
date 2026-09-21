//! Unit-cell geometry: the 3x3 cell matrix and the conversions built on it.

/// Row-major 3x3 cell matrix; row `i` is lattice vector `i` in Cartesian A.
#[derive(Clone, Debug)]
pub struct Lattice {
    pub matrix: [[f64; 3]; 3],
    inverse: [[f64; 3]; 3],
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn invert3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = det3(m);
    if d.abs() < 1e-12 {
        return None;
    }
    let id = 1.0 / d;
    let mut inv = [[0.0; 3]; 3];
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    Some(inv)
}

impl Lattice {
    pub fn new(matrix: [[f64; 3]; 3]) -> Result<Self, String> {
        let inverse = invert3(&matrix)
            .ok_or_else(|| "cell matrix is singular (zero volume)".to_string())?;
        Ok(Lattice { matrix, inverse })
    }

    /// Build the conventional CIF cell: `a` along x, `b` in the xy plane.
    pub fn from_parameters(
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Result<Self, String> {
        if a <= 0.0 || b <= 0.0 || c <= 0.0 {
            return Err("cell lengths must be positive".to_string());
        }
        let (ar, br, gr) = (
            alpha.to_radians(),
            beta.to_radians(),
            gamma.to_radians(),
        );
        let (ca, cb, cg) = (ar.cos(), br.cos(), gr.cos());
        let sg = gr.sin();
        if sg.abs() < 1e-12 {
            return Err("gamma must not be 0 or 180 degrees".to_string());
        }
        let cx = c * cb;
        let cy = c * (ca - cb * cg) / sg;
        let cz2 = c * c - cx * cx - cy * cy;
        if cz2 <= 0.0 {
            return Err("cell angles are geometrically inconsistent".to_string());
        }
        Lattice::new([
            [a, 0.0, 0.0],
            [b * cg, b * sg, 0.0],
            [cx, cy, cz2.sqrt()],
        ])
    }

    #[inline]
    pub fn to_cartesian(&self, f: [f64; 3]) -> [f64; 3] {
        let m = &self.matrix;
        [
            f[0] * m[0][0] + f[1] * m[1][0] + f[2] * m[2][0],
            f[0] * m[0][1] + f[1] * m[1][1] + f[2] * m[2][1],
            f[0] * m[0][2] + f[1] * m[1][2] + f[2] * m[2][2],
        ]
    }

    #[inline]
    pub fn to_fractional(&self, c: [f64; 3]) -> [f64; 3] {
        let m = &self.inverse;
        [
            c[0] * m[0][0] + c[1] * m[1][0] + c[2] * m[2][0],
            c[0] * m[0][1] + c[1] * m[1][1] + c[2] * m[2][1],
            c[0] * m[0][2] + c[1] * m[1][2] + c[2] * m[2][2],
        ]
    }

    pub fn volume(&self) -> f64 {
        det3(&self.matrix).abs()
    }

    pub fn lengths(&self) -> [f64; 3] {
        let m = &self.matrix;
        [0, 1, 2].map(|i| (m[i][0].powi(2) + m[i][1].powi(2) + m[i][2].powi(2)).sqrt())
    }

    /// Spacing between the lattice planes normal to each axis.
    ///
    /// This is what decides how many periodic images a cutoff reaches, not the
    /// cell lengths -- a strongly sheared cell can be much thinner than its
    /// edges suggest.
    pub fn widths(&self) -> [f64; 3] {
        let m = &self.matrix;
        let v = self.volume();
        let cross = |u: [f64; 3], w: [f64; 3]| {
            [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ]
        };
        let norm = |u: [f64; 3]| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        [
            v / norm(cross(m[1], m[2])),
            v / norm(cross(m[0], m[2])),
            v / norm(cross(m[0], m[1])),
        ]
    }
}
