//! Periodic neighbour search on a linked cell list.
//!
//! The grid is built in fractional space and the scan range per axis is
//! derived from the plane spacing, so a cutoff larger than the cell is handled
//! by visiting further image shells rather than by replicating the atoms.

use crate::lattice::Lattice;

pub struct CellList {
    pub n: [usize; 3],
    /// Scan range per axis, in grid cells.
    pub reach: [i64; 3],
    heads: Vec<i32>,
    next: Vec<i32>,
}

impl CellList {
    pub fn build(lattice: &Lattice, fracs: &[[f64; 3]], cutoff: f64) -> CellList {
        let widths = lattice.widths();
        let mut n = [1usize; 3];
        for i in 0..3 {
            // One grid cell must not be thinner than is useful; capping the
            // division count keeps the grid from exploding for long thin cells.
            let d = (widths[i] / cutoff).floor() as i64;
            n[i] = d.clamp(1, 64) as usize;
        }
        let mut reach = [0i64; 3];
        for i in 0..3 {
            let cell_width = widths[i] / n[i] as f64;
            reach[i] = (cutoff / cell_width).ceil() as i64;
        }
        let total = n[0] * n[1] * n[2];
        let mut heads = vec![-1i32; total];
        let mut next = vec![-1i32; fracs.len()];
        for (idx, f) in fracs.iter().enumerate() {
            let c = Self::cell_of(f, &n);
            let flat = (c[0] * n[1] + c[1]) * n[2] + c[2];
            next[idx] = heads[flat];
            heads[flat] = idx as i32;
        }
        CellList { n, reach, heads, next }
    }

    #[inline]
    fn cell_of(f: &[f64; 3], n: &[usize; 3]) -> [usize; 3] {
        let mut c = [0usize; 3];
        for i in 0..3 {
            let w = f[i] - f[i].floor();
            let mut k = (w * n[i] as f64) as usize;
            if k >= n[i] {
                k = n[i] - 1;
            }
            c[i] = k;
        }
        c
    }

    /// Visit every periodic neighbour of `center` within `cutoff`.
    ///
    /// `f(j, dist)` receives the index of the neighbouring site and the
    /// distance. A site is reported once per periodic image in range, and the
    /// image at zero displacement (the atom itself) is skipped.
    #[inline]
    pub fn for_each_neighbor<F>(
        &self,
        lattice: &Lattice,
        fracs: &[[f64; 3]],
        center: usize,
        cutoff: f64,
        mut f: F,
    ) where
        F: FnMut(usize, f64),
    {
        let cut2 = cutoff * cutoff;
        let fc = fracs[center];
        let c0 = Self::cell_of(&fc, &self.n);
        let m = &lattice.matrix;
        for da in -self.reach[0]..=self.reach[0] {
            for db in -self.reach[1]..=self.reach[1] {
                for dc in -self.reach[2]..=self.reach[2] {
                    // Wrap the grid index and record which periodic image the
                    // wrap corresponds to.
                    let (ia, sa) = wrap(c0[0] as i64 + da, self.n[0]);
                    let (ib, sb) = wrap(c0[1] as i64 + db, self.n[1]);
                    let (ic, sc) = wrap(c0[2] as i64 + dc, self.n[2]);
                    let flat = (ia * self.n[1] + ib) * self.n[2] + ic;
                    let mut j = self.heads[flat];
                    while j >= 0 {
                        let ju = j as usize;
                        let fj = fracs[ju];
                        let d = [
                            fj[0] + sa as f64 - fc[0],
                            fj[1] + sb as f64 - fc[1],
                            fj[2] + sc as f64 - fc[2],
                        ];
                        let x = d[0] * m[0][0] + d[1] * m[1][0] + d[2] * m[2][0];
                        let y = d[0] * m[0][1] + d[1] * m[1][1] + d[2] * m[2][1];
                        let z = d[0] * m[0][2] + d[1] * m[1][2] + d[2] * m[2][2];
                        let r2 = x * x + y * y + z * z;
                        if r2 <= cut2 && r2 > 1e-16 {
                            f(ju, r2.sqrt());
                        }
                        j = self.next[ju];
                    }
                }
            }
        }
    }
}

#[inline]
fn wrap(i: i64, n: usize) -> (usize, i64) {
    let n_i = n as i64;
    let mut shift = 0i64;
    let mut k = i;
    while k < 0 {
        k += n_i;
        shift -= 1;
    }
    while k >= n_i {
        k -= n_i;
        shift += 1;
    }
    (k as usize, shift)
}
