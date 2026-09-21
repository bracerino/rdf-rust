//! Partial and total radial distribution functions.
//!
//! Sites carrying several species contribute with the product of their
//! occupancies, which makes the result the ensemble average over an ideally
//! mixed (randomly substituted) lattice. For a fully ordered structure every
//! occupancy is 1 and the expressions below collapse to the usual counting
//! definition, so ordered results are directly comparable with matminer.

use crate::neighbor::CellList;
use crate::structure::Structure;
use rayon::prelude::*;

pub struct PrdfResult {
    pub elements: Vec<String>,
    pub bin_centers: Vec<f64>,
    pub bin_edges: Vec<f64>,
    /// `partials[a * n_el + b]` is g_ab, i.e. b seen from a centre of type a.
    pub partials: Vec<Vec<f64>>,
    pub total: Vec<f64>,
}

/// Bin edges laid out exactly like `numpy.arange(0, cutoff + bin_size, bin_size)`,
/// so the histogram lines up bin for bin with matminer's.
pub fn bin_edges(cutoff: f64, bin_size: f64) -> Vec<f64> {
    let n = ((cutoff + bin_size) / bin_size).ceil() as usize;
    (0..n).map(|i| i as f64 * bin_size).collect()
}

#[inline]
fn bin_of(d: f64, bin_size: f64, edges: &[f64], n_bins: usize) -> Option<usize> {
    let mut k = (d / bin_size).floor() as i64;
    if k < 0 {
        return None;
    }
    // Edges from repeated addition are not exact multiples of the step, so the
    // first guess is nudged until it really brackets the distance -- this is
    // what makes the binning agree with numpy's searchsorted.
    while k > 0 && d < edges[k as usize] {
        k -= 1;
    }
    while (k as usize) + 1 < edges.len() && d >= edges[k as usize + 1] {
        k += 1;
    }
    let k = k as usize;
    if k < n_bins {
        Some(k)
    } else if d == edges[n_bins] {
        Some(n_bins - 1) // numpy closes the final bin on the right
    } else {
        None
    }
}

pub fn compute_prdf(
    structure: &Structure,
    cutoff: f64,
    bin_size: f64,
    n_threads: usize,
) -> Result<PrdfResult, String> {
    // 0 means "use every core", which is rayon's global pool. A specific count
    // gets its own pool so the choice is per call and does not leak into other
    // callers of the library.
    if n_threads == 0 {
        return compute_prdf_inner(structure, cutoff, bin_size);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| format!("could not start {} worker threads: {}", n_threads, e))?;
    pool.install(|| compute_prdf_inner(structure, cutoff, bin_size))
}

fn compute_prdf_inner(
    structure: &Structure,
    cutoff: f64,
    bin_size: f64,
) -> Result<PrdfResult, String> {
    if cutoff <= 0.0 {
        return Err("cutoff must be positive".to_string());
    }
    if bin_size <= 0.0 || bin_size > cutoff {
        return Err("bin_size must be positive and no larger than the cutoff".to_string());
    }
    let edges = bin_edges(cutoff, bin_size);
    let n_bins = edges.len() - 1;
    if n_bins == 0 {
        return Err("cutoff and bin_size give no bins".to_string());
    }

    let elements = structure.elements();
    let n_el = elements.len();
    let el_index = |name: &str| elements.iter().position(|e| e == name).unwrap();

    // Flatten the sites so the hot loop touches plain slices only.
    let fracs: Vec<[f64; 3]> = structure.sites.iter().map(|s| s.frac).collect();
    let mut sp_start = Vec::with_capacity(structure.sites.len() + 1);
    let mut sp_el: Vec<usize> = Vec::new();
    let mut sp_occ: Vec<f64> = Vec::new();
    for site in &structure.sites {
        sp_start.push(sp_el.len());
        for (el, occ) in &site.species {
            sp_el.push(el_index(el));
            sp_occ.push(*occ);
        }
    }
    sp_start.push(sp_el.len());

    let cl = CellList::build(&structure.lattice, &fracs, cutoff);
    let n_sites = fracs.len();
    let stride = n_el * n_bins;

    // Each worker fills its own histogram; the reduction sums them.
    let hist = (0..n_sites)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n_el * stride],
            |mut acc, i| {
                let (ia, ib) = (sp_start[i], sp_start[i + 1]);
                cl.for_each_neighbor(&structure.lattice, &fracs, i, cutoff, |j, d| {
                    if let Some(bin) = bin_of(d, bin_size, &edges, n_bins) {
                        let (ja, jb) = (sp_start[j], sp_start[j + 1]);
                        for a in ia..ib {
                            let base = sp_el[a] * stride;
                            let wa = sp_occ[a];
                            for b in ja..jb {
                                acc[base + sp_el[b] * n_bins + bin] += wa * sp_occ[b];
                            }
                        }
                    }
                });
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n_el * stride],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );

    // Normalise: divide by the shell volume and by the number of centres.
    let shell: Vec<f64> = (0..n_bins)
        .map(|i| 4.0 / 3.0 * std::f64::consts::PI * (edges[i + 1].powi(3) - edges[i].powi(3)))
        .collect();
    let comp = structure.composition();
    let n_alpha: Vec<f64> = elements.iter().map(|e| comp[e]).collect();
    let n_total = structure.n_atoms();

    let mut partials = Vec::with_capacity(n_el * n_el);
    let mut total = vec![0.0f64; n_bins];
    for a in 0..n_el {
        for b in 0..n_el {
            let mut g = vec![0.0f64; n_bins];
            for k in 0..n_bins {
                let c = hist[a * stride + b * n_bins + k];
                // Summed over every ordered pair of species this is the total
                // count in the shell, so the total RDF needs no extra weights.
                total[k] += c / (shell[k] * n_total);
                g[k] = if n_alpha[a] > 0.0 {
                    c / (shell[k] * n_alpha[a])
                } else {
                    0.0
                };
            }
            partials.push(g);
        }
    }

    let centers: Vec<f64> = (0..n_bins).map(|i| edges[i]).collect();
    Ok(PrdfResult {
        elements,
        bin_centers: centers,
        bin_edges: edges,
        partials,
        total,
    })
}
