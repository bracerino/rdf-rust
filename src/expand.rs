//! Turning partial occupancies into an explicit, fully ordered supercell.
//!
//! The weighted RDF in `rdf.rs` needs none of this -- it is the exact ensemble
//! average. Expansion is here for the cases where an actual structure is
//! wanted: exporting a CIF, feeding a code that cannot represent occupancies,
//! or inspecting one particular realisation of the disorder.

use crate::structure::{Site, Structure};
use std::collections::BTreeMap;

/// Deterministic, portable PRNG (xorshift64*), so an expansion is reproducible
/// across platforms and Rust versions in a way `std`'s hasher-seeded RNG is not.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(2685821657736338717).max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(2685821657736338717)
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

pub struct Expansion {
    pub structure: Structure,
    pub scale: [usize; 3],
    pub n_vacancies: usize,
    /// (element, target occupancy, sites in the pool, atoms placed)
    pub report: Vec<(String, f64, usize, usize)>,
    pub warnings: Vec<String>,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Smallest denominator that reproduces `x` to within `tol`, capped at `max_den`.
fn denominator(x: f64, max_den: usize) -> usize {
    let mut best = 1;
    let mut best_err = f64::INFINITY;
    for d in 1..=max_den {
        let err = (x * d as f64 - (x * d as f64).round()).abs() / d as f64;
        if err < best_err - 1e-12 {
            best_err = err;
            best = d;
            if err < 1e-9 {
                break;
            }
        }
    }
    best
}

/// Choose the supercell that reproduces the occupancies, shrinking it to stay
/// under `max_sites`.
pub fn choose_scale(structure: &Structure, max_sites: usize, max_den: usize) -> ([usize; 3], usize) {
    let mut ideal = 1usize;
    for site in &structure.sites {
        if site.is_ordered() {
            continue;
        }
        for (_, occ) in &site.species {
            if *occ > 0.0 && *occ < 1.0 {
                let d = denominator(*occ, max_den);
                ideal = ideal / gcd(ideal, d) * d;
            }
        }
        let t = site.total_occupancy();
        if t < 1.0 - 1e-6 {
            let d = denominator(t, max_den);
            ideal = ideal / gcd(ideal, d) * d;
        }
    }
    let n = structure.sites.len();
    // Prefer an isotropic cell, then fall back to the largest that fits.
    for s in (1..=ideal).rev() {
        if s * s * s * n <= max_sites {
            return ([s, s, s], ideal);
        }
    }
    ([1, 1, 1], ideal)
}

pub fn expand(
    structure: &Structure,
    max_sites: usize,
    seed: u64,
    max_den: usize,
) -> Result<Expansion, String> {
    let (scale, ideal) = choose_scale(structure, max_sites, max_den);
    let super_cell = structure.make_supercell(scale[0], scale[1], scale[2])?;
    if super_cell.sites.len() > max_sites {
        return Err(format!(
            "even a {}x{}x{} cell holds {} sites, above the limit of {}",
            scale[0],
            scale[1],
            scale[2],
            super_cell.sites.len(),
            max_sites
        ));
    }

    // Sites sharing a composition are filled as one pool, which spreads the
    // rounding error over the whole sublattice instead of over each site.
    let mut pools: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut ordered_sites: Vec<Site> = Vec::new();
    for (i, site) in super_cell.sites.iter().enumerate() {
        if site.is_ordered() {
            ordered_sites.push(site.clone());
        } else {
            pools.entry(pool_key(site)).or_default().push(i);
        }
    }

    let mut rng = Rng::new(seed);
    let mut placed: Vec<Site> = Vec::new();
    let mut report: Vec<(String, f64, usize, usize)> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    let mut n_vac = 0usize;

    for (_key, idxs) in pools {
        let n = idxs.len();
        let species = super_cell.sites[idxs[0]].species.clone();
        let wanted: Vec<f64> = species.iter().map(|(_, o)| o * n as f64).collect();
        // Largest-remainder apportionment: floor everything, then hand the
        // leftover sites to the largest fractional parts. Rounding each species
        // on its own would both miscount and, on a tie, always favour the same
        // one.
        let mut counts: Vec<usize> = wanted.iter().map(|w| w.floor() as usize).collect();
        let total_atoms = ((wanted.iter().sum::<f64>()) + 0.5).floor() as usize;
        let total_atoms = total_atoms.min(n);
        let assigned: usize = counts.iter().sum();
        let mut leftover = total_atoms.saturating_sub(assigned);
        let mut order: Vec<usize> = (0..species.len()).collect();
        order.sort_by(|&a, &b| {
            (wanted[b] - counts[b] as f64)
                .partial_cmp(&(wanted[a] - counts[a] as f64))
                .unwrap()
        });
        for &k in order.iter() {
            if leftover == 0 {
                break;
            }
            counts[k] += 1;
            leftover -= 1;
        }

        let mut assignment: Vec<Option<usize>> = Vec::with_capacity(n);
        for (k, &c) in counts.iter().enumerate() {
            for _ in 0..c {
                assignment.push(Some(k));
            }
        }
        while assignment.len() < n {
            assignment.push(None);
        }
        rng.shuffle(&mut assignment);

        for (&site_idx, slot) in idxs.iter().zip(assignment.iter()) {
            match slot {
                Some(k) => placed.push(Site {
                    frac: super_cell.sites[site_idx].frac,
                    species: vec![(species[*k].0.clone(), 1.0)],
                }),
                None => n_vac += 1,
            }
        }
        for (k, (el, occ)) in species.iter().enumerate() {
            report.push((el.clone(), *occ, n, counts[k]));
            if counts[k] == 0 && *occ > 0.0 && !dropped.contains(el) {
                dropped.push(el.clone());
            }
        }
    }

    if !dropped.is_empty() {
        warnings.push(format!(
            "no atoms could be placed for {}: the occupancy is too small for a {}x{}x{} supercell, \
             so these elements are absent from the expanded structure",
            dropped.join(", "),
            scale[0],
            scale[1],
            scale[2]
        ));
    }
    let worst = report
        .iter()
        .map(|(_, target, pool, got)| ((*got as f64 / *pool as f64) - target).abs())
        .fold(0.0f64, f64::max);
    if scale[0] < ideal && worst >= 0.02 {
        warnings.push(format!(
            "the occupancies would need a {0}x{0}x{0} supercell to come out exactly, but that \
             exceeds the {1}-site limit; {2}x{2}x{2} was used and the occupancies are approximated \
             (worst deviation {3:.4})",
            ideal, max_sites, scale[0], worst
        ));
    }

    ordered_sites.extend(placed);
    if ordered_sites.is_empty() {
        return Err("no sites remain after expansion; check the occupancies".to_string());
    }
    let structure = Structure::new(super_cell.lattice.clone(), ordered_sites)?;
    Ok(Expansion {
        structure,
        scale,
        n_vacancies: n_vac,
        report,
        warnings,
    })
}

fn pool_key(site: &Site) -> String {
    let mut parts: Vec<String> = site
        .species
        .iter()
        .map(|(e, o)| format!("{}:{:.4}", e, o))
        .collect();
    parts.sort();
    parts.join("|")
}
