//! A crystal structure whose sites may carry more than one species.

use crate::lattice::Lattice;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Site {
    pub frac: [f64; 3],
    /// (element, occupancy) pairs. More than one entry means a shared site;
    /// occupancies summing below 1 leave the remainder vacant.
    pub species: Vec<(String, f64)>,
}

impl Site {
    pub fn total_occupancy(&self) -> f64 {
        self.species.iter().map(|(_, o)| o).sum()
    }
    pub fn is_ordered(&self) -> bool {
        self.species.len() == 1 && (self.total_occupancy() - 1.0).abs() < 1e-6
    }
}

#[derive(Clone, Debug)]
pub struct Structure {
    pub lattice: Lattice,
    pub sites: Vec<Site>,
}

impl Structure {
    pub fn new(lattice: Lattice, sites: Vec<Site>) -> Result<Self, String> {
        if sites.is_empty() {
            return Err("structure has no sites".to_string());
        }
        for s in &sites {
            let t = s.total_occupancy();
            if t > 1.0 + 1e-3 {
                return Err(format!(
                    "a site has occupancies summing to {:.4}, which exceeds 1",
                    t
                ));
            }
        }
        Ok(Structure { lattice, sites })
    }

    pub fn is_ordered(&self) -> bool {
        self.sites.iter().all(|s| s.is_ordered())
    }

    /// Element -> total occupancy (the number of atoms of that element per cell).
    pub fn composition(&self) -> BTreeMap<String, f64> {
        let mut c = BTreeMap::new();
        for site in &self.sites {
            for (el, occ) in &site.species {
                *c.entry(el.clone()).or_insert(0.0) += occ;
            }
        }
        c
    }

    pub fn elements(&self) -> Vec<String> {
        self.composition().keys().cloned().collect()
    }

    /// Total occupancy over all sites -- the atom count, which for a
    /// partially occupied cell need not be an integer.
    pub fn n_atoms(&self) -> f64 {
        self.sites.iter().map(|s| s.total_occupancy()).sum()
    }

    pub fn cartesian_coords(&self) -> Vec<[f64; 3]> {
        self.sites
            .iter()
            .map(|s| self.lattice.to_cartesian(s.frac))
            .collect()
    }

    pub fn make_supercell(&self, na: usize, nb: usize, nc: usize) -> Result<Structure, String> {
        if na == 0 || nb == 0 || nc == 0 {
            return Err("supercell dimensions must be at least 1".to_string());
        }
        let m = self.lattice.matrix;
        let scaled = [
            [m[0][0] * na as f64, m[0][1] * na as f64, m[0][2] * na as f64],
            [m[1][0] * nb as f64, m[1][1] * nb as f64, m[1][2] * nb as f64],
            [m[2][0] * nc as f64, m[2][1] * nc as f64, m[2][2] * nc as f64],
        ];
        let lattice = Lattice::new(scaled)?;
        let mut sites = Vec::with_capacity(self.sites.len() * na * nb * nc);
        for ia in 0..na {
            for ib in 0..nb {
                for ic in 0..nc {
                    for s in &self.sites {
                        sites.push(Site {
                            frac: [
                                (s.frac[0] + ia as f64) / na as f64,
                                (s.frac[1] + ib as f64) / nb as f64,
                                (s.frac[2] + ic as f64) / nc as f64,
                            ],
                            species: s.species.clone(),
                        });
                    }
                }
            }
        }
        Structure::new(lattice, sites)
    }
}
