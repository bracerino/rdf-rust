//! Python bindings for rdfrust.

mod cif;
mod expand;
mod formats;
mod lattice;
mod neighbor;
mod rdf;
mod structure;
mod symmetry;

use lattice::Lattice;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;
use structure::{Site, Structure};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pyclass(name = "Structure", module = "rdfrust")]
#[derive(Clone)]
pub struct PyStructure {
    inner: Structure,
}

#[pymethods]
impl PyStructure {
    /// Build from a cell matrix, fractional coordinates and per-site species.
    ///
    /// `species` is one dict {element: occupancy} per site.
    #[new]
    #[pyo3(signature = (matrix, frac_coords, species))]
    fn new(
        matrix: Vec<Vec<f64>>,
        frac_coords: Vec<Vec<f64>>,
        species: Vec<BTreeMap<String, f64>>,
    ) -> PyResult<Self> {
        if matrix.len() != 3 || matrix.iter().any(|r| r.len() != 3) {
            return Err(err("matrix must be 3x3".into()));
        }
        let m = [
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]],
        ];
        let lattice = Lattice::new(m).map_err(err)?;
        if frac_coords.len() != species.len() {
            return Err(err(
                "frac_coords and species must have the same length".into()
            ));
        }
        let mut sites = Vec::with_capacity(frac_coords.len());
        for (f, sp) in frac_coords.iter().zip(species.iter()) {
            if f.len() != 3 {
                return Err(err("each coordinate must have 3 components".into()));
            }
            if sp.is_empty() {
                return Err(err("every site needs at least one species".into()));
            }
            sites.push(Site {
                frac: [f[0], f[1], f[2]],
                species: sp.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            });
        }
        Ok(PyStructure {
            inner: Structure::new(lattice, sites).map_err(err)?,
        })
    }

    /// Read a CIF, keeping partial occupancies as they are written.
    #[staticmethod]
    #[pyo3(signature = (path, symprec=1e-4))]
    fn from_cif(path: &str, symprec: f64) -> PyResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("could not read {}: {}", path, e)))?;
        Ok(PyStructure {
            inner: cif::parse_cif(&text, symprec).map_err(err)?,
        })
    }

    /// Read a VASP POSCAR / CONTCAR.
    #[staticmethod]
    fn from_poscar(path: &str) -> PyResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("could not read {}: {}", path, e)))?;
        Ok(PyStructure { inner: formats::parse_poscar(&text).map_err(err)? })
    }

    /// Read an extended XYZ file (needs a `Lattice="..."` field).
    #[staticmethod]
    fn from_xyz(path: &str) -> PyResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("could not read {}: {}", path, e)))?;
        Ok(PyStructure { inner: formats::parse_extxyz(&text).map_err(err)? })
    }

    /// Parse text in a named format: "cif", "poscar" or "xyz".
    #[staticmethod]
    #[pyo3(signature = (text, fmt, symprec=1e-4))]
    fn from_str(text: &str, fmt: &str, symprec: f64) -> PyResult<Self> {
        let inner = match fmt.to_ascii_lowercase().as_str() {
            "cif" => cif::parse_cif(text, symprec),
            "poscar" | "vasp" | "contcar" => formats::parse_poscar(text),
            "xyz" | "extxyz" => formats::parse_extxyz(text),
            other => Err(format!(
                "unknown format '{}'; rdfrust reads cif, poscar and xyz",
                other
            )),
        };
        Ok(PyStructure { inner: inner.map_err(err)? })
    }

    /// Same as `from_cif`, for CIF text already in memory.
    #[staticmethod]
    #[pyo3(signature = (text, symprec=1e-4))]
    fn from_cif_str(text: &str, symprec: f64) -> PyResult<Self> {
        Ok(PyStructure {
            inner: cif::parse_cif(text, symprec).map_err(err)?,
        })
    }

    #[getter]
    fn n_sites(&self) -> usize {
        self.inner.sites.len()
    }

    /// Number of atoms, which is fractional when sites are partly occupied.
    #[getter]
    fn n_atoms(&self) -> f64 {
        self.inner.n_atoms()
    }

    #[getter]
    fn is_ordered(&self) -> bool {
        self.inner.is_ordered()
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.inner.lattice.volume()
    }

    #[getter]
    fn elements(&self) -> Vec<String> {
        self.inner.elements()
    }

    #[getter]
    fn composition(&self) -> BTreeMap<String, f64> {
        self.inner.composition()
    }

    #[getter]
    fn lattice_matrix(&self) -> Vec<Vec<f64>> {
        self.inner
            .lattice
            .matrix
            .iter()
            .map(|r| r.to_vec())
            .collect()
    }

    #[getter]
    fn frac_coords(&self) -> Vec<Vec<f64>> {
        self.inner.sites.iter().map(|s| s.frac.to_vec()).collect()
    }

    #[getter]
    fn cart_coords(&self) -> Vec<Vec<f64>> {
        self.inner
            .cartesian_coords()
            .iter()
            .map(|c| c.to_vec())
            .collect()
    }

    /// Per-site {element: occupancy}.
    #[getter]
    fn species(&self) -> Vec<BTreeMap<String, f64>> {
        self.inner
            .sites
            .iter()
            .map(|s| s.species.iter().cloned().collect())
            .collect()
    }

    /// One entry per partially occupied site: index, coordinates, occupancies.
    ///
    /// Empty for a fully ordered structure, so it doubles as the check a
    /// caller needs before reporting anything to the user.
    fn disorder_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for (i, site) in self.inner.sites.iter().enumerate() {
            if site.is_ordered() {
                continue;
            }
            let d = PyDict::new(py);
            d.set_item("site_index", i)?;
            d.set_item("frac_coords", site.frac.to_vec())?;
            let occ = PyDict::new(py);
            for (el, o) in &site.species {
                occ.set_item(el, *o)?;
            }
            d.set_item("occupancies", occ)?;
            d.set_item("total_occupancy", site.total_occupancy())?;
            d.set_item("vacancy", (1.0 - site.total_occupancy()).max(0.0))?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Number of worker threads rayon will use when `n_threads=0`.
    #[staticmethod]
    fn available_threads() -> usize {
        rayon::current_num_threads()
    }

    fn make_supercell(&self, na: usize, nb: usize, nc: usize) -> PyResult<Self> {
        Ok(PyStructure {
            inner: self.inner.make_supercell(na, nb, nc).map_err(err)?,
        })
    }

    /// Partial and total RDF.
    ///
    /// Returns (r, partials, total) where `partials` maps ("A", "B") to g_AB.
    #[pyo3(signature = (cutoff=10.0, bin_size=0.1, n_threads=0))]
    fn prdf<'py>(
        &self,
        py: Python<'py>,
        cutoff: f64,
        bin_size: f64,
        n_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>, Bound<'py, PyArray1<f64>>)> {
        let res = py
            .allow_threads(|| rdf::compute_prdf(&self.inner, cutoff, bin_size, n_threads))
            .map_err(err)?;
        let n_el = res.elements.len();
        let dict = PyDict::new(py);
        for a in 0..n_el {
            for b in 0..n_el {
                let key = (res.elements[a].clone(), res.elements[b].clone());
                let vals = res.partials[a * n_el + b].clone();
                dict.set_item(key, vals.into_pyarray(py))?;
            }
        }
        Ok((
            res.bin_centers.into_pyarray(py),
            dict,
            res.total.into_pyarray(py),
        ))
    }

    /// Total RDF only.
    #[pyo3(signature = (cutoff=10.0, bin_size=0.1, n_threads=0))]
    fn total_rdf<'py>(
        &self,
        py: Python<'py>,
        cutoff: f64,
        bin_size: f64,
        n_threads: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let res = py
            .allow_threads(|| rdf::compute_prdf(&self.inner, cutoff, bin_size, n_threads))
            .map_err(err)?;
        Ok((res.bin_centers.into_pyarray(py), res.total.into_pyarray(py)))
    }

    /// Expand partial occupancies into an explicit ordered supercell.
    #[pyo3(signature = (max_sites=4096, seed=42, max_denominator=16))]
    fn expand_occupancies<'py>(
        &self,
        py: Python<'py>,
        max_sites: usize,
        seed: u64,
        max_denominator: usize,
    ) -> PyResult<(Self, Bound<'py, PyDict>)> {
        let e = expand::expand(&self.inner, max_sites, seed, max_denominator).map_err(err)?;
        let info = PyDict::new(py);
        info.set_item("scale", e.scale.to_vec())?;
        info.set_item("n_vacancies", e.n_vacancies)?;
        info.set_item("warnings", PyList::new(py, &e.warnings)?)?;
        let rows = PyList::empty(py);
        for (el, target, pool, got) in &e.report {
            let d = PyDict::new(py);
            d.set_item("element", el)?;
            d.set_item("target_occupancy", target)?;
            d.set_item("sites_in_pool", pool)?;
            d.set_item("atoms_placed", got)?;
            d.set_item("actual_occupancy", *got as f64 / *pool as f64)?;
            rows.append(d)?;
        }
        info.set_item("occupancy_report", rows)?;
        Ok((PyStructure { inner: e.structure }, info))
    }

    fn __repr__(&self) -> String {
        let comp: Vec<String> = self
            .inner
            .composition()
            .iter()
            .map(|(e, n)| {
                if (n - n.round()).abs() < 1e-9 {
                    format!("{}{}", e, n.round() as i64)
                } else {
                    format!("{}{:.4}", e, n)
                }
            })
            .collect();
        format!(
            "Structure({} sites, {}{})",
            self.inner.sites.len(),
            comp.join(" "),
            if self.inner.is_ordered() {
                ""
            } else {
                ", disordered"
            }
        )
    }
}

#[pymodule]
fn _rdfrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStructure>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
