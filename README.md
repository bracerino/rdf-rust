# RDF-Rust

Partial and total radial distribution functions (PRDF / RDF) for crystal
structures, with native support for partially occupied sites.

**⚙️ Rust backend, 🐍 Python frontend.** The neighbour search, the histogramming
and the CIF parsing run in compiled Rust, in parallel across cores. You call it
from Python and get NumPy arrays back.

- 🚀 **~100× faster than Matminer on a single core**, and **~340× faster on four
  cores**: 13,720 atoms in 144 ms (42 ms on four cores) instead of 14.3 s.
- ⚡ **~2× faster than Vesin, core for core** (1.5–2.2×).
- 💾 **Under 1 MB of memory at any size.** Matminer needs 3.2 GB and Vesin
  454 MB for the same 13,720 atoms.
- 🧩 **Partial occupancies handled directly**: no supercell, no random seed.

---

## 🌐 Online app

Use it in the browser, nothing to install:
**[prdf-xrdlicious.streamlit.app](https://prdf-xrdlicious.streamlit.app/)**

`prdf_app.py` in this repository is that app: the XRDlicious (P)RDF calculator,
with RDF-Rust, Vesin and Matminer as selectable methods. To run it locally:

```bash
pip install -r requirements.txt
streamlit run prdf_app.py
```

## Install

```bash
pip install rdf-rust
```

## Example

```python
import rdfrust

s = rdfrust.read("BaSrTiO3.cif")                 # reports disorder on stderr
r, partials, total = s.prdf(cutoff=10.0, bin_size=0.1, n_threads=0)

partials[("Ba", "O")]     # numpy array, g_BaO(r)
total                     # number-weighted total RDF
r                         # left edge of each bin
```

`cutoff` is the maximum pair length r_max in Å, `bin_size` the histogram width,
and `n_threads=0` uses every core. A complete runnable script is in
[`examples/run_prdf.py`](https://github.com/bracerino/rdf-rust/blob/main/examples/run_prdf.py).

Every element pair is computed automatically: there is no element list to pass.
A five-element alloy gives all 25 partials, keyed as `(A, B)` tuples.

### 📈 Saving plots

Add `save_plots` and the curves are also written to a folder as clean figures,
with the numbers in a CSV. It is off by default.

```python
s.prdf(cutoff=10.0, bin_size=0.1, save_plots="rdf_plots", title="BaSrTiO3")
```

This writes `total_rdf.png`, `partial_rdfs.png` (all pairs together),
one `partial_A-B.png` per pair, and `rdf.csv`. Needs matplotlib
(`pip install "rdf-rust[plot]"`).

## 🎞️ MD trajectories

Multi-frame XYZ files, as written by MD codes, are read directly. Choose how
often a frame is analysed with `every`:

```python
res = rdfrust.trajectory_prdf(
    "md_run.xyz",
    cutoff=10.0, bin_size=0.05,
    every=5,                  # every 5th frame
    save_plots="md_rdf",      # averages, animations and CSVs
)
res.total, res.partials       # averaged over the frames used
res.total_per_frame           # every frame, shape (n_frames, n_bins)
```

With `save_plots` it writes the **trajectory-averaged** total and partial RDFs
(with the frame-to-frame spread shaded), an **animation** of every analysed
frame (`total_rdf_animation.gif`, `partial_rdfs_animation.gif`), and the data
as CSV.

- Frames with their own `Lattice="..."` keep their own box, so NPT runs work.
- Plain XYZ has no cell: pass it with `cell=[a, b, c]` or a 3×3 matrix.
- Unwrapped coordinates (atoms outside the box) are handled.
- `start` and `stop` limit the frame range; skipped frames are not parsed, so
  `every=10` on a long run stays fast.

A runnable demo that writes and analyses a small trajectory:
[`examples/md_trajectory.py`](https://github.com/bracerino/rdf-rust/blob/main/examples/md_trajectory.py).

## 🚀 Speed and 💾 memory

RDF-Rust on **1 thread** (serial, like Matminer and Vesin) and on **4 threads**:

| 1 thread | 4 threads |
|---|---|
| ![Speed-up, 1 thread](https://raw.githubusercontent.com/bracerino/rdf-rust/main/benchmarks/scaling_speedup_t1.png) | ![Speed-up, 4 threads](https://raw.githubusercontent.com/bracerino/rdf-rust/main/benchmarks/scaling_speedup_t4.png) |
| ![Time, 1 thread](https://raw.githubusercontent.com/bracerino/rdf-rust/main/benchmarks/scaling_time_t1.png) | ![Time, 4 threads](https://raw.githubusercontent.com/bracerino/rdf-rust/main/benchmarks/scaling_time_t4.png) |

![Peak memory](https://raw.githubusercontent.com/bracerino/rdf-rust/main/benchmarks/scaling_memory_t1.png)

| atoms | Matminer | Vesin | ⚡ RDF-Rust, 1 thread | 🚀 RDF-Rust, 4 threads | 💾 RDF-Rust memory |
|------:|---------:|------:|---------------------:|----------------------:|------------------:|
| 320 | 313 ms / 77 MB | 15 ms / 10 MB | **6.3 ms** | **1.8 ms** | **< 1 MB** |
| 2,560 | 2,840 ms / 592 MB | 47 ms / 93 MB | **25 ms** | **6.6 ms** | **< 1 MB** |
| 13,720 | 14,322 ms / 3,178 MB | 276 ms / 454 MB | **144 ms** | **42 ms** | **< 1 MB** |

Memory stays flat because pairs go straight from the neighbour search into the
histogram and are never stored. Matminer and Vesin keep every pair distance,
so their memory grows with the structure.

Results match Matminer to 3.6 × 10⁻¹⁵ (double-precision noise). Full numbers,
structure types, thread scaling and the data behind every figure:
**[BENCHMARKS.md](https://github.com/bracerino/rdf-rust/blob/main/BENCHMARKS.md)**.

## 🧩 Partial occupancies

**How it works:** each site *i* carries an occupancy $c_i^A$ for every species
*A* on it. A pair of sites (*i*, *j*) within the cutoff adds $c_i^A c_j^B$ to
the A–B histogram, instead of 1. Normalising by the number of A atoms,
$N_A = \sum_i c_i^A$, gives

$$g_{AB}(r) = \frac{1}{N_A}\sum_{i \neq j} c_i^A c_j^B \frac{\delta(r - r_{ij})}{4\pi r^2 \Delta r}$$

This is the exact average over every random arrangement of the disordered
sites, so no supercell, random seed or rounding of occupancies is needed. When
all occupancies are 1 it is the ordinary RDF.

If you need an explicit ordered structure instead, for example to export it:
`ordered, info = s.expand_occupancies(max_sites=4096, seed=42)`.

## Formats

| format | partial occupancies |
|---|---|
| CIF | **yes** |
| VASP POSCAR / CONTCAR | no |
| extended XYZ, single frame or trajectory | no |
| plain XYZ trajectory (with `cell=`) | no |
| anything else | read through pymatgen or ASE, if installed |

## 📖 How to cite

RDF-Rust is part of the XRDlicious project. If you use it, please cite:

- **XRDlicious:**
  Lebeda, M. *et al.* XRDlicious: an interactive web-based platform for online
  calculation of diffraction patterns and radial distribution functions from
  crystal structures. *Journal of Applied Crystallography* **58**(5) (2025).
  [journals.iucr.org](https://journals.iucr.org/j/issues/2025/05/00/hat5006/index.html)
- **XRD-Rust:**
  Lebeda, M. *et al.* Rust-accelerated powder X-ray diffraction simulation for
  high-throughput and machine-learning-driven materials science.
  *Journal of Applied Crystallography* **59**(4), 1344–1350 (2026).
  [journals.iucr.org](https://journals.iucr.org/j/issues/2026/04/00/hat5023/index.html)
- **RDF-Rust:** this repository,
  [github.com/bracerino/rdf-rust](https://github.com/bracerino/rdf-rust).

## Tests

```bash
cargo test --lib     # Rust unit tests
pytest tests/ -q     # CIF, formats, neighbours, RDF, disorder physics
```

## License

MIT
