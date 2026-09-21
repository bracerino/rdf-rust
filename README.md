# RDF-Rust

Partial and total radial distribution functions for crystal structures.
Rust backend, Python API, **native support for partially occupied sites**.

- **Fast** — 144 ms on one thread (42 ms on four) for a 13,720-atom cell
  where matminer needs 14.3 s.
- **Light** — memory does not grow with the structure; under 1 MB at any size.
- **Disorder-aware** — a CIF with fractional occupancies is handled directly,
  with no supercell and no random seed.

---

## Install

```bash
pip install rdf-rust
```

The distribution is **`rdf-rust`**; the import name is **`rdfrust`**.

From a checkout:

```bash
pip install maturin
maturin develop --release
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
`n_threads=0` means every core. A complete runnable script is in
[`examples/run_prdf.py`](examples/run_prdf.py):

```bash
python examples/run_prdf.py examples/BaSrTiO3_sym.cif
```

### Elements are never specified

**Every element pair is computed automatically.** There is no element list to
pass — RDF-Rust reads which elements are present and returns all of them. A
five-element alloy gives 25 partials:

```python
s = rdfrust.read("CoCrFeMnNi.cif")
_r, partials, _t = s.prdf(cutoff=6.0, bin_size=0.1)
len(partials)             # 25  = 5 x 5, every ordered combination
sorted(partials)[:3]      # [('Co','Co'), ('Co','Cr'), ('Co','Fe')]
```

Both `(A, B)` and `(B, A)` are present. `g_AB` counts neighbours of type B
around centres of type A, normalised by N_A, so the two are related by
N_A·g_AB = N_B·g_BA rather than being equal.

## Speed

RDF-Rust on **1 thread** (same as Matminer and Vesin) and on **4 threads**:

| 1 thread | 4 threads |
|---|---|
| ![Speed-up, 1 thread](benchmarks/scaling_speedup_t1.png) | ![Speed-up, 4 threads](benchmarks/scaling_speedup_t4.png) |
| ![Time, 1 thread](benchmarks/scaling_time_t1.png) | ![Time, 4 threads](benchmarks/scaling_time_t4.png) |

![Peak memory](benchmarks/scaling_memory_t1.png)

| atoms | Matminer | Vesin | RDF-Rust, 1 thread | RDF-Rust, 4 threads |
|------:|---------:|------:|-------------------:|--------------------:|
| 320 | 313 ms / 77 MB | 15 ms / 10 MB | **6.3 ms** | **1.8 ms** |
| 2,560 | 2,840 ms / 592 MB | 47 ms / 93 MB | **25 ms** | **6.6 ms** |
| 13,720 | 14,322 ms / 3,178 MB | 276 ms / 454 MB | **144 ms** | **42 ms** |

RDF-Rust stays under 1 MB at every size, because pairs go straight from the
cell list into a histogram and are never stored. On one thread it is
1.5–2.2× faster than Vesin; 4 threads add another 3.4–4.0×.

Ordered structures match matminer to 3.6 × 10⁻¹⁵ — double-precision noise —
across cubic, hexagonal, triclinic, sheared and slab-like cells. The one
exception is a lattice whose spacing is an exact multiple of the bin width
(a = 3.600 Å with 0.1 Å bins), where distances sit on bin edges and a few ulp
decide the bin; counts are conserved, only their split between two adjacent
bins moves.

Numbers, per-structure-type results, thermal-disorder tests and every CSV
behind these figures: **[BENCHMARKS.md](BENCHMARKS.md)**.

## Partial occupancies

Most RDF tools need an ordered structure, so a disordered CIF must first become
a random supercell — which rounds the occupancies to whatever the cell allows
and commits to one arrangement. RDF-Rust weights each pair by the product of
its occupancies instead:

$$g_{AB}(r) = \frac{1}{N_A}\sum_{i \neq j} c_i^A c_j^B \frac{\delta(r - r_{ij})}{4\pi r^2 \Delta r}$$

That is the exact ensemble average over an ideally mixed lattice — no
supercell, no seed, no rounding — and it reduces to the ordinary counting
definition when every occupancy is 1, so ordered results match matminer bin for
bin. It is checked against the mean of hundreds of explicitly sampled random
supercells in `tests/test_disorder_physics.py`.

When you do want an explicit structure:

```python
ordered, info = s.expand_occupancies(max_sites=4096, seed=42)
info["scale"], info["occupancy_report"], info["warnings"]
```

## API

```python
rdfrust.read(path, verbose=True, fmt=None)   # -> Structure, reports disorder
rdfrust.from_pymatgen(pmg_structure)
rdfrust.available_threads()
```

| on a `Structure` | |
|---|---|
| `prdf(cutoff, bin_size, n_threads)` | `(r, {(A,B): g_AB}, total)` |
| `total_rdf(cutoff, bin_size, n_threads)` | `(r, total)` |
| `expand_occupancies(max_sites, seed)` | `(ordered Structure, info)` |
| `disorder_summary()` | per-site occupancies; `[]` when ordered |
| `make_supercell(a, b, c)` | |
| `n_sites`, `n_atoms`, `is_ordered`, `elements`, `composition`, `volume` | |
| `frac_coords`, `cart_coords`, `species`, `lattice_matrix` | |

## Formats

| format | partial occupancies |
|---|---|
| CIF | **yes** |
| VASP POSCAR / CONTCAR | no (format cannot express them) |
| extended XYZ (needs `Lattice="..."`) | no |
| anything else | via pymatgen or ASE, if installed |

CIF is the only one that carries fractional occupancies. Space groups must be
given as an explicit symmetry-operation loop; a CIF naming only a
Hermann–Mauguin symbol is read as P1.

## Graphical interface

```bash
pip install "rdfrust[app]"
streamlit run app.py
```

Upload a CIF, set r_max, bin size, an optional supercell and the core count;
see the structure summary, any partial occupancies, the curves, and the time
and memory each run took.

## Online (P)RDF app

`prdf_app.py` is the XRDlicious (P)RDF calculator, with RDF-Rust, Vesin and
Matminer as selectable backends. It runs on its own from this repository:

```bash
pip install -r requirements.txt
streamlit run prdf_app.py
```

`requirements.txt` is also what Streamlit Community Cloud reads when the app is
deployed from GitHub.

## Tests

```bash
cargo test --lib     # symmetry parsing
pytest tests/ -q     # CIF, formats, neighbours, RDF, disorder physics
```

## License

MIT
