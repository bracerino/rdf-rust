# Releasing

One-time setup: store a PyPI API token as a repository secret named
**`PYPI_API_TOKEN`** (GitHub → Settings → Secrets and variables → Actions →
New repository secret). The token must be allowed to upload `rdf-rust`: for
the very first upload that means an account-wide token, since a token scoped
to another project cannot create a new one.

To cut a release:

1. Bump the version in **both** `Cargo.toml` and `pyproject.toml` — they must
   match, or the wheel and its metadata disagree.
2. `cargo test --lib && pytest tests/ -q`
3. Commit, then tag and push:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

The `release.yml` workflow builds wheels for Linux (x86_64, aarch64), macOS
(Intel, Apple silicon) and Windows, plus a source distribution, and publishes
the lot. The wheels are `abi3-py38`, so one wheel per platform covers every
Python from 3.8 up.

## Checking locally first

```bash
maturin build --release        # wheel into target/wheels/
maturin sdist                  # source distribution
pip install target/wheels/rdf_rust-*.whl
python -c "import rdfrust; print(rdfrust.__version__)"   # import name is rdfrust
```
