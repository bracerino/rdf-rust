"""Standalone Streamlit front end for rdfrust.

Run with:  streamlit run app.py

Everything the library reports about a disordered structure is also printed to
the console the server was started from, which is what the --server.headless
workflow reads.
"""
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rdfrust

st.set_page_config(page_title="rdfrust – (P)RDF", layout="wide")

# Validated categorical palette; assigned in fixed order, never cycled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FONT = dict(size=18, color="black")


def log(msg):
    """Echo to the console the server was launched from."""
    print(msg, file=sys.stderr, flush=True)


def measure(fn):
    """(result, seconds, peak MB). malloc_trim first, or glibc's retained
    pages make every run after the first read far too low."""
    import gc
    import threading

    try:
        import ctypes
        import psutil

        proc = psutil.Process()
        trim = getattr(ctypes.CDLL("libc.so.6"), "malloc_trim", None)
    except Exception:
        proc, trim = None, None
    if proc is None:
        t0 = time.perf_counter()
        return fn(), time.perf_counter() - t0, float("nan")
    gc.collect()
    if trim:
        try:
            trim(0)
        except Exception:
            pass
    base = peak = proc.memory_info().rss
    stop = False

    def sample():
        nonlocal peak
        while not stop:
            try:
                peak = max(peak, proc.memory_info().rss)
            except Exception:
                return
            time.sleep(0.001)

    w = threading.Thread(target=sample, daemon=True)
    w.start()
    t0 = time.perf_counter()
    try:
        out = fn()
    finally:
        secs = time.perf_counter() - t0
        stop = True
        w.join(timeout=1.0)
    return out, secs, max(peak - base, 0) / 2 ** 20


st.title("rdfrust — partial & total RDF")
st.caption(
    f"Rust backend, version {rdfrust.__version__} · {rdfrust.available_threads()} cores visible · "
    "partial occupancies handled natively, without building a supercell"
)

with st.sidebar:
    st.header("Structure")
    uploaded = st.file_uploader("CIF file", type=["cif"])
    example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    examples = sorted(f for f in os.listdir(example_dir)) if os.path.isdir(example_dir) else []
    example = st.selectbox("…or an example", ["—"] + examples, index=0)

    st.header("Parameters")
    cutoff = st.number_input(
        "Maximum length r_max (Å)", min_value=1.0, max_value=100.0, value=10.0, step=0.5,
        help="Pairs farther apart than this are not counted.",
    )
    bin_size = st.number_input(
        "Bin size (Å)", min_value=0.001, max_value=2.0, value=0.1, step=0.01, format="%.3f",
    )
    max_cores = rdfrust.available_threads()
    n_threads = st.slider(
        "Number of cores", min_value=0, max_value=max_cores, value=0,
        help="0 uses every core. The calculation is identical either way; only the time changes.",
    )
    st.header("Supercell (optional)")
    rep = st.columns(3)
    na = rep[0].number_input("a", 1, 20, 1, key="na")
    nb = rep[1].number_input("b", 1, 20, 1, key="nb")
    nc = rep[2].number_input("c", 1, 20, 1, key="nc")
    run = st.button("▶️ Calculate", type="primary", width="stretch")

source_name, structure = None, None
if uploaded is not None:
    source_name = uploaded.name
    text = uploaded.read().decode("utf-8", errors="replace")
    try:
        structure = rdfrust.Structure.from_cif_str(text)
    except Exception as exc:
        st.error(f"Could not read **{source_name}**: {exc}")
elif example != "—":
    source_name = example
    try:
        structure = rdfrust.Structure.from_cif(os.path.join(example_dir, example))
    except Exception as exc:
        st.error(f"Could not read **{source_name}**: {exc}")

if structure is None:
    st.info("⬅️ Upload a CIF or pick an example to begin.")
    st.stop()

if na * nb * nc > 1:
    structure = structure.make_supercell(int(na), int(nb), int(nc))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sites", structure.n_sites)
c2.metric("Atoms", f"{structure.n_atoms:g}")
c3.metric("Elements", len(structure.elements))
c4.metric("Cell volume", f"{structure.volume:.1f} Å³")

# Disorder is reported in the page and on the console, once per load.
note = rdfrust.describe_disorder(structure, name=source_name)
if note:
    st.warning("**This structure has partial occupancies.**")
    summary = structure.disorder_summary()
    rows = []
    for e in summary:
        fx, fy, fz = e["frac_coords"]
        rows.append({
            "Site": e["site_index"],
            "Fractional coords": f"({fx:.4f}, {fy:.4f}, {fz:.4f})",
            **{f"{el}": round(o, 4) for el, o in sorted(e["occupancies"].items())},
            "Vacancy": round(e["vacancy"], 4),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "The RDF below is weighted by these occupancies — the exact ensemble average over an "
        "ideally mixed lattice. No supercell is built and no random seed is involved, so the "
        "result is reproducible and free of the finite-size artefacts a random supercell carries."
    )
    if st.session_state.get("_logged") != source_name:
        log(f"\n[rdfrust] {note}")
        st.session_state["_logged"] = source_name
else:
    st.success(f"**{source_name}** is fully ordered — {structure!r}")

if run:
    with st.spinner("Calculating…"):
        (r, partials, total), secs, mem = measure(
            lambda: structure.prdf(cutoff=cutoff, bin_size=bin_size, n_threads=int(n_threads))
        )
    log(
        f"[rdfrust] {source_name}: {structure.n_sites} sites, r_max={cutoff} Å, "
        f"bin={bin_size} Å, threads={n_threads or 'all'} -> {secs * 1000:.1f} ms, "
        f"{mem:.1f} MB peak"
    )
    st.session_state["result"] = (r, {k: np.asarray(v) for k, v in partials.items()},
                                  np.asarray(total), secs, mem, cutoff, bin_size)

if "result" not in st.session_state:
    st.stop()

r, partials, total, secs, mem, used_cut, used_bs = st.session_state["result"]
m1, m2, m3 = st.columns(3)
m1.metric("Time", f"{secs * 1000:.1f} ms")
m2.metric("Peak memory", "n/a" if mem != mem else f"{mem:.1f} MB")
m3.metric("Cores used", n_threads or rdfrust.available_threads())

elements = structure.elements
pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
pairs = [p for p in pairs if p in partials]


def make_fig(title):
    return dict(
        title=dict(text=title, font=FONT),
        xaxis=dict(title=dict(text="r (Å)", font=FONT), tickfont=FONT),
        yaxis=dict(title=dict(text="g(r)", font=FONT), tickfont=FONT, rangemode="tozero"),
        hovermode="x unified",
        font=FONT,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=60, b=80),
    )


tab_p, tab_t, tab_d = st.tabs(["📊 Partial RDF", "📈 Total RDF", "💾 Download"])

with tab_p:
    # One plot per pair by default: with more than eight pairs a single overlay
    # would have to reuse colours, and a repeated hue reads as a repeated series.
    layout = st.radio("Layout", ["One plot per pair", "All pairs in one plot"],
                      horizontal=True, index=0)
    if layout == "All pairs in one plot":
        if len(pairs) > len(PALETTE):
            st.warning(
                f"{len(pairs)} pairs but only {len(PALETTE)} distinct colours — "
                "use one plot per pair to keep them apart."
            )
        fig = go.Figure()
        for i, pair in enumerate(pairs[: len(PALETTE)]):
            fig.add_trace(go.Scatter(x=r, y=partials[pair], mode="lines",
                                     name=f"{pair[0]}–{pair[1]}",
                                     line=dict(color=PALETTE[i], width=2)))
        fig.update_layout(**make_fig(f"Partial RDF — {source_name}"))
        st.plotly_chart(fig, width="stretch")
    else:
        for i, pair in enumerate(pairs):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r, y=partials[pair], mode="lines",
                                     name=f"{pair[0]}–{pair[1]}",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
            fig.update_layout(**make_fig(f"g({pair[0]}–{pair[1]})"))
            st.plotly_chart(fig, width="stretch")

with tab_t:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=total, mode="lines", name="total RDF",
                             line=dict(color=PALETTE[0], width=2)))
    rho = structure.n_atoms / structure.volume
    fig.add_hline(y=rho, line=dict(color="#52514e", width=1, dash="dot"),
                  annotation_text=f"number density ρ = {rho:.4f} Å⁻³",
                  annotation_position="top left")
    fig.update_layout(**make_fig(f"Total RDF — {source_name}"))
    st.plotly_chart(fig, width="stretch")
    st.caption("G(r) approaches the bulk number density ρ at large r.")

with tab_d:
    df = pd.DataFrame({"r_Angstrom": r, "total_RDF": total})
    for pair in pairs:
        df[f"g_{pair[0]}_{pair[1]}"] = partials[pair]
    st.dataframe(df.head(20), width="stretch")
    base = os.path.splitext(source_name or "structure")[0]
    st.download_button("⬇️ Download all curves (CSV)",
                       df.to_csv(index=False).encode(),
                       file_name=f"{base}_rdf.csv", mime="text/csv", type="primary")
