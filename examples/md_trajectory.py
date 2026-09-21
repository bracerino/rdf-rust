"""RDF of an MD trajectory: averaged over frames, plotted and animated.

Writes a small synthetic trajectory first (SrTiO3 heating up, so the peaks
broaden and the box expands frame by frame), then analyses it:

    python examples/md_trajectory.py

With your own run, only the last call is needed:

    rdfrust.trajectory_prdf("my_md.xyz", every=5, save_plots="rdf_out")
"""
import numpy as np

import rdfrust


def write_demo_trajectory(path, n_frames=60, reps=3, seed=0):
    """Extended XYZ with a per-frame Lattice, as ASE and most MD codes write it."""
    rng = np.random.default_rng(seed)
    a0 = 3.905
    base = np.array([[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    symbols = ["Sr", "Ti", "O", "O", "O"]
    cells = np.array([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)])
    frac = ((base[None, :, :] + cells[:, None, :]) / reps).reshape(-1, 3)
    species = symbols * len(cells)
    with open(path, "w") as fh:
        for t in range(n_frames):
            heat = t / (n_frames - 1)
            a = a0 * (1 + 0.01 * heat) * reps            # thermal expansion
            sigma = 0.03 + 0.22 * heat                   # growing vibrations, in A
            pos = frac * a + rng.normal(0.0, sigma, frac.shape)
            fh.write(f"{len(pos)}\n")
            fh.write(f'Lattice="{a} 0 0 0 {a} 0 0 0 {a}" '
                     f"Properties=species:S:1:pos:R:3 step={t * 100}\n")
            for el, p in zip(species, pos):
                fh.write(f"{el} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


if __name__ == "__main__":
    write_demo_trajectory("demo_md.xyz")
    res = rdfrust.trajectory_prdf(
        "demo_md.xyz",
        cutoff=8.0,
        bin_size=0.05,
        every=2,                    # use every 2nd frame
        save_plots="demo_md_rdf",   # averaged plots, animations and CSVs go here
        title="SrTiO3, heating",
    )
    print(f"{len(res.frames)} frames averaged; pairs: {sorted(res.partials)}")
    for name, path in res.files.items():
        print(f"  {name:22} {path}")
