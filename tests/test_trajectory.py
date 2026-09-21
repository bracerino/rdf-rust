"""MD trajectories, plotting, and coordinates outside the cell."""
import os

import numpy as np
import pytest

import rdfrust

A = 3.905
BASE = np.array([[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
SYMBOLS = ["Sr", "Ti", "O", "O", "O"]


def frame_positions(reps, a, sigma, rng):
    cells = np.array([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)])
    frac = ((BASE[None] + cells[:, None]) / reps).reshape(-1, 3)
    return frac * a * reps + rng.normal(0, sigma, frac.shape), SYMBOLS * len(cells)


def write_xyz(path, n_frames, reps=2, lattice=True, props=None, scale_per_frame=0.0,
              garbage_frame=None, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w") as fh:
        for t in range(n_frames):
            a = A * (1 + scale_per_frame * t)
            pos, species = frame_positions(reps, a, 0.05, rng)
            L = a * reps
            fh.write(f"{len(pos)}\n")
            comment = f'Lattice="{L} 0 0 0 {L} 0 0 0 {L}" ' if lattice else ""
            comment += props or ("Properties=species:S:1:pos:R:3" if lattice else "plain")
            fh.write(comment + "\n")
            for el, p in zip(species, pos):
                if t == garbage_frame:
                    fh.write(f"{el} not a number at_all\n")
                elif props and "forces" in props:
                    fh.write(f"{el} 0.1 0.2 0.3 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                else:
                    fh.write(f"{el} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def test_every_start_stop(tmp_path):
    path = tmp_path / "t.xyz"
    write_xyz(path, 20)
    _, idx = rdfrust.read_trajectory(path)
    assert idx == list(range(20))
    _, idx = rdfrust.read_trajectory(path, every=5)
    assert idx == [0, 5, 10, 15]
    _, idx = rdfrust.read_trajectory(path, every=3, start=2, stop=12)
    assert idx == [2, 5, 8, 11]


def test_skipped_frames_are_not_parsed(tmp_path):
    """A malformed frame that is skipped must not stop the read."""
    path = tmp_path / "t.xyz"
    write_xyz(path, 6, garbage_frame=3)
    _, idx = rdfrust.read_trajectory(path, every=2)          # 0, 2, 4
    assert idx == [0, 2, 4]
    with pytest.raises(ValueError, match="expected a number"):
        rdfrust.read_trajectory(path, every=1)


def test_properties_column_order(tmp_path):
    """Positions found through Properties=, even after other columns."""
    a, b = tmp_path / "a.xyz", tmp_path / "b.xyz"
    write_xyz(a, 1, seed=3)
    write_xyz(b, 1, seed=3, props="Properties=species:S:1:forces:R:3:pos:R:3")
    (sa,), _ = rdfrust.read_trajectory(a)
    (sb,), _ = rdfrust.read_trajectory(b)
    assert np.allclose(sa.cart_coords, sb.cart_coords)


def test_plain_xyz_needs_a_cell(tmp_path):
    path = tmp_path / "plain.xyz"
    write_xyz(path, 2, lattice=False)
    with pytest.raises(ValueError, match="no cell was given"):
        rdfrust.read_trajectory(path)
    L = 2 * A
    s1, _ = rdfrust.read_trajectory(path, cell=[L, L, L])
    s2, _ = rdfrust.read_trajectory(path, cell=np.diag([L, L, L]))
    assert abs(s1[0].volume - L ** 3) < 1e-9
    assert np.allclose(s1[1].frac_coords, s2[1].frac_coords)


def test_box_is_followed_per_frame(tmp_path):
    """NPT-style runs: each frame keeps its own Lattice."""
    path = tmp_path / "npt.xyz"
    write_xyz(path, 4, scale_per_frame=0.01)
    frames, _ = rdfrust.read_trajectory(path)
    volumes = [f.volume for f in frames]
    assert all(v2 > v1 for v1, v2 in zip(volumes, volumes[1:]))


def test_average_is_mean_of_frames(tmp_path):
    path = tmp_path / "t.xyz"
    write_xyz(path, 8)
    res = rdfrust.trajectory_prdf(path, cutoff=6.0, bin_size=0.1, every=2, verbose=False)
    frames, _ = rdfrust.read_trajectory(path, every=2)
    totals = [np.asarray(f.prdf(cutoff=6.0, bin_size=0.1)[2]) for f in frames]
    assert res.frames == [0, 2, 4, 6]
    assert res.total_per_frame.shape == (4, len(res.r))
    assert np.allclose(res.total, np.mean(totals, axis=0))
    assert np.allclose(res.total_std, np.std(totals, axis=0, ddof=1))
    g = [np.asarray(f.prdf(cutoff=6.0, bin_size=0.1)[1][("Sr", "O")]) for f in frames]
    assert np.allclose(res.partials[("Sr", "O")], np.mean(g, axis=0))


def test_structures_in_memory(tmp_path):
    path = tmp_path / "t.xyz"
    write_xyz(path, 6)
    frames, _ = rdfrust.read_trajectory(path)
    a = rdfrust.trajectory_prdf(frames, cutoff=6.0, every=2, verbose=False)
    b = rdfrust.trajectory_prdf(path, cutoff=6.0, every=2, verbose=False)
    assert a.frames == b.frames
    assert np.allclose(a.total, b.total)


def test_unwrapped_coordinates(tmp_path):
    """Atoms outside the cell (unwrapped MD output) give the same RDF."""
    m = [[A, 0, 0], [0, A, 0], [0, 0, A]]
    sp = [{e: 1.0} for e in SYMBOLS]
    s = rdfrust.Structure(m, BASE.tolist(), sp).make_supercell(3, 3, 3)
    f = np.array(s.frac_coords)
    shifted = f + np.random.default_rng(1).integers(-4, 5, f.shape)
    u = rdfrust.Structure(s.lattice_matrix, shifted.tolist(), s.species)
    _, pa, ta = s.prdf(cutoff=8.0, bin_size=0.1)
    _, pb, tb = u.prdf(cutoff=8.0, bin_size=0.1)
    assert np.array_equal(np.asarray(ta), np.asarray(tb))
    for k in pa:
        assert np.array_equal(np.asarray(pa[k]), np.asarray(pb[k]))


def test_save_plots_single_structure(tmp_path):
    pytest.importorskip("matplotlib")
    m = [[A, 0, 0], [0, A, 0], [0, 0, A]]
    s = rdfrust.Structure(m, BASE.tolist(), [{e: 1.0} for e in SYMBOLS])
    out = tmp_path / "plots"
    r, p, t = s.prdf(cutoff=6.0, bin_size=0.1, save_plots=str(out), title="test")
    files = set(os.listdir(out))
    assert {"total_rdf.png", "partial_rdfs.png", "rdf.csv"} <= files
    assert {"partial_O-O.png", "partial_O-Sr.png", "partial_Sr-Ti.png"} <= files
    data = np.genfromtxt(out / "rdf.csv", delimiter=",", names=True)
    assert np.allclose(data["total"], t)
    # Off by default.
    before = set(os.listdir(tmp_path))
    s.prdf(cutoff=6.0, bin_size=0.1)
    assert set(os.listdir(tmp_path)) == before


def test_trajectory_plots_and_animation(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    path = tmp_path / "t.xyz"
    write_xyz(path, 10)
    out = tmp_path / "out"
    res = rdfrust.trajectory_prdf(path, cutoff=6.0, bin_size=0.1, every=2,
                                  save_plots=str(out), verbose=False)
    for key in ("total", "partials", "csv", "per_frame_csv",
                "total_animation", "partials_animation", "html"):
        assert os.path.exists(res.files[key]), key
    with Image.open(res.files["total_animation"]) as im:
        assert im.n_frames == 5
    # The HTML player is self-contained: every frame embedded, nothing fetched.
    page = open(res.files["html"], encoding="utf-8").read()
    assert page.count("data:image/png;base64,") == 2 * 5
    assert '"frames": [0, 2, 4, 6, 8]' in page
    assert "http://" not in page and "https://" not in page
    per_frame = np.genfromtxt(res.files["per_frame_csv"], delimiter=",", skip_header=1)
    assert per_frame.shape[0] == 5


def test_animation_is_capped(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    path = tmp_path / "t.xyz"
    write_xyz(path, 12)
    res = rdfrust.trajectory_prdf(path, cutoff=5.0, bin_size=0.2, save_plots=str(tmp_path / "o"),
                                  max_animation_frames=4, verbose=False)
    assert len(res.frames) == 12                   # the average uses them all
    with Image.open(res.files["total_animation"]) as im:
        assert im.n_frames == 4


def test_html_can_be_switched_off(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "t.xyz"
    write_xyz(path, 4)
    res = rdfrust.trajectory_prdf(path, cutoff=5.0, bin_size=0.2, save_plots=str(tmp_path / "o"),
                                  html=False, verbose=False)
    assert "html" not in res.files and "total_animation" in res.files


def test_progress_is_reported(tmp_path, capsys):
    path = tmp_path / "t.xyz"
    write_xyz(path, 6)
    rdfrust.trajectory_prdf(path, cutoff=5.0, bin_size=0.2, every=2)
    err = capsys.readouterr().err
    assert "using 3 of 6 frame(s)" in err
    assert "frames 3/3 (100%)" in err
    assert "finished in" in err
    rdfrust.trajectory_prdf(path, cutoff=5.0, bin_size=0.2, verbose=False)
    assert capsys.readouterr().err == ""
