"""The occupancy-weighted RDF must equal the average over random realisations.

This is the claim that justifies skipping the supercell entirely, so it is
checked against explicitly sampled structures rather than assumed.
"""
import warnings, numpy as np, pytest
warnings.filterwarnings("ignore")
import rdfrust

A = 3.905

def disordered_bsto(n, occ=(("Ba", 0.5), ("Sr", 0.5))):
    """(Ba,Sr)TiO3 supercell with the A site left partially occupied."""
    frac, species = [], []
    for ia in range(n):
        for ib in range(n):
            for ic in range(n):
                o = np.array([ia, ib, ic], dtype=float)
                for pos, sp in [
                    ([0, 0, 0], dict(occ)),
                    ([.5, .5, .5], {"Ti": 1.0}),
                    ([.5, .5, 0], {"O": 1.0}),
                    ([.5, 0, .5], {"O": 1.0}),
                    ([0, .5, .5], {"O": 1.0}),
                ]:
                    frac.append(list((np.array(pos) + o) / n))
                    species.append(sp)
    m = [[A*n, 0, 0], [0, A*n, 0], [0, 0, A*n]]
    return rdfrust.Structure(m, frac, species)

def realisation(template, rng):
    """One structure with each shared site resolved by independent sampling."""
    species = []
    for sp in template.species:
        if len(sp) == 1 and abs(sum(sp.values()) - 1.0) < 1e-12:
            species.append(sp); continue
        els = sorted(sp); p = [sp[e] for e in els]
        u = rng.random(); acc = 0.0; chosen = None
        for e, pe in zip(els, p):
            acc += pe
            if u < acc: chosen = e; break
        species.append({chosen: 1.0} if chosen else {})
    keep = [i for i, s in enumerate(species) if s]
    return rdfrust.Structure(template.lattice_matrix,
                             [template.frac_coords[i] for i in keep],
                             [species[i] for i in keep])

def ensemble_average(template, n_samples, cutoff, bin_size, seed=0):
    """Average g_AB over independently sampled realisations.

    Each realisation is rescaled by its own N_A over the expected N_A. Without
    that, averaging g = hist/(shell*N_A) with a *fluctuating* N_A in the
    denominator biases the mean high by ~(1-p)/N_A -- an artefact of the
    comparison, not of the weighted formula, and one that dies off as the cell
    grows. Rescaling puts both sides over the same denominator so the
    comparison tests the physics instead of Jensen's inequality.
    """
    rng = np.random.default_rng(seed)
    n_expected = {}
    for sp in template.species:
        for el, occ in sp.items():
            n_expected[el] = n_expected.get(el, 0.0) + occ
    acc, count = {}, 0
    for _ in range(n_samples):
        s = realisation(template, rng)
        comp = s.composition
        _, partials, _ = s.prdf(cutoff=cutoff, bin_size=bin_size)
        for k, v in partials.items():
            scale = comp.get(k[0], 0.0) / n_expected[k[0]]
            acc[k] = acc.get(k, 0.0) + np.asarray(v) * scale
        count += 1
    return {k: v / count for k, v in acc.items()}

@pytest.mark.parametrize("occ,label", [
    ((("Ba", 0.5), ("Sr", 0.5)), "50/50 substitution"),
    ((("Ba", 0.7), ("Sr", 0.3)), "70/30 substitution"),
    ((("Ba", 0.75),),            "25% vacancies"),
])
@pytest.mark.parametrize("seed", [1, 2])
def test_weighted_equals_ensemble_average(occ, label, seed):
    # The cell must be wider than the cutoff, otherwise each atom sees its own
    # periodic images -- see test_self_images_are_uncorrelated for why that
    # makes the sampled average disagree on purpose.
    cutoff, bs, n = 6.0, 0.5, 5
    tmpl = disordered_bsto(n, occ)
    _, weighted, _ = tmpl.prdf(cutoff=cutoff, bin_size=bs)
    avg = ensemble_average(tmpl, 400, cutoff, bs, seed=seed)
    for key in weighted:
        w = np.asarray(weighted[key]); a = avg[key]
        mask = w > 0.02 * w.max()          # bins with real signal
        if mask.sum() == 0: continue
        rel = np.abs(a[mask] - w[mask]) / w[mask]
        assert rel.max() < 0.05, f"{label} {key}: max rel dev {rel.max():.3f}"
        assert rel.mean() < 0.025, f"{label} {key}: mean rel dev {rel.mean():.3f}"


def test_self_images_are_uncorrelated():
    """A site and its own periodic images are treated as independent atoms.

    A CIF with fractional occupancies describes a randomly substituted crystal,
    not a periodic superstructure, so two copies of a site one lattice vector
    apart are two different atoms that were occupied independently. The
    weighted RDF therefore weights such a pair by p*p.

    Building an explicit supercell cannot express that: within one periodic
    realisation a site and its image are necessarily the *same* atom, so they
    are perfectly correlated and contribute p instead of p*p. The sampled
    average is then too large by 1/p exactly, which is the artefact the
    weighted route avoids. This test pins that factor down so the intended
    semantics cannot drift.
    """
    cutoff, bs, p = 8.0, 0.2, 0.3
    tmpl = disordered_bsto(1, (("Ba", 1.0 - p), ("Sr", p)))   # cell 3.9 A << cutoff
    _, weighted, _ = tmpl.prdf(cutoff=cutoff, bin_size=bs)
    avg = ensemble_average(tmpl, 3000, cutoff, bs, seed=3)
    w = np.asarray(weighted[("Sr", "Sr")])
    mask = w > 0.02 * w.max()
    ratio = (avg[("Sr", "Sr")][mask] / w[mask])
    assert np.allclose(ratio, 1.0 / p, rtol=0.05), ratio

def test_ordered_sites_unaffected_by_disorder_elsewhere():
    """Ti-Ti and O-O are fully ordered; their PRDF must be exact, not averaged."""
    cutoff, bs = 8.0, 0.2
    tmpl = disordered_bsto(3)
    _, weighted, _ = tmpl.prdf(cutoff=cutoff, bin_size=bs)
    rng = np.random.default_rng(1)
    one = realisation(tmpl, rng)
    _, single, _ = one.prdf(cutoff=cutoff, bin_size=bs)
    for key in [("Ti", "Ti"), ("O", "O"), ("Ti", "O")]:
        assert np.allclose(weighted[key], single[key], atol=1e-9), key

def test_weighted_reduces_to_ordered():
    """With every occupancy at 1 the weighted result is the ordinary one."""
    tmpl = disordered_bsto(2, (("Ba", 1.0),))
    _, w, _ = tmpl.prdf(cutoff=8.0, bin_size=0.1)
    one = realisation(tmpl, np.random.default_rng(0))
    _, s, _ = one.prdf(cutoff=8.0, bin_size=0.1)
    for k in w:
        assert np.array_equal(np.asarray(w[k]), np.asarray(s[k])), k
