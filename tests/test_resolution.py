"""The dimension-free variation estimator and fit-free resolution spine.

Every behavioural test runs at 2, 5 and 8 dimensions with the same assertion.
That is the property under test: the spine must not change meaning with the
input dimension, which is what the Delaunay estimator could not promise.
"""
import numpy as np
import pytest

from benchmarknd.core import Observations, SurfaceND, evaluation_set, score, truth_variation
from benchmarknd.strategies import run_arm
from resolution import fit_free_scores, knn_variation, spine_targets
from resolution.variation import stencil_size

DIMS = (2, 5, 8)


def design(dim, n=400, seed=0):
    return np.random.default_rng(seed).random((n, dim))


@pytest.mark.parametrize("dim", DIMS)
def test_plane_slope_is_recovered_and_its_curvature_is_zero(dim):
    """A steep plane is reconstructed exactly, so it must not attract budget."""
    x = design(dim)
    y = 3.*x[:, 0]
    out = knn_variation(x, y, design(dim, 200, seed=1))
    assert np.allclose(out.gradient, 3., atol=.05)
    assert out.curvature.max() < 1e-3


@pytest.mark.parametrize("dim", DIMS)
def test_curvature_separates_a_narrow_peak_from_a_steep_plane(dim):
    """The ordering slope alone gets wrong: modest peak beats steep plane."""
    x = design(dim)
    query = design(dim, 200, seed=1)
    plane = knn_variation(x, 3.*x[:, 0], query)
    peak = knn_variation(x, np.exp(-.5*np.sum(((x-.5)/.1)**2, axis=1)), query)
    assert plane.gradient.mean() > peak.gradient.mean()
    assert plane.curvature.mean() < peak.curvature.mean()


@pytest.mark.parametrize("dim", DIMS)
def test_a_jump_raises_the_residual_and_branch_restriction_removes_it(dim):
    x = design(dim, 600)
    labels = (x[:, 0] > .5).astype(int)
    y = x[:, 1] + 2.*labels                       # unit-scale jump across x0 = 0.5
    near = x[np.abs(x[:, 0]-.5) < .08][:50]
    spanning = knn_variation(x, y, near)
    restricted = knn_variation(x, y, near, labels=labels)
    assert spanning.residual.mean() > 5*restricted.residual.mean()


@pytest.mark.parametrize("dim", DIMS)
def test_indicator_modes_have_slope_units_and_rank_differently(dim):
    x = design(dim)
    query = design(dim, 100, seed=1)
    out = knn_variation(x, np.exp(-.5*np.sum(((x-.5)/.1)**2, axis=1)), query)
    spacing = np.full(len(query), .1)
    curvature = out.indicator(spacing, "curvature")
    gradient = out.indicator(spacing, "gradient")
    blend = out.indicator(spacing, "blend")
    assert np.allclose(blend, gradient+curvature)
    with pytest.raises(ValueError):
        out.indicator(spacing, "nonsense")


def test_stencil_must_leave_a_residual():
    assert stencil_size(5) >= 7
    with pytest.raises(ValueError):
        stencil_size(5, k=6)


@pytest.mark.parametrize("dim", DIMS)
def test_too_few_observations_is_refused_not_guessed(dim):
    with pytest.raises(ValueError):
        knn_variation(design(dim, dim+1), np.zeros(dim+1), design(dim, 3))


@pytest.mark.parametrize("dim", DIMS)
def test_refining_the_design_lowers_every_spine_quantile(dim):
    probes = design(dim, 500, seed=2)
    variation = np.ones(len(probes))
    coarse = fit_free_scores(design(dim, 60, seed=3), probes, variation, 1.)
    fine = fit_free_scores(design(dim, 600, seed=3), probes, variation, 1.)
    for key in ("fill_p95", "fill_max", "vwfd_p95", "vwfd_max", "vwfd_rms"):
        assert fine[key] < coarse[key], key
    for tau in spine_targets():
        key = f"vwfd_coverage_{int(round(tau*100)):02d}"
        assert 0. <= coarse[key] <= 1. and fine[key] >= coarse[key]


@pytest.mark.parametrize("dim", DIMS)
def test_spine_reports_the_worst_axis_and_refuses_bad_input(dim):
    probes = design(dim, 200, seed=2)
    points = design(dim, 100, seed=3)
    points[:, 1] *= .01                            # collapse one axis
    out = fit_free_scores(points, probes, np.ones(len(probes)), 1.)
    assert out["axis_spread_worst_index"] == 1
    assert out["axis_spread_min"] < out["axis_spread_mean"]
    with pytest.raises(ValueError):
        fit_free_scores(points, probes, np.ones(len(probes)), 0.)
    with pytest.raises(ValueError):
        fit_free_scores(points, probes, np.ones(len(probes)+1), 1.)


def test_truth_variation_is_exact_on_a_known_surface():
    surface = SurfaceND("5d-m1-p1-anis", 0)
    x = design(5, 64, seed=4)
    gradient, curvature = truth_variation(surface, x)
    assert gradient.shape == (64,) and np.isfinite(gradient).all()
    assert curvature.shape == (64,) and np.isfinite(curvature).all()
    assert (gradient > 0).all()


@pytest.mark.parametrize("case", ("5d-m2-rotated", "8d-m2-disjoint"))
def test_resolution_sampling_spends_the_exact_budget_and_declares_its_weights(case):
    surface = SurfaceND(case, 0)
    for arm, uncertainty in (("vwrs", 0.), ("vurs", 1/3)):
        obs = Observations(surface, 2*surface.dim+8, surface.dim, 0)
        predict, metadata = run_arm(arm, obs, 0)
        assert len(obs.x) == obs.budget
        assert metadata["acquisition_weights"]["uncertainty"] == pytest.approx(uncertainty)
        assert sum(metadata["acquisition_weights"].values()) == pytest.approx(1.)
        assert (predict is None) == (arm == "vwrs")


def test_holistic_score_is_not_silently_inherited_across_dimension():
    # A folded case: the fold-band term of H only exists where a fold does.
    surface = SurfaceND("5d-m2-rotated", 0)
    test = evaluation_set(surface, 4096)
    obs = Observations(surface, 32, 5, 0)
    while obs.remaining:
        obs(np.random.default_rng(obs.remaining).random((1, 5)))

    uncalibrated = score(surface, obs, test)
    assert uncalibrated["holistic_uncalibrated"] is True
    assert uncalibrated["holistic_error"] is None

    targets = dict(nmae=.05, band_nmae=.10, p95_error=.15, vwfd_p95=.25)
    calibrated = score(surface, obs, test, targets=targets)
    assert calibrated["holistic_uncalibrated"] is False
    assert calibrated["holistic_driver"] in targets
    assert calibrated["holistic_error"] == pytest.approx(
        max(calibrated[name]/limit for name, limit in targets.items()))
    with pytest.raises(ValueError):
        score(surface, obs, test, targets=dict(does_not_exist=.1))


def test_a_fold_free_control_refuses_a_fold_band_tolerance():
    """The one-mode controls have no fold, so H cannot carry a fold term."""
    surface = SurfaceND("5d-m1-p1-anis", 0)
    test = evaluation_set(surface, 4096)
    obs = Observations(surface, 32, 5, 0)
    while obs.remaining:
        obs(np.random.default_rng(obs.remaining).random((1, 5)))
    assert score(surface, obs, test)["band_nmae"] is None
    with pytest.raises(ValueError):
        score(surface, obs, test, targets=dict(nmae=.05, band_nmae=.10))


def test_the_rbf_evaluator_is_labelled_secondary_in_every_row():
    surface = SurfaceND("5d-m1-p1-anis", 0)
    test = evaluation_set(surface, 4096)
    obs = Observations(surface, 24, 5, 0)
    while obs.remaining:
        obs(np.random.default_rng(100+obs.remaining).random((1, 5)))
    row = score(surface, obs, test)
    assert row["reconstruction_role"] == "secondary"
    assert row["reconstruction"] == "thin-plate-spline-rbf"


def test_declared_tolerances_exist_for_five_and_refuse_eight():
    from benchmarknd.core import tolerances_for
    spine, ported = tolerances_for(5)
    assert set(spine) == {"vwfd_p95", "nonlinear_p95", "fill_p95"}
    assert set(ported) == {"nmae", "band_nmae", "p95_error", "vwfd_p95"}
    assert tolerances_for(5, band_available=False)[1].keys() == {"nmae", "p95_error", "vwfd_p95"}
    # 8D has not been calibrated; inheriting the 5D limits would be the bug.
    with pytest.raises(ValueError, match="no calibrated tolerances"):
        tolerances_for(8)


def test_spine_holistic_is_primary_and_ported_rides_along_labelled():
    from benchmarknd.core import tolerances_for
    surface = SurfaceND("5d-m2-rotated", 0)
    test = evaluation_set(surface, 4096)
    obs = Observations(surface, 40, 5, 0)
    while obs.remaining:
        obs(np.random.default_rng(obs.remaining).random((1, 5)))
    spine, ported = tolerances_for(5)
    row = score(surface, obs, test, targets=spine, secondary_targets=ported)
    assert row["holistic_uncalibrated"] is False
    assert row["holistic_driver"] in spine
    assert row["ported_holistic_uncalibrated"] is False
    assert row["ported_holistic_driver"] in ported
    # The primary score must not contain a reconstruction-based term.
    assert not {"nmae", "band_nmae", "p95_error"} & set(row["holistic_targets"])
