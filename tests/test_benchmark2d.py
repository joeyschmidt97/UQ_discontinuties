"""Numerical contracts that can otherwise silently change the winning arm."""
import numpy as np
import pytest
from benchmark2d.core import (Surface, Observations, BudgetExceeded, evaluation_set,
                              reconstruct, rmse)
from benchmark2d.strategies import run_arm
from benchmark2d.report import qualifying_cost


def test_linear_reconstruction_recovers_affine_surface_everywhere():
    obs = Observations(lambda x: 2*x[:, 0]-3*x[:, 1]+.7, 16)
    q = np.random.default_rng(8).random((512, 2))
    assert np.allclose(reconstruct(obs.x, obs.y, q), 2*q[:, 0]-3*q[:, 1]+.7)


def test_batch_budget_is_atomic_and_cache_does_not_charge_again():
    obs = Observations(Surface("kink"), 9)
    obs([[0., 0.], [.4, .5], [.4, .5]])
    assert len(obs.x) == 5
    before = obs.x.copy()
    with pytest.raises(BudgetExceeded):
        obs([[.1, .1], [.2, .1], [.3, .1], [.4, .1], [.5, .1]])
    assert np.array_equal(before, obs.x)


def test_nonfinite_predictions_are_a_failure_not_zero_error():
    with pytest.raises(ValueError):
        rmse([0., np.nan], [0., 1.])


def test_no_silent_extrapolation_outside_hull():
    with pytest.raises(ValueError):
        reconstruct([[0., 0.], [.5, 0.], [0., .5]], [0., 1., 1.], [[1., 1.]])


def test_offset_peaks_have_known_distances_and_kink_is_continuous():
    s = Surface("offset-peaks", 2)
    assert np.allclose(s.distance(s.centers()), [.17, -.17])
    k = Surface("kink", 2)
    p = np.array([.5, .5])
    assert abs(k([p+1e-9*k.normal])[0]-k([p-1e-9*k.normal])[0]) < 1e-8
    jump = Surface("jump", 2)
    assert abs(jump([p+1e-9*k.normal])[0]-jump([p-1e-9*k.normal])[0]) > .44


@pytest.mark.parametrize("name", ["grid", "sobol", "triangles", "gpr-var", "gpr-grad"])
def test_designs_are_finite_reproducible_and_metered(name):
    runs = []
    for _ in range(2):
        obs = Observations(Surface("on-plane"), 16)
        run_arm(name, obs, seed=3)
        assert 4 <= len(obs.x) <= 16
        test = evaluation_set(Surface("on-plane"), 1024)
        assert np.isfinite(reconstruct(obs.x, obs.y, test[0])).all()
        runs.append(obs.x)
    assert np.array_equal(*runs)


def test_target_must_persist_and_failed_checkpoint_cannot_win():
    def row(n, error, status="ok"):
        return dict(n=n, budget=n, error=error, band_error=error, status=status)
    assert qualifying_cost([row(16, .03), row(32, .2)], .05, .1) is None
    assert qualifying_cost([row(16, .2), row(32, .03)], .05, .1) == 32
    assert qualifying_cost([row(16, .03), row(32, 0., "failed")], .05, .1) is None
    assert qualifying_cost([], .05, .1) is None
