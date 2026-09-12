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
    obs = Observations(Surface("two-plane-four-peaks"), 9)
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


@pytest.mark.parametrize("case,count", [("smooth",1), ("two-plane-four-peaks",4),
                                      ("three-plane-three-peaks",3), ("two-plane-asymmetric",3)])
def test_peaks_are_inside_regions_and_folds_are_continuous(case,count):
    for seed in range(3):
        s = Surface(case, seed)
        assert len(s.centers()) == count
        assert ((s.centers()>0) & (s.centers()<1)).all()
        if case != "smooth":
            assert (s.distance(s.centers()) > .15).all()
            expected = [1,1,1] if count == 3 and "three-plane" in case else ([2,2] if count == 4 else [2,1])
            assert np.bincount(s.region(s.centers())).tolist() == expected
        p = np.array([.5,.5])
        assert abs(s([p+1e-9*s.normal])[0]-s([p-1e-9*s.normal])[0]) < 1e-7


@pytest.mark.parametrize("name", ["grid", "moe", "triangles", "gpr-var", "gpr-grad", "gpr-blend", "moe-tri75", "moe-tri50"])
def test_designs_are_finite_reproducible_and_metered(name):
    runs = []
    for _ in range(2):
        obs = Observations(Surface("two-plane-four-peaks"), 16)
        run_arm(name, obs, seed=3)
        assert len(obs.x) == 16
        test = evaluation_set(Surface("two-plane-four-peaks"), 1024)
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


@pytest.mark.parametrize("name", ["grid", "triangles", "gpr-var", "gpr-grad", "moe", "gpr-blend", "moe-tri75", "moe-tri50"])
def test_budget_does_not_change_earlier_decisions(name):
    small = Observations(Surface("smooth"), 12)
    large = Observations(Surface("smooth"), 16)
    run_arm(name, small, 2)
    run_arm(name, large, 2)
    assert np.array_equal(small.x, large.x[:12])


def test_mixture_gate_uses_only_observed_prequential_errors():
    from benchmark2d.mixture import gating
    query = np.array([[.2,.2],[.8,.8]])
    assert np.allclose(gating(query, [], []), 1/3)
    weights = gating(query, [[.2,.2],[.8,.8]], [[0,1,2],[2,1,0]])
    assert np.allclose(weights.sum(axis=1), 1)
    assert weights[0,0] > weights[0,2]
    assert weights[1,2] > weights[1,0]


def test_plot_does_not_average_an_incomplete_paired_seed_set():
    import matplotlib.pyplot as plt
    from benchmark2d.report import curve
    fig, ax = plt.subplots()
    rows = [dict(budget=9,n=9,error=.1,seed=0),
            dict(budget=10,n=10,error=.08,seed=0),
            dict(budget=10,n=10,error=.06,seed=1)]
    curve(ax, rows, "error", "black", "test", expected_seeds=[0,1])
    assert ax.lines[0].get_xdata().tolist() == [10]
    assert np.allclose(ax.lines[0].get_ydata(), [.07])
    plt.close(fig)


def test_combined_error_pools_squared_errors_and_requires_matched_field():
    from benchmark2d.report import aggregate_errors
    cfg = dict(cases=["a", "b"], seeds=[0, 1], arms=["one", "two"])
    rows = [dict(case=case, seed=seed, arm=arm, n=9, budget=9, status="ok", error=value)
            for arm in cfg["arms"] for case, seed, value in [("a",0,.1),("a",1,.1),("b",0,.3),("b",1,.3)]]
    data = dict(config=cfg, rows=rows)
    result = aggregate_errors(data)
    assert np.isclose(result["one"][0]["error"], np.sqrt(.05))
    assert result["one"][0]["tests"] == 4
    assert aggregate_errors(dict(config=cfg, rows=rows[:-1])) == {"one": [], "two": []}
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_errors(dict(config=cfg, rows=rows+[rows[0]]))
    with pytest.raises(ValueError, match="finite"):
        aggregate_errors(dict(config=cfg, rows=[dict(rows[0], error=float("nan"))]+rows[1:]))


def test_acquisition_blend_normalizes_before_weighting():
    from benchmark2d.strategies import normalized_blend
    assert np.allclose(normalized_blend([10., 0.], [0., 2.], .5), [.5, .5])
    assert np.allclose(normalized_blend([10., 0.], [0., 2.], .75), [.75, .25])
    assert np.allclose(normalized_blend([0., 0.], [0., 0.], .5), 0)
    with pytest.raises(ValueError):
        normalized_blend([1.], [1.], 1.1)
