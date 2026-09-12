"""Numerical contracts that can otherwise silently change the winning arm."""
import json
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


@pytest.mark.parametrize("name", ["grid", "moe", "triangles", "gpr-var", "gpr-grad", "gpr-blend", "moe-tri75", "moe-tri50",
                                  "gpr-u20-g80", "gpr-u30-g70", "gpr-u50-g50", "gpr-u70-g30", "gpr-u80-g20"])
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


@pytest.mark.parametrize("name", ["grid", "triangles", "gpr-var", "gpr-grad", "moe", "gpr-blend", "moe-tri75", "moe-tri50",
                                  "gpr-u20-g80", "gpr-u30-g70", "gpr-u50-g50", "gpr-u70-g30", "gpr-u80-g20"])
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


def test_gp_blend_weights_are_declared_and_ordered():
    from benchmark2d.strategies import GP_BLENDS, ARMS
    assert GP_BLENDS["gpr-blend"] == GP_BLENDS["gpr-u50-g50"] == .5
    assert [GP_BLENDS[f"gpr-u{u}-g{100-u}"] for u in (20, 30, 50, 70, 80)] == [.2, .3, .5, .7, .8]
    assert all(name in ARMS for name in GP_BLENDS)


def test_fifty_fifty_blend_alias_reproduces_the_original_arm():
    old, new = Observations(Surface("two-plane-four-peaks"), 14), Observations(Surface("two-plane-four-peaks"), 14)
    run_arm("gpr-blend", old, 1)
    run_arm("gpr-u50-g50", new, 1)
    assert np.array_equal(old.x, new.x)


def test_blend_share_shifts_the_design_away_from_the_gradient_arm():
    designs = {}
    for name in ("gpr-grad", "gpr-u20-g80", "gpr-u80-g20", "gpr-var"):
        obs = Observations(Surface("two-plane-four-peaks"), 14)
        run_arm(name, obs, 1)
        designs[name] = obs.x
    assert not np.array_equal(designs["gpr-u20-g80"], designs["gpr-u80-g20"])
    assert not np.array_equal(designs["gpr-u20-g80"], designs["gpr-grad"])
    assert not np.array_equal(designs["gpr-u80-g20"], designs["gpr-var"])


def _payload(arms, errors, cases=("smooth",), seeds=(0,), budgets=(8,)):
    rows = [dict(arm=arm, case=case, seed=seed, budget=budget, n=budget, status="ok",
                 error=errors[arm], band_error=errors[arm])
            for arm in arms for case in cases for seed in seeds for budget in budgets]
    return dict(config=dict(cases=list(cases), seeds=list(seeds), budgets=list(budgets), arms=list(arms),
                            test_size=1024, epsilon=.05, band_epsilon=.1, protocol="p"),
                source_hash="h-"+"".join(arms), commit="c", rows=rows)


def test_merge_keeps_only_the_best_added_arms_and_records_provenance(tmp_path):
    from benchmark2d.merge import merge
    base, extra = _payload(["grid"], dict(grid=.04)), _payload(["a", "b", "c"], dict(a=.01, b=.03, c=.02))
    base_path, extra_path = tmp_path/"base.json", tmp_path/"extra.json"
    base_path.write_text(json.dumps(base)); extra_path.write_text(json.dumps(extra))
    merged, scored, selected = merge(base_path, [extra_path], top=2)
    assert [arm for arm, _, _ in scored] == ["a", "c", "b"]
    assert selected == ["a", "c"]
    assert merged["config"]["arms"] == ["grid", "a", "c"]
    assert {r["arm"] for r in merged["rows"]} == {"grid", "a", "c"}
    assert [p["source_hash"] for p in merged["provenance"]] == [base["source_hash"], extra["source_hash"]]


def test_merge_refuses_mismatched_configurations_and_duplicate_arms(tmp_path):
    from benchmark2d.merge import merge
    base = _payload(["grid"], dict(grid=.04))
    base_path = tmp_path/"base.json"
    base_path.write_text(json.dumps(base))
    other = _payload(["a"], dict(a=.01), seeds=(0, 1))
    other_path = tmp_path/"other.json"
    other_path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="seeds"):
        merge(base_path, [other_path], top=1)
    same = tmp_path/"same.json"
    same.write_text(json.dumps(_payload(["grid"], dict(grid=.01))))
    with pytest.raises(ValueError, match="already exist"):
        merge(base_path, [same], top=1)
