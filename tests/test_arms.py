"""
Tests for the arms whose backends are actually installed (sg_lib, GPR).

The SG++ arm is covered separately in `test_sgpp_arm_mock.py`, against a mock,
because pysgpp has no wheel on every platform.

Run:  python tests/test_arms.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from arms import design_metrics, score_arm                      # noqa: E402
from manifold import Manifold                                   # noqa: E402


# ===========================================================================
# sg_lib
# ===========================================================================
def test_sglib_spends_a_real_budget():
    """Regression test for the empty-step bug.

    `do_one_adaption_step_preproc()` returns only the newly ADMISSIBLE
    successors of the index it just retired, so it legitimately returns an empty
    array on many steps. Reading that as termination -- the obvious port of the
    reference script's fixed-step loop -- stopped the method after 3 steps and
    10 points regardless of budget. Anything in that range means the bug is back.
    """
    from arms import SgLibArm
    m = Manifold(mode="argmax", align="gap")
    oracle = m.oracle("gamma")
    arm = SgLibArm().fit(oracle, m.dim, 250)
    assert oracle.n_evals > 100, (
        f"sg_lib spent only {oracle.n_evals} of a 250 budget "
        f"(stopped_on={arm.stopped_on}) -- empty-step regression")
    assert oracle.n_evals <= 250
    assert arm.n_empty_steps > 0, "expected some empty adaption steps in 6-D"


def test_sglib_predict_works_after_a_budget_stop():
    """Regression test: stopping on budget must not poison the interpolant.

    `do_one_adaption_step_preproc()` appends its new multiindices to
    multiindex_set and extends the local basis BEFORE the caller decides whether
    it can pay for them. Breaking out without rolling that back leaves subspaces
    whose nodes were never evaluated, and predict() then raises a KeyError
    naming the unpaid multiindex (e.g. '[5, 1, 1, 1, 1, 2]'). This hit 5 of 8
    sweep runs in the first report build.
    """
    from arms import SgLibArm
    m = Manifold(mode="argmax", align="gap")
    X = np.random.default_rng(0).random((25, m.dim))
    for budget in (60, 100, 150):
        arm = SgLibArm().fit(m.oracle("gamma"), m.dim, budget)
        y = arm.predict(X)                       # must not raise
        assert np.all(np.isfinite(y)), f"non-finite prediction at budget {budget}"


def test_sglib_is_dimension_adaptive():
    """The whole selling point: unequal levels across axes, not a uniform grid."""
    from arms import SgLibArm
    m = Manifold(mode="argmax", align="gap")
    arm = SgLibArm().fit(m.oracle("gamma"), m.dim, 250)
    lv = arm.axis_levels()
    assert len(lv) == m.dim
    assert lv.max() > lv.min(), f"levels are uniform ({lv}) -- not adapting"


def test_sglib_budget_is_respected_when_small():
    from arms import SgLibArm
    m = Manifold(mode="argmax")
    for budget in (20, 60):
        oracle = m.oracle("gamma")
        SgLibArm().fit(oracle, m.dim, budget)
        assert oracle.n_evals <= budget, (oracle.n_evals, budget)


# ===========================================================================
# GPR
# ===========================================================================
def test_gpr_hits_budget_exactly():
    from arms import GPRArm
    m = Manifold(mode="argmax", align="gap")
    for budget in (60, 150):
        oracle = m.oracle("gamma")
        GPRArm("var", batch=8).fit(oracle, m.dim, budget)
        assert oracle.n_evals == budget, (oracle.n_evals, budget)


def test_gpr_drops_failed_runs_instead_of_filling():
    """The structural advantage over both sparse grids: points are free, so a
    failed run is a dropped sample, not a hole to be papered over."""
    from arms import GPRArm
    m = Manifold(mode="argmax", fail_rate=0.25, seed=3)
    oracle = m.oracle("gamma")
    arm = GPRArm("var").fit(oracle, m.dim, 150)
    assert oracle.n_failed > 0
    assert len(arm.X) == oracle.n_evals - arm.n_failed
    assert np.all(np.isfinite(arm.predict(np.random.default_rng(0).random((20, m.dim)))))


def test_gpr_batches_do_not_collapse():
    """Without the distance penalty every point in a batch lands on the same
    argmax. A collapsed design shows up as a near-zero minimum pairwise gap."""
    from arms import GPRArm
    m = Manifold(mode="argmax", align="gap")
    arm = GPRArm("var", batch=8).fit(m.oracle("gamma"), m.dim, 120)
    d = design_metrics(arm.X, m)
    assert d["min_dist"] > 1e-3, f"design collapsed (min_dist={d['min_dist']})"


def test_gpr_acquisitions_differ_in_where_they_look():
    """'ucb' chases the peak, 'grad' chases the transition. If the two produce
    the same targeting profile, the acquisition switch is not doing anything."""
    from arms import GPRArm
    m = Manifold(mode="argmax", align="gap", out="gamma")
    res = {}
    for acq in ("ucb", "grad"):
        oracle = m.oracle("gamma")
        GPRArm(acq, batch=8).fit(oracle, m.dim, 160)
        res[acq] = design_metrics(oracle.design, m)
    assert res["ucb"]["frac_peak"] > res["grad"]["frac_peak"], res
    assert res["grad"]["band_lift"] > res["ucb"]["band_lift"], res


# ===========================================================================
# shared plumbing
# ===========================================================================
def test_design_metrics_flag_a_clumped_design():
    m = Manifold()
    rng = np.random.default_rng(0)
    spread = rng.random((200, m.dim))
    clump = np.clip(0.5 + 0.01 * rng.standard_normal((200, m.dim)), 0, 1)
    assert design_metrics(clump, m)["min_dist"] < design_metrics(spread, m)["min_dist"]
    assert design_metrics(clump, m)["hole"] > design_metrics(spread, m)["hole"]


def test_scoring_is_against_clean_truth():
    """An arm trained on noisy data must not be able to score perfectly by
    memorizing the noise."""
    from arms import RandomNearestArm
    clean = Manifold(mode="argmax")
    noisy = Manifold(mode="argmax", noise_rel=0.30, seed=5)
    test = clean.test_set(n=600)
    s_clean = score_arm(RandomNearestArm(), clean, budget=200, test=test)
    s_noisy = score_arm(RandomNearestArm(), noisy, budget=200, test=test)
    assert s_noisy.rmse > s_clean.rmse, (s_noisy.rmse, s_clean.rmse)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
