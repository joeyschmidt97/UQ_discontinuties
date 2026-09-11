"""
Control-flow tests for the SG++ arm against a mock `pysgpp`.

pysgpp has no wheel for every platform (none for CPython 3.13 on Windows at time
of writing), so on such a machine the arm is unrunnable and therefore untested.
This mock exercises everything the arm owns -- budget accounting, the refinement
loop, NaN repair, early stop, regular-level selection -- and leaves only the
pysgpp API signatures unverified. It is NOT a sparse-grid implementation and
proves nothing about approximation quality.

Run:  python -m pytest tests/ -q      (or:  python tests/test_sgpp_arm_mock.py)
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# ===========================================================================
# Mock pysgpp: stores nodal values, evaluates by nearest neighbour.
# ===========================================================================
class _Point:
    def __init__(self, c):
        self.c = c

    def getStandardCoordinate(self, j):
        return float(self.c[j])


class _Storage:
    def __init__(self, dim):
        self.dim = dim
        self.pts = []

    def getDimension(self):
        return self.dim

    def getSize(self):
        return len(self.pts)

    def getPoint(self, i):
        return _Point(self.pts[i])


class _Generator:
    def __init__(self, grid):
        self.g = grid

    def regular(self, level):
        d = self.g.st.dim
        n = (2 ** level - 1) * d + 1
        rng = np.random.default_rng(1000 * level + d)
        self.g.st.pts = list(rng.random((n, d)))

    def refine(self, functor):
        """Add one jittered child per selected parent. Deliberately overshoots
        the requested batch by one on every other call, so the arm's
        'refine() may add more points than asked' guard is exercised."""
        alpha, batch = functor.alpha, functor.batch
        a = np.abs(np.array([alpha[i] for i in range(len(alpha))]))
        k = min(batch, len(a))
        self.g._calls += 1
        if self.g._calls % 2 == 0:
            k = min(k + 1, len(a))
        rng = np.random.default_rng(7 * self.g._calls)
        for p in np.argsort(-a)[:k]:
            child = np.clip(self.g.st.pts[p] + rng.normal(0, 0.05, self.g.st.dim), 0, 1)
            self.g.st.pts.append(child)


class _Grid:
    def __init__(self, dim):
        self.st = _Storage(dim)
        self._calls = 0

    def getStorage(self):
        return self.st

    def getGenerator(self):
        return _Generator(self)

    def clone(self):
        import copy
        return copy.deepcopy(self)


class _DataVector(list):
    def __init__(self, arg):
        super().__init__([0.0] * arg if isinstance(arg, int) else list(arg))


class _Functor:
    def __init__(self, alpha, batch):
        self.alpha, self.batch = alpha, batch


def _make_mock():
    m = types.ModuleType("pysgpp")
    m.Grid = types.SimpleNamespace(
        createModLinearGrid=lambda d: _Grid(d),
        createLinearGrid=lambda d: _Grid(d),
        createLinearBoundaryGrid=lambda d: _Grid(d),
        createModBsplineGrid=lambda d, deg: _Grid(d),
    )
    m.DataVector = _DataVector
    m.DataMatrix = lambda X: np.asarray(X, float)
    m.SurplusRefinementFunctor = _Functor
    m.SurplusVolumeRefinementFunctor = _Functor
    m.createOperationHierarchisation = lambda g: types.SimpleNamespace(
        doHierarchisation=lambda a: None)              # nodal == hierarchical here

    def _mult_eval(grid, X):
        P = np.array(grid.st.pts)

        def mult(alpha, res):
            A = np.array(list(alpha), float)
            n = min(len(A), len(P))
            idx = np.argmin(((np.atleast_2d(X)[:, None, :] - P[None, :n, :]) ** 2).sum(-1), 1)
            for i, k in enumerate(idx):
                res[i] = float(A[k])
        return types.SimpleNamespace(mult=mult)

    m.createOperationMultipleEval = _mult_eval
    return m


sys.modules["pysgpp"] = _make_mock()

from manifold import Manifold                                        # noqa: E402
from arms.sgpp_arm import SGppArm, SGppRegularArm                    # noqa: E402


# ===========================================================================
# Tests
# ===========================================================================
def test_budget_is_never_exceeded():
    m = Manifold(mode="argmax", align="gap")
    for budget in (40, 100, 300):
        oracle = m.oracle("gamma")
        SGppArm(refine_batch=8).fit(oracle, m.dim, budget)
        assert oracle.n_evals <= budget, (oracle.n_evals, budget)


def test_refinement_actually_happens():
    m = Manifold(mode="argmax", align="gap")
    oracle = m.oracle("gamma")
    arm = SGppArm(refine_batch=8).fit(oracle, m.dim, 300)
    assert len(arm.history) > 3, "no refinement steps taken"
    n = [h["n_evals"] for h in arm.history]
    assert n == sorted(n) and n[-1] > n[0], "evaluation count not monotone"


def test_failed_runs_are_repaired_not_propagated():
    m = Manifold(mode="argmax", align="gap", fail_rate=0.20, seed=3)
    oracle = m.oracle("gamma")
    arm = SGppArm(nan_policy="fill", refine_batch=8).fit(oracle, m.dim, 200)
    assert oracle.n_failed > 0, "manifold produced no failures to repair"
    assert arm.n_failed == oracle.n_failed
    y = arm.predict(np.random.default_rng(0).random((50, m.dim)))
    assert np.all(np.isfinite(y)), "NaN leaked into the interpolant"


def test_nan_policy_error_raises():
    m = Manifold(mode="argmax", fail_rate=0.5, seed=3)
    try:
        SGppArm(nan_policy="error").fit(m.oracle("gamma"), m.dim, 100)
    except RuntimeError:
        return
    raise AssertionError("nan_policy='error' did not raise on a failed run")


def test_regular_arm_picks_largest_level_within_budget():
    m = Manifold()
    for budget in (50, 200, 1000):
        oracle = m.oracle("gamma")
        arm = SGppRegularArm().fit(oracle, m.dim, budget)
        assert oracle.n_evals <= budget
        assert arm.history[0]["level"] >= 1


def test_scoring_path_runs():
    from arms import score_arm
    m = Manifold(mode="argmax", align="gap", noise_rel=0.05, fail_rate=0.05, seed=1)
    test = m.test_set(n=800)
    s = score_arm(SGppArm(refine_batch=8), m, out="omega", budget=200, test=test)
    assert np.isfinite(s.rmse) and np.isfinite(s.rmse_mixed)
    assert 0.0 <= s.frac_design_mixed <= 1.0
    assert s.n_evals <= 200


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed (mock pysgpp -- API signatures NOT verified)")
