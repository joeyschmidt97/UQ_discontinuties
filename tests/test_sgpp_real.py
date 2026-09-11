"""Run separately from test_sgpp_arm_mock.py: require the actual compiled backend."""
import pathlib
import sys
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
pysgpp = pytest.importorskip("pysgpp")
if not getattr(pysgpp, "__file__", None):
    pytest.skip("mock backend is not real SG++", allow_module_level=True)
from arms.sgpp_arm import SGppArm


@pytest.mark.parametrize("budget", [9, 15, 28, 60, 124])
def test_budget_stop_keeps_grid_and_coefficients_consistent(budget):
    calls = []
    def oracle(x):
        calls.extend(x)
        return 1.+np.abs(x[:, 0]+.6*x[:, 1]-.7)
    arm = SGppArm(refine_batch=1, nan_policy="error").fit(oracle, 2, budget)
    assert len(calls) <= budget
    assert arm.grid.getStorage().getSize() == arm.alpha.getSize() == len(calls)
    q = np.random.default_rng(1).random((128, 2))
    assert np.isfinite(arm.predict(q)).all()


def test_refinement_failure_fill_uses_previous_grid():
    count = 0
    def oracle(x):
        nonlocal count
        count += 1
        y = 1.+x[:, 0]**2+x[:, 1]**2
        if count > 1:
            y[0] = np.nan
        return y
    arm = SGppArm(refine_batch=1, nan_policy="fill").fit(oracle, 2, 40)
    assert arm.n_failed > 0
    assert arm.grid.getStorage().getSize() == arm.alpha.getSize()
    assert np.isfinite(arm.predict([[.2, .7]])).all()
