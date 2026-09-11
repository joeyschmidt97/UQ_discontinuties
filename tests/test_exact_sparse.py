"""Exact-N native refinement prefix contracts (requires installed real backends)."""
import numpy as np
import pytest
from benchmark2d.core import Observations, Surface
from benchmark2d.strategies import run_arm

@pytest.mark.parametrize('name', ['sglib', 'sgpp'])
def test_real_sparse_prefix_spends_exact_budget_and_is_causal(name):
    if name == 'sgpp':
        module = pytest.importorskip('pysgpp')
        if not getattr(module, '__file__', None):
            pytest.skip('real compiled backend required; run separately from mocks')
    runs = []
    for budget in (12, 17):
        calls = []
        surface = Surface('two-plane-four-peaks')
        def oracle(x):
            calls.extend(x.tolist())
            return surface(x)
        obs = Observations(oracle, budget)
        _, metadata = run_arm(name, obs, 0)
        assert len(obs.x) == len(calls) == budget
        assert metadata['prefix_of_native_batches']
        runs.append(obs.x)
    assert np.array_equal(runs[0], runs[1][:12])

def test_uniform_geometry_cache_matches_backend_and_returns_copies():
    import arms.sglib_arm as backend
    if not backend.HAVE_SG_LIB:
        pytest.skip('sg_lib is not installed')
    arm = backend.SgLibArm(max_level=7, nan_policy='error')
    arm.fit(lambda x: x[:, 0]+x[:, 1], dim=2, budget=9)
    modules = backend._import_sg_lib()
    original = modules['Grid'](2, 1, 1, np.zeros(2), np.ones(2), [lambda x: 1., lambda x: 1.])
    expected = original.get_1D_points(7, 0., 1.)
    cached = arm.G.get_1D_points(7, 0., 1.)
    assert all(np.array_equal(a,b) for a,b in zip(expected,cached))
    cached[0][0] = -100.
    assert np.array_equal(arm.G.get_1D_points(7, 0., 1.)[0], expected[0])

def test_fixed_budget_continues_other_axes_after_level_limit():
    import arms.sglib_arm as backend
    if not backend.HAVE_SG_LIB:
        pytest.skip('sg_lib is not installed')
    oracle = lambda x: 3*x[:, 0]+x[:, 1]
    conventional = backend.SgLibArm(tol=0., max_level=5, nan_policy='error')
    conventional.fit(oracle, dim=2, budget=25)
    fixed = backend.SgLibArm(tol=0., max_level=5, nan_policy='error', budget_driven=True)
    fixed.fit(oracle, dim=2, budget=25)
    assert conventional.n_evals < 25
    assert fixed.n_evals == 25
    assert np.array_equal(fixed.axis_levels(), [5,5])
    assert np.allclose(fixed.predict([[.2,.3],[.7,.8]]), [.9,2.9], atol=1e-8)
