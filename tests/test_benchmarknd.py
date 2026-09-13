"""Contracts the high-dimensional cases must satisfy before any arm runs."""
import numpy as np
import pytest
from scipy.stats import qmc
from benchmarknd.cases import CASES, strengths, weak_axes
from benchmarknd.core import SurfaceND, Observations, BudgetExceeded, evaluation_set, reconstruct, rmse
from benchmarknd.tables import measured_strength, design_profile


@pytest.mark.parametrize("case", list(CASES))
def test_inert_axes_change_nothing(case):
    surface = SurfaceND(case, 0)
    _, inert = weak_axes(case)
    if not inert:
        pytest.skip("case has no globally inert axis")
    x = qmc.Sobol(surface.dim, scramble=True, seed=2).random(512)
    moved = x.copy()
    moved[:, inert] = np.random.default_rng(3).random((len(x), len(inert)))
    assert np.max(np.abs(surface(x)-surface(moved))) == 0.


@pytest.mark.parametrize("case", list(CASES))
def test_weak_axis_count_matches_the_declared_budget(case):
    per_mode, _ = weak_axes(case)
    allowed = (1, 2, 3) if CASES[case]["dim"] == 5 else (4, 5, 6)
    assert all(len(axes) in allowed for axes in per_mode)
    assert all(sum(1 for v in row if v == 1.) >= 2 for row in strengths(case))


@pytest.mark.parametrize("case", list(CASES))
def test_regions_are_balanced_and_peaks_clear_the_folds(case):
    surface = SurfaceND(case, 0)
    x = qmc.Sobol(surface.dim, scramble=True, seed=11).random(4096)
    shares = np.bincount(surface.region(x), minlength=len(surface.normals))/len(x)
    assert shares.min() > .15
    centers, owners = surface.centers
    for c, m in zip(centers, owners):
        assert surface.region(c)[0] == m
        assert surface.distance(c)[0] >= 3*float(np.min(surface.widths[m]))


def test_fold_is_continuous_but_kinked():
    surface = SurfaceND("5d-m2-rotated", 0)
    normal = surface.normals[0]-surface.normals[1]
    normal = normal/np.linalg.norm(normal)
    x = np.random.default_rng(4).random((256, surface.dim))
    x = np.clip(x-(surface.distance(x)*np.sign((x-.5) @ normal))[:, None]*normal, 0, 1)
    assert np.max(np.abs(surface(x+1e-7*normal)-surface(x-1e-7*normal))) < 1e-5
    slopes = [(surface(x+h*normal)-surface(x))/h for h in (1e-3, -1e-3)]
    assert np.max(np.abs(slopes[0]+slopes[1])) > .1


@pytest.mark.parametrize("case", list(CASES))
def test_measured_strength_ranks_the_declared_axes(case):
    surface = SurfaceND(case, 0)
    x = qmc.Sobol(surface.dim, scramble=True, seed=6).random(2048)
    measured = measured_strength(surface, x)
    declared = np.array(strengths(case), float).max(axis=0)
    assert np.all(measured[declared == 0.] == 0.)
    assert measured[declared == 1.].min() > 4*measured[(declared > 0) & (declared < 1)].max()


def test_design_profile_separates_a_constant_axis_from_a_refined_one():
    n, dim = 512, 5
    space_filling = qmc.Sobol(dim, scramble=True, seed=1).random(n)
    clustered = space_filling.copy()
    clustered[:, 0] = np.clip(np.random.default_rng(0).normal(.5, .02, n), 0, 1)
    constant = space_filling.copy()
    constant[:, 3:] = .5
    assert design_profile(space_filling)["coverage"].min() > .9
    assert design_profile(clustered)["coverage"][0] < .3
    assert design_profile(clustered)["refinement"][0] == 1.
    assert design_profile(constant)["coverage"][3] == 0.
    assert np.all(design_profile(constant)["refinement"][3:] == 0.)


def test_shared_initialization_is_charged_and_budget_is_atomic():
    surface = SurfaceND("5d-m2-rotated", 0)
    obs = Observations(surface, 20, surface.dim)
    assert len(obs.x) == 2*surface.dim+1
    before = obs.x.copy()
    obs(before[:3])
    assert len(obs.x) == len(before)
    with pytest.raises(BudgetExceeded):
        obs(np.random.default_rng(0).random((obs.remaining+1, surface.dim)))
    assert len(obs.x) == len(before)


def test_common_reconstructor_covers_the_whole_box():
    surface = SurfaceND("8d-m2-rotated", 0)
    test = evaluation_set(surface, 4096)
    x = qmc.Sobol(surface.dim, scramble=True, seed=8).random(128)
    predicted = reconstruct(x, surface(x), test["x"])
    assert np.isfinite(predicted).all()
    assert rmse(predicted, test["y"], test["scale"]) < .2
