import numpy as np
import pytest

from benchmark3d.core import CORNERS, Observations, evaluation_set, reconstruct, score
from benchmark3d.strategies import RUNNABLE_ARMS, run_arm
from scripts.datasets import surface_for


def test_cube_corners_cover_affine_truth():
    oracle = lambda x: .3+2*x[:, 0]-x[:, 1]+.5*x[:, 2]
    obs = Observations(oracle, 9)
    query = np.random.default_rng(2).random((128, 3))
    assert len(obs.x) == len(CORNERS) == 8
    assert np.max(np.abs(reconstruct(obs.x, obs.y, query)-oracle(query))) < 1e-12


@pytest.mark.parametrize("arm", RUNNABLE_ARMS)
def test_all_local_3d_arms_spend_exact_smoke_budget(arm):
    oracle = surface_for(3, "ionut-itg-tem-3d-argmax-gamma", 0)
    obs = Observations(oracle, 16)
    run_arm(arm, obs, 0)
    assert len(obs.x) == 16
    assert len({Observations.key(x) for x in obs.x}) == 16


def test_3d_score_has_transition_high_branch_and_resolution_metrics():
    dataset = "data/3d/ionut-itg-tem-3d-argmax-gamma/seed-0"
    truth = evaluation_set(dataset, 1024)
    oracle = surface_for(3, truth["manifest"]["case"], 0)
    obs = Observations(oracle, 16)
    run_arm("grid", obs, 0)
    result = score(obs, truth)
    for key in ("error", "nmae", "p95_error", "transition_error", "transition_nmae",
                "high_error", "high_nmae", "branch0_error", "branch1_error",
                "fill_p95", "vwfd_p95", "holistic_error"):
        assert np.isfinite(result[key])
    assert truth["transition"].any() and truth["high"].any()
