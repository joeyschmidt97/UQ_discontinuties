"""Transition-competition noise, replicate budgeting and noise-aware placement."""
import numpy as np
import pytest

from benchmarknd.noisy import NoisyIonutSurface, NoisyObservations
from benchmarknd.strategies import NOISE_AWARE_ARMS, candidate_noise, run_arm
from resolution import knn_variation
from resolution.noise import TransitionNoise, competition, denoise_residual

CASE = "ionut-itg-tem-argmax-gamma"


def test_competition_is_one_at_a_tie_and_falls_away_from_it():
    tied = competition(np.array([[1., 1.]]), scale=1.)
    separated = competition(np.array([[1., 0.]]), scale=1.)
    assert tied[0] == pytest.approx(1.)
    assert separated[0] < 1e-8
    # Uses the leading pair, not columns 0 and 1, so a third branch cannot hide
    # a tie between the two that actually matter.
    assert competition(np.array([[1., 1., -5.]]), scale=1.)[0] == pytest.approx(1.)


def test_spread_widens_toward_the_transition():
    surface = NoisyIonutSurface(CASE)
    x = np.random.default_rng(0).random((8000, 6))
    truth = surface.truth(x)
    live = truth > .05*truth.max()
    contest = competition(surface.truth.branches(x), scale=surface.scale())
    relative = surface.spread(x)[live]/truth[live]
    hot = relative[(contest > .8)[live]].mean()
    cold = relative[(contest < .05)[live]].mean()
    assert hot > 8*cold
    assert relative.max() <= TransitionNoise().peak + 1e-6


def test_replicates_tighten_the_mean_and_cost_budget():
    surface = NoisyIonutSurface(CASE)
    point = np.random.default_rng(3).random((1, 6))
    reported = []
    for replicates in (1, 4, 16):
        obs = NoisyObservations(surface, 80, 6, 0)
        before = obs.spent
        for _ in range(replicates):
            obs(point)
        assert obs.spent == before + replicates      # a repeat is never free
        index = list(map(obs.key, obs.x)).index(obs.key(point[0]))
        assert obs.replicates[index] == replicates
        reported.append(obs.sigma[index])
    assert reported[1] == pytest.approx(reported[0]/2, rel=.05)
    assert reported[2] == pytest.approx(reported[0]/4, rel=.05)


def test_irreducible_noise_does_not_reward_replication():
    surface = NoisyIonutSurface(CASE, noise=TransitionNoise(reducible=False))
    point = np.random.default_rng(3).random((1, 6))
    obs = NoisyObservations(surface, 80, 6, 0)
    for _ in range(8):
        obs(point)
    index = list(map(obs.key, obs.x)).index(obs.key(point[0]))
    single = surface.spread(point)[0]
    assert obs.sigma[index] == pytest.approx(single, rel=1e-6)


def test_denoising_removes_the_residual_a_flat_noisy_patch_would_show():
    rng = np.random.default_rng(0)
    dim, sigma = 6, .08
    x = rng.random((600, dim))
    plane = 2.*x[:, 0]
    noisy = plane + sigma*rng.standard_normal(len(x))
    query = rng.random((300, dim))
    naive = knn_variation(x, noisy, query)
    aware = knn_variation(x, noisy, query, noise=np.full(len(x), sigma))
    clean = knn_variation(x, plane, query)
    assert clean.curvature.mean() < 1e-6
    assert naive.curvature.mean() > 20*aware.curvature.mean()


def test_denoising_keeps_real_curvature():
    rng = np.random.default_rng(1)
    dim, sigma = 5, .01
    x = rng.random((700, dim))
    peak = np.exp(-.5*np.sum(((x-.5)/.15)**2, axis=1))
    query = rng.random((200, dim))
    clean = knn_variation(x, peak, query)
    aware = knn_variation(x, peak + sigma*rng.standard_normal(len(x)), query,
                          noise=np.full(len(x), sigma))
    assert aware.curvature.mean() > .3*clean.curvature.mean()


def test_denoise_residual_is_never_negative_and_checks_its_stencil():
    out = denoise_residual(np.array([.01]), np.array([5.]), stencil=12, dim=5)
    assert out[0] == 0.
    with pytest.raises(ValueError):
        denoise_residual(np.array([1.]), np.array([1.]), stencil=6, dim=5)


def test_noise_aware_arms_refuse_a_clean_oracle():
    from benchmarknd.core import Observations, SurfaceND
    surface = SurfaceND("5d-m2-rotated", 0)
    for arm in NOISE_AWARE_ARMS:
        obs = Observations(surface, 20, 5, 0)
        with pytest.raises(ValueError, match="reports a spread"):
            run_arm(arm, obs, 0)


def test_candidate_noise_reads_only_what_was_paid_for():
    surface = NoisyIonutSurface(CASE)
    obs = NoisyObservations(surface, 40, 6, 0)
    candidates = np.random.default_rng(5).random((50, 6))
    estimate = candidate_noise(obs, candidates)
    single = obs.sigma*np.sqrt(obs.replicates)
    assert estimate.shape == (50,)
    # Every estimate is one of the observed single-shot spreads: no oracle peek.
    assert set(np.round(estimate, 12)) <= set(np.round(single, 12))


@pytest.mark.parametrize("arm", NOISE_AWARE_ARMS)
def test_noise_aware_arms_spend_the_exact_budget(arm):
    surface = NoisyIonutSurface(CASE)
    obs = NoisyObservations(surface, 22, 6, 0)
    predict, metadata = run_arm(arm, obs, 0)
    assert obs.spent == obs.budget
    if arm != "gpr-n":
        assert metadata["noise_aware"] is True


def test_allocation_weight_is_negative_on_the_aleatoric_term():
    from benchmarknd.strategies import VURS_A_WEIGHTS
    assert VURS_A_WEIGHTS["aleatoric"] < 0
    assert sum(v for k, v in VURS_A_WEIGHTS.items() if k != "aleatoric") == pytest.approx(1.)
