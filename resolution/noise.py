"""Transition-competition noise: gamma and omega as values with a spread.

Physical motivation. In GENE the nrg and field time traces near a mode
transition oscillate around a value rather than settling, because two branches
have nearly equal growth rates and neither dominates. The reported growth rate
and frequency are then averages over a trace that has not converged, and they
carry a spread that widens as the branch gap closes. Away from a transition one
branch wins early, the trace settles and the spread collapses to the ordinary
numerical floor.

Model. With branch growth rates G, define the competition field

    s(x) = exp( -|G_0(x) - G_1(x)| / (T * ptp(G)) )     in (0, 1]

so s -> 1 where the branches are tied and s -> 0 where one dominates. The
relative spread is then interpolated between a floor and a peak,

    sigma_rel(x) = floor + (peak - floor) * s(x)
    sigma(x)     = sigma_rel(x) * |y(x)| + absolute_floor

giving the "gamma +/- %" form directly. T defaults to 0.05, matching the
softmax temperature the proxies already use for mode selection, which puts
roughly an eighth of the domain in strong competition.

Reducibility is the decision that matters, and it is the one Ionut's note flags:
"distinguish irreducible difficulty from uncertainty that additional samples
could resolve." Here the spread is *reducible*, because a longer GENE run
averages the oscillation down. Repeating an evaluation at the same point draws
again and the mean tightens as 1/sqrt(m). That makes replicate-versus-explore a
real decision an acquisition rule has to make, rather than a trap that swallows
budget forever. Set `reducible=False` to get the opposite regime on purpose --
the one where uncertainty-seeking methods should be punished.

Everything here is a phenomenological stand-in for non-convergence, not a model
of gyrokinetic saturation physics.
"""
from dataclasses import dataclass

import numpy as np

DEFAULT_TEMPERATURE = .05       # matches the proxies' own softmax temperature
DEFAULT_FLOOR = .01             # 1% relative spread far from any transition
DEFAULT_PEAK = .25              # 25% relative spread at a tied branch gap
ABSOLUTE_FLOOR = 1e-6           # keeps sigma > 0 where the response passes zero


def competition(branches, temperature=DEFAULT_TEMPERATURE, scale=None):
    """Closeness of the two leading branches, 1 when tied and 0 when separated."""
    g = np.atleast_2d(np.asarray(branches, float))
    if g.shape[1] < 2:
        raise ValueError("a competition field needs at least two branches")
    if temperature <= 0:
        raise ValueError("positive temperature required")
    ordered = np.sort(g, axis=1)
    gap = ordered[:, -1]-ordered[:, -2]          # leading pair, not columns 0 and 1
    span = float(np.ptp(g)) if scale is None else float(scale)
    if span <= 0:
        return np.ones(len(g))
    return np.exp(-gap/(span*temperature))


@dataclass(frozen=True)
class TransitionNoise:
    """Heteroscedastic spread tied to how contested the mode selection is."""
    floor: float = DEFAULT_FLOOR
    peak: float = DEFAULT_PEAK
    temperature: float = DEFAULT_TEMPERATURE
    reducible: bool = True

    def __post_init__(self):
        if not 0 <= self.floor <= self.peak:
            raise ValueError("require 0 <= floor <= peak relative spread")

    def relative(self, branches, scale=None):
        s = competition(branches, self.temperature, scale)
        return self.floor + (self.peak-self.floor)*s

    def std(self, value, branches, scale=None):
        """Absolute spread of one observation, in response units."""
        value = np.asarray(value, float)
        return self.relative(branches, scale)*np.abs(value) + ABSOLUTE_FLOOR

    def draw(self, value, branches, rng, replicates=1, scale=None):
        """Observed mean of `replicates` draws, and the spread of that mean.

        When the spread is reducible, averaging m draws tightens the reported
        mean as 1/sqrt(m), which is what running a simulation longer buys. When
        it is not, replicates cost budget and return the same uncertainty --
        the regime where an uncertainty-hungry rule should be seen to waste.
        """
        value = np.asarray(value, float)
        replicates = int(replicates)
        if replicates < 1:
            raise ValueError("at least one replicate per evaluation")
        single = self.std(value, branches, scale)
        effective = single/np.sqrt(replicates) if self.reducible else single
        return value + effective*rng.standard_normal(value.shape), effective


def denoise_residual(residual, noise, stencil, dim, method="hard"):
    """Remove the part of a local-linear residual that is only observation noise.

    A weighted linear fit of k points on d+1 coefficients leaves a fraction
    (k-d-1)/k of the noise variance in its residual, so a purely noisy but
    perfectly flat neighbourhood still produces one. That expectation E has to
    come out, or VWRS reads noise as curvature and refines the loudest region
    rather than the most structured one.

    How it comes out matters, and the honest answer turned out to be the blunt
    one. Hard subtraction, sqrt(max(0, R^2 - E)), is unbiased in expectation and
    clips to exactly zero wherever the realized R^2 falls below E. At a 60%
    relative spread that is 56% of the domain (4.2% at a 5% spread, 24.4% at
    25%). A zero is not a failure there: it is the correct statement that no
    structure is detectable above the noise, and VWRS's additive rule then
    degenerates gracefully to its coverage term, which is an interpretable
    fallback rather than a silent one.

    A Wiener-style shrinkage, R^2 * R^2/(R^2 + E), was tried as the default to
    avoid those zeros and was worse on both counts that matter. On a pure-noise
    plane it left 0.107 against hard subtraction's 0.005 -- because at S = 0 it
    returns E/2 rather than 0 -- and on a noisy peak it returned 148% of the
    clean curvature, inflating the estimate with the noise it was meant to
    remove. It is retained as `method="shrink"` for comparison only.

    What the zeros do break is any rule that compares a *magnitude* against
    this deficit, since a zero deficit is won by anything. That is a constraint
    on such rules, not a reason to blur the estimator.
    """
    residual = np.asarray(residual, float)
    noise = np.asarray(noise, float)
    stencil = int(stencil)
    if stencil <= dim+1:
        raise ValueError("stencil must exceed the linear-fit coefficient count")
    expected = noise**2*(stencil-dim-1)/stencil
    square = residual**2
    if method == "hard":
        return np.sqrt(np.clip(square-expected, 0., None))
    if method != "shrink":
        raise ValueError(method)
    return np.sqrt(square*square/np.maximum(square+expected, 1e-300))
