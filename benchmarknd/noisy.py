"""Noisy proxies and a budget that lets a method pay for a tighter answer.

Two things change relative to the deterministic benchmark.

First the oracle returns a value *and* its spread, because a simulator knows
roughly how badly its trace was converging and reports it. That is the
"gamma +/- %" the sampler actually sees, and it is what lets an acquisition
rule tell a contested region from a quiet one before spending anything.

Second a repeated evaluation is no longer free. The deterministic
`Observations` caches duplicates and charges nothing, which is right when the
oracle is a function. Here a repeat is another simulation: it costs a paid
evaluation and returns a fresh draw, and the running mean tightens as
1/sqrt(m). Replicate-versus-explore becomes a decision the rule must make,
which is the exploration/exploitation question in its sharpest form.

Grading is unchanged and uses the noiseless surface. A method is judged on
recovering the truth despite noisy observations, never on reproducing its own
noisy samples.
"""
import numpy as np

from resolution.noise import TransitionNoise
from .core import BudgetExceeded
from .ionut import IonutSurface


class NoisyIonutSurface:
    """A native proxy whose observations carry transition-competition spread.

    The underlying deterministic surface stays available as `truth`, so the
    evaluation set, the truth variation and every score are built from the
    noiseless response exactly as before.
    """
    dim = IonutSurface.dim

    def __init__(self, case, noise=None, seed=0):
        self.truth = IonutSurface(case)
        self.case = case
        self.seed = 0
        self.noise = noise or TransitionNoise()
        self._rng = np.random.default_rng(20260921+seed)
        self._scale = None

    # The deterministic interface, delegated so scoring paths are shared.
    band_threshold = IonutSurface.band_threshold

    def __call__(self, x):
        return self.truth(x)

    def region(self, x):
        return self.truth.region(x)

    def distance(self, x):
        return self.truth.distance(x)

    def peak_distance(self, x):
        return self.truth.peak_distance(x)

    @property
    def normals(self):
        return self.truth.normals

    def scale(self):
        """One fixed branch scale, so sigma does not drift with the query set."""
        if self._scale is None:
            probe = np.random.default_rng(4242).random((4096, self.dim))
            self._scale = float(np.ptp(self.truth.branches(probe)))
        return self._scale

    def spread(self, x):
        """The spread a single evaluation here would report, before drawing."""
        x = np.atleast_2d(np.asarray(x, float))
        return self.noise.std(self.truth(x), self.truth.branches(x), self.scale())

    def observe(self, x, replicates=1):
        x = np.atleast_2d(np.asarray(x, float))
        return self.noise.draw(self.truth(x), self.truth.branches(x), self._rng,
                               replicates=replicates, scale=self.scale())


class NoisyObservations:
    """Metered noisy evaluations where a replicate is a paid evaluation.

    `x` and `y` present the running means, so an arm written against the
    deterministic interface still works and simply never replicates. `sigma`
    exposes the reported spread of each mean and `replicates` the count behind
    it, which is what a noise-aware rule needs.
    """
    def __init__(self, surface, budget, dim, seed=0, shared=None):
        if budget < 2*dim+2:
            raise ValueError("budget must leave room for the shared initialization")
        self.surface = surface
        self.budget, self.dim = int(budget), int(dim)
        self.requests = 0
        self._order = []
        self._sum = {}
        self._count = {}
        self._single = {}
        self.spent = 0
        from .core import Observations
        self(Observations.initial_design(dim, seed) if shared is None else shared)

    @staticmethod
    def key(x):
        return tuple(np.round(x, 12))

    @property
    def x(self):
        return np.array(self._order)

    @property
    def y(self):
        return np.array([self._sum[k]/self._count[k] for k in map(self.key, self._order)])

    @property
    def sigma(self):
        """Spread of each running mean: the single-shot spread over sqrt(m)."""
        keys = [self.key(p) for p in self._order]
        single = np.array([self._single[k] for k in keys])
        counts = np.array([self._count[k] for k in keys], float)
        return single/np.sqrt(counts) if self.surface.noise.reducible else single

    @property
    def replicates(self):
        return np.array([self._count[self.key(p)] for p in self._order])

    @property
    def remaining(self):
        return self.budget - self.spent

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[1] != self.dim or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("finite unit-box coordinates of the right dimension required")
        if len(x) > self.remaining:
            raise BudgetExceeded("batch would exceed the paid-evaluation budget")
        drawn, single = self.surface.observe(x)
        for point, value, spread in zip(x, drawn, np.atleast_1d(single)):
            key = self.key(point)
            if key not in self._sum:
                self._order.append(np.asarray(point, float))
                self._sum[key], self._count[key] = 0., 0
            self._sum[key] += float(value)
            self._count[key] += 1
            self._single[key] = float(spread)
            self.spent += 1
        self.requests += len(x)
        return np.array([self._sum[self.key(p)]/self._count[self.key(p)] for p in x])
