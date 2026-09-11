"""Analytic truth, metered observations and independent reconstruction scoring."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator

CORNERS = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
CASES = ("smooth", "kink", "on-plane", "offset-peaks", "jump")


@dataclass(frozen=True)
class Surface:
    case: str
    seed: int = 0

    def __post_init__(self):
        if self.case not in CASES:
            raise ValueError(self.case)

    @property
    def normal(self):
        # Paired geometry realizations; never given to acquisition policies.
        theta = (27 + 19 * self.seed % 110) * np.pi / 180
        return np.array([np.cos(theta), np.sin(theta)])

    def distance(self, x):
        return (np.asarray(x) - .5) @ self.normal

    def centers(self):
        tangent = np.array([-self.normal[1], self.normal[0]])
        offset = .17 if self.case == "offset-peaks" else 0.
        return np.array([.5 + .19 * tangent + offset * self.normal,
                         .5 - .19 * tangent - offset * self.normal])

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        h = self.distance(x)
        base = .25 + .15 * x[:, 0] + .08 * x[:, 1]
        if self.case == "smooth":
            return base + .25 * np.sin(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1])
        y = base + .65 * np.abs(h)
        if self.case in ("on-plane", "offset-peaks", "jump"):
            tangent = np.array([-self.normal[1], self.normal[0]])
            for center, amplitude, width in zip(self.centers(), (.9, .55), (.045, .07)):
                delta = x - center
                y += amplitude * np.exp(-.5*((delta@self.normal/width)**2
                                            + (delta@tangent/.09)**2))
        if self.case == "jump":
            y += .45 * (h >= 0)
        return y


class BudgetExceeded(RuntimeError):
    pass


class Observations:
    """Exact deterministic oracle; charge unique evaluations, cache duplicates.

    A whole requested batch is priced before evaluating anything. Four common
    corners are charged to every arm. Acquisition receives only this callback.
    """
    def __init__(self, evaluate, budget):
        if budget < 9:
            raise ValueError("budget must be >= 9")
        self._evaluate = evaluate
        self.budget = int(budget)
        self._values = {}
        self.requests = 0
        self(CORNERS)

    @staticmethod
    def key(x):
        return tuple(np.round(x, 12))

    @property
    def x(self):
        return np.array(list(self._values))

    @property
    def y(self):
        return np.array(list(self._values.values()))

    @property
    def remaining(self):
        return self.budget - len(self._values)

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[1] != 2 or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("finite 2-D unit-box coordinates required")
        keys = [self.key(p) for p in x]
        new = list(dict.fromkeys(k for k in keys if k not in self._values))
        if len(new) > self.remaining:
            raise BudgetExceeded("batch would exceed unique-evaluation budget")
        if new:
            values = np.asarray(self._evaluate(np.array(new)), float)
            if values.shape != (len(new),) or not np.isfinite(values).all():
                raise ValueError("clean pilot oracle must return finite scalar observations")
            self._values.update(zip(new, values.tolist()))
        self.requests += len(x)
        return np.array([self._values[k] for k in keys])


def reconstruct(x, y, query, kind="linear"):
    if kind == "linear":
        result = LinearNDInterpolator(x, y)(query)
    elif kind == "rbf":
        result = RBFInterpolator(x, y, smoothing=0.)(query)
    else:
        raise ValueError(kind)
    if not np.isfinite(result).all():
        raise ValueError("reconstruction failed to cover the entire evaluation set")
    return result


def rmse(predicted, truth, scale=1.):
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if predicted.shape != truth.shape or not truth.size:
        raise ValueError("nonempty matching predictions/truth required")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all() or scale <= 0:
        raise ValueError("nonfinite predictions must not disappear from scoring")
    return float(np.sqrt(np.mean((predicted-truth)**2))/scale)


def evaluation_set(surface, n=16384):
    from scipy.stats import qmc
    if n < 1024 or n & (n-1):
        raise ValueError("test size must be a power of two >= 1024")
    x = qmc.Sobol(2, scramble=True, seed=91479).random_base2(n.bit_length()-1)
    y = surface(x)
    scale = float(np.ptp(y))
    band = np.abs(surface.distance(x)) < .06
    peak = np.min(np.linalg.norm(x[:, None, :]-surface.centers()[None, :, :], axis=2), axis=1) < .1
    return x, y, scale, band, peak


def score(surface, observations, test):
    x, truth, scale, band, peak = test
    yhat = reconstruct(observations.x, observations.y, x)
    rbf = reconstruct(observations.x, observations.y, x, "rbf")
    return dict(n=len(observations.x), n_requests=observations.requests,
                error=rmse(yhat, truth, scale), band_error=rmse(yhat[band], truth[band], scale),
                peak_error=rmse(yhat[peak], truth[peak], scale),
                rbf_error=rmse(rbf, truth, scale), scale=scale,
                x=observations.x.tolist(), y=observations.y.tolist())
