"""Analytic truth, metered observations and independent reconstruction scoring."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree, distance

CORNERS = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
CASES = ("smooth", "two-plane-four-peaks", "three-plane-three-peaks", "two-plane-asymmetric")
TRANSITION_PEAK_OFFSET = .08
HOLISTIC_TARGETS = dict(nmae=.05, band_nmae=.10, p95_error=.15, vwfd_p95=.25)


@dataclass(frozen=True)
class Surface:
    """Continuous affine envelopes with peaks that probe their mode boundaries."""
    case: str
    seed: int = 0

    def __post_init__(self):
        if self.case not in CASES:
            raise ValueError(self.case)

    @property
    def normal(self):
        theta = np.deg2rad(27 + 19*self.seed % 110)
        return np.array([np.cos(theta), np.sin(theta)])

    @property
    def normals(self):
        if self.case == "three-plane-three-peaks":
            theta = np.arctan2(self.normal[1], self.normal[0])+np.arange(3)*2*np.pi/3
            return np.column_stack([np.cos(theta), np.sin(theta)])
        return np.array([self.normal, -self.normal])

    def region(self, x):
        return np.argmax((np.atleast_2d(x)-.5) @ self.normals.T, axis=1)

    def distance(self, x):
        x = np.atleast_2d(x)
        if self.case == "smooth":
            return np.zeros(len(x))
        values = (x-.5) @ self.normals.T
        winner = values.argmax(axis=1)
        distances = []
        for j in range(len(self.normals)):
            denom = np.linalg.norm(self.normals[winner]-self.normals[j], axis=1)
            distances.append(np.divide(values[np.arange(len(x)), winner]-values[:, j], denom,
                                       out=np.full(len(x), np.inf), where=denom>0))
        return np.min(distances, axis=0)

    def centers(self):
        if self.case == "smooth":
            return np.array([.5+.13*self.normal])
        if self.case == "three-plane-three-peaks":
            # Modes 0 and 1 carry a matched pair immediately across their
            # shared boundary. Mode 2 keeps one isolated interior peak as a
            # control: transition-focused samplers should not forget it.
            n0, n1, n2 = self.normals
            boundary_ray = (n0+n1)/np.linalg.norm(n0+n1)
            across = (n0-n1)/np.linalg.norm(n0-n1)
            base = .5+.22*boundary_ray
            return np.array([base+TRANSITION_PEAK_OFFSET*across,
                             base-TRANSITION_PEAK_OFFSET*across, .5+.27*n2])
        tangent = np.array([-self.normal[1], self.normal[0]])
        # Paired peaks straddle the fold at two tangential locations. Their
        # centers are close enough to make peak and transition objectives
        # compete, while remaining assigned to opposite modes.
        positive = [.5+TRANSITION_PEAK_OFFSET*self.normal+v*tangent for v in (-.18, .18)]
        negative = ([.5-TRANSITION_PEAK_OFFSET*self.normal+v*tangent for v in (-.18, .18)]
                    if self.case == "two-plane-four-peaks"
                    else [.5-TRANSITION_PEAK_OFFSET*self.normal-.18*tangent])
        return np.array(positive+negative)

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        y = .25+.15*x[:, 0]+.08*x[:, 1]
        if self.case != "smooth":
            y += .65*np.max((x-.5) @ self.normals.T, axis=1)
        # Shared additive bumps preserve the affine-envelope switch boundaries.
        for i, center in enumerate(self.centers()):
            width = .085 if self.case == "smooth" else (.055, .065, .06, .05)[i]
            y += (.9, .7, .8, .6)[i]*np.exp(-.5*np.sum(((x-center)/width)**2, axis=1))
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
    grad2 = np.zeros(n)
    for axis in range(2):
        plus, minus = x.copy(), x.copy()
        plus[:, axis] = np.minimum(1., plus[:, axis]+1e-4)
        minus[:, axis] = np.maximum(0., minus[:, axis]-1e-4)
        grad2 += ((surface(plus)-surface(minus))/(plus[:, axis]-minus[:, axis]))**2
    return x, y, scale, band, peak, np.sqrt(grad2)


def score(surface, observations, test, secondary=True, targets=None):
    x, truth, scale, band, peak, truth_variation = test
    yhat = reconstruct(observations.x, observations.y, x)
    rbf = reconstruct(observations.x, observations.y, x, "rbf") if secondary else None
    absolute = np.abs(yhat-truth)/scale
    nearest = cKDTree(observations.x).query(x)[0]
    vwfd = nearest*truth_variation/scale
    separation = (float(distance.pdist(observations.x).min())
                  if len(observations.x) > 1 else None)
    out = dict(n=len(observations.x), n_requests=observations.requests,
               error=float(np.sqrt(np.mean(absolute**2))),
               nmae=float(np.mean(absolute)), p95_error=float(np.quantile(absolute, .95)),
               band_error=rmse(yhat[band], truth[band], scale),
               band_nmae=float(np.mean(absolute[band])),
               peak_error=rmse(yhat[peak], truth[peak], scale),
               peak_nmae=float(np.mean(absolute[peak])),
               fill_p95=float(np.quantile(nearest, .95)), fill_max=float(nearest.max()),
               min_separation=separation,
               vwfd_rms=float(np.sqrt(np.mean(vwfd**2))),
               vwfd_p95=float(np.quantile(vwfd, .95)),
               vwfd_coverage_02=float(np.mean(vwfd <= .02)),
               vwfd_coverage_05=float(np.mean(vwfd <= .05)),
               vwfd_coverage_10=float(np.mean(vwfd <= .10)),
               rbf_error=rmse(rbf, truth, scale) if secondary else None, scale=scale,
               x=observations.x.tolist(), y=observations.y.tolist())
    limits = HOLISTIC_TARGETS if targets is None else targets
    out["vwfd_coverage_target"] = float(np.mean(vwfd <= limits["vwfd_p95"]))
    out["holistic_error"] = float(max(out[name]/limits[name] for name in HOLISTIC_TARGETS))
    out["holistic_driver"] = max(HOLISTIC_TARGETS, key=lambda name: out[name]/limits[name])
    return out
