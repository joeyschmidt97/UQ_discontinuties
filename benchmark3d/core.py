"""Frozen truth, metered observations and common tetrahedral scoring in 3D."""
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree, distance

from scripts.datasets import load_dataset, surface_for


CORNERS = np.asarray(list(product((0., 1.), repeat=3)))
TARGETS = dict(nmae=.05, transition_nmae=.10, p95_error=.15, vwfd_p95=.25)


class BudgetExceeded(RuntimeError):
    pass


class Observations:
    def __init__(self, evaluate, budget, shared=None):
        if budget < len(CORNERS) + 1:
            raise ValueError("budget must exceed the eight shared cube corners")
        self._evaluate = evaluate
        self.budget = int(budget)
        self.dim = 3
        self._values = {}
        self.requests = 0
        self(CORNERS if shared is None else shared)

    @staticmethod
    def key(x):
        return tuple(np.round(x, 12))

    @property
    def x(self):
        return np.asarray(list(self._values))

    @property
    def y(self):
        return np.asarray(list(self._values.values()))

    @property
    def remaining(self):
        return self.budget - len(self._values)

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[1] != 3 or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("finite 3D unit-box coordinates required")
        keys = [self.key(point) for point in x]
        new = list(dict.fromkeys(key for key in keys if key not in self._values))
        if len(new) > self.remaining:
            raise BudgetExceeded("batch would exceed the unique-evaluation budget")
        if new:
            values = np.asarray(self._evaluate(np.asarray(new)), float)
            if values.shape != (len(new),) or not np.isfinite(values).all():
                raise ValueError("oracle must return finite scalar observations")
            self._values.update(zip(new, values.tolist()))
        self.requests += len(x)
        return np.asarray([self._values[key] for key in keys])


def reconstruct(x, y, query, kind="linear"):
    if kind == "linear":
        result = LinearNDInterpolator(x, y)(query)
    elif kind == "rbf":
        result = RBFInterpolator(x, y, smoothing=0.)(query)
    else:
        raise ValueError(kind)
    if not np.isfinite(result).all():
        raise ValueError("common reconstruction did not cover the full reference set")
    return result


def normalized_rmse(predicted, truth, scale):
    return float(np.sqrt(np.mean((np.asarray(predicted)-np.asarray(truth))**2))/scale)


@lru_cache(maxsize=None)
def evaluation_set(dataset_string, count=16384):
    dataset = Path(dataset_string)
    manifest, _, reference = load_dataset(dataset)
    if count > len(reference["x"]):
        raise ValueError("requested reference count exceeds the frozen archive")
    take = slice(0, count)
    x = reference["x"][take]
    truth = reference["y"][take]
    branches = reference["G"][take]
    scale = float(manifest["scale"])
    gap = np.abs(branches[:, 0]-branches[:, 1])
    transition = gap <= .05*float(np.ptp(branches))
    high = truth >= np.quantile(truth, .9)
    region = np.argmax(branches, axis=1)
    oracle = surface_for(3, manifest["case"], 0)
    gradient2 = np.zeros(count)
    for axis in range(3):
        plus, minus = x.copy(), x.copy()
        plus[:, axis] = np.minimum(1., plus[:, axis]+1e-4)
        minus[:, axis] = np.maximum(0., minus[:, axis]-1e-4)
        gradient2 += ((oracle(plus)-oracle(minus))/(plus[:, axis]-minus[:, axis]))**2
    return dict(manifest=manifest, x=x, y=truth, scale=scale, transition=transition,
                high=high, region=region, variation=np.sqrt(gradient2))


def score(observations, test, secondary=False, targets=None):
    truth, scale = test["y"], test["scale"]
    predicted = reconstruct(observations.x, observations.y, test["x"])
    absolute = np.abs(predicted-truth)/scale
    nearest = cKDTree(observations.x).query(test["x"])[0]
    vwfd = nearest*test["variation"]/scale
    out = dict(
        n=len(observations.x), n_requests=observations.requests, scale=scale,
        error=float(np.sqrt(np.mean(absolute**2))),
        nmae=float(np.mean(absolute)), p95_error=float(np.quantile(absolute, .95)),
        transition_error=normalized_rmse(predicted[test["transition"]], truth[test["transition"]], scale),
        transition_nmae=float(np.mean(absolute[test["transition"]])),
        high_error=normalized_rmse(predicted[test["high"]], truth[test["high"]], scale),
        high_nmae=float(np.mean(absolute[test["high"]])),
        fill_p95=float(np.quantile(nearest, .95)), fill_max=float(nearest.max()),
        min_separation=float(distance.pdist(observations.x).min()),
        vwfd_rms=float(np.sqrt(np.mean(vwfd**2))), vwfd_p95=float(np.quantile(vwfd, .95)),
        vwfd_coverage_02=float(np.mean(vwfd <= .02)),
        vwfd_coverage_05=float(np.mean(vwfd <= .05)),
        vwfd_coverage_10=float(np.mean(vwfd <= .10)),
        rbf_error=(normalized_rmse(reconstruct(observations.x, observations.y, test["x"], "rbf"),
                                   truth, scale) if secondary else None),
    )
    for branch in range(len(test["manifest"]["branches"])):
        mask = test["region"] == branch
        out[f"branch{branch}_error"] = normalized_rmse(predicted[mask], truth[mask], scale)
        out[f"branch{branch}_nmae"] = float(np.mean(absolute[mask]))
    limits = TARGETS if targets is None else targets
    out["vwfd_coverage_target"] = float(np.mean(vwfd <= limits["vwfd_p95"]))
    out["holistic_error"] = float(max(out[name]/limits[name] for name in TARGETS))
    out["holistic_driver"] = max(TARGETS, key=lambda name: out[name]/limits[name])
    return out
