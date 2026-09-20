"""Analytic truth, metered observations and the common RBF reconstruction score.

Delaunay reconstruction is not usable here: at d=8, N=512 it costs minutes and
leaves ~85% of an independent test set outside the convex hull, which the 2-D
scorer treats as a hard failure. One fixed radial-basis interpolant therefore
scores every strategy, exactly as the 2-D secondary cross-check did.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc
from resolution import fit_free_scores, knn_variation
from .cases import CASES, strengths

# Holistic tolerances, per input dimension, measured against the `space-filling`
# reference arm at the declared budget (results/nd-calibration-2026-09-20).
# Nothing is defaulted across dimensions: an absent dimension raises rather than
# inheriting a foreign limit.
#
# SPINE is primary above three dimensions. Every term is fit-free, so none can
# be inverted by a concentrated design and all three survive to 8D unchanged.
# PORTED keeps the 2D/3D four-term shape as a labelled secondary so continuity
# with the published low-dimensional scores stays visible; three of its terms
# come from the demoted thin-plate-spline evaluator and it must not decide a
# ranking.
#
# Calibration rule: the reference arm's worst case at the declared budget,
# rounded up. One declared asymmetry -- `fill_p95` is set at 1.5x the reference
# worst case rather than at it, because `space-filling` optimizes fill distance
# by construction and its own optimum cannot serve as a bar any other method
# could clear. All values are provisional normalized tolerances at the stated
# budget, not fitted accuracy claims.
SPINE_TOLERANCES = {
    5: dict(vwfd_p95=.70, nonlinear_p95=2.50, fill_p95=.40),
}
PORTED_TOLERANCES = {
    5: dict(nmae=.035, band_nmae=.035, p95_error=.13, vwfd_p95=.70),
}
CALIBRATION_BUDGET = {5: 512}

NARROW, WIDE = .06, .5          # peak sigma at full strength and at weak strength
ENVELOPE, BASE = .65, .15       # affine-envelope and base-slope amplitudes
AMPLITUDES = (.9, .7, .8, .6)
KERNEL, SMOOTHING = "thin_plate_spline", 0.


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class SurfaceND:
    """Max of affine modes plus anisotropic Gaussian peaks inside each mode.

    Every mode normal, the shared base slope and every peak width follow the
    declared strength matrix, so an inert axis changes nothing about the value.
    The seed moves peak centers and the signs of the non-leading axes; it never
    changes which axes are strong, so the declared strength table is stable.
    """
    case: str
    seed: int = 0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.case not in CASES:
            raise ValueError(self.case)

    @property
    def dim(self):
        return CASES[self.case]["dim"]

    @property
    def normals(self):
        """Unit mode normals; magnitudes reproduce the declared strengths."""
        if "normals" in self._cache:
            return self._cache["normals"]
        rows = np.array(strengths(self.case), float)
        rng = np.random.default_rng(1000+self.seed)
        signs = np.where(rng.random(self.dim) < .5, -1., 1.)
        signs[int(np.argmax(rows[0]))] = 1.
        rows = rows*signs
        fold = CASES[self.case]["fold"]
        if fold == "axis":
            lead = int(np.argmax(np.abs(rows[0])))
            rows[1] = rows[0].copy()
            rows[1][lead] *= -1          # fold hyperplane perpendicular to one axis
        elif fold == "dense":
            rows[1] = -rows[0]           # fold normal is the full strength vector
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        self._cache["normals"] = rows/norms
        return self._cache["normals"]

    @property
    def offsets(self):
        """Mode offsets balancing the region volumes; solved on a fixed design."""
        if "offsets" in self._cache:
            return self._cache["offsets"]
        k = len(self.normals)
        c = np.zeros(k)
        if k > 1:
            x = qmc.Sobol(self.dim, scramble=True, seed=777).random_base2(14)
            values = (x-.5) @ self.normals.T
            for _ in range(600):
                share = np.bincount(np.argmax(values+c, axis=1), minlength=k)/len(x)
                c = c - .5*ENVELOPE*(share-1/k)
                c = c - c.mean()
        self._cache["offsets"] = c
        return c

    @property
    def widths(self):
        """Peak sigma per mode and axis: narrow on strong axes, infinite on inert."""
        rows = np.array(strengths(self.case), float)
        sigma = NARROW + (WIDE-NARROW)*(1-rows)
        return np.where(rows > 0, sigma, np.inf)

    @property
    def centers(self):
        """One center per peak, kept inside its own mode and off every fold."""
        if "centers" in self._cache:
            return self._cache["centers"]
        rng = np.random.default_rng(self.seed)
        out, owners = [], []
        for m, count in enumerate(CASES[self.case]["peaks"]):
            finite = np.isfinite(self.widths[m])
            narrow = np.where(finite, self.widths[m], 1.)
            chosen = []
            for _ in range(count):
                point = None
                # An axis this mode ignores still decides which region a point
                # lies in whenever another mode is strong there, so pinning the
                # inert coordinates can leave no admissible center at all. The
                # second pass frees them; that never changes the surface value,
                # because their peak width is infinite either way.
                for free_inert in (False, True):
                    for _ in range(20000):
                        draw = rng.uniform(.2, .8, self.dim)
                        p = draw if free_inert else np.where(finite, draw, .5)
                        if self.region(p)[0] != m:
                            continue
                        if self.distance(p)[0] < 3*float(np.min(narrow)):
                            continue
                        if any(np.max(np.abs(p-q)/narrow) < 3 for q in chosen):
                            continue
                        point = p
                        break
                    if point is not None:
                        break
                if point is None:
                    raise RuntimeError(f"{self.case}: no admissible peak center for mode {m}")
                chosen.append(point)
            out.extend(chosen)
            owners.extend([m]*count)
        self._cache["centers"] = (np.array(out), np.array(owners))
        return self._cache["centers"]

    def region(self, x):
        return np.argmax((np.atleast_2d(x)-.5) @ self.normals.T + self.offsets, axis=1)

    def distance(self, x):
        """Distance to the nearest fold; +inf for a single-mode case."""
        x = np.atleast_2d(x)
        if len(self.normals) == 1:
            return np.full(len(x), np.inf)
        values = (x-.5) @ self.normals.T + self.offsets
        winner = values.argmax(axis=1)
        out = np.full(len(x), np.inf)
        for j in range(len(self.normals)):
            gap = np.linalg.norm(self.normals[winner]-self.normals[j], axis=1)
            out = np.minimum(out, np.divide(values[np.arange(len(x)), winner]-values[:, j], gap,
                                            out=np.full(len(x), np.inf), where=gap > 0))
        return out

    def peak_distance(self, x):
        """Distance to the nearest peak in that peak own width units."""
        x = np.atleast_2d(x)
        centers, owners = self.centers
        scaled = [np.sqrt(np.sum(np.square((x-c)/self.widths[m]), axis=1))
                  for c, m in zip(centers, owners)]
        return np.min(scaled, axis=0)

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        union = np.abs(np.array(strengths(self.case), float)).max(axis=0)
        union = union/max(float(np.linalg.norm(union)), 1e-12)
        y = .25 + BASE*(x @ union)
        if len(self.normals) > 1:
            y = y + ENVELOPE*np.max((x-.5) @ self.normals.T + self.offsets, axis=1)
        centers, owners = self.centers
        for i, (c, m) in enumerate(zip(centers, owners)):
            y = y + AMPLITUDES[i % len(AMPLITUDES)]*np.exp(
                -.5*np.sum(np.square((x-c)/self.widths[m]), axis=1))
        return y


class Observations:
    """Exact deterministic oracle charging unique evaluations, caching duplicates.

    A shared Sobol initialization replaces the 2-D corner charge: 2^d corners
    would consume 256 of an 8-D budget before any strategy makes a decision.
    """
    def __init__(self, evaluate, budget, dim, seed=0, shared=None):
        if budget < 2*dim+2:
            raise ValueError("budget must leave room for the shared initialization")
        self._evaluate = evaluate
        self.budget, self.dim = int(budget), int(dim)
        self._values = {}
        self.requests = 0
        self(self.initial_design(dim, seed) if shared is None else shared)

    @staticmethod
    def initial_design(dim, seed=0):
        # Latin hypercube, not Sobol: 2d+1 is never a power of two.
        return qmc.LatinHypercube(dim, seed=20260912+seed).random(2*dim+1)

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
        if x.shape[1] != self.dim or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("finite unit-box coordinates of the right dimension required")
        keys = [self.key(p) for p in x]
        new = list(dict.fromkeys(k for k in keys if k not in self._values))
        if len(new) > self.remaining:
            raise BudgetExceeded("batch would exceed the unique-evaluation budget")
        if new:
            values = np.asarray(self._evaluate(np.array(new)), float)
            if values.shape != (len(new),) or not np.isfinite(values).all():
                raise ValueError("oracle must return finite scalar observations")
            self._values.update(zip(new, values.tolist()))
        self.requests += len(x)
        return np.array([self._values[k] for k in keys])


def reconstruct(x, y, query):
    """The one common reconstructor; identical settings for every strategy."""
    result = RBFInterpolator(np.asarray(x), np.asarray(y), kernel=KERNEL, smoothing=SMOOTHING)(query)
    if not np.isfinite(result).all():
        raise ValueError("reconstruction produced nonfinite values")
    return result


def rmse(predicted, truth, scale=1.):
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if predicted.shape != truth.shape or not truth.size:
        raise ValueError("nonempty matching predictions/truth required")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all() or scale <= 0:
        raise ValueError("nonfinite predictions must not disappear from scoring")
    return float(np.sqrt(np.mean((predicted-truth)**2))/scale)


def truth_variation(surface, x, step=1e-4):
    """Exact finite-difference slope and curvature magnitude of the reference.

    Grading variation comes from the analytic surface, never from a sampler's
    own estimate, so VWRS and VURS cannot grade the estimator they sample with.
    """
    x = np.atleast_2d(np.asarray(x, float))
    gradient2 = np.zeros(len(x))
    curvature2 = np.zeros(len(x))
    base = surface(x)
    for axis in range(surface.dim):
        plus, minus = x.copy(), x.copy()
        plus[:, axis] = np.minimum(1., plus[:, axis]+step)
        minus[:, axis] = np.maximum(0., minus[:, axis]-step)
        span = plus[:, axis]-minus[:, axis]
        up, down = surface(plus), surface(minus)
        gradient2 += ((up-down)/span)**2
        curvature2 += (4*(up-2*base+down)/span**2)**2
    return np.sqrt(gradient2), np.sqrt(curvature2)


def evaluation_set(surface, n=65536, band=.06, peak=2.):
    if n < 4096 or n & (n-1):
        raise ValueError("test size must be a power of two >= 4096")
    x = qmc.Sobol(surface.dim, scramble=True, seed=91479).random_base2(n.bit_length()-1)
    y = surface(x)
    variation, curvature = truth_variation(surface, x)
    return dict(x=x, y=y, scale=float(np.ptp(y)), band=surface.distance(x) < band,
                peak=surface.peak_distance(x) < peak, region=surface.region(x),
                variation=variation, curvature=curvature)


def score(surface, observations, test, targets=None, secondary_targets=None,
          observed_variation=False):
    """Fit-free spine first, reconstruction-based errors second.

    `targets` are the primary holistic tolerances for *this* input dimension,
    and above three dimensions they are the fit-free spine set. They are
    deliberately not defaulted: the 2D values were calibrated against a regular
    grid at N=128 in two dimensions and carry no meaning elsewhere. Passing None
    reports every component and marks the holistic score uncalibrated rather
    than silently reusing a foreign tolerance. `secondary_targets` records the
    ported 2D/3D four-term shape alongside it, labelled, so continuity with the
    published low-dimensional scores is visible without letting a term from the
    demoted evaluator decide a ranking.

    The reconstruction-based family below uses the common thin-plate-spline RBF,
    which is demoted to a secondary cross-check: its normalized error rises with
    budget on concentrated designs (`5d-m1-p1-anis`, 0.1213 -> 0.2887 between
    N=218 and N=512). Read `error` and its relatives as dimension-local.
    """
    scale, truth = test["scale"], test["y"]
    observed = None
    if observed_variation:
        observed = knn_variation(observations.x, observations.y, test["x"])
    out = dict(n=len(observations.x), n_requests=observations.requests, scale=scale)
    out.update(fit_free_scores(observations.x, test["x"], test["variation"], scale,
                               curvature=test["curvature"], observed=observed))

    yhat = reconstruct(observations.x, observations.y, test["x"])
    absolute = np.abs(yhat-truth)/scale
    out.update(reconstruction="thin-plate-spline-rbf", reconstruction_role="secondary",
               error=rmse(yhat, truth, scale), nmae=float(np.mean(absolute)),
               p95_error=float(np.quantile(absolute, .95)),
               band_error=rmse(yhat[test["band"]], truth[test["band"]], scale) if test["band"].any() else None,
               band_nmae=float(np.mean(absolute[test["band"]])) if test["band"].any() else None,
               peak_error=rmse(yhat[test["peak"]], truth[test["peak"]], scale),
               peak_nmae=float(np.mean(absolute[test["peak"]])))
    for m in range(len(surface.normals)):
        mask = test["region"] == m
        out[f"mode{m}_error"] = rmse(yhat[mask], truth[mask], scale)
        out[f"mode{m}_nmae"] = float(np.mean(absolute[mask]))

    _apply_targets(out, targets, "holistic")
    _apply_targets(out, secondary_targets, "ported_holistic")
    return out


def _apply_targets(out, targets, prefix):
    out.update({prefix + "_error": None, prefix + "_driver": None,
                prefix + "_targets": None, prefix + "_uncalibrated": True})
    if targets is None:
        return
    missing = [name for name in targets if out.get(name) is None]
    if missing:
        raise ValueError(f"{prefix} targets name unavailable scores: {missing}")
    ratios = {name: out[name]/limit for name, limit in targets.items()}
    out.update({prefix + "_error": float(max(ratios.values())),
                prefix + "_driver": max(ratios, key=ratios.get),
                prefix + "_targets": dict(targets), prefix + "_uncalibrated": False})


def tolerances_for(dim, band_available=True):
    """The declared tolerance pair for this dimension, or a refusal.

    `band_available` drops the fold-band term for the one-mode controls, which
    have no fold for it to measure.
    """
    if dim not in SPINE_TOLERANCES:
        raise ValueError(
            f"no calibrated tolerances for d={dim}; run the reference-arm "
            f"calibration first, as was done for d=5 at budget "
            f"{CALIBRATION_BUDGET.get(5)}")
    ported = dict(PORTED_TOLERANCES[dim])
    if not band_available:
        ported.pop("band_nmae", None)
    return dict(SPINE_TOLERANCES[dim]), ported
