"""Placement policies for the high-dimensional cases.

Ported from the 2-D study with three changes forced by dimension: the triangle
policy and MoE's triangle/bilinear-grid experts are gone (Delaunay is unusable
above ~6-D and a bilinear grid needs side**d coefficients), the candidate pool
and the duplicate radius scale with d, and the progressive grid is replaced by a
Sobol space-filling baseline. Policies see sampled values only.
"""
import warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from resolution import knn_variation
from resolution.variation import stencil_size

# GP-uncertainty share of the blended acquisition; the rest weights the
# gradient-weighted merit. The three ratios are the 2-D sweep winners.
GP_BLENDS = {"gpr-u50-g50": .5, "gpr-u70-g30": .7, "gpr-u30-g70": .3}

ARMS = (("space-filling", "gpr-var", "gpr-grad", "moe", "sglib", "sgpp",
         "gpr-m05-var", "gpr-m05-grad", "gpr-m05-blend", "vwrs", "vurs")
        + tuple(GP_BLENDS))
RUNNABLE_ARMS = tuple(arm for arm in ARMS if arm != "sgpp")

# VWRS/VURS acquisition weights, matching the preregistered 2D and 3D defaults:
# VWRS splits coverage and variation evenly, VURS adds GP uncertainty in equal
# thirds. Fixed baselines, not tuned winners.
VWRS_WEIGHTS = dict(coverage=.5, variation=.5, uncertainty=0.)
VURS_WEIGHTS = dict(coverage=1/3, variation=1/3, uncertainty=1/3)


def candidate_count(dim):
    """1,024 candidates was calibrated at d=2; hold candidates per axis fixed.

    Rounded to a power of two so the Sobol pool keeps its balance property.
    """
    return int(2**round(np.log2(min(16384, 512*dim))))


def duplicate_radius(dim, n):
    """A tenth of the typical neighbour distance at this size and dimension."""
    return .1*n**(-1/dim)


def fit_gp(obs, seed, nu=1.5):
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (1e-3, 1e3))*Matern([.3]*obs.dim, (1e-2, 1e2), nu=nu),
        alpha=1e-8, normalize_y=True, random_state=seed, n_restarts_optimizer=1)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(obs.x, obs.y)
    return gp, len(messages)


def normalized_blend(first, second, first_weight):
    if not 0 <= first_weight <= 1:
        raise ValueError("blend weight must be between zero and one")
    a = np.asarray(first)/max(float(np.max(first)), 1e-12)
    b = np.asarray(second)/max(float(np.max(second)), 1e-12)
    return first_weight*a + (1-first_weight)*b


def sobol_candidates(rng, dim, count):
    return qmc.Sobol(dim, scramble=True, seed=int(rng.integers(2**31))).random(count)


def space_filling(obs, seed):
    """Baseline: extend one scrambled Sobol sequence, skipping paid points."""
    rng = np.random.default_rng(seed)
    while obs.remaining:
        pool = sobol_candidates(rng, obs.dim, candidate_count(obs.dim))
        distance = cKDTree(obs.x).query(pool)[0]
        obs(pool[int(distance.argmax())])
    return None, dict(pool=candidate_count(obs.dim))


def gpr(obs, seed, gradient=False, blend=None, nu=1.5):
    """GP posterior standard deviation, optionally weighted by predicted gradient."""
    if blend is not None and not 0 <= blend <= 1:
        raise ValueError("uncertainty share must be between zero and one")
    rng = np.random.default_rng(seed)
    warning_count = 0
    while obs.remaining:
        gp, count = fit_gp(obs, seed, nu)
        warning_count += count
        candidates = sobol_candidates(rng, obs.dim, candidate_count(obs.dim))
        nearest = cKDTree(obs.x).query(candidates)[0]
        _, sd = gp.predict(candidates, return_std=True)
        merit = sd.copy()
        if gradient or blend is not None:
            grad = np.zeros(len(candidates))
            for axis in range(obs.dim):
                plus, minus = candidates.copy(), candidates.copy()
                plus[:, axis] = np.minimum(1., plus[:, axis]+.002)
                minus[:, axis] = np.maximum(0., minus[:, axis]-.002)
                grad += ((gp.predict(plus)-gp.predict(minus))/(plus[:, axis]-minus[:, axis]))**2
            merit *= .1 + np.sqrt(grad)
            if blend is not None:
                merit = normalized_blend(merit, sd, 1-blend)
        merit[nearest < duplicate_radius(obs.dim, len(obs.x))] = -np.inf
        obs(candidates[int(np.argmax(merit))])
    gp, count = fit_gp(obs, seed, nu)
    return gp.predict, dict(fit_warnings=warning_count+count, kernel=str(gp.kernel_),
                            matern_nu=nu, candidates=candidate_count(obs.dim),
                            acquisition_weights=None if blend is None else
                            {"gpr-grad": 1-blend, "gpr-var": blend})


def resolution_sampling(obs, seed, uncertainty=False, mode="curvature"):
    """VWRS and VURS with a dimension-free variation estimator.

    The 2D and 3D implementations weight fill distance by an observed Delaunay
    simplex gradient. Delaunay does not survive these dimensions, so the
    estimator is replaced by the shared k-nearest-neighbour weighted
    least-squares fit while the acquisition rule itself is unchanged:

        A(x) = lc*h~(x) + lv*q~(x) [+ lu*sigma~(x)]

    Normalization is by candidate-cloud maximum, matching the 2D and 3D arms,
    so the fixed weights keep their preregistered meaning.
    """
    weights = VURS_WEIGHTS if uncertainty else VWRS_WEIGHTS
    rng = np.random.default_rng(seed)
    warning_count = 0
    while obs.remaining:
        candidates = sobol_candidates(rng, obs.dim, candidate_count(obs.dim))
        spacing = cKDTree(obs.x).query(candidates)[0]
        local = knn_variation(obs.x, obs.y, candidates)
        variation = spacing*local.indicator(spacing, mode)
        merit = (weights["coverage"]*spacing/max(float(spacing.max()), 1e-12)
                 + weights["variation"]*variation/max(float(variation.max()), 1e-12))
        if uncertainty:
            gp, count = fit_gp(obs, seed, nu=.5)
            warning_count += count
            sd = gp.predict(candidates, return_std=True)[1]
            merit = merit + weights["uncertainty"]*sd/max(float(sd.max()), 1e-12)
        merit[spacing < duplicate_radius(obs.dim, len(obs.x))] = -np.inf
        obs(candidates[int(np.argmax(merit))])
    metadata = dict(acquisition_weights=dict(weights), variation_mode=mode,
                    variation="fill distance times knn weighted-least-squares "
                              "local-linear residual curvature",
                    stencil=stencil_size(obs.dim), candidates=candidate_count(obs.dim))
    if uncertainty:
        gp, count = fit_gp(obs, seed, nu=.5)
        metadata.update(fit_warnings=warning_count+count, kernel=str(gp.kernel_), matern_nu=.5)
        return gp.predict, metadata
    return None, metadata


def mixture(obs, seed):
    """Locally gated GP, quadratic-ANOVA and nearest-neighbour linear experts.

    The 2-D triangle and bilinear-grid experts do not survive the dimension, and
    neither replacement shares a form with the common reconstructor. Gates learn
    only errors predicted before the new observation is paid for.
    """
    rng = np.random.default_rng(seed)
    history_x, history_errors, gate_history = [], [], []
    quadratic = PolynomialFeatures(2, include_bias=False)

    def local_linear(query):
        tree = cKDTree(obs.x)
        k = min(len(obs.x), 2*obs.dim+2)
        _, index = tree.query(query, k=k)
        out = np.empty(len(query))
        for i, rows in enumerate(np.atleast_2d(index)):
            a = np.column_stack([obs.x[rows]-query[i], np.ones(len(rows))])
            out[i] = np.linalg.lstsq(a, obs.y[rows], rcond=None)[0][-1]
        return out

    def fit():
        gp, _ = fit_gp(obs, seed)
        design = quadratic.fit_transform(obs.x)
        ridge = Ridge(alpha=1e-3).fit(design, obs.y)
        def experts(q):
            return np.column_stack([local_linear(q), ridge.predict(quadratic.transform(q)), gp.predict(q)])
        return gp, experts

    while obs.remaining:
        gp, experts = fit()
        candidates = sobol_candidates(rng, obs.dim, candidate_count(obs.dim))
        predictions = experts(candidates)
        gates = gating(candidates, history_x, history_errors, obs.dim)
        distance = cKDTree(obs.x).query(candidates)[0]
        sd = gp.predict(candidates, return_std=True)[1]
        merits = np.column_stack([np.abs(predictions[:, 2]-predictions[:, 0])*distance, distance, sd])
        merits /= np.maximum(merits.max(axis=0), 1e-12)
        disagreement = predictions.std(axis=1)
        merit = np.sum(gates*merits, axis=1)+.25*disagreement/max(disagreement.max(), 1e-12)
        if len(history_x) % 5 == 0:
            merit = distance
        merit[distance < duplicate_radius(obs.dim, len(obs.x))] = -np.inf
        index = int(merit.argmax())
        point, before = candidates[index], predictions[index].copy()
        value = obs(point)[0]
        history_x.append(point.tolist())
        history_errors.append(((before-value)**2).tolist())
        gate_history.append(gates[index].tolist())
    _, experts = fit()

    def predict(query):
        return np.sum(gating(query, history_x, history_errors, obs.dim)*experts(query), axis=1)
    return predict, dict(experts=["knn-linear", "quadratic-anova", "gpr"], gate_history=gate_history,
                         shared_observations=True, candidates=candidate_count(obs.dim),
                         prequential_locations=history_x, prequential_squared_errors=history_errors)


def gating(query, history_x, history_errors, dim):
    if not len(history_x):
        return np.full((len(query), 3), 1/3)
    bandwidth = .2*np.sqrt(dim/2)
    weights = np.exp(-np.sum((query[:, None]-np.asarray(history_x)[None])**2, axis=2)/(2*bandwidth**2))+.01
    errors = weights @ np.asarray(history_errors)/weights.sum(axis=1, keepdims=True)
    inverse = 1/(errors+1e-4)
    return .9*inverse/inverse.sum(axis=1, keepdims=True)+.1/3


def run_arm(name, obs, seed):
    if name == "space-filling":
        return space_filling(obs, seed)
    if name == "moe":
        return mixture(obs, seed)
    if name in ("gpr-var", "gpr-grad") or name in GP_BLENDS:
        return gpr(obs, seed, gradient=name == "gpr-grad", blend=GP_BLENDS.get(name))
    if name in ("gpr-m05-var", "gpr-m05-grad", "gpr-m05-blend"):
        return gpr(obs, seed, gradient=name == "gpr-m05-grad",
                   blend=.5 if name == "gpr-m05-blend" else None, nu=.5)
    if name in ("vwrs", "vurs"):
        return resolution_sampling(obs, seed, uncertainty=name == "vurs")
    if name == "sglib":
        from arms.sglib_arm import SgLibArm
        arm = SgLibArm(tol=0., max_level=20, budget_driven=True, nan_policy="error")
    elif name == "sgpp":
        from arms.sgpp_arm import SGppArm
        arm = SGppArm(basis="modlinear", refine="surplus", refine_batch=1, target_surplus=-1., nan_policy="error")
    else:
        raise ValueError(name)
    from .core import BudgetExceeded
    batch_sizes = []

    def prescribed(points):
        points = np.atleast_2d(points)
        batch_sizes.append(len(points))
        values = []
        for point in points:
            values.append(obs(point)[0])
            if not obs.remaining:
                raise BudgetExceeded("exact paid-point prefix complete")
        return np.array(values)
    try:
        arm.fit(prescribed, dim=obs.dim, budget=16*obs.budget)
    except BudgetExceeded:
        if obs.remaining:
            raise
    if obs.remaining:
        raise RuntimeError(f"native refinement stalled with {obs.remaining} unpaid points")
    return None, dict(batch_sizes=batch_sizes, prefix_of_native_batches=True)
