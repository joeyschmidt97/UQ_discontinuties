"""Policies see sampled values only; no access to truth, masks or test points."""
import warnings
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

ARMS = ("grid", "moe", "sglib", "sgpp", "gpr-var", "gpr-grad", "triangles",
        "gpr-blend", "moe-tri75", "moe-tri50")


def initialize(obs, seed):
    obs(qmc.LatinHypercube(2, seed=seed).random(min(5, obs.remaining)))


def fit_gp(obs, seed):
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (1e-3, 1e3))*Matern([.2, .2], (1e-2, 10.), nu=1.5),
        alpha=1e-8, normalize_y=True, random_state=seed, n_restarts_optimizer=0)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(obs.x, obs.y)
    return gp, len(messages)


def normalized_blend(first, second, first_weight):
    """Blend acquisition scores after independent max normalization."""
    if not 0 <= first_weight <= 1:
        raise ValueError("blend weight must be between zero and one")
    a = np.asarray(first) / max(float(np.max(first)), 1e-12)
    b = np.asarray(second) / max(float(np.max(second)), 1e-12)
    return first_weight*a + (1-first_weight)*b


def gpr(obs, seed, gradient=False, blend=False):
    initialize(obs, seed)
    rng = np.random.default_rng(seed)
    warning_count = 0
    while obs.remaining:
        gp, count = fit_gp(obs, seed)
        warning_count += count
        candidates = rng.random((1024, 2))
        nearest = cKDTree(obs.x).query(candidates)[0]
        _, sd = gp.predict(candidates, return_std=True)
        merit = sd.copy()
        if gradient or blend:
            grad = np.zeros(len(candidates))
            for axis in range(2):
                plus, minus = candidates.copy(), candidates.copy()
                plus[:, axis] = np.minimum(1., plus[:, axis]+.002)
                minus[:, axis] = np.maximum(0., minus[:, axis]-.002)
                grad += ((gp.predict(plus)-gp.predict(minus))/(plus[:, axis]-minus[:, axis]))**2
            # A small exploration floor avoids a zero score on a flat GP mean.
            merit *= .1 + np.sqrt(grad)
            if blend:
                merit = normalized_blend(merit, sd, .5)
        merit[nearest < 1e-6] = -np.inf
        batch = []
        for _ in range(min(1, obs.remaining)):
            index = int(np.argmax(merit))
            batch.append(candidates[index])
            merit[np.linalg.norm(candidates-candidates[index], axis=1) < .06] = -np.inf
        obs(batch)
    gp, count = fit_gp(obs, seed)
    return gp.predict, dict(fit_warnings=warning_count+count, kernel=str(gp.kernel_),
                            acquisition_weights={"gpr-grad": .5, "gpr-var": .5} if blend else None)


def triangles(obs, seed):
    """Proposed heuristic: area times neighboring-gradient disagreement.

    Gradients are fitted to each triangle's observed vertex values. Every
    fifth acquisition explores the largest triangle, including flat regions.
    This is not an implementation of published high-order SSC.
    """
    initialize(obs, seed)
    step = 0
    while obs.remaining:
        tri = Delaunay(obs.x)
        vertices = obs.x[tri.simplices]
        edges = vertices[:, 1:]-vertices[:, :1]
        areas = np.abs(edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0])/2
        matrices = np.concatenate([vertices, np.ones((*vertices.shape[:2], 1))], axis=2)
        gradients = np.linalg.solve(matrices, obs.y[tri.simplices][..., None])[..., 0][:, :2]
        variation = np.zeros(len(vertices))
        for i, neighbors in enumerate(tri.neighbors):
            valid = neighbors[neighbors >= 0]
            if len(valid):
                variation[i] = np.max(np.linalg.norm(gradients[valid]-gradients[i], axis=1))
        merit = areas if step % 5 == 0 else areas*(.05+np.sqrt(areas)*variation)
        obs(vertices[int(np.argmax(merit))].mean(axis=0))
        step += 1
    return None, dict(exploration_every=5)


def run_arm(name, obs, seed):
    if name == "grid":
        level = 1
        while obs.remaining:
            axis = np.linspace(0, 1, 2**level+1)
            a, b = np.meshgrid(axis, axis)
            candidates = np.column_stack([a.ravel(), b.ravel()])
            while obs.remaining:
                distance = cKDTree(obs.x).query(candidates)[0]
                if distance.max() < 1e-10:
                    break
                obs(candidates[distance.argmax()])
            level += 1
        return None, dict(nested=True, ordering="maximin within dyadic grid level")
    if name in ("moe", "moe-tri75", "moe-tri50"):
        from .mixture import mixture
        return mixture(obs, seed, triangle_weight={"moe": 0., "moe-tri75": .25, "moe-tri50": .5}[name])
    if name in ("gpr-var", "gpr-grad", "gpr-blend"):
        return gpr(obs, seed, gradient=name == "gpr-grad", blend=name == "gpr-blend")
    if name == "triangles":
        return triangles(obs, seed)
    if name == "sglib":
        from arms.sglib_arm import SgLibArm
        arm = SgLibArm(tol=0., max_level=20, budget_driven=True, nan_policy="error")
    elif name == "sgpp":
        from arms.sgpp_arm import SGppArm
        arm = SGppArm(basis="modlinear", refine="surplus", refine_batch=1, target_surplus=-1., nan_policy="error")
    else:
        raise ValueError(name)
    # External methods prescribe their own nodes. The common corners count
    # toward reconstruction/cost but do not alter those methods' native grids.
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
        # Native grids select batches; pay and expose only the prefix affordable
        # under the common unique-evaluation budget. No future values are read.
        arm.fit(prescribed, dim=2, budget=16*obs.budget)
    except BudgetExceeded:
        if obs.remaining:
            raise
    if obs.remaining:
        raise RuntimeError(f"native refinement stalled with {obs.remaining} unpaid points")
    return None, dict(batch_sizes=batch_sizes, prefix_of_native_batches=True,
                      native_prediction_unavailable="budget may end inside refinement batch")
