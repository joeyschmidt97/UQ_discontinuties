"""Three-dimensional versions of every primary 2D placement policy."""
import warnings

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


ARMS = ("grid", "sglib", "sgpp", "triangles", "gpr-var", "gpr-grad", "gpr-blend",
        "gpr-m05-var", "gpr-m05-grad", "gpr-m05-blend", "vwrs", "vurs")
RUNNABLE_ARMS = tuple(arm for arm in ARMS if arm != "sgpp")


def initialize(obs, seed):
    obs(qmc.LatinHypercube(obs.dim, seed=seed).random(min(5, obs.remaining)))


def fit_gp(obs, seed, nu=1.5):
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (1e-3, 1e3))*Matern(np.full(obs.dim, .2), (1e-2, 10.), nu=nu),
        alpha=1e-8, normalize_y=True, random_state=seed, n_restarts_optimizer=0)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(obs.x, obs.y)
    return gp, len(messages)


def normalized_blend(first, second, first_weight):
    a = np.asarray(first)/max(float(np.max(first)), 1e-12)
    b = np.asarray(second)/max(float(np.max(second)), 1e-12)
    return first_weight*a+(1-first_weight)*b


def gpr(obs, seed, gradient=False, blend=None, nu=1.5):
    initialize(obs, seed)
    rng = np.random.default_rng(seed)
    warnings_seen = 0
    while obs.remaining:
        gp, count = fit_gp(obs, seed, nu)
        warnings_seen += count
        candidates = rng.random((2048, obs.dim))
        nearest = cKDTree(obs.x).query(candidates)[0]
        _, sd = gp.predict(candidates, return_std=True)
        merit = sd.copy()
        if gradient or blend is not None:
            grad2 = np.zeros(len(candidates))
            for axis in range(obs.dim):
                plus, minus = candidates.copy(), candidates.copy()
                plus[:, axis] = np.minimum(1., plus[:, axis]+.002)
                minus[:, axis] = np.maximum(0., minus[:, axis]-.002)
                grad2 += ((gp.predict(plus)-gp.predict(minus))/(plus[:, axis]-minus[:, axis]))**2
            weighted = sd*(.1+np.sqrt(grad2))
            merit = weighted if blend is None else normalized_blend(weighted, sd, 1-blend)
        merit[nearest < 1e-6] = -np.inf
        obs(candidates[int(np.argmax(merit))])
    gp, count = fit_gp(obs, seed, nu)
    return gp.predict, dict(fit_warnings=warnings_seen+count, kernel=str(gp.kernel_), matern_nu=nu,
                            acquisition_weights=None if blend is None else
                            {"gpr-grad": 1-blend, "gpr-var": blend})


def simplex_gradient(obs, candidates):
    tri = Delaunay(obs.x)
    vertices = obs.x[tri.simplices]
    matrices = np.concatenate([vertices, np.ones((*vertices.shape[:2], 1))], axis=2)
    coefficients = np.linalg.solve(matrices, obs.y[tri.simplices][..., None])[..., 0]
    which = tri.find_simplex(candidates)
    if (which < 0).any():
        raise RuntimeError("shared cube corners must cover every candidate")
    return np.linalg.norm(coefficients[which, :obs.dim], axis=1)


def resolution_sampling(obs, seed, uncertainty=False):
    initialize(obs, seed)
    rng = np.random.default_rng(seed)
    warnings_seen = 0
    while obs.remaining:
        candidates = rng.random((2048, obs.dim))
        spacing = cKDTree(obs.x).query(candidates)[0]
        variation = spacing*(.1+simplex_gradient(obs, candidates))
        merit = normalized_blend(spacing, variation, .5)
        if uncertainty:
            gp, count = fit_gp(obs, seed, nu=.5)
            warnings_seen += count
            sd = gp.predict(candidates, return_std=True)[1]
            merit = (spacing/max(float(spacing.max()), 1e-12)
                     + variation/max(float(variation.max()), 1e-12)
                     + sd/max(float(sd.max()), 1e-12))/3
        merit[spacing < 1e-6] = -np.inf
        obs(candidates[int(np.argmax(merit))])
    metadata = dict(coverage_weight=1/3 if uncertainty else .5,
                    variation_weight=1/3 if uncertainty else .5,
                    uncertainty_weight=1/3 if uncertainty else 0.,
                    variation="distance times observed tetrahedral gradient")
    if uncertainty:
        gp, count = fit_gp(obs, seed, nu=.5)
        metadata.update(fit_warnings=warnings_seen+count, kernel=str(gp.kernel_), matern_nu=.5)
        return gp.predict, metadata
    return None, metadata


def triangles(obs, seed):
    initialize(obs, seed)
    step = 0
    while obs.remaining:
        tri = Delaunay(obs.x)
        vertices = obs.x[tri.simplices]
        edges = vertices[:, 1:]-vertices[:, :1]
        volumes = np.abs(np.linalg.det(edges))/6
        matrices = np.concatenate([vertices, np.ones((*vertices.shape[:2], 1))], axis=2)
        gradients = np.linalg.solve(matrices, obs.y[tri.simplices][..., None])[..., 0][:, :obs.dim]
        variation = np.zeros(len(vertices))
        for index, neighbors in enumerate(tri.neighbors):
            valid = neighbors[neighbors >= 0]
            if len(valid):
                variation[index] = np.max(np.linalg.norm(gradients[valid]-gradients[index], axis=1))
        merit = volumes if step % 5 == 0 else volumes*(.05+np.cbrt(volumes)*variation)
        obs(vertices[int(np.argmax(merit))].mean(axis=0))
        step += 1
    return None, dict(exploration_every=5, geometry="tetrahedral volume and gradient disagreement")


def grid(obs):
    level = 1
    while obs.remaining:
        axis = np.linspace(0, 1, 2**level+1)
        meshes = np.meshgrid(axis, axis, axis, indexing="ij")
        candidates = np.column_stack([mesh.ravel() for mesh in meshes])
        while obs.remaining:
            spacing = cKDTree(obs.x).query(candidates)[0]
            if spacing.max() < 1e-10:
                break
            obs(candidates[int(spacing.argmax())])
        level += 1
    return None, dict(nested=True, ordering="maximin within 3D dyadic grid level")


def external_grid(name, obs):
    if name == "sglib":
        from arms.sglib_arm import SgLibArm
        arm = SgLibArm(tol=0., max_level=20, budget_driven=True, nan_policy="error")
    else:
        from arms.sgpp_arm import SGppArm
        arm = SGppArm(basis="modlinear", refine="surplus", refine_batch=1,
                      target_surplus=-1., nan_policy="error")
    from .core import BudgetExceeded
    batches = []

    def prescribed(points):
        points = np.atleast_2d(points)
        batches.append(len(points))
        values = []
        for point in points:
            values.append(obs(point)[0])
            if not obs.remaining:
                raise BudgetExceeded("exact paid-point prefix complete")
        return np.asarray(values)
    try:
        arm.fit(prescribed, dim=obs.dim, budget=16*obs.budget)
    except BudgetExceeded:
        if obs.remaining:
            raise
    if obs.remaining:
        raise RuntimeError(f"native refinement stalled with {obs.remaining} unpaid points")
    return None, dict(batch_sizes=batches, prefix_of_native_batches=True,
                      native_prediction_unavailable="budget may end inside a refinement batch")


def run_arm(name, obs, seed):
    if name == "grid":
        return grid(obs)
    if name == "triangles":
        return triangles(obs, seed)
    if name in ("gpr-var", "gpr-grad", "gpr-blend"):
        return gpr(obs, seed, gradient=name == "gpr-grad", blend=.5 if name == "gpr-blend" else None)
    if name in ("gpr-m05-var", "gpr-m05-grad", "gpr-m05-blend"):
        return gpr(obs, seed, gradient=name == "gpr-m05-grad",
                   blend=.5 if name == "gpr-m05-blend" else None, nu=.5)
    if name in ("vwrs", "vurs"):
        return resolution_sampling(obs, seed, uncertainty=name == "vurs")
    if name in ("sglib", "sgpp"):
        return external_grid(name, obs)
    raise ValueError(name)
