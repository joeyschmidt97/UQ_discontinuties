"""Policies see sampled values only; no access to truth, masks or test points."""
import warnings
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

ARMS = ("grid", "sobol", "sglib", "sgpp", "gpr-var", "gpr-grad", "triangles")


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


def gpr(obs, seed, gradient=False):
    initialize(obs, seed)
    rng = np.random.default_rng(seed)
    warning_count = 0
    while obs.remaining:
        gp, count = fit_gp(obs, seed)
        warning_count += count
        candidates = rng.random((2048, 2))
        nearest = cKDTree(obs.x).query(candidates)[0]
        _, sd = gp.predict(candidates, return_std=True)
        merit = sd.copy()
        if gradient:
            grad = np.zeros(len(candidates))
            for axis in range(2):
                plus, minus = candidates.copy(), candidates.copy()
                plus[:, axis] = np.minimum(1., plus[:, axis]+.002)
                minus[:, axis] = np.maximum(0., minus[:, axis]-.002)
                grad += ((gp.predict(plus)-gp.predict(minus))/(plus[:, axis]-minus[:, axis]))**2
            # A small exploration floor avoids a zero score on a flat GP mean.
            merit *= .1 + np.sqrt(grad)
        merit[nearest < 1e-6] = -np.inf
        batch = []
        for _ in range(min(4, obs.remaining)):
            index = int(np.argmax(merit))
            batch.append(candidates[index])
            merit[np.linalg.norm(candidates-candidates[index], axis=1) < .06] = -np.inf
        obs(batch)
    gp, count = fit_gp(obs, seed)
    return gp.predict, dict(fit_warnings=warning_count+count, kernel=str(gp.kernel_))


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
        side = int(np.sqrt(obs.budget))
        a, b = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        obs(np.column_stack([a.ravel(), b.ravel()]))
        return None, dict(grid_side=side, nested=False)
    if name == "sobol":
        # Keep power-of-two Sobol blocks intact; four charged corners are extra.
        exponent = (obs.remaining).bit_length()-1
        obs(qmc.Sobol(2, scramble=True, seed=seed).random_base2(exponent))
        return None, dict(sobol_points=2**exponent)
    if name in ("gpr-var", "gpr-grad"):
        return gpr(obs, seed, gradient=name == "gpr-grad")
    if name == "triangles":
        return triangles(obs, seed)
    if name == "sglib":
        from arms.sglib_arm import SgLibArm
        arm = SgLibArm(tol=1e-10, nan_policy="error")
    elif name == "sgpp":
        from arms.sgpp_arm import SGppArm
        arm = SGppArm(basis="modlinear", refine="surplus", refine_batch=1, nan_policy="error")
    else:
        raise ValueError(name)
    # External methods prescribe their own nodes. The common corners count
    # toward reconstruction/cost but do not alter those methods' native grids.
    arm.fit(obs, dim=2, budget=obs.remaining)
    return arm.predict, arm.summary()
