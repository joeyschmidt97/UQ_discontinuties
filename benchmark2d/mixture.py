"""Shared-budget mixture of triangle, grid-basis and GP experts.

Shortlist frozen from the previous pilot (b4899ad); no test-set selection.
Gates learn only errors predicted BEFORE the new observation is paid for.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from .strategies import initialize, fit_gp, normalized_blend

EXPERTS = ("triangles", "grid", "gpr-var")


def grid_features(x, side):
    axis = np.linspace(0, 1, side)
    h = 1/(side-1)
    a = np.maximum(0, 1-np.abs(x[:, 0, None]-axis)/h)
    b = np.maximum(0, 1-np.abs(x[:, 1, None]-axis)/h)
    return (a[:, :, None]*b[:, None, :]).reshape(len(x), -1)


def gating(query, history_x, history_errors):
    if not len(history_x):
        return np.full((len(query), 3), 1/3)
    weights = np.exp(-np.sum((query[:, None]-np.asarray(history_x)[None])**2, axis=2)/(2*.2**2))+.01
    errors = weights @ np.asarray(history_errors)/weights.sum(axis=1, keepdims=True)
    # Error floor plus a 10% uniform gate avoids permanently starving an expert.
    inverse = 1/(errors+1e-4)
    return .9*inverse/inverse.sum(axis=1, keepdims=True)+.1/3


def mixture(obs, seed, triangle_weight=0.):
    if not 0 <= triangle_weight <= 1:
        raise ValueError("triangle weight must be between zero and one")
    initialize(obs, seed)
    rng = np.random.default_rng(seed)
    history_x, history_errors, gate_history = [], [], []
    def fit():
        gp, _ = fit_gp(obs, seed)
        linear = LinearNDInterpolator(obs.x, obs.y)
        side = max(2, min(8, int(np.sqrt(len(obs.x)/2))))
        design = grid_features(obs.x, side)
        coef = np.linalg.solve(design.T@design+1e-5*np.eye(side*side), design.T@obs.y)
        def experts(q):
            return np.column_stack([linear(q), grid_features(q, side)@coef, gp.predict(q)])
        return gp, experts
    while obs.remaining:
        gp, experts = fit()
        tri = Delaunay(obs.x)
        vertices = obs.x[tri.simplices]
        centroids = vertices.mean(axis=1)
        candidates = np.vstack([rng.random((768, 2)), centroids])
        predictions = experts(candidates)
        gates = gating(candidates, history_x, history_errors)
        distance = cKDTree(obs.x).query(candidates)[0]
        sd = gp.predict(candidates, return_std=True)[1]
        # Triangle nonlinearity is estimated using the GP and observed linear
        # interpolant; no oracle is consulted for candidate scoring.
        tri_merit = np.abs(predictions[:, 2]-predictions[:, 0])*distance
        merits = np.column_stack([tri_merit, distance, sd])
        merits /= np.maximum(merits.max(axis=0), 1e-12)
        disagreement = predictions.std(axis=1)
        merit = np.sum(gates*merits, axis=1)+.25*disagreement/max(disagreement.max(), 1e-12)
        if len(history_x) % 5 == 0:
            merit = distance
        if triangle_weight:
            # The actual triangle policy scores its centroid proposals only.
            # Both policies use the same observations; no extra objective calls.
            edges = vertices[:, 1:]-vertices[:, :1]
            areas = np.abs(edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0])/2
            matrices = np.concatenate([vertices, np.ones((*vertices.shape[:2], 1))], axis=2)
            gradients = np.linalg.solve(matrices, obs.y[tri.simplices][..., None])[..., 0][:, :2]
            variation = np.zeros(len(vertices))
            for i, neighbors in enumerate(tri.neighbors):
                valid = neighbors[neighbors >= 0]
                if len(valid):
                    variation[i] = np.max(np.linalg.norm(gradients[valid]-gradients[i], axis=1))
            triangle_merit = areas if len(history_x) % 5 == 0 else areas*(.05+np.sqrt(areas)*variation)
            triangle_scores = np.concatenate([np.zeros(768), triangle_merit])
            merit = normalized_blend(merit, triangle_scores, 1-triangle_weight)
        merit[distance < 1e-8] = -np.inf
        index = merit.argmax()
        point, before = candidates[index], predictions[index].copy()
        value = obs(point)[0]
        history_x.append(point.tolist())
        history_errors.append(((before-value)**2).tolist())
        gate_history.append(gates[index].tolist())
    _, experts = fit()
    def predict(query):
        return np.sum(gating(query, history_x, history_errors)*experts(query), axis=1)
    return predict, dict(experts=list(EXPERTS), selection_source="previous pilot b4899ad; frozen shortlist",
                         shared_observations=True, gate_history=gate_history,
                         acquisition_weights={"moe": 1-triangle_weight, "triangles": triangle_weight},
                         blend_selection="weights frozen before new runs; components selected from 1da61c4 ranking",
                         prequential_locations=history_x, prequential_squared_errors=history_errors,
                         grid_expert="ridge fit of bilinear grid basis to shared paid observations")
