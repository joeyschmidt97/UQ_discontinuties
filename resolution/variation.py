"""Local response variation from observed samples alone, in any dimension.

`benchmark2d` and `benchmark3d` estimate the local variation that VWRS and VURS
need from a Delaunay simplex gradient. That is the single dimension-bound piece
of the design: the method definition in the vault note leaves the estimator open
("gradient magnitude, curvature, local oscillation, local-linear residual, or a
discontinuity indicator"), and only the instantiation assumes a triangulation.

One k-nearest-neighbour weighted least-squares linear fit per query point yields
every estimator that definition names, in a single pass, at any input dimension:

    y(x_j) ~ a + g . (x_j - q),  weights tricube in ||x_j - q|| / r

  * `gradient`     = ||g||, the fitted slope magnitude.
  * `curvature`    = 2 * (weighted RMS residual) / r**2. The residual is what a
                     linear reconstruction of this neighbourhood fails to
                     capture; dividing by r**2 turns it into an effective second
                     derivative, so `curvature * h` has slope units and
                     `h**2 * curvature` has response units.
  * `disagreement` = spread of the neighbours' own fitted gradients, the
                     neighbouring-gradient-disagreement indicator.

Why the residual leads the slope for refinement: under piecewise-linear
reconstruction a steep plane is reproduced exactly while a modest narrow peak is
not, so slope alone sends budget to structure that is already resolved. A
discontinuity produces a large residual and is therefore flagged by the same
quantity. Note that the residual is a refinement *signal*, not an error bound --
a finite smoothness estimate cannot bound behaviour across a jump.

Branch restriction is available for truth-known scoring, where region labels
exist. Acquisition deliberately runs unrestricted: a stencil that spans a fold
reports the jump, which is the signal the sampler needs.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

RIDGE = 1e-10           # stabilizes the normal equations on degenerate stencils
MIN_RADIUS = 1e-12


def stencil_size(dim, k=None):
    """Enough neighbours for a linear fit plus a residual to exist.

    A d-dimensional linear fit has d+1 coefficients, so d+2 points are the
    minimum that can leave any residual at all; 2(d+1) gives the fit a usable
    amount of redundancy without making the neighbourhood non-local.
    """
    if k is not None:
        if k < dim + 2:
            raise ValueError("stencil must exceed the linear-fit coefficient count")
        return int(k)
    return 2*(dim + 1)


def tricube(u):
    return np.clip(1 - np.clip(u, 0, 1)**3, 0, None)**3


@dataclass(frozen=True)
class LocalVariation:
    """Per-query-point variation estimates, all with declared units."""
    gradient: np.ndarray        # slope units: response per unit input
    curvature: np.ndarray       # response per unit input squared
    disagreement: np.ndarray    # slope units
    residual: np.ndarray        # response units
    radius: np.ndarray          # input units: distance to the furthest neighbour

    def indicator(self, spacing, mode="curvature"):
        """Refinement signal in slope units, for weighting by fill distance.

        "curvature" is the default and the note's preference: it ignores a
        resolved steep plane and fires on peaks, folds and jumps. "gradient"
        reproduces what the 2D and 3D Delaunay implementations used, so the
        back-port can run both and show the difference rather than assert it.
        """
        spacing = np.asarray(spacing, float)
        if mode == "curvature":
            return self.curvature*spacing
        if mode == "gradient":
            return self.gradient
        if mode == "blend":
            return self.gradient + self.curvature*spacing
        raise ValueError(mode)


def knn_variation(x, y, query, k=None, labels=None, query_labels=None):
    """Fit one weighted local linear model per query point.

    `labels` restricts each stencil to observed points carrying the same label
    as the query. Used for truth-known scoring only; leave it None to let a
    stencil span a discontinuity and report it.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    query = np.atleast_2d(np.asarray(query, float))
    if x.ndim != 2 or y.shape != (len(x),) or query.shape[1] != x.shape[1]:
        raise ValueError("matching observation matrix, value vector and query matrix required")
    dim = x.shape[1]
    k = stencil_size(dim, k)
    if len(x) < dim + 2:
        raise ValueError("too few observations for a local linear fit with a residual")

    if labels is None:
        return _fit(x, y, query, min(k, len(x)), dim)

    labels = np.asarray(labels)
    if labels.shape != (len(x),):
        raise ValueError("one label per observation required")
    if query_labels is None:
        query_labels = labels[cKDTree(x).query(query)[1]]
    query_labels = np.asarray(query_labels)
    if query_labels.shape != (len(query),):
        raise ValueError("one label per query point required")

    parts = {}
    for label in np.unique(query_labels):
        rows = query_labels == label
        members = labels == label
        # A branch with too few samples cannot support a restricted fit; fall
        # back to the unrestricted stencil rather than inventing a value.
        if members.sum() < dim + 2:
            parts[label] = _fit(x, y, query[rows], min(k, len(x)), dim)
        else:
            parts[label] = _fit(x[members], y[members], query[rows],
                                min(k, int(members.sum())), dim)
    out = {}
    for field in ("gradient", "curvature", "disagreement", "residual", "radius"):
        merged = np.empty(len(query))
        for label, part in parts.items():
            merged[query_labels == label] = getattr(part, field)
        out[field] = merged
    return LocalVariation(**out)


def _fit(x, y, query, k, dim):
    if not len(query):
        empty = np.empty(0)
        return LocalVariation(empty, empty, empty, empty, empty)
    distance, index = cKDTree(x).query(query, k=k)
    distance = np.atleast_2d(distance)
    index = np.atleast_2d(index)
    radius = np.maximum(distance[:, -1], MIN_RADIUS)

    offsets = x[index] - query[:, None, :]                  # (m, k, dim)
    values = y[index]                                       # (m, k)
    weights = tricube(distance/radius[:, None]) + 1e-6      # (m, k)

    design = np.concatenate([np.ones((*offsets.shape[:2], 1)), offsets], axis=2)
    weighted = design*weights[:, :, None]
    normal = np.einsum("mkj,mkl->mjl", weighted, design)
    normal[:, np.arange(dim + 1), np.arange(dim + 1)] += RIDGE
    target = np.einsum("mkj,mk->mj", weighted, values)
    # NumPy 2 treats a 2-D right-hand side as a single matrix, so keep the
    # batch explicit with a trailing axis and drop it again.
    coefficients = np.linalg.solve(normal, target[..., None])[..., 0]   # (m, dim+1)

    prediction = np.einsum("mkj,mj->mk", design, coefficients)
    weight_sum = weights.sum(axis=1)
    residual = np.sqrt(np.einsum("mk,mk->m", weights, (values - prediction)**2)/weight_sum)
    gradient = np.linalg.norm(coefficients[:, 1:], axis=1)
    curvature = 2*residual/radius**2

    # Neighbour disagreement reuses the same fits: each observed point's own
    # gradient, spread across the query's stencil. Fitting at the observations
    # costs one extra pass of the same shape, not a second algorithm.
    own = _observation_gradients(x, y, k, dim)
    neighbour = own[index]                                  # (m, k, dim)
    disagreement = np.linalg.norm(
        neighbour - coefficients[:, None, 1:], axis=2).max(axis=1)

    return LocalVariation(gradient=gradient, curvature=curvature,
                          disagreement=disagreement, residual=residual, radius=radius)


def _observation_gradients(x, y, k, dim):
    distance, index = cKDTree(x).query(x, k=min(k + 1, len(x)))
    distance = np.atleast_2d(distance)[:, 1:]
    index = np.atleast_2d(index)[:, 1:]
    if not index.size:
        return np.zeros((len(x), dim))
    radius = np.maximum(distance[:, -1], MIN_RADIUS)
    offsets = x[index] - x[:, None, :]
    weights = tricube(distance/radius[:, None]) + 1e-6
    design = np.concatenate([np.ones((*offsets.shape[:2], 1)), offsets], axis=2)
    weighted = design*weights[:, :, None]
    normal = np.einsum("mkj,mkl->mjl", weighted, design)
    normal[:, np.arange(dim + 1), np.arange(dim + 1)] += RIDGE
    target = np.einsum("mkj,mk->mj", weighted, y[index])
    return np.linalg.solve(normal, target[..., None])[..., 0][:, 1:]
