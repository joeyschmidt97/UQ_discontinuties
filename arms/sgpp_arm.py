"""
SG++ arm -- spatially adaptive sparse grid with local hierarchical bases.

This is the contender Ionut pointed at as the discontinuity-capable alternative
to `sg_lib`: local bases let refinement chase a kink instead of ringing around
it, and SG++ is licensed for use with credit, which the `sg_lib` reference clone
is not.

Differences from the two scripts this repo started with (`sgpp_examples/`),
each of which was blocking a fair comparison:

  * budget is metered in FUNCTION EVALUATIONS, not grid points, and is a hard
    cap -- so an arm that stalls on a kink stops at the same cost as one that
    converges, instead of running to a different `max_points`;
  * `createOperationMultipleEval` replaces the per-point `createOperationEvalNaive`
    loop; with RMSE tracked every refinement step, that loop was the runtime;
  * FAILED EVALUATIONS are handled explicitly. A sparse grid prescribes its
    nodes, so a NaN is a hole in the interpolant, not a dropped sample -- the
    single property most likely to decide this against GPR in a semi-autonomous
    run. `nan_policy` makes the choice visible instead of crashing;
  * the refinement functor and basis are switchable, since surplus-vs-volume
    refinement and linear-vs-B-spline bases are exactly the SG++ knobs whose
    value on a discontinuous target is the open question.

Requires `pysgpp`. Install: `pip install pysgpp` (wheels exist for common
platforms; otherwise build SG++ from source with the Python bindings enabled).
"""

from __future__ import annotations

import numpy as np

from .base import Arm

try:
    import pysgpp
    HAVE_PYSGPP = True
except ImportError:                                    # pragma: no cover
    pysgpp = None
    HAVE_PYSGPP = False


_MISSING = ("pysgpp is not installed. `pip install pysgpp`, or build SG++ with "
            "the Python bindings. The manifold and the scoring harness run "
            "without it; only this arm needs it.")


# ===========================================================================
# numpy <-> pysgpp plumbing
# ===========================================================================
def _to_matrix(X):
    X = np.ascontiguousarray(np.atleast_2d(np.asarray(X, float)))
    try:
        return pysgpp.DataMatrix(X)                    # newer bindings take numpy
    except Exception:
        dm = pysgpp.DataMatrix(X.shape[0], X.shape[1])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                dm.set(i, j, float(X[i, j]))
        return dm


def _from_vector(v, n):
    try:
        return np.asarray(v.array(), float).copy()[:n]
    except Exception:
        return np.array([v[i] for i in range(n)], float)


def grid_coords(gs, start=0):
    """Standard coordinates of grid points [start, size) as an (m, dim) array."""
    d = gs.getDimension()
    idx = range(start, gs.getSize())
    P = np.empty((len(idx), d))
    for r, i in enumerate(idx):
        gp = gs.getPoint(i)
        for j in range(d):
            P[r, j] = gp.getStandardCoordinate(j)
    return P


_GRID_FACTORY = {
    "modlinear": lambda d, deg: pysgpp.Grid.createModLinearGrid(d),
    "linear": lambda d, deg: pysgpp.Grid.createLinearGrid(d),
    "linearboundary": lambda d, deg: pysgpp.Grid.createLinearBoundaryGrid(d),
    "modbspline": lambda d, deg: pysgpp.Grid.createModBsplineGrid(d, deg),
}


def _make_functor(kind, alpha, batch):
    if kind == "surplus":
        return pysgpp.SurplusRefinementFunctor(alpha, batch)
    if kind == "volume":
        f = getattr(pysgpp, "SurplusVolumeRefinementFunctor", None)
        if f is None:
            raise RuntimeError("this pysgpp build has no SurplusVolumeRefinementFunctor")
        return f(alpha, batch)
    raise ValueError("refine must be 'surplus' or 'volume'")


# ===========================================================================
# Arm
# ===========================================================================
class SGppArm(Arm):
    """Surplus-refined sparse-grid interpolant on [0,1]^dim.

    basis        : 'modlinear' (default; no boundary points, the right choice
                   when the box corners are not cheap) | 'linear' |
                   'linearboundary' | 'modbspline'
    degree       : B-spline degree, ignored for the linear bases
    refine       : 'surplus' (magnitude) | 'volume' (magnitude x support volume)
    init_level   : regular starting level
    refine_batch : points added per refinement step
    nan_policy   : what to do with a failed (NaN) evaluation
                     'fill'  - substitute the current interpolant's own value,
                               giving that node zero surplus. The node stops
                               attracting refinement and the hole is papered
                               over: honest about the fact that a sparse grid
                               cannot simply skip a prescribed node.
                     'zero'  - substitute 0.0. Worst case; kept for contrast.
                     'error' - raise. Use to confirm a clean run really is clean.
    """

    def __init__(self, basis="modlinear", degree=3, refine="surplus",
                 init_level=2, refine_batch=8, target_surplus=0.0,
                 nan_policy="fill", name=None):
        if not HAVE_PYSGPP:
            raise ImportError(_MISSING)
        self.basis = basis
        self.degree = degree
        self.refine = refine
        self.init_level = init_level
        self.refine_batch = refine_batch
        self.target_surplus = target_surplus
        self.nan_policy = nan_policy
        self.name = name or f"sgpp-{basis}-{refine}"
        self.history = []
        self.grid = None
        self.alpha = None
        self.n_failed = 0

    # -- internals --------------------------------------------------------
    def _new_grid(self, dim):
        try:
            factory = _GRID_FACTORY[self.basis]
        except KeyError:
            raise ValueError(f"basis must be one of {sorted(_GRID_FACTORY)}")
        return factory(dim, self.degree)

    def _hierarchise(self, fx):
        alpha = pysgpp.DataVector(list(fx))
        pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(alpha)
        return alpha

    def _repair(self, y, coords, alpha, grid=None):
        """Replace NaN evaluations per `nan_policy`. Returns (values, n_bad)."""
        bad = ~np.isfinite(y)
        n_bad = int(bad.sum())
        if not n_bad:
            return y, 0
        if self.nan_policy == "error":
            raise RuntimeError(f"{n_bad} failed evaluations and nan_policy='error'")
        if self.nan_policy == "zero":
            y = np.where(bad, 0.0, y)
        elif self.nan_policy == "fill":
            fill = (self._eval(coords[bad], alpha, grid=grid) if alpha is not None
                    else np.zeros(n_bad))
            y = y.copy()
            y[bad] = fill
        else:
            raise ValueError("nan_policy must be 'fill', 'zero' or 'error'")
        return y, n_bad

    def _eval(self, X, alpha=None, grid=None):
        X = np.atleast_2d(np.asarray(X, float))
        if len(X) == 0:
            return np.empty(0)
        alpha = self.alpha if alpha is None else alpha
        # The SWIG operation borrows DataMatrix storage. Keep it alive until
        # mult() finishes; an inline temporary can cause a native segfault.
        data = _to_matrix(X)
        op = pysgpp.createOperationMultipleEval(self.grid if grid is None else grid, data)
        res = pysgpp.DataVector(len(X))
        op.mult(alpha, res)
        return _from_vector(res, len(X))

    # -- Arm interface ----------------------------------------------------
    def fit(self, oracle, dim, budget, **kw):
        self.grid = self._new_grid(dim)
        gs = self.grid.getStorage()
        self.history = []
        self.n_failed = 0

        # start at the largest regular level that fits inside the budget
        level = self.init_level
        while level > 1:
            probe = self._new_grid(dim)
            probe.getGenerator().regular(level)
            if probe.getStorage().getSize() <= budget:
                break
            level -= 1
        self.grid.getGenerator().regular(level)

        coords = grid_coords(gs)
        if len(coords) > budget:
            raise ValueError("budget cannot fund the initial SG++ grid")
        y, nb = self._repair(np.asarray(oracle(coords), float), coords, None)
        self.n_failed += nb
        fx = list(y)
        self.alpha = self._hierarchise(fx)

        n_evals = len(fx)
        newest_max = max(abs(self.alpha[i]) for i in range(gs.getSize()))
        self.history.append(dict(n_evals=n_evals, n_basis=gs.getSize(),
                                 newest_surplus=float(newest_max), n_failed=self.n_failed))

        while n_evals < budget and newest_max > self.target_surplus:
            n0 = gs.getSize()
            previous_grid = self.grid.clone()
            # never overshoot the budget on the last step
            batch = int(min(self.refine_batch, max(1, budget - n_evals)))
            self.grid.getGenerator().refine(_make_functor(self.refine, self.alpha, batch))
            n1 = gs.getSize()
            if n1 == n0:
                break                                   # nothing left to refine

            new_coords = grid_coords(gs, start=n0)
            if n_evals + len(new_coords) > budget:
                # refine() can add more than `batch` points (parent completion);
                # stop rather than silently spend past the cap.
                self.grid = previous_grid
                gs = self.grid.getStorage()
                break

            yv, nb = self._repair(np.asarray(oracle(new_coords), float),
                                  new_coords, self.alpha, grid=previous_grid)
            self.n_failed += nb
            fx.extend(yv.tolist())
            n_evals += len(yv)
            self.alpha = self._hierarchise(fx)

            newest_max = max(abs(self.alpha[i]) for i in range(n0, n1))
            self.history.append(dict(n_evals=n_evals, n_basis=n1,
                                     newest_surplus=float(newest_max),
                                     n_failed=self.n_failed))
        return self

    def predict(self, X):
        if self.alpha is None:
            raise RuntimeError("call fit() first")
        return self._eval(X)

    def summary(self):
        return dict(name=self.name, basis=self.basis, refine=self.refine,
                    nan_policy=self.nan_policy,
                    n_basis=int(self.grid.getStorage().getSize()) if self.grid else 0,
                    n_repaired=int(self.n_failed),
                    n_steps=len(self.history))


class SGppRegularArm(SGppArm):
    """Non-adaptive regular sparse grid at the largest level fitting the budget.

    The baseline that says how much of an adaptive arm's score comes from
    adaptivity rather than from sparse grids as such.
    """

    def __init__(self, **kw):
        kw.setdefault("name", "sgpp-regular")
        super().__init__(**kw)

    def fit(self, oracle, dim, budget, **kw):
        level = 1
        while True:
            probe = self._new_grid(dim)
            probe.getGenerator().regular(level + 1)
            if probe.getStorage().getSize() > budget:
                break
            level += 1
        self.grid = self._new_grid(dim)
        self.grid.getGenerator().regular(level)
        coords = grid_coords(self.grid.getStorage())
        if len(coords) > budget:
            raise ValueError("budget cannot fund the initial SG++ grid")
        y, nb = self._repair(np.asarray(oracle(coords), float), coords, None)
        self.n_failed = nb
        self.alpha = self._hierarchise(list(y))
        self.history = [dict(n_evals=len(y), n_basis=len(y), level=level,
                             newest_surplus=float("nan"), n_failed=nb)]
        return self
