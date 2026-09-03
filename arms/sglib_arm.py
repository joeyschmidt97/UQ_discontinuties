"""
sg_lib arm -- sensitivity-driven dimension-adaptive sparse grid (Farcas lineage).

Wraps the reference implementation at
`C:\\Users\\joesc\\git\\sensitivity-driven-sparse-grid-approx` (Farcas et al.,
J. Comput. Phys. 410 (2020) 109394, arXiv:1812.00080). Not vendored: the clone
carries no license, so this arm imports it from an external path exactly the way
`TPED`'s `sparse_scan_driver.py` does.

How it picks next points -- and why that differs in kind from SG++:

  * refinement is over MULTIINDICES (subspaces), not individual points. Each
    adaption step scores every admissible candidate subspace by the variance its
    spectral coefficients carry, then admits the winners. So the budget flows to
    whole DIRECTIONS, not to locations.
  * that makes the refinement AXIS-ALIGNED. A mode boundary running oblique
    through the box cannot be chased directly; the method can only buy more
    resolution in each contributing direction, which is the structural reason the
    method's own author expects it to struggle here.
  * a subspace is atomic: you cannot evaluate half of one and still have an
    interpolant. Budget granularity is therefore a subspace, not a point, and
    this arm stops BEFORE admitting a subspace it cannot pay for in full. The
    reported `n_evals` will usually undershoot the budget -- that is a real
    property of the method, not a bookkeeping slip.
  * global orthogonal-polynomial basis, so a jump discontinuity is felt
    everywhere (ringing), not just locally.

Free output nothing else gives: Sobol indices, hence the Stage-A axis ranking.
"""

from __future__ import annotations

import copy
import os
import pathlib
import sys

import numpy as np

from .base import Arm

DEFAULT_SG_LIB_PATH = os.environ.get(
    "SG_LIB_PATH", r"C:\Users\joesc\git\sensitivity-driven-sparse-grid-approx")

_IMPORTED = {}


def _import_sg_lib(path=None):
    """Import sg_lib from an external checkout. Cached per path."""
    path = str(pathlib.Path(path or DEFAULT_SG_LIB_PATH))
    if path in _IMPORTED:
        return _IMPORTED[path]
    if not pathlib.Path(path, "sg_lib").is_dir():
        raise ImportError(
            f"sg_lib not found at {path!r}. Clone "
            "https://github.com/ionutfarcas/sensitivity-driven-sparse-grid-approx "
            "or set SG_LIB_PATH.")
    if path not in sys.path:
        sys.path.insert(0, path)
    from sg_lib.adaptivity.spectral_scores import SpectralScores
    from sg_lib.algebraic.multiindex import Multiindex
    from sg_lib.grid.grid import Grid
    from sg_lib.operation.interpolation_to_spectral import InterpolationToSpectral
    mod = dict(Grid=Grid, Multiindex=Multiindex,
               InterpolationToSpectral=InterpolationToSpectral,
               SpectralScores=SpectralScores)
    _IMPORTED[path] = mod
    return mod


try:
    _import_sg_lib()
    HAVE_SG_LIB = True
except ImportError:                                        # pragma: no cover
    HAVE_SG_LIB = False


class SgLibArm(Arm):
    """Sensitivity-driven dimension-adaptive sparse grid.

    tol         : per-direction termination tolerance
    max_level   : maximum level any direction may reach
    grid_level  : starting level (sg_lib always starts at 1)
    nan_policy  : 'fill' substitutes the current interpolant's value for a failed
                  node (a prescribed node cannot be skipped); 'zero' substitutes
                  0.0; 'error' raises. Mirrors `sparse_scan_driver.tell()`,
                  which rejects non-finite values outright.
    """

    def __init__(self, tol=1e-6, max_level=20, grid_level=1, max_steps=4000,
                 nan_policy="fill", sg_lib_path=None, name="sglib"):
        if not HAVE_SG_LIB:
            raise ImportError("sg_lib not importable; see DEFAULT_SG_LIB_PATH")
        self.tol = tol
        self.max_level = max_level
        self.grid_level = grid_level
        self.max_steps = max_steps
        self.nan_policy = nan_policy
        self.sg_lib_path = sg_lib_path
        self.name = name
        self.history = []
        self.stopped_on = None

    # -- internals --------------------------------------------------------
    def _repair(self, pts, vals):
        bad = ~np.isfinite(vals)
        n_bad = int(bad.sum())
        if not n_bad:
            return vals, 0
        if self.nan_policy == "error":
            raise RuntimeError(f"{n_bad} failed evaluations and nan_policy='error'")
        vals = np.asarray(vals, float).copy()
        if self.nan_policy == "zero":
            vals[bad] = 0.0
        elif self.nan_policy == "fill":
            for i in np.flatnonzero(bad):
                try:
                    vals[i] = self._eval_one(pts[i])
                except Exception:
                    vals[i] = 0.0
        else:
            raise ValueError("nan_policy must be 'fill', 'zero' or 'error'")
        return vals, n_bad

    # State that `do_one_adaption_step_preproc()` mutates. Same attribute list
    # sg_lib's own serialize_data() persists.
    _ADAPT_STATE = ("_key_O", "_O", "_key_A", "_A", "_key_local_error",
                    "_local_error", "_multiindex_set", "_local_basis_local",
                    "_local_basis_global")

    def _snapshot(self):
        return {k: copy.deepcopy(getattr(self.A, k)) for k in self._ADAPT_STATE}

    def _restore(self, snap):
        for k, v in snap.items():
            setattr(self.A, k, v)

    def _eval_one(self, x):
        return float(self.I.eval_operation_sg(self.A.multiindex_set,
                                              np.asarray(x, float)))

    def _feed(self, oracle, points, multiindex=None):
        """Evaluate a block of prescribed nodes and push them into sg_lib."""
        pts = np.atleast_2d(np.asarray(points, float))
        vals, n_bad = self._repair(pts, np.asarray(oracle(pts), float))
        self.n_failed += n_bad
        for p, v in zip(pts, vals):
            self.I.update_sg_evals_all_lut(p, float(v))
        if multiindex is not None:
            self.I.update_sg_evals_multiindex_lut(multiindex, self.G)
        return len(pts)

    # -- Arm interface ----------------------------------------------------
    def fit(self, oracle, dim, budget, **kw):
        m = _import_sg_lib(self.sg_lib_path)
        lo, hi = np.zeros(dim), np.ones(dim)
        weights = [lambda x: 1.0 for _ in range(dim)]

        self.G = m["Grid"](dim, self.grid_level, 1, lo, hi, weights)
        self.M = m["Multiindex"](dim)
        self.I = m["InterpolationToSpectral"](dim, 1, lo, hi, weights,
                                              self.max_level, self.G)
        init_multiindex = np.ones(dim, dtype=int)
        self.A = m["SpectralScores"](dim, self.tol * np.ones(dim + 1),
                                     init_multiindex, self.max_level, 1, self.I)

        self.history = []
        self.n_failed = 0
        self.stopped_on = None
        n_evals = 0

        init_set = self.M.get_std_total_degree_mindex(self.grid_level)
        init_pts = self.G.get_std_sg_surplus_points(init_set)
        self.I.get_local_global_basis(self.A)
        n_evals += self._feed(oracle, init_pts, init_multiindex)
        self.A.init_adaption()
        self.history.append(dict(n_evals=n_evals, n_basis=n_evals, step=0,
                                 n_new_subspaces=len(init_set), n_failed=0))

        self.n_empty_steps = 0
        for step in range(1, self.max_steps + 1):
            snap = self._snapshot()
            new_multiindices = self.A.do_one_adaption_step_preproc()

            # An EMPTY step is normal, not termination. preproc() moves the
            # highest-scoring index from the active set to the old set and
            # returns only its newly ADMISSIBLE successors -- which is often
            # none, because a successor still needs its other predecessors
            # admitted first. Treating empty as "done" (the obvious reading of
            # the reference script's fixed-step loop) stops the method after ~3
            # steps and ~10 points, under-spending the budget by ~20x.
            if not len(new_multiindices):
                self.n_empty_steps += 1
                self.A.check_termination_criterion()
                if self.A.stop_adaption or not len(self.A.A):
                    self.stopped_on = "exhausted"
                    break
                continue

            # price the whole step before spending any of it: a subspace is
            # atomic, so a partially-paid step would leave a broken interpolant
            blocks = [(mi, self.G.get_sg_surplus_points_multiindex(mi))
                      for mi in new_multiindices]
            cost = sum(len(p) for _, p in blocks)
            if n_evals + cost > budget:
                # preproc() has ALREADY appended these multiindices to
                # multiindex_set and extended the local basis. Breaking here
                # without rolling back leaves subspaces in the set whose nodes
                # were never evaluated, and eval_operation_sg() then raises a
                # KeyError naming the unpaid multiindex. Roll back to the
                # snapshot taken before preproc so the interpolant only ever
                # spans subspaces that were paid for in full.
                self._restore(snap)
                self.stopped_on = "budget"
                break

            for mi, pts in blocks:
                n_evals += self._feed(oracle, pts, mi)
            self.A.do_one_adaption_step_postproc(new_multiindices)
            self.A.check_termination_criterion()

            self.history.append(dict(n_evals=n_evals, n_basis=n_evals, step=step,
                                     n_new_subspaces=len(new_multiindices),
                                     n_empty_steps=self.n_empty_steps,
                                     n_failed=self.n_failed))
            if self.A.stop_adaption:
                self.stopped_on = "tolerance"
                break
        else:
            self.stopped_on = "max_steps"

        self.I.get_local_global_basis(self.A)
        self.n_evals = n_evals
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        ms = self.A.multiindex_set
        return np.array([float(self.I.eval_operation_sg(ms, x)) for x in X])

    # -- the free readout -------------------------------------------------
    def sobol(self):
        """Total-effect Sobol indices per axis, or None if unavailable.

        This is what the method gives away that nothing else does, and it is the
        Stage-A axis-pruning readout in the survey plan.
        """
        for getter in ("get_total_sobol_indices", "get_first_order_sobol_indices"):
            fn = getattr(self.I, getter, None)
            if fn is None:
                continue
            for args in ((self.A.multiindex_set,), ()):
                try:
                    return np.asarray(fn(*args), float).ravel()
                except Exception:
                    continue
        return None

    def axis_levels(self):
        """Max level reached per direction -- how the budget was split across
        axes. The dimension-adaptive analogue of 'where did the points go'."""
        ms = np.atleast_2d(np.asarray(self.A.multiindex_set, int))
        return ms.max(axis=0)

    def summary(self):
        s = dict(name=self.name, stopped_on=self.stopped_on,
                 n_empty_steps=int(getattr(self, "n_empty_steps", 0)),
                 n_subspaces=int(len(np.atleast_2d(self.A.multiindex_set))),
                 n_steps=len(self.history), n_repaired=int(self.n_failed))
        s["axis_levels"] = self.axis_levels().tolist()
        sob = self.sobol()
        if sob is not None:
            s["sobol_total"] = np.round(sob, 5).tolist()
        return s
