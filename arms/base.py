"""
Common interface and scoring for benchmark arms.

Every contender -- SG++ (spatially adaptive), sg_lib (sensitivity/dimension
adaptive), GPR (free placement) -- implements the same two methods:

    arm.fit(oracle, dim, budget)   # oracle is a CountingOracle; budget in EVALS
    arm.predict(X)                 # X on the unit cube -> yhat

Budget is metered in function evaluations rather than basis functions or grid
points, because that is the only quantity the three methods share: an SG++ grid
point, an sg_lib Leja node and a GPR acquisition are all exactly one GENE run.

Scoring is always against the manifold's CLEAN truth, even when the arm was
trained on noisy, partly-failed observations. That separation is the whole point
of the noise/failure knobs: an arm that fits noise gets punished here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


class Arm:
    """Contender interface. Subclasses set `name` and implement fit/predict."""

    name = "arm"

    def fit(self, oracle, dim, budget, **kw):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    #: list of dicts logged per refinement step, each with at least "n_evals"
    history: list = []

    def summary(self):
        return {"name": self.name}


# ===========================================================================
# Design geometry -- "how do they fill the space?"
# ===========================================================================
def design_metrics(design, manifold, ref=None, mix_thresh=0.30, top_q=0.90):
    """Geometry of WHERE a method spent its budget, independent of accuracy.

    An arm can be accurate on average and still be useless for a sensitivity
    analysis if its points never reach the competing-mode band -- and it can be
    perfectly space-filling and still miss it. These separate the two.

      min_dist    smallest pairwise gap. Small => the design clumped (the
                  classic naive-batch-active-learning failure).
      hole        largest distance from a reference point to its nearest design
                  point: the biggest unsampled void. Small => good coverage.
      coverage    std/mean of nearest-design-point distance over the reference
                  sample (Gunzburger's measure). 0 => perfectly regular.
      frac_band   fraction of the design inside the competing-mode band.
                  Compare to the band's own volume fraction: above it means the
                  method is genuinely TARGETING mode competition, at it means
                  the method is merely space-filling.
      band_lift   frac_band divided by that volume fraction. 1.0 = no targeting.
      frac_peak   fraction of the design in the top decile of |QoI|, i.e. is it
                  finding the max-growth-rate region.
      axis_spread per-axis standard deviation of the design, normalized by the
                  0.2887 of a uniform design. Below 1 on an axis means the
                  method concentrated there; a dimension-adaptive method should
                  show a clearly uneven profile.
    """
    D = np.atleast_2d(np.asarray(design, float))
    out = dict(n_design=int(len(D)))
    if len(D) < 2:
        return out

    if ref is None:
        ref = np.random.default_rng(999).random((4000, D.shape[1]))
    d2 = ((ref[:, None, :] - D[None, :, :]) ** 2).sum(-1)
    nn = np.sqrt(d2.min(axis=1))

    pw = ((D[:, None, :] - D[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(pw, np.inf)

    ev = manifold.evaluate(D)
    ev_ref = manifold.evaluate(ref)
    band = float(np.mean(ev["mix"] >= mix_thresh))
    ref_band = float(np.mean(ev_ref["mix"] >= mix_thresh))
    q = np.quantile(np.abs(ev_ref[manifold.out]), top_q)

    out.update(
        min_dist=float(np.sqrt(pw.min())),
        hole=float(nn.max()),
        coverage=float(nn.std() / max(nn.mean(), 1e-12)),
        frac_band=band,
        band_lift=float(band / ref_band) if ref_band > 0 else float("nan"),
        frac_peak=float(np.mean(np.abs(ev[manifold.out]) >= q)),
        axis_spread=[float(v / 0.288675) for v in D.std(axis=0)],
    )
    return out


# ===========================================================================
# Scoring
# ===========================================================================
@dataclass
class Score:
    arm: str
    out: str
    n_evals: int
    n_failed: int
    rmse: float
    nrmse: float
    mae: float
    max_err: float
    rmse_mixed: float
    rmse_calm: float
    frac_design_mixed: float
    frac_test_mixed: float
    fit_seconds: float
    design: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        d = dict(self.__dict__)
        d["extra"] = dict(self.extra)
        d["design"] = dict(self.design)
        return d

    def __str__(self):
        return (f"{self.arm:>12s} out={self.out:<6s} n={self.n_evals:5d} "
                f"fail={self.n_failed:3d} nRMSE={self.nrmse:.3e} "
                f"mixed={self.rmse_mixed:.3e} calm={self.rmse_calm:.3e} "
                f"band_lift={self.design.get('band_lift', float('nan')):.2f} "
                f"hole={self.design.get('hole', float('nan')):.3f} "
                f"[{self.fit_seconds:.1f}s]")


def score_arm(arm, manifold, out=None, budget=400, test=None, mix_thresh=0.30,
              **fit_kw):
    """Fit one arm on one manifold under one budget and score it.

    Reported errors:
      rmse / nrmse     -- overall; nrmse normalized by the clean range of the QoI
                          so gamma (~0.3) and omega (~6) are comparable
      rmse_mixed       -- restricted to the competing-mode band. This is the
                          number the whole exercise is about: a method can look
                          fine on the bulk and be useless exactly where the
                          sensitivity analysis needs to land.
      rmse_calm        -- the complement, for contrast
      frac_design_mixed-- what fraction of the BUDGET the sampler spent inside
                          the mixed band. Measures targeting, not accuracy.
    """
    out = out or manifold.out
    if test is None:
        test = manifold.test_set()
    U_test, ev = test
    y_true = ev[out]
    mixed = ev["mix"] >= mix_thresh

    oracle = manifold.oracle(out)
    t0 = time.perf_counter()
    arm.fit(oracle, manifold.dim, budget, **fit_kw)
    dt = time.perf_counter() - t0

    yhat = np.asarray(arm.predict(U_test), float)
    err = yhat - y_true
    rng = float(np.ptp(y_true)) or 1.0

    design = oracle.design
    frac_design_mixed = (float(np.mean(manifold.mixed_mask(design, mix_thresh)))
                         if len(design) else float("nan"))
    dm = design_metrics(design, manifold, ref=U_test, mix_thresh=mix_thresh)

    def _rmse(mask):
        return float(np.sqrt(np.mean(err[mask] ** 2))) if np.any(mask) else float("nan")

    return Score(
        arm=arm.name, out=out,
        n_evals=oracle.n_evals, n_failed=oracle.n_failed,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        nrmse=float(np.sqrt(np.mean(err ** 2))) / rng,
        mae=float(np.mean(np.abs(err))),
        max_err=float(np.max(np.abs(err))),
        rmse_mixed=_rmse(mixed), rmse_calm=_rmse(~mixed),
        frac_design_mixed=frac_design_mixed,
        frac_test_mixed=float(np.mean(mixed)),
        fit_seconds=dt,
        design=dm,
        extra=arm.summary(),
    )


# ===========================================================================
# Reference arm -- numpy only, so the harness is testable without pysgpp
# ===========================================================================
class RandomNearestArm(Arm):
    """Space-filling random design + nearest-neighbour lookup.

    Deliberately the dumbest thing that satisfies the interface. It exists as a
    floor: any contender that cannot beat random-plus-nearest-neighbour at equal
    budget is not a contender. Also lets the scoring path be exercised on a
    machine without pysgpp installed.
    """

    name = "random-nn"

    def __init__(self, seed=0):
        self.seed = seed
        self.history = []

    def fit(self, oracle, dim, budget, **kw):
        X = np.random.default_rng(self.seed).random((budget, dim))
        y = np.asarray(oracle(X), float)
        ok = np.isfinite(y)
        self.X, self.y = X[ok], y[ok]
        self.history = [{"n_evals": budget, "n_basis": int(ok.sum())}]
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        idx = np.argmin(((X[:, None, :] - self.X[None, :, :]) ** 2).sum(-1), axis=1)
        return self.y[idx]

    def summary(self):
        return {"name": self.name, "n_kept": int(len(self.y))}
