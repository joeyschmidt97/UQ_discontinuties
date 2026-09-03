"""
GPR arm -- Gaussian-process regression with a batch active-learning loop.

Ionut's recommendation for the boundary half of the survey. Structurally the
opposite contract to both sparse grids:

  * points are FREE. Nothing is prescribed, so a failed run is a dropped sample
    rather than a hole in an interpolant -- the property most likely to decide
    this in a semi-autonomous run.
  * refinement is not axis-aligned. Acquisition can follow an oblique mode
    boundary directly.
  * a nugget (WhiteKernel) absorbs convergence scatter instead of interpolating
    it. Both sparse-grid arms must fit noise exactly.
  * cost: O(n^3) refits, anisotropic length scales to tune, and degradation past
    ~10 input dimensions.

Three acquisition strategies, because "how it picks next points" is the whole
question and the right answer depends on the survey objective:

  'var'   pure predictive standard deviation. Space-filling early, then drawn to
          wherever the response is hardest to predict -- which for a kinked
          target is the transition itself.
  'ucb'   mu + kappa*sigma. Targets MAX GROWTH RATE: spends budget where the
          instability is strongest, per "target max growth rate instabilities".
  'grad'  sigma * |grad mu|. Boundary seeking: the largest predictive
          uncertainty ON A STEEP SLOPE, i.e. the mode-transition and
          mixed-mode manifold rather than a smooth peak.

Batches are chosen greedily with a distance penalty, otherwise every point in a
batch lands on the same argmax and the batch is wasted -- the standard failure
of naive batch active learning.
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import Arm

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    HAVE_SKLEARN = True
except ImportError:                                        # pragma: no cover
    HAVE_SKLEARN = False

try:
    from scipy.stats import qmc
    HAVE_QMC = True
except ImportError:                                        # pragma: no cover
    HAVE_QMC = False


def _lhs(n, dim, seed):
    if HAVE_QMC:
        return qmc.LatinHypercube(d=dim, seed=seed).random(n)
    rng = np.random.default_rng(seed)                       # stratified fallback
    return (rng.permuted(np.tile(np.arange(n), (dim, 1)), axis=1).T
            + rng.random((n, dim))) / n


class GPRArm(Arm):
    """GP regression + batch active learning on [0,1]^dim.

    acquisition : 'var' | 'ucb' | 'grad'
    n_init      : size of the initial Latin-hypercube design
    batch       : points acquired per refit
    pool        : candidate pool size scored at each step
    kappa       : UCB exploitation weight (acquisition='ucb' only)
    nu          : Matern smoothness. 1.5 is the right default here -- the
                  standard 2.5 assumes a smoother field than a kinked target has,
                  and the RBF limit assumes analyticity outright.
    nugget      : initial WhiteKernel level; tuned by marginal likelihood
    """

    def __init__(self, acquisition="var", n_init=None, batch=8, pool=6000,
                 kappa=1.0, nu=1.5, nugget=1e-4, normalize_y=True,
                 seed=0, name=None):
        if not HAVE_SKLEARN:
            raise ImportError("scikit-learn is required for the GPR arm")
        self.acquisition = acquisition
        self.n_init = n_init
        self.batch = batch
        self.pool = pool
        self.kappa = kappa
        self.nu = nu
        self.nugget = nugget
        self.normalize_y = normalize_y
        self.seed = seed
        self.name = name or f"gpr-{acquisition}"
        self.history = []

    # -- internals --------------------------------------------------------
    def _kernel(self, dim):
        return (ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.full(dim, 0.3),
                         # wide bounds on purpose: with the default (1e-2, 1e2)
                         # the fitted scales pin to both rails, which silently
                         # means "this axis is irrelevant" / "this axis varies
                         # faster than I can resolve" without saying so
                         length_scale_bounds=(1e-3, 1e4), nu=self.nu)
                + WhiteKernel(self.nugget, (1e-10, 1e1)))

    def _refit(self, dim):
        self.gp = GaussianProcessRegressor(
            kernel=self._kernel(dim), normalize_y=self.normalize_y,
            n_restarts_optimizer=1, random_state=self.seed)
        with warnings.catch_warnings():
            # length scales pinning to a bound is information, not a fault; it
            # is reported through length_scales() instead of as console spam
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(self.X, self.y)

    def _grad_norm(self, C, h=1e-3):
        """|grad mu| by central differences on the posterior mean. Cheap enough:
        2*dim extra predict() calls on the pool, no kernel refit."""
        dim = C.shape[1]
        g = np.zeros(len(C))
        for j in range(dim):
            Cp, Cm = C.copy(), C.copy()
            Cp[:, j] = np.clip(Cp[:, j] + h, 0, 1)
            Cm[:, j] = np.clip(Cm[:, j] - h, 0, 1)
            step = (Cp[:, j] - Cm[:, j])
            step[step == 0] = h
            g += ((self.gp.predict(Cp) - self.gp.predict(Cm)) / step) ** 2
        return np.sqrt(g)

    def _acquire(self, dim, rng, k):
        """Score a fresh candidate pool and greedily take k points with a
        distance penalty, so a batch spreads instead of collapsing onto one
        argmax."""
        C = rng.random((self.pool, dim))
        mu, sd = self.gp.predict(C, return_std=True)
        if self.acquisition == "var":
            score = sd
        elif self.acquisition == "ucb":
            score = mu + self.kappa * sd
        elif self.acquisition == "grad":
            score = sd * self._grad_norm(C)
        else:
            raise ValueError("acquisition must be 'var', 'ucb' or 'grad'")

        # exclusion radius ~ mean nearest-neighbour spacing of the design so far
        r = 0.5 / max(len(self.X), 1) ** (1.0 / dim)
        picked = []
        s = score.astype(float).copy()
        for _ in range(k):
            i = int(np.argmax(s))
            if not np.isfinite(s[i]):
                break
            picked.append(C[i])
            s[np.linalg.norm(C - C[i], axis=1) < r] = -np.inf
        return np.array(picked) if picked else C[:k]

    # -- Arm interface ----------------------------------------------------
    def fit(self, oracle, dim, budget, **kw):
        rng = np.random.default_rng(self.seed)
        n_init = self.n_init or max(2 * dim, min(40, budget // 4))
        n_init = int(min(n_init, budget))

        X = _lhs(n_init, dim, self.seed)
        y = np.asarray(oracle(X), float)
        ok = np.isfinite(y)
        self.n_failed = int((~ok).sum())          # failures are simply dropped
        self.X, self.y = X[ok], y[ok]
        n_evals = len(X)
        self.history = [dict(n_evals=n_evals, n_basis=len(self.X), step=0,
                             n_failed=self.n_failed)]
        if len(self.X) < 2:
            raise RuntimeError("initial design produced fewer than 2 usable points")
        self._refit(dim)

        step = 0
        while n_evals < budget:
            step += 1
            k = int(min(self.batch, budget - n_evals))
            Xn = self._acquire(dim, rng, k)
            yn = np.asarray(oracle(Xn), float)
            n_evals += len(Xn)
            ok = np.isfinite(yn)
            self.n_failed += int((~ok).sum())
            if np.any(ok):
                self.X = np.vstack([self.X, Xn[ok]])
                self.y = np.concatenate([self.y, yn[ok]])
                self._refit(dim)
            self.history.append(dict(n_evals=n_evals, n_basis=len(self.X),
                                     step=step, n_failed=self.n_failed))
        return self

    def predict(self, X):
        return self.gp.predict(np.atleast_2d(np.asarray(X, float)))

    def predict_std(self, X):
        return self.gp.predict(np.atleast_2d(np.asarray(X, float)), return_std=True)[1]

    def length_scales(self):
        """Fitted anisotropic length scales -- the GP's own answer to 'which
        axes matter', and the closest thing it has to a Sobol ranking."""
        for p, v in self.gp.kernel_.get_params().items():
            if p.endswith("length_scale") and np.ndim(v) >= 1:
                return np.asarray(v, float).ravel()
        return None

    def summary(self):
        s = dict(name=self.name, acquisition=self.acquisition,
                 n_kept=int(len(self.X)), n_dropped=int(self.n_failed),
                 n_steps=len(self.history), kernel=str(self.gp.kernel_)[:160])
        ls = self.length_scales()
        if ls is not None:
            s["length_scales"] = np.round(ls, 4).tolist()
        return s
