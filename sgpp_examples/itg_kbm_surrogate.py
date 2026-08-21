"""
Spatially-adaptive sparse-grid surrogate for the stellarator ITG/KBM transition.

Connects a physics growth-rate model  gamma = f(X)  to an SG++ (pysgpp)
surplus-refined ModLinear sparse grid, and reports the diagnostics that matter
for a *localized-feature* target: parity, convergence vs. a regular grid, and
the clustering of grid points on the transition front.

Key changes vs. a naive pipeline:
  * function is evaluated ONLY on newly created points each refinement
    (nodal values are cached) -- this is the whole point of a surrogate, since
    the real f may be an expensive GK run;
  * convergence is judged on the surplus of the *newest* points, so refining an
    already-refined parent can't stall the loop;
  * a hard max_points cap guards against a too-small target_surplus;
  * ModLinear + createOperationEvalNaive for robust evaluation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt

from pysgpp import (Grid, DataVector, createOperationHierarchisation,
                    createOperationEvalNaive, SurplusRefinementFunctor)


# ---------------------------------------------------------------------------
# 1. PHYSICS MODEL 
# ---------------------------------------------------------------------------
def test_model(X, beta_c=0.32, delta_beta=0.04, eps=0.025,
               alpha_n=0.5, alpha_tau=0.3, grad_shift=0.05):
    """Stellarator-style ITG/KBM growth-rate model on [0,1]^6.
    Columns: [nref, Tref, aLTi, aLTe, aLn, tau]."""
    X = np.atleast_2d(np.asarray(X, float))
    nref, Tref, aLTi, aLTe, aLn, tau = (X[:, k] for k in range(6))

    beta_eff   = nref * Tref                               # effective beta
    beta_c_eff = beta_c - grad_shift * (aLn + aLTi)        # gradient-shifted KBM threshold

    # smooth ITG -> KBM regime weight (the sharp-but-C-infinity front)
    S = 0.5 * (1.0 + np.tanh((beta_eff - beta_c_eff) / (2.0 * delta_beta)))

    def soft_clip(x):                                      # smooth ReLU, allows small gamma<0
        return eps * np.log1p(np.exp(x / eps)) - eps * np.log(2.0)

    gamma_ITG = soft_clip(aLTi - alpha_n * aLn - alpha_tau * (tau - 0.5))
    gamma_KBM = soft_clip(beta_eff - beta_c_eff) * (aLn + aLTi + aLTe)

    return (1.0 - S) * gamma_ITG + S * gamma_KBM


# ---------------------------------------------------------------------------
# 2. SPARSE-GRID HELPERS
# ---------------------------------------------------------------------------
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


def eval_surrogate(grid, alpha, X):
    """Evaluate the interpolant at rows of X."""
    X = np.atleast_2d(np.asarray(X, float))
    op = createOperationEvalNaive(grid)
    p = DataVector(grid.getStorage().getDimension())
    y = np.empty(len(X))
    for i, row in enumerate(X):
        for j in range(len(row)):
            p[j] = row[j]
        y[i] = op.eval(alpha, p)
    return y


# ---------------------------------------------------------------------------
# 3. ADAPTIVE SURROGATE BUILDER
# ---------------------------------------------------------------------------
def build_adaptive_surrogate(f, dim, init_level=2, target_surplus=5e-4,
                             refine_batch=5, max_points=2500,
                             test_X=None, test_y=None, verbose=True):
    """Build a surplus-refined ModLinear surrogate of f on [0,1]^dim.

    Returns (grid, alpha, history) where history is a list of
    (n_points, newest_max_surplus, test_rmse) tuples logged per refinement.
    """
    grid = Grid.createModLinearGrid(dim)
    gs = grid.getStorage()
    grid.getGenerator().regular(init_level)

    # nodal-value cache: evaluate f on ALL current points once
    fx = list(f(grid_coords(gs)))
    alpha = DataVector(fx)
    createOperationHierarchisation(grid).doHierarchisation(alpha)

    def _rmse():
        if test_X is None:
            return np.nan
        return float(np.sqrt(np.mean((eval_surrogate(grid, alpha, test_X) - test_y) ** 2)))

    # initial "newest" surplus = max over all points (to enter the loop)
    newest_max = max(abs(alpha[i]) for i in range(gs.getSize()))
    history = [(gs.getSize(), newest_max, _rmse())]

    while newest_max > target_surplus and gs.getSize() < max_points:
        n_before = gs.getSize()
        grid.getGenerator().refine(SurplusRefinementFunctor(alpha, refine_batch))
        n_after = gs.getSize()
        if n_after == n_before:                      # nothing left to refine
            break

        # evaluate f ONLY on the newly created points, extend the cache
        new_coords = grid_coords(gs, start=n_before)
        fx.extend(f(new_coords).tolist())

        alpha = DataVector(fx)
        createOperationHierarchisation(grid).doHierarchisation(alpha)

        # convergence judged on the newest points only
        newest_max = max(abs(alpha[i]) for i in range(n_before, n_after))
        history.append((n_after, newest_max, _rmse()))
        if verbose and len(history) % 10 == 0:
            print(f"  n={n_after:5d}  newest|surplus|={newest_max:.2e}  "
                  f"test_RMSE={history[-1][2]:.2e}")

    if verbose:
        print(f"final grid size: {gs.getSize()}  (newest|surplus|={newest_max:.2e})")
    return grid, alpha, history


def build_regular_surrogate(f, dim, level):
    """Non-adaptive regular ModLinear grid at a given level (for comparison)."""
    grid = Grid.createModLinearGrid(dim)
    grid.getGenerator().regular(level)
    fx = list(f(grid_coords(grid.getStorage())))
    alpha = DataVector(fx)
    createOperationHierarchisation(grid).doHierarchisation(alpha)
    return grid, alpha, grid.getStorage().getSize()


# ---------------------------------------------------------------------------
# 4. DEMO / DIAGNOSTICS
# ---------------------------------------------------------------------------
def main():
    dim = 6
    rng = np.random.default_rng(0)
    n_test = 3000
    test_X = rng.random((n_test, dim))
    test_y = test_model(test_X)

    print("Building adaptive surrogate ...")
    grid, alpha, hist = build_adaptive_surrogate(
        test_model, dim, init_level=2, target_surplus=4e-4,
        refine_batch=5, max_points=2200, test_X=test_X, test_y=test_y)

    y_pred = eval_surrogate(grid, alpha, test_X)
    rmse = np.sqrt(np.mean((y_pred - test_y) ** 2))
    maxerr = np.max(np.abs(y_pred - test_y))
    n_final = grid.getStorage().getSize()
    print(f"adaptive: n={n_final}  RMSE={rmse:.3e}  max|err|={maxerr:.3e}")

    # regular grids for comparison
    reg = []
    for lvl in range(2, 6):
        g, a, ng = build_regular_surrogate(test_model, dim, lvl)
        yr = eval_surrogate(g, a, test_X)
        reg.append((ng, np.sqrt(np.mean((yr - test_y) ** 2))))
        print(f"regular L{lvl}: n={ng:5d}  RMSE={reg[-1][1]:.3e}")

    # ---- styling (usetex off for portability; re-enable locally) ----
    charcoal = [0, 0, 0]
    c_ref, c_sg = "#1b9e77", "#d95f02"
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})

    # (A) parity plot
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lo, hi = test_y.min(), test_y.max()
    ax.plot([lo, hi], [lo, hi], color=charcoal, lw=1, zorder=1)
    ax.scatter(test_y, y_pred, s=8, alpha=0.5, color=c_sg, zorder=2,
               label=f"adaptive SG (n={n_final})")
    ax.set_xlabel(r"reference $\gamma$")
    ax.set_ylabel(r"surrogate $\gamma$")
    ax.set_title(f"Parity: RMSE={rmse:.2e}, max err={maxerr:.2e}")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout(); fig.savefig("sg_parity.png"); plt.close(fig)

    # (B) convergence: adaptive path vs regular grids
    h = np.array([(n, r) for n, _, r in hist if np.isfinite(r)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(h[:, 0], h[:, 1], "-", color=c_sg, lw=2, label="spatially adaptive")
    rn = np.array(reg)
    ax.loglog(rn[:, 0], rn[:, 1], "s--", color=c_ref, lw=1.5, ms=6, label="regular (level 2–5)")
    ax.set_xlabel("number of grid points")
    ax.set_ylabel("test RMSE")
    ax.set_title("Adaptive vs. regular sparse grid on the ITG/KBM front")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("sg_convergence.png"); plt.close(fig)

    # (C) grid clustering on the transition front, over a gamma slice
    #     slice: vary (nref, Tref); fix aLTi=aLTe=aLn=tau=0.5
    gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    mid = 0.5 * np.ones_like(gx.ravel())
    Xsl = np.column_stack([gx.ravel(), gy.ravel(), mid, mid, mid, mid])
    gam = test_model(Xsl).reshape(gx.shape)

    P = grid_coords(grid.getStorage())
    beta_c_eff_mid = 0.32 - 0.05 * (0.5 + 0.5)            # at aLn=aLTi=0.5
    tref_front = beta_c_eff_mid / np.clip(np.linspace(0.02, 1, 200), 1e-3, None)

    fig, ax = plt.subplots(figsize=(6, 5))
    pc = ax.pcolormesh(gx, gy, gam, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax, label=r"$\gamma$ (slice at $a_{LTi}=a_{LTe}=a_{Ln}=\tau=0.5$)")
    ax.plot(np.linspace(0.02, 1, 200), tref_front, "w--", lw=1.5,
            label=r"$n\,T=\beta_c^{\rm eff}$ (front)")
    ax.scatter(P[:, 0], P[:, 1], s=5, color="red", alpha=0.35, label="grid pts (all, projected)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$n_{\rm ref}$ (col 0)"); ax.set_ylabel(r"$T_{\rm ref}$ (col 1)")
    ax.set_title("Adaptive points cluster along the ITG/KBM front")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout(); fig.savefig("sg_clustering.png"); plt.close(fig)

    print("saved: sg_parity.png, sg_convergence.png, sg_clustering.png")


if __name__ == "__main__":
    main()
