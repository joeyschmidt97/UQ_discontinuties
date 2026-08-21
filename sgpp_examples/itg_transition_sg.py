"""
Reduced ITG/TEM/KBM dispersion emulator  +  spatially-adaptive sparse-grid
surrogate (pysgpp), with a REALISTIC, SMOOTH mode combination.

Two ways to combine the competing branches into an observable (gamma, omega_r):

  * 'argmax'  : dominant = fastest-growing branch. Physically the zero-coupling
                limit. Gives a C0 KINK in gamma and a JUMP in omega_r at the
                crossover -> hard for any interpolant (surpluses stall on the
                crossover manifold; the omega_r jump never resolves).

  * 'softmax' : finite-coupling / hybridization blend (DEFAULT). Weights
                w_i = softmax(gamma_i / T); observables are w-weighted averages.
                For two modes this is EXACTLY the tanh blend
                    w_B = 0.5*(1 + tanh((gamma_B - gamma_A)/(2 T))),
                and T -> 0 recovers argmax. Both gamma and omega_r are C-infinity,
                so the surrogate converges far faster.

The branch-level phenomenology (critical-gradient onsets ~ sqrt(drive-drive_c),
collisional TEM stabilization, finite-beta ITG stabilization + KBM threshold,
ion/electron diamagnetic omega_r signs, FLR ky-envelope) is unchanged from the
uploaded emulator. Only the mode-combination step is made smooth/realistic.
This is NOT a gyrokinetic eigenvalue solve.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysgpp import (Grid, DataVector, createOperationHierarchisation,
                    createOperationEvalNaive, SurplusRefinementFunctor)


# ===========================================================================
# 1. PHYSICS BRANCHES 
# ===========================================================================
BASELINE = dict(
    RLTi=6.0, RLTe=5.0, RLn=2.0,
    beta=0.0, nu=0.0,
    eps=0.18, tau=1.0,
    RLTi_crit=4.0, TEM_crit=3.0, beta_crit=0.9, beta_ITG_stab=1.3,
    a_Te=1.0, a_n=0.6, nu_c=0.30,
    C_ITG=0.60, C_TEM=1.20, C_KBM=0.50,
    C_wi=0.25, C_we=0.30, C_wp=0.20,
)


def _shape(ky, ky0):
    return ky * np.exp(-(ky / ky0) ** 2)


def branch_ITG(p, ky):
    drive = np.maximum(0.0, p["RLTi"] - p["RLTi_crit"])
    beta_stab = np.maximum(0.0, 1.0 - p["beta"] / p["beta_ITG_stab"])
    gamma = p["C_ITG"] * np.sqrt(drive) * _shape(ky, 0.8) * beta_stab / np.sqrt(1.0 + p["tau"])
    omega = -p["C_wi"] * ky * (1.0 + 0.5 * p["RLTi"])
    return gamma, omega


def branch_TEM(p, ky):
    drive = np.maximum(0.0, p["a_Te"] * p["RLTe"] + p["a_n"] * p["RLn"] - p["TEM_crit"])
    coll = 1.0 / (1.0 + (p["nu"] / p["nu_c"]) ** 2)
    trapped = np.sqrt(np.maximum(p["eps"], 1e-3))
    gamma = p["C_TEM"] * np.sqrt(drive) * coll * trapped * _shape(ky, 1.1)
    omega = +p["C_we"] * ky * (1.0 + 0.5 * p["RLTe"])
    return gamma, omega


def branch_KBM(p, ky):
    drive = np.maximum(0.0, p["beta"] - p["beta_crit"])
    RLp = 0.3 * (p["RLTi"] + p["RLTe"] + p["RLn"])
    gamma = p["C_KBM"] * np.sqrt(drive) * (1.0 + RLp) * _shape(ky, 0.6)
    omega = -p["C_wp"] * ky * RLp
    return gamma, omega


_BRANCHES = {"ITG": branch_ITG, "TEM": branch_TEM, "KBM": branch_KBM}


# ===========================================================================
# 2. MODE COMBINATION  (argmax  vs  smooth softmax/tanh hybridization)
# ===========================================================================
def combine_modes(params, ky, branches=("ITG", "TEM"), mode="softmax", T=0.05):
    """Combine branch (gamma, omega_r) into a single observable.

    mode='softmax' (default): realistic, C-infinity hybridization blend.
    mode='argmax'           : hard fastest-growing selection (kink + jump).
    T : hybridization width [c_s/a]; smaller = sharper transition, T->0 => argmax.
    Returns dict with 'gamma', 'omega', 'share' (weight of the LAST branch),
    plus stacked per-branch 'G','W' and 'names'.
    """
    p = {**BASELINE, **params}
    shp = np.broadcast_shapes(np.shape(ky), *(np.shape(v) for v in p.values()))

    G, W = [], []
    for name in branches:
        g, w = _BRANCHES[name](p, ky)
        G.append(np.broadcast_to(g, shp).astype(float))
        W.append(np.broadcast_to(w, shp).astype(float))
    G, W = np.stack(G), np.stack(W)                       # (n_branch, ...)

    if mode == "argmax":
        k = np.argmax(G, axis=0)
        gamma = np.take_along_axis(G, k[None], 0)[0]
        omega = np.take_along_axis(W, k[None], 0)[0]
        share = (k == (len(branches) - 1)).astype(float)
    elif mode == "softmax":
        z = (G - G.max(axis=0, keepdims=True)) / T        # stable softmax
        w = np.exp(z)
        w /= w.sum(axis=0, keepdims=True)                 # responsibilities
        gamma = (w * G).sum(0)
        omega = (w * W).sum(0)
        share = w[-1]
    else:
        raise ValueError("mode must be 'argmax' or 'softmax'")

    return dict(gamma=gamma, omega=omega, share=share, G=G, W=W, names=branches)


# Parameter-space test function for the surrogate: X in [0,1]^6 -> gamma.
_COLS = ("RLTi", "RLTe", "RLn", "nu", "beta", "ky_scale")
_RANGES = dict(RLTi=(3.0, 9.0), RLTe=(2.0, 8.0), RLn=(0.5, 3.5),
               nu=(0.0, 0.8), beta=(0.0, 1.5))


def testfunc(X, kind="ITG_KBM", ky=0.30, mode="softmax", T=0.05, out="gamma"):
    """X:(N,6) in [0,1] -> observable (default dominant growth rate).
    kind selects which two branches compete."""
    X = np.atleast_2d(np.asarray(X, float))
    params = {k: lo + (hi - lo) * X[:, i]
              for i, (k, (lo, hi)) in enumerate(_RANGES.items())}
    ky_pt = ky * (0.5 + X[:, 5])
    branches = ("ITG", "TEM") if kind == "ITG_TEM" else ("ITG", "KBM")
    return combine_modes(params, ky_pt, branches, mode=mode, T=T)[out]


# ===========================================================================
# 3. PYSGPP ADAPTIVE SURROGATE  (efficient: eval f only on new points)
# ===========================================================================
def grid_coords(gs, start=0):
    d = gs.getDimension()
    idx = range(start, gs.getSize())
    P = np.empty((len(idx), d))
    for r, i in enumerate(idx):
        gp = gs.getPoint(i)
        for j in range(d):
            P[r, j] = gp.getStandardCoordinate(j)
    return P


def eval_surrogate(grid, alpha, X):
    X = np.atleast_2d(np.asarray(X, float))
    op = createOperationEvalNaive(grid)
    p = DataVector(grid.getStorage().getDimension())
    y = np.empty(len(X))
    for i, row in enumerate(X):
        for j in range(len(row)):
            p[j] = row[j]
        y[i] = op.eval(alpha, p)
    return y


def build_adaptive_surrogate(f, dim, init_level=2, target_surplus=3e-4,
                             refine_batch=8, max_points=1400,
                             test_X=None, test_y=None, verbose=False):
    grid = Grid.createModLinearGrid(dim)
    gs = grid.getStorage()
    grid.getGenerator().regular(init_level)

    fx = list(f(grid_coords(gs)))
    alpha = DataVector(fx)
    createOperationHierarchisation(grid).doHierarchisation(alpha)

    def _rmse():
        if test_X is None:
            return np.nan
        return float(np.sqrt(np.mean((eval_surrogate(grid, alpha, test_X) - test_y) ** 2)))

    newest_max = max(abs(alpha[i]) for i in range(gs.getSize()))
    history = [(gs.getSize(), newest_max, _rmse())]

    while newest_max > target_surplus and gs.getSize() < max_points:
        n0 = gs.getSize()
        grid.getGenerator().refine(SurplusRefinementFunctor(alpha, refine_batch))
        n1 = gs.getSize()
        if n1 == n0:
            break
        fx.extend(f(grid_coords(gs, start=n0)).tolist())
        alpha = DataVector(fx)
        createOperationHierarchisation(grid).doHierarchisation(alpha)
        newest_max = max(abs(alpha[i]) for i in range(n0, n1))
        history.append((n1, newest_max, _rmse()))

    return grid, alpha, history


def build_regular_surrogate(f, dim, level):
    grid = Grid.createModLinearGrid(dim)
    grid.getGenerator().regular(level)
    fx = list(f(grid_coords(grid.getStorage())))
    alpha = DataVector(fx)
    createOperationHierarchisation(grid).doHierarchisation(alpha)
    return grid, alpha, grid.getStorage().getSize()


# ===========================================================================
# 4. FIGURES
# ===========================================================================
def fig_sweeps():
    """argmax vs smooth blend, 1D sweeps for both transitions (gamma & omega)."""
    KY, T = 0.30, 0.05
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    # -- ITG<->TEM sweeping collisionality --
    nu = np.linspace(0.0, 0.6, 500)
    hard = combine_modes({"nu": nu}, KY, ("ITG", "TEM"), mode="argmax")
    soft = combine_modes({"nu": nu}, KY, ("ITG", "TEM"), mode="softmax", T=T)
    ax[0, 0].plot(nu, hard["G"][0], ":", color="C3", lw=1.2, label=r"$\gamma_{\rm ITG}$")
    ax[0, 0].plot(nu, hard["G"][1], ":", color="C0", lw=1.2, label=r"$\gamma_{\rm TEM}$")
    ax[0, 0].plot(nu, hard["gamma"], "-", color="0.6", lw=1.6, label="argmax (kink)")
    ax[0, 0].plot(nu, soft["gamma"], "-", color="k", lw=2.4, label="smooth (tanh)")
    ax[0, 0].set_ylabel(r"$\gamma$ [$c_s/a$]"); ax[0, 0].set_title(r"ITG$\leftrightarrow$TEM vs $\nu$")
    ax[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    ax[0, 1].axhline(0, color="0.7", lw=0.8)
    ax[0, 1].plot(nu, hard["omega"], "-", color="0.6", lw=1.6, label="argmax (jump)")
    ax[0, 1].plot(nu, soft["omega"], "-", color="k", lw=2.4, label="smooth")
    ax[0, 1].set_ylabel(r"$\omega_r$ [$c_s/a$]"); ax[0, 1].set_title(r"$\omega_r$ sign flip")
    ax[0, 1].legend(frameon=False, fontsize=8)

    # -- ITG<->KBM sweeping beta --
    beta = np.linspace(0.0, 1.5, 500)
    hard = combine_modes({"beta": beta}, KY, ("ITG", "KBM"), mode="argmax")
    soft = combine_modes({"beta": beta}, KY, ("ITG", "KBM"), mode="softmax", T=T)
    ax[1, 0].plot(beta, hard["G"][0], ":", color="C3", lw=1.2, label=r"$\gamma_{\rm ITG}$")
    ax[1, 0].plot(beta, hard["G"][1], ":", color="C2", lw=1.2, label=r"$\gamma_{\rm KBM}$")
    ax[1, 0].plot(beta, hard["gamma"], "-", color="0.6", lw=1.6, label="argmax (kink)")
    ax[1, 0].plot(beta, soft["gamma"], "-", color="k", lw=2.4, label="smooth (tanh)")
    ax[1, 0].set_xlabel(r"$\beta$ [%]"); ax[1, 0].set_ylabel(r"$\gamma$ [$c_s/a$]")
    ax[1, 0].set_title(r"ITG$\leftrightarrow$KBM vs $\beta$ (valley)")
    ax[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    ax[1, 1].axhline(0, color="0.7", lw=0.8)
    ax[1, 1].plot(beta, hard["omega"], "-", color="0.6", lw=1.6, label="argmax")
    ax[1, 1].plot(beta, soft["omega"], "-", color="k", lw=2.4, label="smooth")
    ax[1, 1].set_xlabel(r"$\beta$ [%]"); ax[1, 1].set_ylabel(r"$\omega_r$ [$c_s/a$]")
    ax[1, 1].set_title(r"$\omega_r$ (both ion dir.)"); ax[1, 1].legend(frameon=False, fontsize=8)

    for a in ax.ravel():
        a.spines[["right", "top"]].set_visible(False)
    fig.tight_layout(); fig.savefig("emu_sweeps_argmax_vs_smooth.png", dpi=140); plt.close(fig)


def fig_surrogate():
    """Build SG surrogates for the smooth target; compare to argmax target & regular grids."""
    dim = 6
    rng = np.random.default_rng(1)
    test_X = rng.random((3000, dim))
    kind = "ITG_KBM"

    f_soft = lambda X: testfunc(X, kind=kind, mode="softmax", T=0.05)
    f_hard = lambda X: testfunc(X, kind=kind, mode="argmax")
    y_soft = f_soft(test_X)
    y_hard = f_hard(test_X)

    g_s, a_s, h_s = build_adaptive_surrogate(f_soft, dim, test_X=test_X, test_y=y_soft)
    g_h, a_h, h_h = build_adaptive_surrogate(f_hard, dim, test_X=test_X, test_y=y_hard)

    yp_s = eval_surrogate(g_s, a_s, test_X)
    rmse_s = np.sqrt(np.mean((yp_s - y_soft) ** 2))
    rmse_h_pred = eval_surrogate(g_h, a_h, test_X)
    rmse_h = np.sqrt(np.mean((rmse_h_pred - y_hard) ** 2))
    n_s, n_h = g_s.getStorage().getSize(), g_h.getStorage().getSize()
    print(f"[ITG_KBM] smooth : n={n_s}  RMSE={rmse_s:.3e}")
    print(f"[ITG_KBM] argmax : n={n_h}  RMSE={rmse_h:.3e}")

    reg = []
    for lvl in range(2, 6):
        g, a, ng = build_regular_surrogate(f_soft, dim, lvl)
        yr = eval_surrogate(g, a, test_X)
        reg.append((ng, np.sqrt(np.mean((yr - y_soft) ** 2))))
    reg = np.array(reg)

    # ---- convergence ----
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    hs = np.array([(n, r) for n, _, r in h_s if np.isfinite(r)])
    hh = np.array([(n, r) for n, _, r in h_h if np.isfinite(r)])
    ax.loglog(hs[:, 0], hs[:, 1], "-", color="#1b9e77", lw=2.4, label="adaptive, smooth target")
    ax.loglog(hh[:, 0], hh[:, 1], "-", color="#d95f02", lw=2.0, label="adaptive, argmax target")
    ax.loglog(reg[:, 0], reg[:, 1], "s--", color="0.4", lw=1.4, ms=6, label="regular, smooth target")
    ax.set_xlabel("number of grid points"); ax.set_ylabel("test RMSE")
    ax.set_title(r"ITG$\leftrightarrow$KBM surrogate: smooth blend is far cheaper")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.spines[["right", "top"]].set_visible(False); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig("emu_convergence.png", dpi=140); plt.close(fig)

    # ---- parity (smooth) ----
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lo, hi = y_soft.min(), y_soft.max()
    ax.plot([lo, hi], [lo, hi], "k", lw=1)
    ax.scatter(y_soft, yp_s, s=8, alpha=0.45, color="#1b9e77", label=f"smooth SG (n={n_s})")
    ax.set_xlabel(r"reference $\gamma$"); ax.set_ylabel(r"surrogate $\gamma$")
    ax.set_title(f"Parity (smooth target): RMSE={rmse_s:.2e}")
    ax.spines[["right", "top"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("emu_parity.png", dpi=140); plt.close(fig)

    # ---- 2D slice (RLTi vs beta) + grid clustering ----
    gx, gy = np.meshgrid(np.linspace(0, 1, 220), np.linspace(0, 1, 220))
    mid = 0.5 * np.ones_like(gx.ravel())
    Xsl = np.column_stack([gx.ravel(), mid, mid, mid, gy.ravel(), mid])
    gam = f_soft(Xsl).reshape(gx.shape)
    P = grid_coords(g_s.getStorage())

    fig, ax = plt.subplots(figsize=(6.2, 5))
    pc = ax.pcolormesh(3 + 6 * gx, 1.5 * gy, gam, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax, label=r"$\gamma$ (smooth), slice")
    ax.scatter(3 + 6 * P[:, 0], 1.5 * P[:, 4], s=5, color="red", alpha=0.35,
               label="grid pts (projected)")
    ax.axhline(0.9, color="w", ls="--", lw=1.3, label=r"$\beta_{\rm crit}$ (KBM onset)")
    ax.axvline(4.0, color="w", ls=":", lw=1.3, label=r"$R/L_{Ti,\rm crit}$ (ITG onset)")
    ax.set_xlabel(r"$R/L_{Ti}$ (col 0)"); ax.set_ylabel(r"$\beta$ [%] (col 4)")
    ax.set_title("Adaptive points cluster on onsets + transition")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig("emu_clustering.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    fig_sweeps()
    fig_surrogate()
    print("saved: emu_sweeps_argmax_vs_smooth.png, emu_convergence.png, "
          "emu_parity.png, emu_clustering.png")
