"""Data-only functions extracted from Ionut Farcas upstream/main 13f87b9.
Phenomenological proxies, not gyrokinetic solves. Soft selection does not
remove the square-root branch-onset singularities. See data/README.md.
"""

import numpy as np

BASELINE = dict(RLTi=6.0, RLTe=5.0, RLn=2.0, beta=0.0, nu=0.0, eps=0.18, tau=1.0, RLTi_crit=4.0, TEM_crit=3.0, beta_crit=0.9, beta_ITG_stab=1.3, a_Te=1.0, a_n=0.6, nu_c=0.3, C_ITG=0.6, C_TEM=1.2, C_KBM=0.5, C_wi=0.25, C_we=0.3, C_wp=0.2)

def _shape(ky, ky0):
    return ky * np.exp(-(ky / ky0) ** 2)

def branch_ITG(p, ky):
    drive = np.maximum(0.0, p['RLTi'] - p['RLTi_crit'])
    beta_stab = np.maximum(0.0, 1.0 - p['beta'] / p['beta_ITG_stab'])
    gamma = p['C_ITG'] * np.sqrt(drive) * _shape(ky, 0.8) * beta_stab / np.sqrt(1.0 + p['tau'])
    omega = -p['C_wi'] * ky * (1.0 + 0.5 * p['RLTi'])
    return (gamma, omega)

def branch_TEM(p, ky):
    drive = np.maximum(0.0, p['a_Te'] * p['RLTe'] + p['a_n'] * p['RLn'] - p['TEM_crit'])
    coll = 1.0 / (1.0 + (p['nu'] / p['nu_c']) ** 2)
    trapped = np.sqrt(np.maximum(p['eps'], 0.001))
    gamma = p['C_TEM'] * np.sqrt(drive) * coll * trapped * _shape(ky, 1.1)
    omega = +p['C_we'] * ky * (1.0 + 0.5 * p['RLTe'])
    return (gamma, omega)

def branch_KBM(p, ky):
    drive = np.maximum(0.0, p['beta'] - p['beta_crit'])
    RLp = 0.3 * (p['RLTi'] + p['RLTe'] + p['RLn'])
    gamma = p['C_KBM'] * np.sqrt(drive) * (1.0 + RLp) * _shape(ky, 0.6)
    omega = -p['C_wp'] * ky * RLp
    return (gamma, omega)

_BRANCHES = {'ITG': branch_ITG, 'TEM': branch_TEM, 'KBM': branch_KBM}

def combine_modes(params, ky, branches=('ITG', 'TEM'), mode='softmax', T=0.05):
    """Combine phenomenological branches; softmax smooths selection, not branch onsets."""
    p = {**BASELINE, **params}
    shp = np.broadcast_shapes(np.shape(ky), *(np.shape(v) for v in p.values()))
    G, W = ([], [])
    for name in branches:
        g, w = _BRANCHES[name](p, ky)
        G.append(np.broadcast_to(g, shp).astype(float))
        W.append(np.broadcast_to(w, shp).astype(float))
    G, W = (np.stack(G), np.stack(W))
    if mode == 'argmax':
        k = np.argmax(G, axis=0)
        gamma = np.take_along_axis(G, k[None], 0)[0]
        omega = np.take_along_axis(W, k[None], 0)[0]
        share = (k == len(branches) - 1).astype(float)
    elif mode == 'softmax':
        z = (G - G.max(axis=0, keepdims=True)) / T
        w = np.exp(z)
        w /= w.sum(axis=0, keepdims=True)
        gamma = (w * G).sum(0)
        omega = (w * W).sum(0)
        share = w[-1]
    else:
        raise ValueError("mode must be 'argmax' or 'softmax'")
    return dict(gamma=gamma, omega=omega, share=share, G=G, W=W, names=branches)

_COLS = ('RLTi', 'RLTe', 'RLn', 'nu', 'beta', 'ky_scale')

_RANGES = dict(RLTi=(3.0, 9.0), RLTe=(2.0, 8.0), RLn=(0.5, 3.5), nu=(0.0, 0.8), beta=(0.0, 1.5))

def testfunc(X, kind='ITG_KBM', ky=0.3, mode='softmax', T=0.05, out='gamma'):
    """X:(N,6) in [0,1] -> observable (default dominant growth rate).
    kind selects which two branches compete."""
    X = np.atleast_2d(np.asarray(X, float))
    params = {k: lo + (hi - lo) * X[:, i] for i, (k, (lo, hi)) in enumerate(_RANGES.items())}
    ky_pt = ky * (0.5 + X[:, 5])
    branches = ('ITG', 'TEM') if kind == 'ITG_TEM' else ('ITG', 'KBM')
    return combine_modes(params, ky_pt, branches, mode=mode, T=T)[out]

def test_model(X, beta_c=0.32, delta_beta=0.04, eps=0.025, alpha_n=0.5, alpha_tau=0.3, grad_shift=0.05):
    """Stellarator-style ITG/KBM growth-rate model on [0,1]^6.
    Columns: [nref, Tref, aLTi, aLTe, aLn, tau]."""
    X = np.atleast_2d(np.asarray(X, float))
    nref, Tref, aLTi, aLTe, aLn, tau = (X[:, k] for k in range(6))
    beta_eff = nref * Tref
    beta_c_eff = beta_c - grad_shift * (aLn + aLTi)
    S = 0.5 * (1.0 + np.tanh((beta_eff - beta_c_eff) / (2.0 * delta_beta)))

    def soft_clip(x):
        return eps * np.log1p(np.exp(x / eps)) - eps * np.log(2.0)
    gamma_ITG = soft_clip(aLTi - alpha_n * aLn - alpha_tau * (tau - 0.5))
    gamma_KBM = soft_clip(beta_eff - beta_c_eff) * (aLn + aLTi + aLTe)
    return (1.0 - S) * gamma_ITG + S * gamma_KBM
