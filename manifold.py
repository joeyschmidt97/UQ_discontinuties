"""
Test manifold for the sparse-grid / GPR discontinuity benchmark.

A free, cheap oracle standing in for a linear GENE k_y scan over the NSTX
pedestal electron-channel scan axes. Its job is NOT to be a gyrokinetic solve;
it is to reproduce, controllably, the four properties that decide whether a
sensitivity-driven sparse grid (sg_lib), a spatially-adaptive sparse grid
(SG++) or a GP can drive a semi-autonomous scan:

  1. mode competition          -> kink in gamma, jump in omega_r at crossings
  2. tunable sharpness         -> hybridization width T; T -> 0 is a true argmax
  3. band-gap discontinuity    -> rational-surface alignment gate for MTM
  4. imperfect evaluations     -> convergence noise + failed (NaN) runs

Axes are the frozen first-campaign electron-channel knobs (pedestal-top values
and widths as +-30% scale factors), plus T_i/T_e and a k_y selector. Derived
physical quantities (normalized gradients, beta_e, collisionality) are computed
from them, so the axes are coupled the way the real scan's are.

Design contracts the benchmark depends on:

  * `evaluate()` is the CLEAN truth; `observe()` adds noise and failures.
    Error metrics always score a noisily-trained surrogate against clean truth.
  * Noise and failure are a FROZEN FIELD, not a stream: both are a deterministic
    hash of the point coordinates. Two different samplers that happen to visit
    the same point see the same value, so arms are compared on identical data.
  * `CountingOracle` meters budget in FUNCTION EVALUATIONS, the only currency
    that is fair across a point-adaptive grid (SG++), a dimension-adaptive grid
    (sg_lib) and a free-placement acquisition loop (GPR).

Not a gyrokinetic eigenvalue solve. Coefficients are plausible, not calibrated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Axis", "CAMPAIGN_AXES", "NOMINAL", "Manifold", "CountingOracle",
    "BRANCH_NAMES", "derived_params", "mode_competition", "MIX_T_REF",
]


# ===========================================================================
# 1. AXES  --  first-campaign electron channel (+-30%), plus tau and k_y
# ===========================================================================
@dataclass(frozen=True)
class Axis:
    """One scan axis. `lo`/`hi` are in the axis's own units; the manifold is
    always called on the unit cube and maps internally."""
    name: str
    lo: float
    hi: float
    note: str = ""

    def to_phys(self, u):
        return self.lo + (self.hi - self.lo) * np.asarray(u, float)


# +-30% scale factors on the mtanh step amplitude / width, per the frozen
# campaign scope (roadmap: "electron channel only, at +-30% max").
CAMPAIGN_AXES = (
    Axis("Te_ped_scale", 0.70, 1.30, "pedestal-top T_e scale factor"),
    Axis("ne_ped_scale", 0.70, 1.30, "pedestal-top n_e scale factor"),
    Axis("w_Te_scale",   0.70, 1.30, "T_e pedestal width scale factor"),
    Axis("w_ne_scale",   0.70, 1.30, "n_e pedestal width scale factor"),
    Axis("Ti_Te",        0.50, 2.00, "tau = T_i/T_e (out of campaign 1, in full spec)"),
    Axis("ky_scale",     0.00, 1.00, "selects k_y rho_s in [KY_LO, KY_HI]"),
)

# NSTX-ish pre-ELM pedestal anchor. Absolute values only set the scale of the
# derived quantities; the thresholds below are tuned so the +-30% box straddles
# every onset, which is the point of a benchmark manifold.
NOMINAL = dict(
    Te_ped_keV=0.35,      # pedestal-top T_e
    ne_ped_e19=4.0,       # pedestal-top n_e [1e19 m^-3]
    Te_sep_keV=0.030,     # separatrix T_e, held fixed -> couples to Te_ped
    ne_sep_e19=0.60,      # separatrix n_e, held fixed
    w_Te_m=0.045,         # T_e pedestal width [m]
    w_ne_m=0.055,         # n_e pedestal width [m]
    a_m=0.60,             # minor radius
    R_m=0.85,             # major radius
    B_T=0.45,             # toroidal field
    q_ped=4.5,            # safety factor at the pedestal
    shear=3.0,            # magnetic shear in the pedestal
    n_tor=8.0,            # toroidal mode number the alignment gate is judged at
)

KY_LO, KY_HI = 0.05, 1.20

# Reference hybridization width used ONLY to define the competing-mode band.
# The band has to be a property of the manifold's GEOMETRY -- where branches are
# within a few percent of each other -- not of the combination rule in force:
# under argmax the weights are one-hot, so a weight-derived indicator is
# identically zero exactly on the hard cases, and "error inside the mixed
# region" becomes unmeasurable there. Fixing the reference also makes the band
# identical across cases, so the cases are comparable to each other.
MIX_T_REF = 0.10

BRANCH_NAMES = ("ITG", "TEM", "KBM", "MTM")


# ===========================================================================
# 2. DERIVED PHYSICS  --  axes -> dimensionless drives
# ===========================================================================
def derived_params(U, axes=CAMPAIGN_AXES, nominal=None):
    """Unit-cube rows -> dict of dimensionless drives, all shape (N,).

    Normalized gradients use the mtanh peak-gradient estimate
        a/L_X ~= (a / w_X) * (1 - X_sep / X_ped),
    so raising the pedestal top steepens the gradient (separatrix pinned) and
    widening the pedestal flattens it. beta_e and collisionality follow from
    the same two pedestal-top values, which is what makes the axes coupled.
    """
    nom = dict(NOMINAL if nominal is None else nominal)
    U = np.atleast_2d(np.asarray(U, float))
    if U.shape[1] != len(axes):
        raise ValueError(f"expected {len(axes)} columns, got {U.shape[1]}")
    phys = {ax.name: ax.to_phys(U[:, i]) for i, ax in enumerate(axes)}

    Te = nom["Te_ped_keV"] * phys["Te_ped_scale"]
    ne = nom["ne_ped_e19"] * phys["ne_ped_scale"]
    w_Te = nom["w_Te_m"] * phys["w_Te_scale"]
    w_ne = nom["w_ne_m"] * phys["w_ne_scale"]
    tau = phys["Ti_Te"]
    a, R, B = nom["a_m"], nom["R_m"], nom["B_T"]

    aLTe = (a / w_Te) * (1.0 - nom["Te_sep_keV"] / Te)
    aLne = (a / w_ne) * (1.0 - nom["ne_sep_e19"] / ne)
    # ion channel rides the electron temperature profile shape, scaled by tau
    aLTi = aLTe * (0.85 + 0.15 * tau)

    # beta_e [%] = 0.403 * n[1e19] * T[keV] / B[T]^2
    beta_e = 0.403 * ne * Te / B ** 2
    # collisionality proxy, normalized to 1 at the nominal point: nu* ~ n / T^2
    nu = 0.30 * (ne / nom["ne_ped_e19"]) / (Te / nom["Te_ped_keV"]) ** 2

    ky = KY_LO + (KY_HI - KY_LO) * phys["ky_scale"]

    return dict(
        Te=Te, ne=ne, w_Te=w_Te, w_ne=w_ne, tau=tau,
        aLTe=aLTe, aLTi=aLTi, aLne=aLne,
        beta_e=beta_e, nu=nu, ky=ky,
        eps=a / R, q=nom["q_ped"], shear=nom["shear"], n_tor=nom["n_tor"],
    )


def _shape(ky, ky0):
    """FLR envelope: rises linearly, cut off above ky0."""
    return ky * np.exp(-(ky / ky0) ** 2)


def _softplus(x, eps=0.02):
    """Smooth ReLU, strictly positive. Keeps sqrt() arguments differentiable at
    onset, so any residual non-smoothness in the target is the mode competition
    and the alignment gate -- not the threshold algebra. No -eps*log(2) shift:
    a negative drive would make sqrt(drive) NaN just below threshold."""
    return eps * np.log1p(np.exp(np.clip(x / eps, -60.0, 60.0)))


# --- coefficients --------------------------------------------------------
# Tuned so the +-30% campaign box STRADDLES every onset and no single branch
# owns the box -- the point of a competition benchmark. Plausible, not calibrated.
# Amplitudes fixed by a grid search for near-uniform dominant-branch occupancy
# over the box (ITG 0.28 / TEM 0.31 / KBM 0.33 / MTM 0.09). MTM's smaller share
# is structural, not a mistune: it is a low-k_y mode, so it can only win in the
# low-k_y end of the k_y axis.
COEFF = dict(
    aLTi_crit=12.0, C_ITG=0.75, C_wi=0.22, beta_ITG_stab=5.0,
    TEM_crit=18.0, a_Te=1.0, a_n=0.55, nu_c=0.35, C_TEM=0.45, C_we=0.28,
    beta_crit=4.40, grad_shift=0.090, C_KBM=0.24, C_wp=0.20, C_pKBM=0.08,
    aLTe_crit=12.0, C_MTM=3.60, nu_m=0.45, beta_MTM=0.9, C_wm=0.55,
    mu_w=0.32, gamma_ref=0.020,
)


def branch_ITG(d, c=COEFF):
    drive = _softplus(d["aLTi"] - c["aLTi_crit"] * (1.0 + 0.20 * (d["tau"] - 1.0)))
    beta_stab = _softplus(1.0 - d["beta_e"] / c["beta_ITG_stab"])
    gamma = (c["C_ITG"] * np.sqrt(drive) * _shape(d["ky"], 0.80)
             * beta_stab / np.sqrt(1.0 + d["tau"]))
    omega = -c["C_wi"] * d["ky"] * (1.0 + 0.5 * d["aLTi"])     # ion direction
    return gamma, omega


def branch_TEM(d, c=COEFF):
    drive = _softplus(c["a_Te"] * d["aLTe"] + c["a_n"] * d["aLne"] - c["TEM_crit"])
    coll = 1.0 / (1.0 + (d["nu"] / c["nu_c"]) ** 2)            # collisions stabilize
    trapped = np.sqrt(np.maximum(d["eps"], 1e-3))
    gamma = c["C_TEM"] * np.sqrt(drive) * coll * trapped * _shape(d["ky"], 1.10)
    omega = +c["C_we"] * d["ky"] * (1.0 + 0.5 * d["aLTe"])     # electron direction
    return gamma, omega


def branch_KBM(d, c=COEFF):
    aLp = (d["aLTe"] + d["aLTi"] + 2.0 * d["aLne"]) / 4.0
    beta_crit_eff = c["beta_crit"] - c["grad_shift"] * aLp     # gradient-shifted onset
    drive = _softplus(d["beta_e"] - beta_crit_eff)
    gamma = c["C_KBM"] * np.sqrt(drive) * (1.0 + c["C_pKBM"] * aLp) * _shape(d["ky"], 0.60)
    omega = -c["C_wp"] * d["ky"] * aLp                         # ion direction
    return gamma, omega


def alignment_mu(d):
    """Rational-surface alignment coordinate, in [0, 1].

    Stand-in for the SLiM-style mu feature (distance of the nearest rational
    surface to the omega_*e peak). The number of rational surfaces inside the
    pedestal goes like n_tor * q * shear * w / a, so the fractional part of that
    count sweeps through alignment as the WIDTH and PEDESTAL-TOP axes move. That
    is what makes the band structure a function of the scan axes rather than a
    decoration: mu = 0 is aligned (MTM allowed), mu = 1 is anti-aligned.
    """
    m_count = d["n_tor"] * d["q"] * d["shear"] * d["w_Te"] / NOMINAL["a_m"]
    m_count = m_count * (1.0 + 0.30 * (d["aLTe"] / 10.0))      # omega_*e peak shifts
    frac = m_count - np.floor(m_count)
    return 2.0 * np.abs(frac - 0.5)                            # distance to integer


def branch_MTM(d, align="smooth", c=COEFF):
    """Microtearing. Needs finite beta (magnetic flutter), sits on a
    collisionality resonance, electron-diamagnetic, low k_y.

    `align` controls the rational-surface gate and is the manifold's
    discontinuity knob:
      'off'    - no gate (smooth branch, control case)
      'smooth' - Gaussian gate in mu (band structure, C-infinity)
      'gap'    - hard gate (true JUMP in gamma: the band-gap case)
    """
    drive = _softplus(d["aLTe"] - c["aLTe_crit"])
    beta_gate = d["beta_e"] / (d["beta_e"] + c["beta_MTM"])    # needs finite beta
    r = d["nu"] / c["nu_m"]
    coll_res = 2.0 * r / (1.0 + r ** 2)                        # peaks at nu = nu_m
    gamma = c["C_MTM"] * np.sqrt(drive) * beta_gate * coll_res * _shape(d["ky"], 0.25)

    mu = alignment_mu(d)
    if align == "off":
        gate = 1.0
    elif align == "smooth":
        gate = np.exp(-(mu / c["mu_w"]) ** 2)
    elif align == "gap":
        gate = (mu < c["mu_w"]).astype(float)
    else:
        raise ValueError("align must be 'off', 'smooth' or 'gap'")

    omega = +c["C_wm"] * d["ky"] * (1.0 + 0.5 * d["aLTe"])     # electron direction
    return gamma * gate, omega


_BRANCHES = {"ITG": branch_ITG, "TEM": branch_TEM, "KBM": branch_KBM}


# ===========================================================================
# 3. MODE COMBINATION
# ===========================================================================
def combine(G, W, mode="softmax", T=0.05, relative=True, floor=1e-3):
    """Branch (gamma, omega) stacks -> observables.

    'argmax'  : dominant = fastest growing. C0 KINK in gamma, JUMP in omega.
    'softmax' : finite-coupling hybridization. Weights w_i = softmax(gamma_i/T).
                For two branches this is exactly the tanh blend; T -> 0 recovers
                argmax. Both observables are C-infinity, so T is a continuous
                dial from "trivially interpolable" to "worst case".

    `relative=True` (default) measures T as a FRACTION of the local peak growth
    rate rather than in absolute c_s/a. Without it, T is not a portable
    benchmark knob: gamma varies by an order of magnitude across the box (the
    k_y envelope alone does that), so one absolute T means "hard selection" at
    high-gamma corners and "everything blended" at low-gamma ones, and the
    sweep over T stops being a clean sharpness axis.
    """
    if mode == "argmax":
        k = np.argmax(G, axis=0)
        w = np.zeros_like(G)
        np.put_along_axis(w, k[None], 1.0, axis=0)
    elif mode == "softmax":
        peak = G.max(axis=0, keepdims=True)
        scale = np.maximum(peak, floor) if relative else 1.0
        z = (G - peak) / np.maximum(T * scale, 1e-12)
        w = np.exp(z)
        w /= w.sum(axis=0, keepdims=True)
    else:
        raise ValueError("mode must be 'argmax' or 'softmax'")

    gamma = (w * G).sum(0)
    omega = (w * W).sum(0)
    return gamma, omega, w


def mode_competition(G, T_ref=None, gamma_ref=None):
    """Competing-mode indicator in [0, 1], from the BRANCH growth rates alone.

    Independent of `mode`/`T`, so the same physical band is flagged whether the
    manifold is hybridizing or hard-selecting. This is the mask the
    "target mixed-mode regions" objective is scored against.
    """
    w = np.exp((G - G.max(axis=0, keepdims=True))
               / np.maximum((MIX_T_REF if T_ref is None else T_ref)
                            * np.maximum(G.max(axis=0, keepdims=True), 1e-3), 1e-12))
    w /= w.sum(axis=0, keepdims=True)
    return _margin_mix(w, peak=G.max(axis=0), gamma_ref=gamma_ref)


def _margin_mix(w, peak=None, gamma_ref=None):
    """Competing-mode indicator in [0, 1]: 1 - (top weight - runner-up weight),
    damped by how unstable the point actually is.

    1.0 means two branches are exactly tied at a strongly unstable point (peak
    hybridization); 0.0 means one branch owns the point outright, OR the point
    is quiescent. The damping matters: in the stable corner every branch sits at
    gamma ~ 0 and the weights tie trivially, which would otherwise flag the
    DULLEST region of the box as the most interesting one and send an
    acquisition loop chasing it. Damping is a smooth ramp, so it introduces no
    discontinuity of its own.

    This is the scalar the acquisition side is supposed to chase ("mixed-mode
    regions, peak and complex competing-mode areas"), so it is a first-class
    output rather than a diagnostic.
    """
    srt = np.sort(w, axis=0)
    raw = 1.0 - (srt[-1] - srt[-2])
    if peak is None:
        return raw
    g_ref = COEFF["gamma_ref"] if gamma_ref is None else gamma_ref
    return raw * (peak / (peak + g_ref))


# ===========================================================================
# 4. FROZEN NOISE / FAILURE FIELD
# ===========================================================================
def _row_seeds(U, salt, decimals=12):
    """Deterministic per-row seed from the coordinates.

    Rounded before hashing so dyadic sparse-grid coordinates and Leja nodes hash
    stably. Two arms visiting the same point get the same draw, which is what
    makes a cross-arm comparison on noisy data meaningful.
    """
    Q = np.ascontiguousarray(np.round(np.atleast_2d(U), decimals)) + 0.0  # kill -0.0
    key = salt.encode()
    out = np.empty(len(Q), dtype=np.uint64)
    for i, row in enumerate(Q):
        h = hashlib.blake2b(row.tobytes() + key, digest_size=8).digest()
        out[i] = int.from_bytes(h, "little")
    return out


def _frozen_uniform(U, salt, n=1):
    """n independent U(0,1) draws per row, frozen to the coordinates."""
    seeds = _row_seeds(U, salt)
    out = np.empty((len(seeds), n))
    for i, s in enumerate(seeds):
        out[i] = np.random.default_rng(int(s)).random(n)
    return out


# ===========================================================================
# 5. MANIFOLD
# ===========================================================================
@dataclass
class Manifold:
    """The oracle. Call on the unit cube; every knob below is a benchmark axis.

    branches   : which branches compete
    mode       : 'softmax' (hybridized) or 'argmax' (hard selection)
    T          : hybridization width; smaller = sharper. T -> 0 == argmax
    align      : MTM rational-surface gate: 'off' | 'smooth' | 'gap'
    noise_rel  : relative Gaussian noise (convergence scatter)
    noise_abs  : absolute Gaussian noise floor
    fail_rate  : fraction of evaluations returned as NaN (non-converged run)
    out        : default output of __call__ -- 'gamma' | 'omega' | 'mix' | 'share'
    """
    axes: tuple = CAMPAIGN_AXES
    branches: tuple = BRANCH_NAMES
    mode: str = "softmax"
    T: float = 0.05
    T_relative: bool = True
    align: str = "smooth"
    noise_rel: float = 0.0
    noise_abs: float = 0.0
    fail_rate: float = 0.0
    seed: int = 0
    out: str = "gamma"
    nominal: dict = field(default_factory=lambda: dict(NOMINAL))
    #: per-instance overrides layered on COEFF. Use this to widen the band gap
    #: (`mu_w`), move an onset, or kill a branch, WITHOUT mutating the module
    #: global -- two Manifolds in the same process must not affect each other.
    coeff: dict = field(default_factory=dict)

    def __post_init__(self):
        self._c = {**COEFF, **self.coeff}

    @property
    def dim(self):
        return len(self.axes)

    # -- clean truth ------------------------------------------------------
    def evaluate(self, U):
        """Clean, noise-free, never-failing. Returns every observable at once --
        one k_y-scan node yields all candidate QoIs, mirroring the real scan."""
        U = np.atleast_2d(np.asarray(U, float))
        d = derived_params(U, self.axes, self.nominal)

        G, W = [], []
        for name in self.branches:
            if name == "MTM":
                g, w = branch_MTM(d, align=self.align, c=self._c)
            else:
                g, w = _BRANCHES[name](d, c=self._c)
            G.append(np.broadcast_to(g, (U.shape[0],)).astype(float))
            W.append(np.broadcast_to(w, (U.shape[0],)).astype(float))
        G, W = np.stack(G), np.stack(W)

        gamma, omega, w = combine(G, W, mode=self.mode, T=self.T,
                                  relative=self.T_relative)
        return dict(
            gamma=gamma, omega=omega,
            weights=w, G=G, W=W,
            label=np.argmax(G, axis=0),
            mix=mode_competition(G, gamma_ref=self._c["gamma_ref"]),
            share=w[-1],
            mu=alignment_mu(d),
            names=self.branches,
        )

    def truth(self, U, out=None):
        return self.evaluate(U)[out or self.out]

    # -- what a sampler actually gets -------------------------------------
    def observe(self, U, out=None):
        """Clean truth + frozen noise, with a frozen subset returned as NaN."""
        U = np.atleast_2d(np.asarray(U, float))
        y = np.asarray(self.truth(U, out), float).copy()
        if self.noise_rel or self.noise_abs:
            z = _frozen_uniform(U, f"noise-{self.seed}", 2)
            # Box-Muller from the frozen uniforms -> frozen standard normal
            g = np.sqrt(-2.0 * np.log(np.clip(z[:, 0], 1e-12, 1.0))) * np.cos(2 * np.pi * z[:, 1])
            y = y * (1.0 + self.noise_rel * g) + self.noise_abs * g
        if self.fail_rate:
            u = _frozen_uniform(U, f"fail-{self.seed}", 1)[:, 0]
            y = np.where(u < self.fail_rate, np.nan, y)
        return y

    def __call__(self, U):
        return self.observe(U)

    # -- benchmark helpers -------------------------------------------------
    def mixed_mask(self, U, thresh=0.50):
        """Points inside the competing-mode band -- the region the sensitivity
        analysis is supposed to resolve."""
        return self.evaluate(U)["mix"] >= thresh

    def test_set(self, n=4000, seed=12345):
        """Frozen test design shared by every arm."""
        U = np.random.default_rng(seed).random((n, self.dim))
        ev = self.evaluate(U)
        return U, ev

    def oracle(self, out=None):
        return CountingOracle(self, out or self.out)

    def describe(self):
        return (f"Manifold(dim={self.dim}, branches={'+'.join(self.branches)}, "
                f"mode={self.mode}, T={self.T}, align={self.align}, "
                f"noise_rel={self.noise_rel}, fail_rate={self.fail_rate}, out={self.out})")


class CountingOracle:
    """Meters budget in function evaluations -- the only currency that compares
    a point-adaptive grid, a dimension-adaptive grid and a free-placement
    acquisition loop on equal terms. Also logs every design point, so
    'where did the budget land' is answerable after the fact."""

    def __init__(self, manifold: Manifold, out: str):
        self.m = manifold
        self.out = out
        self.n_calls = 0
        self.n_evals = 0
        self.n_failed = 0
        self._X = []

    def __call__(self, U):
        U = np.atleast_2d(np.asarray(U, float))
        y = self.m.observe(U, self.out)
        self.n_calls += 1
        self.n_evals += len(U)
        self.n_failed += int(np.count_nonzero(~np.isfinite(y)))
        self._X.append(U.copy())
        return y

    @property
    def design(self):
        return np.vstack(self._X) if self._X else np.empty((0, self.m.dim))

    def reset(self):
        self.n_calls = self.n_evals = self.n_failed = 0
        self._X = []


# ===========================================================================
# 6. SMOKE TEST  (numpy only -- runs without pysgpp)
# ===========================================================================
def _smoke():
    rng = np.random.default_rng(0)
    U = rng.random((2000, len(CAMPAIGN_AXES)))

    print("axes:", ", ".join(a.name for a in CAMPAIGN_AXES))
    d = derived_params(U)
    for k in ("aLTe", "aLTi", "aLne", "beta_e", "nu", "ky"):
        print(f"  {k:7s} range [{d[k].min():7.3f}, {d[k].max():7.3f}]")

    for mode, T, align in [("softmax", 0.20, "smooth"),
                           ("softmax", 0.02, "smooth"),
                           ("argmax", 0.0, "smooth"),
                           ("argmax", 0.0, "gap")]:
        m = Manifold(mode=mode, T=T, align=align)
        ev = m.evaluate(U)
        occ = np.bincount(ev["label"], minlength=len(BRANCH_NAMES))
        frac_mixed = float(np.mean(ev["mix"] >= 0.5))
        print(f"{m.describe()}\n   gamma[{ev['gamma'].min():.3f},{ev['gamma'].max():.3f}] "
              f"omega[{ev['omega'].min():.3f},{ev['omega'].max():.3f}] "
              f"mixed={frac_mixed:.3f} occupancy={dict(zip(BRANCH_NAMES, occ))}")

    # frozen field: same points -> identical draws, on separate calls
    m = Manifold(noise_rel=0.05, fail_rate=0.05, seed=1)
    a, b = m.observe(U[:50]), m.observe(U[:50])
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = np.isfinite(a)
    assert np.allclose(a[ok], b[ok]), "noise field is not frozen"
    # ... and under row permutation
    p = rng.permutation(50)
    c = m.observe(U[:50][p])
    ok = np.isfinite(c)
    assert np.allclose(c[ok], a[p][ok]), "noise field is not coordinate-keyed"
    print(f"frozen field OK: {np.count_nonzero(~ok)}/50 failed, "
          f"noise std={np.nanstd(a - m.truth(U[:50])):.4f}")


if __name__ == "__main__":
    _smoke()
