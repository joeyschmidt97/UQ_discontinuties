"""Ionut's native 6D microinstability proxies as a high-dimensional surface.

The 3D work used declared conditional slices of these formulas. This runs them
in their own dimension, with no coordinate held fixed, so the sampler faces the
full parameter space the proxy was written over:

    RLTi, RLTe, RLn, nu, beta   (scripts/ionut_proxies._RANGES)
    ky scale                    (the sixth axis, 0.30*(0.5 + x5))

Available branch pairs are ITG-TEM and ITG-KBM. There is no ETG or MTM branch
in the upstream formulas, so no ETG case can be run here; `branch_ITG`,
`branch_TEM` and `branch_KBM` are the whole set.

These are phenomenological proxies, not gyrokinetic solves. The transition mask
is derived from the stored branch growth rates rather than from the hard/soft
selection weights, so argmax and softmax variants of one pair share a physical
mask and stay comparable.
"""
import numpy as np

from scripts.generate_ionut_data import CASES as NATIVE_CASES, values

# The blend case carries no branch decomposition, so it has no transition mask
# and no per-branch error; excluded from the matched field rather than scored
# with a silently different metric set.
CASES = tuple(case for case in NATIVE_CASES if case != "ionut-stellarator-itg-kbm")
TRANSITION_FRACTION = .05       # branch-gap width counted as "at the transition"


class IonutSurface:
    """Adapter presenting a native 6D proxy through the SurfaceND interface.

    `distance` and `peak_distance` carry the same meaning their synthetic
    counterparts do -- distance to the feature the mask selects -- so the
    existing band and peak masks keep working without a second scoring path.
    """
    dim = 6
    band_threshold = TRANSITION_FRACTION

    def __init__(self, case, seed=0):
        if case not in CASES:
            raise ValueError(f"unknown or unscoreable Ionut case: {case}")
        if seed != 0:
            raise ValueError("the proxy formulas carry no surface seed; use seed 0")
        self.case = case
        self.seed = 0

    def __call__(self, x):
        return values(self.case, np.atleast_2d(np.asarray(x, float)))["y"]

    def branches(self, x):
        return values(self.case, np.atleast_2d(np.asarray(x, float)))["G"]

    def region(self, x):
        """Dominant branch by growth rate, independent of the selection rule."""
        return np.argmax(self.branches(x), axis=1)

    def distance(self, x):
        """Normalized branch-growth gap: small means near the transition."""
        g = self.branches(x)
        spread = float(np.ptp(g))
        if spread <= 0:
            return np.full(len(g), np.inf)
        return np.abs(g[:, 0]-g[:, 1])/spread

    def peak_distance(self, x):
        """Rank distance into the high-response tail, in the same units the
        synthetic peak mask uses: below 2 selects the top decile."""
        y = self(x)
        cut = float(np.quantile(y, .9))
        top = float(np.max(y))
        if top <= cut:
            return np.full(len(y), np.inf)
        return 2.*np.clip((cut-y)/(top-cut)+1., 0., None)

    @property
    def normals(self):
        """Two branches, so per-branch errors are reported for both."""
        return np.zeros((2, self.dim))
