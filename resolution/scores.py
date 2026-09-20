"""Fit-free resolution scores: the part of the grade that is dimension-free.

These require no reconstruction and no triangulation, so their definition is
literally the same expression at d=2 and d=8. That is what makes them the
cross-dimension spine; every reconstruction-based error is dimension-local
because it inherits whichever interpolant that dimension can support.

    h_N(x) = min_i ||x - x_i||                      local spacing (fill distance)
    q_N(x) = h_N(x) * V_N(x) / scale                variation-weighted fill distance
    C_N(t) = Pr_rho[ q_N(x) <= t ]                  coverage at resolution tolerance t

Quantiles of h_N are reported separately from q_N so a geometric hole cannot
hide inside a variation-weighted average, and the nonlinear indicator
h_N**2 * H_N / scale is reported separately from q_N so a narrow peak cannot
hide behind a steep plane.

Grading variation is the exact finite-difference variation of the reference
surface, not an algorithm's own estimate -- otherwise VWRS and VURS would grade
the quality of the estimator they sample with. Observed-data variation is
reported alongside it, prefixed `observed_`, as a diagnostic of how well the
sampler's own view matches the truth.
"""
import numpy as np
from scipy.spatial import cKDTree, distance as spdistance

# Declared resolution tolerances. Fixed across dimension on purpose: they are
# the cross-dimension comparison, so they must not be recalibrated per run.
# The holistic score's tolerances are a separate, dimension-local object.
SPINE_TAUS = (.02, .05, .10, .25)


def spine_targets():
    return tuple(SPINE_TAUS)


def _quantiles(values, weights=None):
    values = np.asarray(values, float)
    if weights is None:
        return (float(np.sqrt(np.mean(values**2))), float(np.mean(values)),
                float(np.quantile(values, .95)), float(values.max()))
    weights = np.asarray(weights, float)
    total = weights.sum()
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])/total
    p95 = float(values[order][int(np.searchsorted(cumulative, .95))])
    return (float(np.sqrt(np.sum(weights*values**2)/total)),
            float(np.sum(weights*values)/total), p95, float(values.max()))


def fit_free_scores(design, probes, variation, scale, curvature=None,
                    weights=None, observed=None, prefix=""):
    """Every score in the spine, for one design against one probe cloud.

    `variation` is a slope-unit array at the probe points; `curvature` is the
    optional second-derivative-unit array for the nonlinear indicator. `scale`
    is the response range, making every reported quantity dimensionless.
    """
    design = np.atleast_2d(np.asarray(design, float))
    probes = np.atleast_2d(np.asarray(probes, float))
    variation = np.asarray(variation, float)
    if design.shape[1] != probes.shape[1]:
        raise ValueError("design and probes must share the input dimension")
    if variation.shape != (len(probes),):
        raise ValueError("one variation value per probe point required")
    if not np.isfinite(variation).all() or not np.isfinite(design).all():
        raise ValueError("nonfinite variation or design must not disappear from scoring")
    if scale <= 0:
        raise ValueError("positive response scale required")

    spacing = cKDTree(design).query(probes)[0]
    q = spacing*variation/scale

    fill_rms, fill_mean, fill_p95, fill_max = _quantiles(spacing, weights)
    q_rms, q_mean, q_p95, q_max = _quantiles(q, weights)
    out = {
        prefix + "fill_rms": fill_rms, prefix + "fill_mean": fill_mean,
        prefix + "fill_p95": fill_p95, prefix + "fill_max": fill_max,
        prefix + "vwfd_rms": q_rms, prefix + "vwfd_mean": q_mean,
        prefix + "vwfd_p95": q_p95, prefix + "vwfd_max": q_max,
    }
    for tau in SPINE_TAUS:
        key = prefix + f"vwfd_coverage_{int(round(tau*100)):02d}"
        out[key] = (float(np.mean(q <= tau)) if weights is None else
                    float(np.sum(np.asarray(weights, float)*(q <= tau))/np.sum(weights)))

    if curvature is not None:
        # The folded surfaces are kinked, so the exact second difference is
        # unbounded on the fold itself and `nonlinear_max` diverges with a
        # smaller finite-difference step. Rank on `nonlinear_p95`; read the RMS
        # and maximum as diagnostics, the same rule the note applies to maximum
        # absolute error.
        curvature = np.asarray(curvature, float)
        if curvature.shape != (len(probes),):
            raise ValueError("one curvature value per probe point required")
        nonlinear = spacing**2*curvature/scale
        n_rms, n_mean, n_p95, n_max = _quantiles(nonlinear, weights)
        out.update({prefix + "nonlinear_rms": n_rms, prefix + "nonlinear_mean": n_mean,
                    prefix + "nonlinear_p95": n_p95, prefix + "nonlinear_max": n_max})

    if not prefix:
        out["min_separation"] = float(spdistance.pdist(design).min()) if len(design) > 1 else 0.
        spread = design.max(axis=0) - design.min(axis=0)
        out["axis_spread_min"] = float(spread.min())
        out["axis_spread_mean"] = float(spread.mean())
        # A design that collapses onto a subspace is cheap to miss in any
        # scalar; report the worst axis explicitly.
        out["axis_spread_worst_index"] = int(np.argmin(spread))

    if observed is not None:
        out.update(fit_free_scores(design, probes, observed.gradient, scale,
                                   curvature=observed.curvature, weights=weights,
                                   prefix=prefix + "observed_"))
    return out
