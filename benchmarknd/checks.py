"""Geometry and scorer validation for the high-dimensional cases.

Run before any strategy does, so that a later ranking cannot be blamed on a
degenerate surface or an unusable scorer. Every number printed here is a
property of the case or of the common reconstructor, never of a sampler.
"""
import time
import numpy as np
from scipy.stats import qmc
from .cases import CASES, weak_axes
from .core import SurfaceND, Observations, evaluation_set, reconstruct, rmse
from .tables import measured_strength


def inert_axis_leak(surface, n=4096, seed=5):
    """Largest value change produced by moving only the globally inert axes."""
    _, inert = weak_axes(surface.case)
    if not inert:
        return 0., 0
    rng = np.random.default_rng(seed)
    x = qmc.Sobol(surface.dim, scramble=True, seed=13).random(n)
    moved = x.copy()
    moved[:, inert] = rng.random((n, len(inert)))
    return float(np.max(np.abs(surface(x)-surface(moved)))), len(inert)


def fold_continuity(surface, n=2048, eps=1e-7, seed=7):
    """Value gap across the fold; the envelope is kinked but continuous."""
    if len(surface.normals) == 1:
        return 0.
    rng = np.random.default_rng(seed)
    x = rng.random((n, surface.dim))
    gaps = []
    for i in range(len(surface.normals)):
        for j in range(i+1, len(surface.normals)):
            normal = surface.normals[i]-surface.normals[j]
            normal = normal/np.linalg.norm(normal)
            shift = (surface.distance(x))[:, None]*normal
            on_fold = np.clip(x-shift*np.sign((x-.5) @ normal)[:, None], 0, 1)
            gaps.append(np.max(np.abs(surface(on_fold+eps*normal)-surface(on_fold-eps*normal))))
    return float(np.max(gaps))


def peak_margins(surface):
    """Fold clearance of every peak center, in that peak narrowest width."""
    centers, owners = surface.centers
    out = []
    for c, m in zip(centers, owners):
        narrow = float(np.min(surface.widths[m]))
        out.append(float(surface.distance(c)[0]/narrow))
    return out


def monte_carlo_error(values, scale):
    """Relative standard error of an RMS estimate from a finite test set."""
    squared = np.square(values)
    return float(np.std(squared, ddof=1)/np.sqrt(len(squared))/(2*np.mean(squared))*np.sqrt(np.mean(squared))/scale)


def scorer_report(surface, test, budgets):
    """Cost and reconstruction floor of the common RBF scorer at fixed budgets."""
    rows = []
    for budget in budgets:
        x = qmc.Sobol(surface.dim, scramble=True, seed=4242).random(budget)
        y = surface(x)
        start = time.perf_counter()
        predicted = reconstruct(x, y, test["x"])
        seconds = time.perf_counter()-start
        residual = float(np.max(np.abs(reconstruct(x, y, x)-y)))
        rows.append(dict(budget=budget, seconds=seconds, floor=rmse(predicted, test["y"], test["scale"]),
                         interpolation_residual=residual,
                         mc_standard_error=monte_carlo_error(predicted-test["y"], test["scale"])))
    return rows


def main(test_size=65536, budgets=(128, 256, 512, 1024)):
    for case, spec in CASES.items():
        surface = SurfaceND(case, 0)
        test = evaluation_set(surface, test_size)
        leak, inert = inert_axis_leak(surface)
        regions = np.bincount(test["region"], minlength=len(surface.normals))/len(test["x"])
        measured = measured_strength(surface, test["x"][:4096])
        print(f"\n{case}  (d={spec['dim']}, modes={len(spec['modes'])}, peaks={spec['peaks']}, fold={spec['fold']})")
        print(f"  region shares      {np.round(regions, 3).tolist()}")
        print(f"  fold band / peak   {100*test['band'].mean():.1f}% / {100*test['peak'].mean():.2f}% of the test set")
        print(f"  inert axes         {inert} leak {leak:.2e}")
        print(f"  fold continuity    {fold_continuity(surface):.2e}")
        print(f"  peak fold margin   {np.round(peak_margins(surface), 1).tolist()} narrow widths")
        print(f"  measured strength  {np.round(measured, 3).tolist()}")
        sizes = [b for b in budgets if b >= 4*spec["dim"]]
        for row in scorer_report(surface, test, sizes):
            print(f"  rbf N={row['budget']:5} {row['seconds']:6.2f}s  floor {row['floor']:.4f}"
                  f"  +-{row['mc_standard_error']:.1e}  interpolation residual {row['interpolation_residual']:.1e}")


if __name__ == "__main__":
    main()
