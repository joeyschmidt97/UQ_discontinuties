"""Per-axis strength and design-profile tables: the high-dimensional replacement
for the 2-D manifold, placement and residual sheets.

Three things are worth seeing per case and they are all axis-indexed, so they
share one layout: what the generator declared, what the realized surface
actually responds to, and how much resolution each strategy spent per axis.
A 3-D render has no counterpart above two dimensions; this does.
"""
import pathlib

import numpy as np
from scipy.spatial import cKDTree
from .cases import CASES, strengths


def measured_strength(surface, x, step=2e-3, region=None):
    """Normalized RMS partial derivative per axis over the given points.

    Declared strength is the generator setting; this is what the surface does,
    peaks included. The two differ wherever a peak narrows an axis the envelope
    treats as moderate, which is exactly what the table should expose.
    """
    x = np.atleast_2d(np.asarray(x, float))
    if region is not None:
        x = x[surface.region(x) == region]
        if not len(x):
            raise ValueError("no test points fall in that mode region")
    out = np.zeros(surface.dim)
    for axis in range(surface.dim):
        plus, minus = x.copy(), x.copy()
        plus[:, axis] = np.minimum(1., plus[:, axis]+step)
        minus[:, axis] = np.maximum(0., minus[:, axis]-step)
        gradient = (surface(plus)-surface(minus))/(plus[:, axis]-minus[:, axis])
        out[axis] = np.sqrt(np.mean(np.square(gradient)))
    return out/max(float(out.max()), 1e-12)


UNIFORM_SPREAD = 1/np.sqrt(12)


def design_profile(x):
    """Per-axis coverage and local refinement of a design, versus uniform.

    Two rows are needed, not one. A design that holds an axis constant and a
    design that resolves it infinitely finely have the same neighbour gap, so
    gap alone is ambiguous; coverage separates them. Coverage is the marginal
    spread relative to a uniform design, refinement the uniform neighbour gap
    relative to this design. Refinement is zero where nothing is covered.
    """
    x = np.atleast_2d(np.asarray(x, float))
    if len(x) < 3:
        raise ValueError("a design profile needs at least three points")
    coverage = x.std(axis=0)/UNIFORM_SPREAD
    neighbour = cKDTree(x).query(x, k=2)[1][:, 1]
    gaps = np.median(np.abs(x-x[neighbour]), axis=0)
    reference = np.median(np.abs(np.diff(np.sort(np.random.default_rng(0).random((len(x), 1)), axis=0), axis=0)))
    refinement = np.where(coverage > 1e-6, reference/np.maximum(gaps, 1e-12), 0.)
    return dict(coverage=np.minimum(coverage, 1.), refinement=refinement/max(float(refinement.max()), 1e-12))


def case_table(surface, test, designs=None):
    """Rows: declared per mode, measured per mode, measured overall, per design."""
    rows, labels = [], []
    declared = np.array(strengths(surface.case), float)
    for m, row in enumerate(declared):
        rows.append(row)
        labels.append(f"declared mode {m}")
    for m in range(len(declared)):
        rows.append(measured_strength(surface, test["x"][:4096], region=m))
        labels.append(f"measured mode {m}")
    rows.append(measured_strength(surface, test["x"][:4096]))
    labels.append("measured overall")
    for name, x in (designs or {}).items():
        profile = design_profile(x)
        rows.append(profile["coverage"])
        labels.append(f"coverage {name}")
        rows.append(profile["refinement"])
        labels.append(f"refinement {name}")
    return np.array(rows), labels


def render_table(surface, table, labels, path, title=None):
    """One heatmap per case: axes across, declared/measured/design rows down."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dim = surface.dim
    fig, ax = plt.subplots(figsize=(1.05*dim+3.2, .52*len(labels)+2.1), layout="constrained")
    image = ax.imshow(table, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(dim), yticks=range(len(labels)),
           xticklabels=[f"x{j+1}" for j in range(dim)], yticklabels=labels)
    for i in range(table.shape[0]):
        for j in range(dim):
            value = table[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value < .55 else "black")
    ax.set_title(title or f"{surface.case} | seed {surface.seed} | axis strength", fontsize=13, pad=12)
    fig.colorbar(image, ax=ax, shrink=.85, label="strength / coverage / refinement (1 = highest in that row block)")
    fig.savefig(path, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=.2)
    plt.close(fig)
    return path


def case_summary():
    return {case: dict(dim=spec["dim"], modes=len(spec["modes"]), peaks=spec["peaks"],
                       fold=spec["fold"], patterns=spec["modes"]) for case, spec in CASES.items()}


def main():
    """Render one strength table per case into results/<dim>d/figures."""
    import argparse
    from scipy.stats import qmc
    from .core import SurfaceND, evaluation_set
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=16384)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results"))
    args = parser.parse_args()
    for case in args.cases:
        surface = SurfaceND(case, args.seed)
        test = evaluation_set(surface, args.test_size)
        n = 512 if surface.dim == 5 else 1024
        strength = measured_strength(surface, test["x"][:4096])
        active = [j for j in range(surface.dim) if strength[j] > .1]
        sobol = qmc.Sobol(surface.dim, scramble=True, seed=4242).random(n)
        oracle = np.random.default_rng(2).random((n, surface.dim))
        oracle[:, active] = qmc.Sobol(len(active), scramble=True, seed=99).random(n)
        table, labels = case_table(surface, test, {"space-filling": sobol, "active-subspace oracle": oracle})
        figures = args.output/f"{surface.dim}d"/"figures"
        figures.mkdir(parents=True, exist_ok=True)
        path = figures/f"strength-{case.split('-', 1)[1]}.png"
        render_table(surface, table, labels, path,
                     title=f"{case}  |  seed {args.seed}  |  declared vs measured axis strength, "
                           f"and design profiles at N={n}")
        print(path)


if __name__ == "__main__":
    main()
