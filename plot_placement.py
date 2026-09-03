"""
How each method fills the space and picks its next points.

This is the placement question, kept deliberately separate from accuracy: an arm
can be accurate on average and still never put a point where the sensitivity
analysis needs one. Only designs are needed here, not predictions, so this runs
fast even for arms whose interpolant is expensive to query.

Figures:
  fig_placement_<case>.png   design projected onto the two axes that carry the
                             transition, coloured by ACQUISITION ORDER, over the
                             competing-mode ridge. One column per arm.
  fig_marginals_<case>.png   per-axis marginal density of each design -- the
                             readout for dimension adaptivity: a flat profile is
                             space-filling, a peaked one means the method
                             concentrated on that axis.
  fig_targeting_curve_<case>.png  running fraction of the design inside the
                             competing-mode band and in the top-decile of the
                             QoI, vs evaluation index. Shows WHEN a method stops
                             exploring and starts exploiting.

Run:  python plot_placement.py --case argmax+gap --budget 300
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manifold import CAMPAIGN_AXES, Manifold
from run_pilot import CASES, _make_arm

OUT = pathlib.Path(__file__).parent / "outputs"
DIM = len(CAMPAIGN_AXES)
NAMES = [a.name for a in CAMPAIGN_AXES]

# Te_ped_scale vs w_Te_scale: the pair carrying the oblique kink ridge.
IX, IY = 0, 2


def collect(case, budget, arm_names, out="gamma", mix_thresh=0.30):
    """Fit each arm once and keep its design in acquisition order."""
    m = Manifold(out=out, **CASES[case])
    designs = {}
    for nm in arm_names:
        try:
            arm = _make_arm(nm)
        except ImportError as exc:
            print(f"  [skip] {nm}: {exc}")
            continue
        oracle = m.oracle(out)
        try:
            arm.fit(oracle, m.dim, budget)
        except Exception as exc:
            print(f"  [fail] {nm}: {exc}")
            continue
        D = oracle.design
        ev = m.evaluate(D)
        designs[nm] = dict(D=D, mix=ev["mix"], y=ev[out],
                           n_evals=oracle.n_evals, n_failed=oracle.n_failed,
                           summary=arm.summary())
        print(f"  {nm:14s} n={oracle.n_evals:5d} failed={oracle.n_failed:3d} "
              f"in-band={np.mean(ev['mix'] >= mix_thresh):.3f}")
    return m, designs


def fig_placement(m, designs, case, mix_thresh):
    n = len(designs)
    gx, gy = np.meshgrid(np.linspace(0, 1, 240), np.linspace(0, 1, 240))
    U = np.full((gx.size, DIM), 0.5)
    U[:, IX], U[:, IY] = gx.ravel(), gy.ravel()
    MIX = m.evaluate(U)["mix"].reshape(gx.shape)
    X, Y = CAMPAIGN_AXES[IX].to_phys(gx), CAMPAIGN_AXES[IY].to_phys(gy)

    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.3), squeeze=False)
    for ax, (nm, d) in zip(axes[0], designs.items()):
        ax.pcolormesh(X, Y, MIX, shading="auto", cmap="Greys", vmin=0, vmax=1)
        ax.contour(X, Y, MIX, levels=[mix_thresh], colors="#1b9e77", linewidths=1.6)
        D = d["D"]
        order = np.arange(len(D))
        sc = ax.scatter(CAMPAIGN_AXES[IX].to_phys(D[:, IX]),
                        CAMPAIGN_AXES[IY].to_phys(D[:, IY]),
                        c=order, cmap="plasma", s=16, edgecolor="k",
                        linewidth=0.25, alpha=0.9)
        fig.colorbar(sc, ax=ax, label="acquisition order")
        ax.set_title(f"{nm}  (n={d['n_evals']}, in-band "
                     f"{np.mean(d['mix'] >= mix_thresh):.2f})", fontsize=9)
        ax.set_xlabel(NAMES[IX])
        ax.set_ylabel(NAMES[IY])
    fig.suptitle(f"Where the budget went — case '{case}'. Grey = competing-mode "
                 f"indicator, green contour = band edge.\n"
                 f"Points are projections of a {DIM}-D design; the other four "
                 f"axes are NOT at 0.5, so off-ridge points can still be on it.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_placement_{case.replace('+', '_')}.png", dpi=140)
    plt.close(fig)


def fig_marginals(designs, case):
    n = len(designs)
    fig, axes = plt.subplots(n, DIM, figsize=(2.0 * DIM, 2.1 * n),
                             squeeze=False, sharex=True)
    for i, (nm, d) in enumerate(designs.items()):
        D = d["D"]
        for j in range(DIM):
            ax = axes[i, j]
            ax.hist(D[:, j], bins=16, range=(0, 1), color="#d95f02")
            ax.axhline(len(D) / 16, color="k", ls="--", lw=1.0)
            if i == 0:
                ax.set_title(NAMES[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(nm, fontsize=8)
            ax.set_yticks([])
            ax.spines[["right", "top", "left"]].set_visible(False)
    fig.suptitle(f"Per-axis marginals — case '{case}'. Dashed line = uniform. "
                 "Peaked = the method concentrated on that axis; flat = space-filling.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_marginals_{case.replace('+', '_')}.png", dpi=140)
    plt.close(fig)


def fig_targeting_curve(m, designs, case, mix_thresh, out="gamma"):
    ref = np.random.default_rng(4).random((6000, DIM))
    ev_ref = m.evaluate(ref)
    band_vol = float(np.mean(ev_ref["mix"] >= mix_thresh))
    q90 = float(np.quantile(np.abs(ev_ref[out]), 0.90))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for nm, d in designs.items():
        in_band = (d["mix"] >= mix_thresh).astype(float)
        in_peak = (np.abs(d["y"]) >= q90).astype(float)
        k = np.arange(1, len(in_band) + 1)
        axes[0].plot(k, np.cumsum(in_band) / k, lw=1.8, label=nm)
        axes[1].plot(k, np.cumsum(in_peak) / k, lw=1.8, label=nm)
    axes[0].axhline(band_vol, color="k", ls="--", lw=1.2,
                    label=f"band volume {band_vol:.3f} (no targeting)")
    axes[1].axhline(0.10, color="k", ls="--", lw=1.2, label="0.10 (no targeting)")
    axes[0].set_ylabel("running fraction of design in the competing-mode band")
    axes[1].set_ylabel(f"running fraction in the top decile of |{out}|")
    for a in axes:
        a.set_xlabel("evaluation index")
        a.legend(frameon=False, fontsize=7)
        a.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"When does each method stop exploring and start exploiting? "
                 f"— case '{case}'", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_targeting_curve_{case.replace('+', '_')}.png", dpi=140)
    plt.close(fig)
    return band_vol, q90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="argmax+gap", choices=list(CASES))
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--out", default="gamma")
    ap.add_argument("--mix-thresh", type=float, default=0.30)
    ap.add_argument("--arms", nargs="+",
                    default=["sglib", "gpr-var", "gpr-ucb", "gpr-grad",
                             "sgpp-surplus", "random-nn"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"case={args.case} budget={args.budget} out={args.out}")
    m, designs = collect(args.case, args.budget, args.arms, args.out,
                         args.mix_thresh)
    if not designs:
        print("no arms produced a design")
        return

    fig_placement(m, designs, args.case, args.mix_thresh)
    fig_marginals(designs, args.case)
    band_vol, q90 = fig_targeting_curve(m, designs, args.case, args.mix_thresh,
                                        args.out)

    from arms import design_metrics
    rows = {}
    print(f"\n{'arm':14s} {'n':>5s} {'band':>6s} {'lift':>5s} {'peak':>6s} "
          f"{'minD':>6s} {'hole':>6s} {'cov':>5s}")
    for nm, d in designs.items():
        dm = design_metrics(d["D"], m, mix_thresh=args.mix_thresh)
        rows[nm] = dict(dm, n_evals=d["n_evals"], n_failed=d["n_failed"],
                        summary=d["summary"])
        print(f"{nm:14s} {d['n_evals']:5d} {dm['frac_band']:6.3f} "
              f"{dm['band_lift']:5.2f} {dm['frac_peak']:6.3f} "
              f"{dm['min_dist']:6.3f} {dm['hole']:6.3f} {dm['coverage']:5.2f}")

    path = OUT / f"placement_{args.case.replace('+', '_')}.json"
    path.write_text(json.dumps(
        dict(case=args.case, budget=args.budget, out=args.out,
             band_volume=band_vol, q90=q90, arms=rows), indent=2, default=str))
    print(f"\nfigures + {path.name} -> {OUT}")


if __name__ == "__main__":
    main()
