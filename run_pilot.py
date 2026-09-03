"""
Pilot benchmark: all contender arms on the test manifold.

Sweeps the manifold's difficulty knobs (hybridization width, band-gap
discontinuity, evaluation noise and failures) against evaluation budget, and
scores every arm on clean truth. Results serialize to JSON so the figures
regenerate without re-running the fits.

The five cases, in increasing order of what they break:

  soft-T0.20   softmax hybridization, wide       -- C-infinity, easy control
  soft-T0.05   softmax hybridization, narrow     -- C-infinity but thin feature
  argmax       hard mode selection               -- C0 kink in gamma, JUMP in omega
  argmax+gap   + rational-surface band gap       -- genuine jump discontinuity
  argmax+gap+noisy  + 5% noise, 5% failed runs   -- the semi-autonomous reality

Two QoIs, because the choice between them is an open roadmap decision:
`gamma` is only kinked under argmax, whereas `omega` genuinely jumps when the
dominant branch flips diamagnetic direction. An arm that looks acceptable on
gamma can be hopeless on omega.

Usage:
    python run_pilot.py                  # default sweep
    python run_pilot.py --quick          # 2 budgets, gamma only
    python run_pilot.py --figures-only   # re-plot from results.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from manifold import Manifold
from arms import RandomNearestArm, score_arm

OUT_DIR = pathlib.Path(__file__).parent / "outputs"

# name -> Manifold kwargs
CASES = {
    "soft-T0.20":       dict(mode="softmax", T=0.20, align="smooth"),
    "soft-T0.05":       dict(mode="softmax", T=0.05, align="smooth"),
    "argmax":           dict(mode="argmax", align="smooth"),
    "argmax+gap":       dict(mode="argmax", align="gap"),
    "argmax+gap+noisy": dict(mode="argmax", align="gap",
                             noise_rel=0.05, fail_rate=0.05, seed=1),
}


def build_arms(which):
    """Instantiate arms by name. Arms whose backend is missing are skipped with
    a warning rather than killing the sweep."""
    out = []
    for nm in which:
        try:
            out.append(_make_arm(nm))
        except ImportError as exc:
            print(f"  [skip] {nm}: {exc}", file=sys.stderr)
    return out


def _make_arm(nm):
    if nm == "random-nn":
        return RandomNearestArm(seed=0)
    if nm.startswith("sgpp"):
        from arms import SGppArm, SGppRegularArm
        if nm == "sgpp-regular":
            return SGppRegularArm(basis="modlinear")
        if nm == "sgpp-surplus":
            return SGppArm(basis="modlinear", refine="surplus")
        if nm == "sgpp-volume":
            return SGppArm(basis="modlinear", refine="volume")
        if nm == "sgpp-bspline":
            return SGppArm(basis="modbspline", degree=3, refine="surplus")
    if nm == "sglib":
        from arms import SgLibArm
        return SgLibArm()
    if nm.startswith("gpr-"):
        from arms import GPRArm
        return GPRArm(acquisition=nm.split("-", 1)[1])
    raise ValueError(f"unknown arm {nm}")


ALL_ARMS = ["sglib", "gpr-var", "gpr-ucb", "gpr-grad",
            "sgpp-surplus", "sgpp-volume", "sgpp-regular", "random-nn"]


def run(cases, outs, budgets, arm_names, n_test=4000, mix_thresh=0.30):
    rows = []
    for case in cases:
        for out in outs:
            m = Manifold(out=out, **CASES[case])
            # one frozen test design per (case, out): every arm and budget is
            # scored on identical points
            test = m.test_set(n=n_test)
            print(f"\n== {case} / {out} == "
                  f"(mixed fraction of test set: "
                  f"{np.mean(test[1]['mix'] >= mix_thresh):.3f})")
            for budget in budgets:
                for arm in build_arms(arm_names):
                    try:
                        s = score_arm(arm, m, out=out, budget=budget, test=test,
                                      mix_thresh=mix_thresh)
                    except Exception as exc:              # one bad arm must not
                        print(f"  [fail] {arm.name} @ {budget}: {exc}")  # kill the sweep
                        continue
                    print("  " + str(s))
                    rows.append(dict(case=case, budget=budget, **s.as_dict()))
    return rows


# ===========================================================================
# Figures
# ===========================================================================
def make_figures(rows, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cases = list(dict.fromkeys(r["case"] for r in rows))
    outs = list(dict.fromkeys(r["out"] for r in rows))
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    colors = {a: f"C{i}" for i, a in enumerate(arms)}

    # (A) convergence: nRMSE vs budget, one panel per case
    for out in outs:
        fig, axes = plt.subplots(1, len(cases), figsize=(3.3 * len(cases), 3.6),
                                 sharey=True, squeeze=False)
        for ax, case in zip(axes[0], cases):
            for a in arms:
                sel = sorted((r for r in rows if r["case"] == case
                              and r["out"] == out and r["arm"] == a),
                             key=lambda r: r["n_evals"])
                if not sel:
                    continue
                ax.loglog([r["n_evals"] for r in sel], [r["nrmse"] for r in sel],
                          "o-", color=colors[a], lw=1.8, ms=4, label=a)
            ax.set_title(case, fontsize=9)
            ax.set_xlabel("function evaluations")
            ax.grid(True, which="both", ls=":", alpha=0.4)
            ax.spines[["right", "top"]].set_visible(False)
        axes[0][0].set_ylabel(f"normalized RMSE ({out})")
        axes[0][-1].legend(frameon=False, fontsize=7)
        fig.suptitle(f"Budget vs error, QoI = {out}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_convergence_{out}.png", dpi=140)
        plt.close(fig)

    # (B) where the error lives: mixed band vs the calm bulk, at max budget
    bmax = max(r["n_evals"] for r in rows)
    for out in outs:
        sel = [r for r in rows if r["out"] == out and r["n_evals"] >= 0.8 * bmax]
        if not sel:
            continue
        fig, ax = plt.subplots(figsize=(1.7 * len(cases) + 3, 4))
        width = 0.8 / max(len(arms), 1)
        x = np.arange(len(cases))
        for i, a in enumerate(arms):
            mix = [next((r["rmse_mixed"] for r in sel
                         if r["case"] == c and r["arm"] == a), np.nan) for c in cases]
            calm = [next((r["rmse_calm"] for r in sel
                          if r["case"] == c and r["arm"] == a), np.nan) for c in cases]
            ax.bar(x + i * width, mix, width * 0.92, color=colors[a], label=f"{a} (mixed)")
            ax.plot(x + i * width, calm, "_", ms=11, mew=2.2, color="k",
                    label="calm bulk" if i == 0 else None)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(cases, rotation=20, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(f"RMSE ({out})")
        ax.set_title(f"Error concentrates in the competing-mode band "
                     f"(bars) vs the bulk (ticks), budget ~{bmax}")
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_mixed_vs_calm_{out}.png", dpi=140)
        plt.close(fig)

    # (C) targeting: what fraction of the budget landed in the mixed band
    fig, ax = plt.subplots(figsize=(1.7 * len(cases) + 3, 4))
    x = np.arange(len(cases))
    width = 0.8 / max(len(arms), 1)
    ref = [next((r["frac_test_mixed"] for r in rows if r["case"] == c), np.nan)
           for c in cases]
    for i, a in enumerate(arms):
        vals = [np.nanmean([r["frac_design_mixed"] for r in rows
                            if r["case"] == c and r["arm"] == a] or [np.nan])
                for c in cases]
        ax.bar(x + i * width, vals, width * 0.92, color=colors[a], label=a)
    ax.plot(x + 0.4 - width / 2, ref, "k_", ms=18, mew=2.5,
            label="volume fraction of the band (random baseline)")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(cases, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("fraction of design points in the mixed band")
    ax.set_title("Targeting: does refinement actually go to the competing-mode region?")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_targeting.png", dpi=140)
    plt.close(fig)

    print(f"figures -> {out_dir}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--outs", nargs="+", default=["gamma", "omega"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[100, 200, 400, 800])
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--mix-thresh", type=float, default=0.30)
    ap.add_argument("--quick", action="store_true",
                    help="2 budgets, gamma only, 3 cases")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = OUT_DIR / "results.json"

    if args.figures_only:
        rows = json.loads(results.read_text())["rows"]
        make_figures(rows)
        return

    if args.quick:
        args.budgets = [100, 400]
        args.outs = ["gamma"]
        args.cases = ["soft-T0.20", "argmax", "argmax+gap"]

    rows = run(args.cases, args.outs, args.budgets, args.arms,
               n_test=args.n_test, mix_thresh=args.mix_thresh)
    results.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print(f"\nresults -> {results}  ({len(rows)} fits)")
    if rows:
        make_figures(rows)


if __name__ == "__main__":
    main()
