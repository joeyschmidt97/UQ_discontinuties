"""
Visual verification sheet — every assertion, rendered.

`test_arms.py` answers "did it pass?". This answers "what was actually
measured, and how close to the line was it?" Each panel plots the quantity an
assertion checks, draws the pass threshold, and annotates PASS / FAIL. A
borderline pass looks different from a comfortable one, which a boolean cannot
show you.

Runs as a test (asserts at the end, non-zero exit on failure) AND as a report
generator (writes figures/verification_sheet.png). Use it to eyeball the
benchmark before trusting any number in report.ipynb.

Run:  python tests/test_visual.py            # sheet + assertions
      python tests/test_visual.py --no-sglib # skip the slow arm (~40 s)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arms import GPRArm, RandomNearestArm, design_metrics, score_arm   # noqa: E402
from manifold import Manifold                                          # noqa: E402

FIGDIR = ROOT / "figures"
BUDGET = 150
OK, BAD = "#1b9e77", "#d62728"


def _verdict(ax, passed, note=""):
    ax.text(.02, .96, ("PASS " + note) if passed else ("FAIL " + note),
            transform=ax.transAxes, va="top", fontsize=9, fontweight="bold",
            color=OK if passed else BAD,
            bbox=dict(fc="w", ec=OK if passed else BAD, lw=1.2, alpha=.9))
    return passed


# ===========================================================================
def build_sheet(use_sglib=True):
    results = {}
    m = Manifold(mode="argmax", align="gap", out="gamma")

    arms = {"gpr-var": GPRArm("var", batch=8), "gpr-ucb": GPRArm("ucb", batch=8),
            "gpr-grad": GPRArm("grad", batch=8), "random-nn": RandomNearestArm()}
    if use_sglib:
        from arms import SgLibArm
        arms["sglib"] = SgLibArm()

    designs, spent = {}, {}
    for nm, arm in arms.items():
        oracle = m.oracle("gamma")
        arm.fit(oracle, m.dim, BUDGET)
        designs[nm] = oracle.design
        spent[nm] = oracle.n_evals
        print(f"  fitted {nm:11s} n={oracle.n_evals}")

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))

    # -- 1. budget is never exceeded -------------------------------------
    ax = axes[0, 0]
    nm_ = list(spent)
    ax.bar(nm_, [spent[n] for n in nm_], color=[OK if spent[n] <= BUDGET else BAD for n in nm_])
    ax.axhline(BUDGET, color="k", ls="--", lw=1.4, label=f"budget {BUDGET}")
    for i, n in enumerate(nm_):
        ax.text(i, spent[n] + 2, str(spent[n]), ha="center", fontsize=8)
    ax.set_ylabel("evaluations actually spent"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("1. Budget respected\n(sg_lib undershoots: subspaces are atomic)", fontsize=9)
    ax.tick_params(axis="x", rotation=20, labelsize=7)
    results["budget"] = _verdict(ax, all(spent[n] <= BUDGET for n in nm_))

    # -- 2. frozen noise field -------------------------------------------
    ax = axes[0, 1]
    mn = Manifold(mode="argmax", align="gap", noise_rel=.05, fail_rate=.05, seed=1)
    U = np.random.default_rng(0).random((400, mn.dim))
    a, b = mn.observe(U), mn.observe(U[::-1])[::-1]          # same points, reordered
    fin = np.isfinite(a) & np.isfinite(b)
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    identical = np.allclose(a[fin], b[fin])
    ax.scatter(a[fin], b[fin], s=14, color=OK, alpha=.6)
    lo, hi = np.nanmin(a[fin]), np.nanmax(a[fin])
    ax.plot([lo, hi], [lo, hi], "k", lw=1)
    ax.set_xlabel("observe(X)"); ax.set_ylabel("observe(X reordered)")
    ax.set_title("2. Noise/failure field is frozen to coordinates\n"
                 "(off-diagonal here = arms compared on different data)", fontsize=9)
    results["frozen"] = _verdict(ax, same_nan and identical,
                                 f"{np.sum(~np.isfinite(a))} NaN, both calls agree")

    # -- 3. noise absorbed, not memorized --------------------------------
    ax = axes[0, 2]
    clean, noisy = Manifold(mode="argmax"), Manifold(mode="argmax", noise_rel=.30, seed=5)
    test = clean.test_set(n=600)
    s_c = score_arm(RandomNearestArm(), clean, budget=200, test=test)
    s_n = score_arm(RandomNearestArm(), noisy, budget=200, test=test)
    ax.bar(["clean", "30% noise"], [s_c.rmse, s_n.rmse], color=[OK, "#d95f02"])
    for i, v in enumerate([s_c.rmse, s_n.rmse]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSE vs CLEAN truth")
    ax.set_title("3. Scoring is against clean truth\n"
                 "(noisy must score worse — else noise is being memorized)", fontsize=9)
    results["clean_truth"] = _verdict(ax, s_n.rmse > s_c.rmse)

    # -- 4. batches do not collapse --------------------------------------
    ax = axes[1, 0]
    dm = {n: design_metrics(D, m) for n, D in designs.items()}
    vals = [dm[n]["min_dist"] for n in nm_]
    ax.bar(nm_, vals, color=[OK if v > 1e-3 else BAD for v in vals])
    ax.axhline(1e-3, color="k", ls="--", lw=1.4, label="collapse threshold")
    ax.set_yscale("log"); ax.set_ylabel("min pairwise distance")
    ax.set_title("4. Designs do not clump\n(a collapsed batch sits on the line)", fontsize=9)
    ax.legend(frameon=False, fontsize=8); ax.tick_params(axis="x", rotation=20, labelsize=7)
    results["no_collapse"] = _verdict(ax, all(v > 1e-3 for v in vals))

    # -- 5. acquisitions actually differ ---------------------------------
    ax = axes[1, 1]
    x = np.arange(2); w = .38
    peak = [dm["gpr-ucb"]["frac_peak"], dm["gpr-grad"]["frac_peak"]]
    lift = [dm["gpr-ucb"]["band_lift"], dm["gpr-grad"]["band_lift"]]
    ax.bar(x - w/2, peak, w, color="#7570b3", label="frac_peak")
    ax.bar(x + w/2, lift, w, color="#e7298a", label="band_lift")
    ax.axhline(.10, color="#7570b3", ls=":", lw=1.3)
    ax.axhline(1.0, color="#e7298a", ls=":", lw=1.3)
    ax.set_xticks(x); ax.set_xticklabels(["gpr-ucb", "gpr-grad"])
    ax.set_title("5. The acquisition switch does something\n"
                 "ucb wins peaks, grad wins the band. Dotted = random.", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    results["acq_differ"] = _verdict(
        ax, peak[0] > peak[1] and lift[1] > lift[0],
        f"peak {peak[0]:.2f}>{peak[1]:.2f}, lift {lift[1]:.2f}>{lift[0]:.2f}")

    # -- 6. sg_lib dimension adaptivity + budget-stop regression ----------
    ax = axes[1, 2]
    if use_sglib:
        from arms import SgLibArm
        lv = arms["sglib"].axis_levels()
        ax.bar(np.arange(len(lv)), lv, color=OK if lv.max() > lv.min() else BAD)
        ax.set_xticks(np.arange(len(lv)))
        ax.set_xticklabels([a.name for a in __import__("manifold").CAMPAIGN_AXES],
                           rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("max level reached")
        # budget-stop must not poison the interpolant (KeyError regression)
        Xp = np.random.default_rng(0).random((25, m.dim)); finite = True
        for b in (60, 100):
            try:
                finite &= bool(np.all(np.isfinite(SgLibArm().fit(m.oracle("gamma"), m.dim, b)
                                                 .predict(Xp))))
            except Exception:
                finite = False
        ax.set_title("6. sg_lib: axis-adaptive, and predict() survives a\n"
                     "budget stop (the unpaid-subspace KeyError)", fontsize=9)
        results["sglib_adaptive"] = _verdict(ax, bool(lv.max() > lv.min()) and finite,
                                             f"levels {lv.tolist()}, predict finite={finite}")
    else:
        ax.text(.5, .5, "sg_lib skipped (--no-sglib)", ha="center", transform=ax.transAxes)
        ax.set_axis_off()

    for a in axes.ravel():
        a.spines[["right", "top"]].set_visible(False)
    n_pass = sum(results.values())
    fig.suptitle(f"Verification sheet — {n_pass}/{len(results)} checks passed "
                 f"(manifold: argmax + band gap, budget {BUDGET})", fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "verification_sheet.png", dpi=130)
    plt.close(fig)
    return results


def test_verification_sheet():
    res = build_sheet(use_sglib=True)
    failed = [k for k, v in res.items() if not v]
    assert not failed, f"visual checks failed: {failed}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sglib", action="store_true", help="skip the slow arm")
    args = ap.parse_args()

    res = build_sheet(use_sglib=not args.no_sglib)
    print()
    for k, v in res.items():
        print(f"  {'ok  ' if v else 'FAIL'}  {k}")
    print(f"{sum(res.values())}/{len(res)} passed")
    print(f"sheet -> {FIGDIR / 'verification_sheet.png'}")
    sys.exit(0 if all(res.values()) else 1)
