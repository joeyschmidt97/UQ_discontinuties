"""Compact multi-sheet report for the matched 3D microinstability benchmark."""
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from scripts.render_ionut_3d_report import render_reference
from .strategies import ARMS


NAMES = dict(grid="Progressive grid", sglib="Ionut / sg_lib", sgpp="SG++",
             triangles="Tetrahedra", **{"gpr-var":"GP uncertainty", "gpr-grad":"GP gradient",
             "gpr-blend":"GP grad/unc 50/50", "gpr-m05-var":"GP Matern .5 uncertainty",
             "gpr-m05-grad":"GP Matern .5 gradient", "gpr-m05-blend":"GP Matern .5 grad/unc",
             "vwrs":"VWRS", "vurs":"VURS"})
COLORS = {arm: plt.get_cmap("tab20")(index) for index, arm in enumerate(ARMS)}
METRICS = (("error", "Global normalized RMS"), ("nmae", "Global NMAE"),
           ("p95_error", "Normalized point-error P95"),
           ("transition_error", "Transition normalized RMS"),
           ("transition_nmae", "Transition NMAE"),
           ("high_error", "High-response normalized RMS"),
           ("high_nmae", "High-response NMAE"), ("vwfd_p95", "VWFD P95"),
           ("holistic_error", "Worst-case holistic H"))
DESIGN_METRICS = (("branch0_error", "Branch 0 normalized RMS"),
                  ("branch1_error", "Branch 1 normalized RMS"),
                  ("branch0_nmae", "Branch 0 NMAE"), ("branch1_nmae", "Branch 1 NMAE"),
                  ("fill_p95", "Fill-distance P95"), ("fill_max", "Maximum fill distance"),
                  ("min_separation", "Minimum sample separation"),
                  ("rbf_error", "Common RBF cross-check"))


def good(payload):
    return [row for row in payload["rows"] if row["status"] == "ok"]


def grouped(rows, arm, metric, case=None):
    values = {}
    for row in rows:
        if row["arm"] != arm or (case is not None and row["case"] != case):
            continue
        value = row.get(metric)
        if value is not None:
            values.setdefault(row["n"], []).append(value)
    return (np.asarray(sorted(values)),
            np.asarray([np.median(values[n]) for n in sorted(values)]))


def save(fig, figures, filename):
    fig.savefig(figures/filename, dpi=145, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return filename


def placement(rows, figures, cases, arms):
    chosen = [case for case in cases if case.endswith("argmax-gamma")]
    fig = plt.figure(figsize=(3.2*len(arms), 6.8), constrained_layout=True)
    for row_index, case in enumerate(chosen):
        for column, arm in enumerate(arms):
            ax = fig.add_subplot(len(chosen), len(arms), row_index*len(arms)+column+1, projection="3d")
            candidates = [r for r in rows if r["case"] == case and r["arm"] == arm and r["seed"] == 0 and "x" in r]
            if candidates:
                x = np.asarray(candidates[-1]["x"])
                ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=np.arange(len(x)), cmap="viridis", s=5)
            ax.set(xticks=(0,1), yticks=(0,1), zticks=(0,1))
            if row_index == 0:
                ax.set_title(NAMES[arm], fontsize=9)
            if column == 0:
                ax.set_ylabel(case.replace("ionut-", ""), labelpad=8)
    fig.suptitle("Representative paid-point placement at N=256 (seed 0)", fontsize=16)
    return save(fig, figures, "02-point-placement-3d.png")


def case_curves(rows, figures, cases, arms):
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)
    for ax, case in zip(axes.ravel(), cases):
        for arm in arms:
            n, y = grouped(rows, arm, "error", case)
            if len(n):
                ax.plot(n, y, color=COLORS[arm], label=NAMES[arm], linewidth=1.7)
        ax.set(xscale="log", yscale="log", title=case.replace("ionut-", ""),
               xlabel="Paid evaluations", ylabel="Normalized RMS")
        ax.grid(True, which="both", alpha=.18)
    axes[0, 0].legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Common tetrahedral reconstruction error by 3D proxy", fontsize=17)
    return save(fig, figures, "03-error-versus-points.png")


def metric_sheet(rows, figures, arms, metrics, filename, title):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    for ax, (metric, panel_title) in zip(axes.ravel(), metrics):
        for arm in arms:
            n, y = grouped(rows, arm, metric)
            if len(n):
                ax.plot(n, y, color=COLORS[arm], label=NAMES[arm], linewidth=1.7)
        ax.set(xscale="log", yscale="log", title=panel_title, xlabel="Paid evaluations")
        ax.grid(True, which="both", alpha=.18)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    axes[-1, 1].legend(ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(.5, -.25), loc="upper center")
    fig.suptitle(title, fontsize=17)
    return save(fig, figures, filename)


def scorecard(rows, figures, cases, arms):
    final_n = max(row["n"] for row in rows)
    matrix = np.full((len(arms), len(cases)), np.nan)
    for i, arm in enumerate(arms):
        for j, case in enumerate(cases):
            values = [row["holistic_error"] for row in rows
                      if row["arm"] == arm and row["case"] == case and row["n"] == final_n]
            if values:
                matrix[i, j] = np.median(values)
    valid = matrix[np.isfinite(matrix)]
    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="magma_r",
                      norm=LogNorm(vmin=max(float(valid.min()), 1e-3), vmax=float(valid.max())))
    ax.set(xticks=np.arange(len(cases)), xticklabels=[c.replace("ionut-", "") for c in cases],
           yticks=np.arange(len(arms)), yticklabels=[NAMES[a] for a in arms],
           title=f"Median holistic score H at N={final_n} (lower is better)")
    ax.tick_params(axis="x", rotation=35)
    for i in range(len(arms)):
        for j in range(len(cases)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if image.norm(matrix[i,j]) > .6 else "black")
    fig.colorbar(image, ax=ax, label="Holistic H (log scale)")
    return save(fig, figures, "05-final-scorecard.png")


def render(payload, data_root, out):
    out, data_root = Path(out), Path(data_root)
    figures = out/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    reference = [render_reference(data_root, figures, family) for family in ("itg-tem", "itg-kbm")]
    rows = good(payload)
    cases, arms = payload["config"]["cases"], payload["config"]["arms"]
    available = [arm for arm in arms if any(row["arm"] == arm for row in rows)]
    seeds = sorted({row["seed"] for row in rows})
    pooled_title = f"Prediction and resolution errors pooled across {len(cases)} surfaces × {len(seeds)} acquisition seeds"
    design_title = f"Branch and design diagnostics pooled across {len(cases)} surfaces × {len(seeds)} acquisition seeds"
    sheets = [(reference[0], "ITG–TEM references"), (reference[1], "ITG–KBM references"),
              (placement(rows, figures, cases, available), "Point placement"),
              (case_curves(rows, figures, cases, available), "Error by case"),
              (metric_sheet(rows, figures, available, METRICS, "04-all-error-diagnostics.png", pooled_title),
               "Prediction and resolution errors"),
              (metric_sheet(rows, figures, available, DESIGN_METRICS, "05-branch-design-diagnostics.png", design_title),
               "Branch and design diagnostics"),
              (scorecard(rows, figures, cases, available), "Final holistic scorecard")]
    failures = sorted({f"{r['arm']}: {r['status']} — {r['reason']}" for r in payload["rows"] if r["status"] != "ok"})
    document = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Matched Ionut 3D benchmark</title>',
                '<style>body{max-width:1700px;margin:28px auto;padding:0 24px;font:16px/1.5 system-ui;background:#f7f9fb;color:#223}img{width:100%;background:white}section{margin:48px 0}pre{white-space:pre-wrap}</style><body>',
                '<h1>Matched 3D ITG–TEM/ITG–KBM benchmark</h1>',
                '<p>All methods share the same paid-point budgets, frozen reference points, and common piecewise-linear tetrahedral reconstruction. Native GP errors remain secondary diagnostics. Curves are medians across three acquisition seeds.</p>']
    for filename, title in sheets:
        document.append(f'<section><h2>{html.escape(title)}</h2><a href="figures/{filename}"><img src="figures/{filename}"></a></section>')
    if failures:
        document.append('<h2>Unavailable or failed arms</h2><pre>'+html.escape("\n".join(failures))+'</pre>')
    document.append('</body></html>')
    (out/"index.html").write_text("\n".join(document), encoding="utf-8")
