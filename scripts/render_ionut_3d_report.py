"""Render reference slice atlases and native-GP score curves for Ionut 3D slices."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from scripts.datasets import load_dataset
from scripts.generate_ionut_slices import values


FAMILIES = ("itg-tem", "itg-kbm")
VARIANTS = (("argmax", "gamma"), ("argmax", "omega"),
            ("softmax", "gamma"), ("softmax", "omega"))
CONFIGS = (("m15", "Matérn 3/2 blend"), ("m05", "Matérn 1/2 blend"))


def case_name(family, mode, output):
    return f"ionut-{family}-3d-{mode}-{output}"


def plane(case, pair, fixed_axis, count=121):
    grid = np.linspace(0., 1., count)
    a, b = np.meshgrid(grid, grid, indexing="xy")
    x = np.full((count * count, 3), .5)
    x[:, pair[0]], x[:, pair[1]] = a.ravel(), b.ravel()
    result = values(case, x)
    return grid, result["y"].reshape(count, count), (result["G"][:, 0] - result["G"][:, 1]).reshape(count, count)


def render_reference(data_root, figures, family):
    pairs = ((0, 1), (0, 2), (1, 2))
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), constrained_layout=True)
    for row, (mode, output) in enumerate(VARIANTS):
        case = case_name(family, mode, output)
        manifest, _, _ = load_dataset(data_root / case / "seed-0")
        columns = manifest["columns"]
        rendered = [plane(case, pair, ({0, 1, 2} - set(pair)).pop()) for pair in pairs]
        lo = min(float(field.min()) for _, field, _ in rendered)
        hi = max(float(field.max()) for _, field, _ in rendered)
        image = None
        for column, (pair, (grid, field, gap)) in enumerate(zip(pairs, rendered)):
            ax = axes[row, column]
            image = ax.pcolormesh(grid, grid, field, shading="auto", cmap="viridis", vmin=lo, vmax=hi)
            if float(gap.min()) <= 0 <= float(gap.max()):
                ax.contour(grid, grid, gap, levels=[0.], colors="white", linewidths=1.2)
            fixed = ({0, 1, 2} - set(pair)).pop()
            ax.set(xlabel=columns[pair[0]], ylabel=columns[pair[1]],
                   title=f"{columns[fixed]} = midpoint")
        axes[row, 0].text(-.34, .5, f"{mode}\n{output}", transform=axes[row, 0].transAxes,
                          ha="center", va="center", fontsize=12, fontweight="bold")
        fig.colorbar(image, ax=axes[row, :], shrink=.72, label=f"{output} proxy")
    label = "ITG–TEM" if family == "itg-tem" else "ITG–KBM"
    fig.suptitle(f"{label} 3D reference — orthogonal midpoint planes (white: equal branch growth)", fontsize=16)
    path = figures / f"01-{family}-reference-slices.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path.name


def load_results(results_root):
    records = {}
    for path in results_root.glob("*.json"):
        payload = json.loads(path.read_text())
        name = Path(payload["dataset"]).parent.name
        tag = "m05" if payload["nu"] == .5 else "m15"
        records[(name, tag)] = payload
    return records


def render_scores(records, figures):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True, constrained_layout=True)
    colors = {"m15": "#126782", "m05": "#e85d04"}
    for family_row, family in enumerate(FAMILIES):
        for variant_column, (mode, output) in enumerate(VARIANTS):
            ax = axes[family_row, variant_column]
            case = case_name(family, mode, output)
            for tag, label in CONFIGS:
                payload = records.get((case, tag))
                if payload:
                    ax.plot([r["n"] for r in payload["rows"]], [r["normalized_rmse"] for r in payload["rows"]],
                            marker="o", color=colors[tag], label=label)
            ax.set(xscale="log", yscale="log", title=f"{mode} {output}", xlabel="Paid evaluations")
            if variant_column == 0:
                ax.set_ylabel(("ITG–TEM" if family == "itg-tem" else "ITG–KBM") + "\nNormalized RMSE")
            ax.grid(True, which="both", alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Native GP prediction error on independent frozen 3D reference points", fontsize=17)
    path = figures / "03-gp-score-curves.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path.name


def final_table(records):
    rows = []
    for family in FAMILIES:
        for mode, output in VARIANTS:
            case = case_name(family, mode, output)
            for tag, label in CONFIGS:
                payload = records.get((case, tag))
                if not payload:
                    continue
                score = payload["rows"][-1]
                rows.append((case, label, score))
    return rows


def render_final_metrics(rows, figures):
    keys = ("normalized_rmse", "nmae", "normalized_p95",
            "transition_normalized_rmse", "high_response_normalized_rmse")
    labels = ("NRMSE", "NMAE", "P95", "Transition NRMSE", "High-response NRMSE")
    matrix = np.asarray([[row[key] for key in keys] for _, _, row in rows], float)
    row_labels = [case.replace("ionut-", "").replace("-3d-", "\n") + " — " + model.replace(" blend", "")
                  for case, model, _ in rows]
    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="magma_r",
                      norm=LogNorm(vmin=max(float(matrix.min()), 1e-4), vmax=float(matrix.max())))
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
           yticks=np.arange(len(row_labels)), yticklabels=row_labels,
           title="Final-budget normalized errors (N=256; lower is better)")
    ax.tick_params(axis="x", rotation=25)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            normalized = image.norm(matrix[i, j])
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if normalized > .58 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=.8, label="Normalized error (log color scale)")
    path = figures / "04-final-metric-scorecard.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/3d"))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    refs = [render_reference(args.data, figures, family) for family in FAMILIES]
    records = load_results(args.results)
    score_figure = render_scores(records, figures)
    table_rows = final_table(records)
    metric_figure = render_final_metrics(table_rows, figures)
    headings = ("Case", "Model", "NRMSE", "NMAE", "P95", "Transition NRMSE", "High-response NRMSE")
    body = []
    for case, model, row in table_rows:
        values_ = [case, model] + [f"{row[key]:.5f}" for key in
                  ("normalized_rmse", "nmae", "normalized_p95", "transition_normalized_rmse", "high_response_normalized_rmse")]
        body.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values_) + "</tr>")
    sections = [(refs[0], "ITG–TEM reference slices"), (refs[1], "ITG–KBM reference slices"),
                (score_figure, "GP error versus paid evaluations"),
                (metric_figure, "Final-budget multi-metric scorecard")]
    document = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Ionut 3D proxy report</title>',
                '<style>body{max-width:1500px;margin:28px auto;padding:0 24px;font:16px/1.5 system-ui;color:#223;background:#f7f9fb}img{width:100%;background:white}section{margin:44px 0}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #ccd;text-align:right}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}</style><body>',
                '<h1>Ionut conditional 3D microinstability proxies</h1>',
                '<p>These are declared 3D slices of the unchanged native 6D phenomenological formulas. White contours mark equal branch growth. Scores use each GP native predictor on independent frozen reference points and are separate from the 2D Delaunay leaderboard.</p>']
    for filename, title in sections:
        document.append(f'<section><h2>{html.escape(title)}</h2><a href="figures/{filename}"><img src="figures/{filename}"></a></section>')
    document.append('<h2>Final-budget scores</h2><table><tr>' + ''.join(f'<th>{h}</th>' for h in headings) + '</tr>' + ''.join(body) + '</table>')
    document.append('</body></html>')
    (args.output / "index.html").write_text("\n".join(document), encoding="utf-8")
    print((args.output / "index.html").resolve())


if __name__ == "__main__":
    main()
