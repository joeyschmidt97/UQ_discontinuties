"""Render the 2D noise sweep: error against noise, and error against placement.

    python -m scripts.render_noise_2d --run results/noise-2d-2026-09-21

Sheets:
  01  placement error against N per arm, one line per noise level, with the
      end-to-end error dashed beside it -- how much noise costs and where.
  02  final-budget table of both scores per arm and noise level.
  03+ placement maps for one case and seed. Each panel colours the surface by
      the absolute reconstruction error (log scale, shared across panels) and
      overlays the sampled points, so a region of high error can be read off
      against where the sampler did and did not look. Fold lines are drawn in
      white and replicated points are ringed.

The maps use the *placement* reconstruction -- true values at the sampled
points -- so colour differences between noise levels are caused by where the
sampler went, not by noisy values. A matching end-to-end map sheet shows what
the noisy values then add on top.

Trajectories recorded before designs were stored are replayed from their seed.
Both the surface noise and every arm are seeded, so a replay reproduces the
original design exactly; the replayed score is checked against the stored one
and the map is refused if they disagree.
"""
import argparse
import json
import math
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from threadpoolctl import threadpool_limits

from benchmark2d.core import evaluation_set, reconstruct, rmse
from benchmarknd.noisy import NoisyObservations
from benchmarknd.strategies import run_arm
from scripts.run_noise_2d import CORNERS, Noisy2DSurface

GRID = 161
MAP_PEAKS = (0., .20)


def pooled(values):
    values = [v for v in values if v is not None]
    return math.sqrt(sum(v*v for v in values)/len(values)) if values else None


def style(index):
    return plt.get_cmap("viridis")(index)


def convergence_sheet(rows, path):
    arms = sorted({r["arm"] for r in rows})
    peaks = sorted({r["noise_peak"] for r in rows})
    columns = 3
    figure, axes = plt.subplots(math.ceil(len(arms)/columns), columns,
                                figsize=(14, 3.6*math.ceil(len(arms)/columns)),
                                squeeze=False, sharex=True, sharey=True)
    for axis, arm in zip(axes.ravel(), arms):
        for peak in peaks:
            color = style(peak/max(max(peaks), 1e-12))
            for key, line in (("placement_error", "-"), ("end_to_end_error", "--")):
                series = defaultdict(list)
                for r in rows:
                    if r["arm"] == arm and r["noise_peak"] == peak:
                        series[r["n"]].append(r[key])
                if not series:
                    continue
                n = sorted(series)
                axis.plot(n, [pooled(series[k]) for k in n], line, color=color,
                          linewidth=1.6 if line == "-" else 1.1,
                          label=(f"{peak:.0%} peak" if key == "placement_error" else None))
        axis.set_title(arm)
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.grid(alpha=.25, which="both")
    for axis in axes.ravel()[len(arms):]:
        axis.axis("off")
    for axis in axes[:, 0]:
        axis.set_ylabel("normalized RMS")
    for axis in axes[-1]:
        axis.set_xlabel("paid evaluations $N$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                  title="solid: placement (true values at sampled points)   "
                        "dashed: end-to-end (noisy values)")
    figure.suptitle("2D noise sweep — error against budget, cases and seeds pooled (RMS)")
    figure.tight_layout(rect=(0, .07, 1, .97))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def final_table(rows):
    final = [r for r in rows if r["n"] == r["budget"]]
    arms = sorted({r["arm"] for r in final})
    peaks = sorted({r["noise_peak"] for r in final})
    table = []
    for arm in arms:
        cells = {}
        for peak in peaks:
            subset = [r for r in final if r["arm"] == arm and r["noise_peak"] == peak]
            if subset:
                cells[peak] = (pooled([r["placement_error"] for r in subset]),
                               pooled([r["end_to_end_error"] for r in subset]), len(subset))
        table.append((arm, cells))
    return peaks, table


def design_for(row):
    """The stored design, or an exact seeded replay checked against the score."""
    surface = Noisy2DSurface(row["case"], row["seed"], peak=row["noise_peak"])
    if row.get("x") is not None:
        return surface, np.asarray(row["x"]), np.asarray(row["y_observed"]), \
            np.asarray(row["replicates"]), False
    obs = NoisyObservations(surface, row["budget"], 2, row["seed"], shared=CORNERS)
    with threadpool_limits(limits=1):
        run_arm(row["arm"], obs, row["seed"])
    test = evaluation_set(surface.truth)
    replayed = rmse(reconstruct(obs.x, surface.truth(obs.x), test[0]), test[1], test[2])
    if not math.isclose(replayed, row["placement_error"], rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError(f"replay of {row['arm']} does not reproduce the stored score "
                           f"({replayed:.6g} vs {row['placement_error']:.6g})")
    return surface, obs.x, obs.y, obs.replicates, True


def map_sheet(rows, case, seed, path, mode="placement"):
    here = [r for r in rows if r["n"] == r["budget"] and r["case"] == case and r["seed"] == seed]
    available = sorted({r["noise_peak"] for r in here})
    # The cleanest and the noisiest level finished so far, so a partial run
    # still shows the contrast the maps exist to show.
    wanted = (available[0], available[-1]) if available else ()
    final = [r for r in here if r["noise_peak"] in wanted]
    arms = sorted({r["arm"] for r in final})
    if not arms:
        return None
    axis_values = np.linspace(0, 1, GRID)
    gx, gy = np.meshgrid(axis_values, axis_values)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    panels, errors, replayed = {}, [], 0
    for row in final:
        surface, design, observed, reps, was_replayed = design_for(row)
        replayed += was_replayed
        truth = surface.truth(grid)
        scale = float(np.ptp(truth))
        values = surface.truth(design) if mode == "placement" else observed
        error = np.abs(reconstruct(design, values, grid)-truth)/scale
        panels[(row["arm"], row["noise_peak"])] = (error.reshape(GRID, GRID), design, reps,
                                                   surface.truth.region(grid).reshape(GRID, GRID),
                                                   row[f"{mode}_error"])
        errors.append(error)
    everything = np.concatenate(errors)
    norm = LogNorm(vmin=max(float(np.quantile(everything, .02)), 1e-5),
                   vmax=float(np.quantile(everything, .998)))

    peaks = sorted({k[1] for k in panels})
    figure, axes = plt.subplots(len(peaks), len(arms), figsize=(2.6*len(arms)+1.2, 2.8*len(peaks)+.9),
                                squeeze=False)
    image = None
    for i, peak in enumerate(peaks):
        for j, arm in enumerate(arms):
            axis = axes[i, j]
            axis.set_xticks([]); axis.set_yticks([])
            if (arm, peak) not in panels:
                axis.text(.5, .5, "not run\n(clean twin)" if peak == 0 else "pending",
                          ha="center", va="center", transform=axis.transAxes, color="#888")
                axis.set_facecolor("#f2f2f2")
                if i == 0:
                    axis.set_title(arm, fontsize=9)
                continue
            error, design, reps, region, score = panels[(arm, peak)]
            image = axis.imshow(error, origin="lower", extent=(0, 1, 0, 1), cmap="magma",
                                norm=norm, interpolation="nearest")
            axis.contour(axis_values, axis_values, region, levels=np.arange(region.max())+.5,
                         colors="white", linewidths=.8, alpha=.8)
            single = reps <= 1
            axis.scatter(design[single, 0], design[single, 1], s=5, c="#39d2ff",
                         edgecolors="black", linewidths=.25)
            if (~single).any():
                axis.scatter(design[~single, 0], design[~single, 1], s=28, facecolors="none",
                             edgecolors="#39ff88", linewidths=1.)
            axis.set_title(f"{arm}\n{peak:.0%} peak — RMS {score:.4f}", fontsize=8.5)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.85, pad=.01,
                    label="|reconstruction − truth| / range  (log)")
    label = ("placement reconstruction — true values at the sampled points"
             if mode == "placement" else "end-to-end reconstruction — the noisy values sampled")
    figure.suptitle(f"{case}, seed {seed}: error surface and sampled points\n{label}", fontsize=11)
    figure.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(figure)
    return replayed


def noise_field_sheet(case, seed, path):
    """What the sampler is up against: the surface, and the spread it sees."""
    axis_values = np.linspace(0, 1, GRID)
    gx, gy = np.meshgrid(axis_values, axis_values)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    surface = Noisy2DSurface(case, seed, peak=.20)
    truth = surface.truth(grid).reshape(GRID, GRID)
    relative = ((surface.spread(grid)-1e-6)/np.maximum(np.abs(surface.truth(grid)), 1e-9)
                ).reshape(GRID, GRID)
    region = surface.truth.region(grid).reshape(GRID, GRID)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for axis, field, title, cmap in ((axes[0], truth, "true response", "viridis"),
                                     (axes[1], relative, "relative spread at a 20% peak", "cividis")):
        image = axis.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap=cmap)
        axis.contour(axis_values, axis_values, region, levels=np.arange(region.max())+.5,
                     colors="white", linewidths=.9)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=.85)
    figure.suptitle(f"{case}, seed {seed} — surface and noise field (white: mode folds)")
    figure.tight_layout()
    figure.savefig(path, dpi=125)
    plt.close(figure)


def html(payload, rows, sheets, peaks, table, path):
    expected = None
    config = payload.get("config", {})
    finished = len({(r["case"], r["seed"], r["arm"], r["noise_peak"]) for r in rows
                    if r["n"] == r["budget"]})
    head = "".join(f"<th colspan=2>{p:.0%} peak</th>" for p in peaks)
    sub = "".join("<th>placement</th><th>end-to-end</th>" for _ in peaks)
    body = ""
    for arm, cells in table:
        body += f"<tr><td>{arm}</td>"
        for p in peaks:
            if p in cells:
                a, b, n = cells[p]
                body += f"<td>{a:.4f}</td><td class='e2e'>{b:.4f}</td>"
            else:
                body += "<td>—</td><td>—</td>"
        body += "</tr>"
    images = "".join(f"<h2>{caption}</h2><img src='figures/{name}'>" for name, caption in sheets)
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>2D noise sweep</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1300px;color:#111;padding:0 16px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{border:1px solid #ccc;padding:.3rem .55rem;text-align:right}}
td:first-child{{text-align:left}}.e2e{{color:#777}}img{{max-width:100%;border:1px solid #ddd}}
.warn{{background:#fff4e5;padding:.6rem .9rem;border-left:3px solid #e8a33d}}</style>
<h1>2D noise sweep</h1>
<p class="warn">Partial run: {finished} completed trajectories. Commit <code>{str(payload.get('commit','?'))[:10]}</code>.</p>
<p>The sampler sees values carrying a relative spread that rises from a 1% floor to the stated peak at the mode folds.
<b>Placement</b> reconstructs from the true response at the sampled points, isolating where the sampler looked.
<b>End-to-end</b> (grey) reconstructs from the noisy values it actually saw. Both are graded on the noiseless surface
with the common Delaunay evaluator. Values are pooled RMS over cases and seeds at the final budget.</p>
<table><tr><th rowspan=2>arm</th>{head}</tr><tr>{sub}</tr>{body}</table>
{images}""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--map-case", default="two-plane-four-peaks")
    parser.add_argument("--map-seed", type=int, default=0)
    args = parser.parse_args()
    payload = json.loads((args.run/"results.json").read_text(encoding="utf-8"))
    rows = [r for r in payload["rows"] if r["status"] == "ok"]
    figures = args.run/"figures"
    figures.mkdir(exist_ok=True)

    convergence_sheet(rows, figures/"01-error-vs-budget.png")
    noise_field_sheet(args.map_case, args.map_seed, figures/"02-surface-and-noise-field.png")
    replayed = map_sheet(rows, args.map_case, args.map_seed, figures/"03-placement-error-map.png")
    map_sheet(rows, args.map_case, args.map_seed, figures/"04-end-to-end-error-map.png",
              mode="end_to_end")
    peaks, table = final_table(rows)
    sheets = [("01-error-vs-budget.png", "Error against budget, by noise level"),
              ("02-surface-and-noise-field.png", "Surface and noise field"),
              ("03-placement-error-map.png", "Placement: error surface with sampled points"),
              ("04-end-to-end-error-map.png", "End-to-end: error surface with sampled points")]
    html(payload, rows, sheets, peaks, table, args.run/"index.html")
    print(f"wrote {(args.run/'index.html').resolve()} ({replayed or 0} map designs replayed)")


if __name__ == "__main__":
    main()
