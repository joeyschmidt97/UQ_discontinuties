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


_REPLAYS = {}


def design_for(row):
    """The stored design, or an exact seeded replay checked against the score.

    Replays are cached, because several sheets draw the same design and a GP
    trajectory at the full budget takes minutes to reproduce.
    """
    key = (row["case"], row["seed"], row["arm"], row["noise_peak"])
    if key not in _REPLAYS:
        _REPLAYS[key] = _design_for(row)
        surface, x, observed, reps, replayed = _REPLAYS[key]
        if replayed and REPLAY_CACHE is not None:
            cache = (json.loads(REPLAY_CACHE.read_text(encoding="utf-8"))
                     if REPLAY_CACHE.exists() else {})
            cache["|".join(map(str, key))] = dict(x=x.tolist(), y_observed=observed.tolist(),
                                                  replicates=reps.tolist())
            REPLAY_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return _REPLAYS[key]


# Replayed designs are written beside the results so they are paid for once.
# Entries are keyed by case, seed, arm and noise peak, and are only ever
# written after the replay reproduced the stored placement score.
REPLAY_CACHE = None


def _design_for(row):
    surface = Noisy2DSurface(row["case"], row["seed"], peak=row["noise_peak"])
    if row.get("x") is None and REPLAY_CACHE is not None and REPLAY_CACHE.exists():
        cached = json.loads(REPLAY_CACHE.read_text(encoding="utf-8")).get(
            "|".join(map(str, (row["case"], row["seed"], row["arm"], row["noise_peak"]))))
        if cached is not None:
            row = dict(row, **cached)
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


# At zero noise a noise-aware arm runs its clean twin's algorithm: the spread
# collapses to the absolute floor, so denoising, the heteroscedastic alpha and
# the aleatoric penalty are all constant and change no decision. The zero-noise
# row therefore shows the twin's design rather than an empty panel.
CLEAN_TWIN = {"vwrs-n": "vwrs", "vurs-n": "vurs", "vurs-a": "vurs", "gpr-n": "gpr-u50-g50"}
ARM_ORDER = ("space-filling", "gpr-var", "gpr-u50-g50", "gpr-n", "vwrs", "vwrs-n",
             "vurs", "vurs-n", "vurs-a")


def placement_sheet(rows, case, seed, path, noisy_peak=.20):
    """Point placement in the style of the 2D benchmark's placement sheet.

    Row 1, no noise: grey isocontours of the true surface, teal mode folds,
    points coloured by acquisition order.
    Row 2, noisy, same style: directly comparable with row 1.
    Row 3, noisy, coloured by error: the filled isocontours show the absolute
    spread one evaluation carries there -- red where it is worst, which is high
    response close to a fold -- and each point is coloured on the same scale by
    the error of the value it actually returned.
    """
    final = {(r["arm"], r["noise_peak"]): r for r in rows
             if r["n"] == r["budget"] and r["case"] == case and r["seed"] == seed}
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in final)]
    axis_values = np.linspace(0, 1, GRID)
    gx, gy = np.meshgrid(axis_values, axis_values)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    clean = Noisy2DSurface(case, seed, peak=0.)
    truth = clean.truth(grid).reshape(GRID, GRID)
    region = clean.truth.region(grid).reshape(GRID, GRID)
    low, high = float(truth.min()), float(truth.max())
    scale = high-low
    noisy = Noisy2DSurface(case, seed, peak=noisy_peak)
    spread = (noisy.spread(grid)/scale).reshape(GRID, GRID)

    designs = {}
    for arm in arms:
        for peak, source in ((0., CLEAN_TWIN.get(arm, arm)), (noisy_peak, arm)):
            row = final.get((source, peak))
            if row is not None:
                _, x, observed, reps, _ = design_for(row)
                designs[(arm, peak)] = (x, observed, reps, source != arm, row)
    # The ceiling is the field's own largest one-sigma spread, not the largest
    # returned error: a single Gaussian tail draw otherwise sets the scale and
    # washes the field and nearly every point out to pale. Returned errors above
    # the ceiling are clipped to the darkest red, and a power scale keeps the
    # moderate spread near the fold readable.
    top = float(spread.max())
    norm = matplotlib.colors.PowerNorm(gamma=.6, vmin=0, vmax=top, clip=True)

    labels = ("no noise\nacquisition order",
              f"{noisy_peak:.0%} noise\nacquisition order",
              f"{noisy_peak:.0%} noise\ncoloured by error")
    figure, axes = plt.subplots(3, len(arms), figsize=(2.9*len(arms), 9.6),
                                squeeze=False, layout="constrained")
    order_image = error_image = None
    for j, arm in enumerate(arms):
        for i, label in enumerate(labels):
            axis = axes[i, j]
            peak = 0. if i == 0 else noisy_peak
            axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[0, .5, 1],
                     yticks=[0, .5, 1])
            axis.tick_params(labelsize=7)
            if i < 2:
                axis.contourf(gx, gy, truth, levels=np.linspace(low, high, 18),
                              cmap="Greys", alpha=.30)
            else:
                error_image = axis.contourf(gx, gy, spread, levels=np.linspace(0, top, 17),
                                            cmap="YlOrRd", norm=norm, extend="max")
                axis.contour(gx, gy, truth, levels=np.linspace(low, high, 12),
                             colors="#555555", linewidths=.35, alpha=.6)
            axis.contour(gx, gy, region, levels=np.arange(region.max())+.5,
                         colors="#008a9a", linewidths=1)
            if (arm, peak) not in designs:
                axis.text(.5, .5, "pending", transform=axis.transAxes, ha="center",
                          color="#888888")
            else:
                x, observed, reps, twin, row = designs[(arm, peak)]
                if i < 2:
                    order_image = axis.scatter(x[:, 0], x[:, 1], c=np.linspace(0, 1, len(x)),
                                               cmap="plasma", vmin=0, vmax=1, s=9,
                                               edgecolors="none")
                else:
                    error = np.abs(observed-noisy.truth(x))/scale
                    axis.scatter(x[:, 0], x[:, 1], c=error, cmap="YlOrRd", norm=norm, s=13,
                                 edgecolors="black", linewidths=.35)
                repeated = reps > 1
                if repeated.any():
                    axis.scatter(x[repeated, 0], x[repeated, 1], s=36, facecolors="none",
                                 edgecolors="#00c853", linewidths=1.1)
                note = f"RMS {row['placement_error']:.4f}"
                if twin:
                    note += f"\n= {CLEAN_TWIN[arm]} at 0%"
                axis.text(.03, .97, note, transform=axis.transAxes, va="top", fontsize=7.5,
                          bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=1.5))
            if i == 0:
                axis.set_title(arm, fontsize=11, weight="bold")
            if j == 0:
                axis.set_ylabel(label, fontsize=9)
    if order_image is not None:
        figure.colorbar(order_image, ax=axes[:2, :].ravel().tolist(), shrink=.7, pad=.01,
                        label="sample order: early (dark) to late (yellow)")
    if error_image is not None:
        figure.colorbar(error_image, ax=axes[2, :].ravel().tolist(), shrink=.9, pad=.01,
                        label="1σ spread (field), returned error (points)\n/ response range")
    figure.suptitle(f"Where every method puts its points — {case}, seed {seed}.  "
                    "Teal: mode folds.  Green rings: replicated points.  "
                    "RMS is the placement score.", fontsize=12)
    figure.savefig(path, dpi=115)
    plt.close(figure)


def _replay_worker(row):
    return row, _design_for(row)


def prefetch_designs(rows, case, seed, workers=4):
    """Replay every uncached design for one case and seed in parallel.

    Each replay is a full trajectory; done serially at the full budget they take
    minutes apiece. Results go through the same checked cache as design_for.
    """
    from concurrent.futures import ProcessPoolExecutor
    cache = (json.loads(REPLAY_CACHE.read_text(encoding="utf-8"))
             if REPLAY_CACHE is not None and REPLAY_CACHE.exists() else {})
    needed = [r for r in rows if r["n"] == r["budget"] and r["case"] == case
              and r["seed"] == seed and r.get("x") is None
              and "|".join(map(str, (r["case"], r["seed"], r["arm"], r["noise_peak"]))) not in cache]
    if not needed:
        return 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row, result in pool.map(_replay_worker, needed):
            key = (row["case"], row["seed"], row["arm"], row["noise_peak"])
            _REPLAYS[key] = result
            _, x, observed, reps, _ = result
            cache["|".join(map(str, key))] = dict(x=x.tolist(), y_observed=observed.tolist(),
                                                  replicates=reps.tolist())
    if REPLAY_CACHE is not None:
        REPLAY_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return len(needed)


def noise_ladder_sheets(rows, case, seed, order_path, error_path):
    """Placement at every noise level, in the 2D benchmark's style.

    Sheet 1: one row per noise level, grey isocontours of the true surface,
    teal folds, points coloured by acquisition order.
    Sheet 2: one row per noisy level, the filled isocontours showing that
    level's one-sigma spread and each point coloured by the error of the value
    it returned. Every row shares one colour scale, set by the largest level's
    spread, so the growth of the noise from row to row is read off directly
    rather than renormalized away.
    """
    final = {(r["arm"], r["noise_peak"]): r for r in rows
             if r["n"] == r["budget"] and r["case"] == case and r["seed"] == seed}
    peaks = sorted({k[1] for k in final})
    noisy_peaks = [p for p in peaks if p > 0]
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in final)]
    axis_values = np.linspace(0, 1, GRID)
    gx, gy = np.meshgrid(axis_values, axis_values)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    clean = Noisy2DSurface(case, seed, peak=0.)
    truth = clean.truth(grid).reshape(GRID, GRID)
    region = clean.truth.region(grid).reshape(GRID, GRID)
    low, high = float(truth.min()), float(truth.max())
    scale = high-low

    def design(arm, peak):
        source = CLEAN_TWIN.get(arm, arm) if peak == 0 else arm
        row = final.get((source, peak))
        if row is None:
            return None
        _, x, observed, reps, _ = design_for(row)
        return x, observed, reps, source != arm, row

    def frame(axis):
        axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[0, .5, 1], yticks=[0, .5, 1])
        axis.tick_params(labelsize=7)

    def annotate(axis, reps, x, row, twin, arm):
        repeated = reps > 1
        if repeated.any():
            axis.scatter(x[repeated, 0], x[repeated, 1], s=36, facecolors="none",
                         edgecolors="#00c853", linewidths=1.1)
        note = f"RMS {row['placement_error']:.4f}"
        if twin:
            note += f"\n= {CLEAN_TWIN[arm]} at 0%"
        axis.text(.03, .97, note, transform=axis.transAxes, va="top", fontsize=7.5,
                  bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=1.5))

    figure, axes = plt.subplots(len(peaks), len(arms), figsize=(2.9*len(arms), 3.05*len(peaks)+.6),
                                squeeze=False, layout="constrained")
    image = None
    for i, peak in enumerate(peaks):
        for j, arm in enumerate(arms):
            axis = axes[i, j]
            frame(axis)
            axis.contourf(gx, gy, truth, levels=np.linspace(low, high, 18), cmap="Greys", alpha=.30)
            axis.contour(gx, gy, region, levels=np.arange(region.max())+.5, colors="#008a9a",
                         linewidths=1)
            found = design(arm, peak)
            if found is None:
                axis.text(.5, .5, "pending", transform=axis.transAxes, ha="center", color="#888888")
            else:
                x, observed, reps, twin, row = found
                image = axis.scatter(x[:, 0], x[:, 1], c=np.linspace(0, 1, len(x)), cmap="plasma",
                                     vmin=0, vmax=1, s=9, edgecolors="none")
                annotate(axis, reps, x, row, twin, arm)
            if i == 0:
                axis.set_title(arm, fontsize=11, weight="bold")
            if j == 0:
                axis.set_ylabel(f"{peak:.0%} noise", fontsize=10)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.6, pad=.01,
                        label="sample order: early (dark) to late (yellow)")
    figure.suptitle(f"Placement at every noise level — {case}, seed {seed}, acquisition order.  "
                    "Teal: mode folds.  RMS is the placement score.", fontsize=12)
    figure.savefig(order_path, dpi=110)
    plt.close(figure)

    if not noisy_peaks:
        return
    fields = {p: (Noisy2DSurface(case, seed, peak=p).spread(grid)/scale).reshape(GRID, GRID)
              for p in noisy_peaks}
    top = max(float(f.max()) for f in fields.values())
    norm = matplotlib.colors.PowerNorm(gamma=.6, vmin=0, vmax=top, clip=True)
    figure, axes = plt.subplots(len(noisy_peaks), len(arms),
                                figsize=(2.9*len(arms), 3.05*len(noisy_peaks)+.6),
                                squeeze=False, layout="constrained")
    image = None
    for i, peak in enumerate(noisy_peaks):
        truth_noisy = Noisy2DSurface(case, seed, peak=peak).truth
        for j, arm in enumerate(arms):
            axis = axes[i, j]
            frame(axis)
            image = axis.contourf(gx, gy, fields[peak], levels=np.linspace(0, top, 17),
                                  cmap="YlOrRd", norm=norm, extend="max")
            axis.contour(gx, gy, truth, levels=np.linspace(low, high, 12), colors="#555555",
                         linewidths=.35, alpha=.6)
            axis.contour(gx, gy, region, levels=np.arange(region.max())+.5, colors="#008a9a",
                         linewidths=1)
            found = design(arm, peak)
            if found is None:
                axis.text(.5, .5, "pending", transform=axis.transAxes, ha="center", color="#888888")
            else:
                x, observed, reps, twin, row = found
                axis.scatter(x[:, 0], x[:, 1], c=np.abs(observed-truth_noisy(x))/scale,
                             cmap="YlOrRd", norm=norm, s=13, edgecolors="black", linewidths=.35)
                annotate(axis, reps, x, row, twin, arm)
            if i == 0:
                axis.set_title(arm, fontsize=11, weight="bold")
            if j == 0:
                axis.set_ylabel(f"{peak:.0%} noise", fontsize=10)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.7, pad=.01,
                    label="1σ spread (field), returned error (points)\n/ response range")
    figure.suptitle(f"Placement at every noise level — {case}, seed {seed}, coloured by error "
                    "(one shared scale across rows).  Teal: mode folds.", fontsize=12)
    figure.savefig(error_path, dpi=110)
    plt.close(figure)


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
    global REPLAY_CACHE
    REPLAY_CACHE = args.run/"replayed-designs.json"

    convergence_sheet(rows, figures/"01-error-vs-budget.png")
    noise_field_sheet(args.map_case, args.map_seed, figures/"02-surface-and-noise-field.png")
    prefetch_designs(rows, args.map_case, args.map_seed)
    placement_sheet(rows, args.map_case, args.map_seed, figures/"03-point-placement.png")
    noise_ladder_sheets(rows, args.map_case, args.map_seed,
                        figures/"06-placement-by-noise-order.png",
                        figures/"07-placement-by-noise-error.png")
    replayed = map_sheet(rows, args.map_case, args.map_seed, figures/"04-placement-error-map.png")
    map_sheet(rows, args.map_case, args.map_seed, figures/"05-end-to-end-error-map.png",
              mode="end_to_end")
    peaks, table = final_table(rows)
    sheets = [("01-error-vs-budget.png", "Error against budget, by noise level"),
              ("02-surface-and-noise-field.png", "Surface and noise field"),
              ("03-point-placement.png", "Where every method puts its points"),
              ("04-placement-error-map.png",
               "Reconstruction error with sampled points (placement)"),
              ("05-end-to-end-error-map.png",
               "Reconstruction error with sampled points (end-to-end)"),
              ("06-placement-by-noise-order.png",
               "Placement at every noise level — acquisition order"),
              ("07-placement-by-noise-error.png",
               "Placement at every noise level — coloured by error, shared scale")]
    html(payload, rows, sheets, peaks, table, args.run/"index.html")
    print(f"wrote {(args.run/'index.html').resolve()} ({replayed or 0} map designs replayed)")


if __name__ == "__main__":
    main()
