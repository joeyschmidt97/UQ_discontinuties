"""Static scientific figures plus a self-contained HTML comparison report."""
import html
import hashlib
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .core import Surface, reconstruct

COLORS = dict(grid="#758398", moe="#343a40", sglib="#b07813", sgpp="#9b59b6",
              **{"gpr-var": "#007c91", "gpr-grad": "#dd5f42", "triangles": "#318448", "gpr-blend": "#e7298a",
                 "moe-tri75": "#3b4cc0", "moe-tri50": "#a51c30",
                 # Uncertainty/gradient ramp: warm at a gradient-dominated mix,
                 # cool as the uncertainty share grows.
                 "gpr-u20-g80": "#d94801", "gpr-u30-g70": "#fdae61", "gpr-u50-g50": "#e7298a",
                 "gpr-u70-g30": "#6a3d9a", "gpr-u80-g20": "#1f78b4"})


def curve(ax, rows, metric, color, label, expected_seeds=None, **kw):
    # Group by requested cap; plot actual measured cost, not nominal budget.
    points = []
    for budget in sorted({r["budget"] for r in rows}):
        group = [r for r in rows if r["budget"] == budget]
        if expected_seeds is not None and {r["seed"] for r in group} != set(expected_seeds):
            continue
        points.append((np.median([r["n"] for r in group]),
                       np.median([r[metric] for r in group]),
                       np.quantile([r[metric] for r in group], [.25, .75])))
    if not points:
        return
    points.sort(key=lambda p: p[0])
    x, y, spread = zip(*points)
    ax.plot(x, y, "-", color=color, label=label, markersize=2, **kw)
    ax.fill_between(x, np.array(spread)[:, 0], np.array(spread)[:, 1], color=color, alpha=.10)


def format_axis(ax, epsilon):
    ax.axhline(epsilon, color="#666666", ls=":", lw=1, label=f"global target {epsilon:g}")
    ax.set(xlabel="Unique evaluations (corners included)", ylabel="RMS / fixed truth range", yscale="log", xscale="log")
    ax.grid(alpha=.2, which="both")
    ax.spines[["right", "top"]].set_visible(False)


def qualifying_cost(rows, epsilon, band_epsilon):
    # Measured checkpoint only; all subsequent tested point counts must pass.
    ordered = sorted(rows, key=lambda r: r["budget"])
    for i, row in enumerate(ordered):
        if all(r["status"] == "ok" and r["error"] <= epsilon and r["band_error"] <= band_epsilon for r in ordered[i:]):
            return row["n"]
    return None


def aggregate_errors(payload):
    """Pool normalized squared errors, equally weighting every case/seed.

    Only point counts complete for the entire configured field are comparable.
    Regional errors are not added: their evaluation sets overlap global error.
    """
    cfg = payload["config"]
    expected = {(case, seed) for case in cfg["cases"] for seed in cfg["seeds"]}
    groups = {}
    for row in payload["rows"]:
        if row["arm"] not in cfg["arms"] or (row["case"], row["seed"]) not in expected:
            continue
        if row["n"] != row["budget"] or row["status"] != "ok":
            continue
        key = (row["arm"], row["n"])
        group = groups.setdefault(key, {})
        identity = (row["case"], row["seed"])
        if identity in group:
            raise ValueError("duplicate case/seed at the same point count")
        error = float(row["error"])
        if not np.isfinite(error) or error < 0:
            raise ValueError("combined scores require finite nonnegative errors")
        group[identity] = error
    counts = sorted({n for arm, n in groups})
    common = [n for n in counts if all(set(groups.get((arm, n), {})) == expected for arm in cfg["arms"])]
    return {arm: [dict(n=n, error=float(np.sqrt(np.mean(np.square(list(groups[arm, n].values()))))),
                       tests=len(expected)) for n in common] for arm in cfg["arms"]}


def render(payload, out):
    """Five consolidated sheets. Acquisition/scoring results remain untouched."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    cfg, rows = payload["config"], payload["rows"]
    cases, arms = cfg["cases"], cfg["arms"]
    seed, cap = min(cfg["seeds"]), max(cfg["budgets"])
    ok = [r for r in rows if r["status"] == "ok"]
    epsilon, band_epsilon = cfg["epsilon"], cfg["band_epsilon"]
    names = {"grid": "Progressive grid", "moe": "Mixture of experts", "sglib": "Ionut / sg_lib",
             "sgpp": "SG++", "gpr-var": "GP uncertainty", "gpr-grad": "GP gradient", "triangles": "Triangles", "gpr-blend": "GP grad/unc 50/50",
             "moe-tri75": "MoE/tri 75/25", "moe-tri50": "MoE/tri 50/50",
             "gpr-u20-g80": "GP unc/grad 20/80", "gpr-u30-g70": "GP unc/grad 30/70",
             "gpr-u50-g50": "GP unc/grad 50/50", "gpr-u70-g30": "GP unc/grad 70/30",
             "gpr-u80-g20": "GP unc/grad 80/20"}
    a, b = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    query = np.column_stack([a.ravel(), b.ravel()])
    surfaces = {case: Surface(case, seed) for case in cases}
    truth = {case: surfaces[case](query).reshape(a.shape) for case in cases}
    chosen = {(r["case"], r["arm"]): r for r in ok if r["seed"] == seed and r["budget"] == cap}
    sheets = []

    def save_figure(fig, filename, title, caption):
        fig.savefig(figures / filename, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=.25)
        plt.close(fig)
        sheets.append((filename, title, caption))

    # 1: Reference geometry, shared elevation and color scale.
    low = min(float(z.min()) for z in truth.values())
    high = max(float(z.max()) for z in truth.values())
    fig = plt.figure(figsize=(5*len(cases), 5.8), layout="constrained")
    axes = []
    for col, case in enumerate(cases):
        ax = fig.add_subplot(1, len(cases), col+1, projection="3d")
        axes.append(ax)
        ax.plot_surface(a, b, truth[case], cmap="viridis", vmin=low, vmax=high,
                        linewidth=0, rcount=101, ccount=101, antialiased=True)
        ax.set(title=case, xlabel="Parameter 1", ylabel="Parameter 2", zlabel="Growth-rate proxy",
               xlim=(0, 1), ylim=(0, 1), zlim=(low, high))
        ax.view_init(35, -20)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"01  True 3-D manifolds | geometry seed {seed} | shared height and color scales", fontsize=17)
    fig.colorbar(ScalarMappable(Normalize(low, high), "viridis"), ax=axes,
                 location="bottom", shrink=.35, aspect=55, label="True growth-rate proxy")
    save_figure(fig, "01-manifold-reference.png", "1. Start with the true manifolds",
                "Exact synthetic surfaces, before any method samples them. All views share the same camera, height range and color scale. All peaks lie inside mode regions, away from continuous folds. Peak counts are 1, 4, 3 and 3.")

    # 2: All strategies x all cases in one placement sheet.
    fig, axes = plt.subplots(len(cases), len(arms), figsize=(3.25*len(arms), 3.15*len(cases)),
                             squeeze=False, layout="constrained")
    for i, case in enumerate(cases):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            ax.contourf(a, b, truth[case], levels=np.linspace(low, high, 18), cmap="Greys", alpha=.30)
            if case != "smooth":
                ax.contour(a, b, surfaces[case].region(query).reshape(a.shape), levels=np.arange(len(surfaces[case].normals)-1)+.5, colors="#008a9a", linewidths=1)
            row = chosen.get((case, arm))
            if row:
                x = np.array(row["x"])
                ax.scatter(x[:, 0], x[:, 1], c=np.linspace(0, 1, len(x)), cmap="plasma",
                           vmin=0, vmax=1, s=10, edgecolors="none")
                ax.text(.03, .97, f"N = {row['n']}", transform=ax.transAxes, va="top", fontsize=9,
                        bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
            else:
                ax.text(.5, .5, "Unavailable / failed", transform=ax.transAxes, ha="center")
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(names[arm], color=COLORS[arm], fontsize=12, weight="bold")
            if j == 0:
                ax.set_ylabel(case+"\nParameter 2", fontsize=11)
            if i == len(cases)-1:
                ax.set_xlabel("Parameter 1")
    fig.suptitle(f"02  Where every method puts its points | seed {seed} | exactly {cap} paid points per method", fontsize=17)
    fig.colorbar(ScalarMappable(Normalize(0, 1), "plasma"), ax=axes.ravel().tolist(),
                 location="bottom", shrink=.40, aspect=70, label="Sample order: early (dark) to late (yellow)")
    save_figure(fig, "02-point-placement.png", "2. Compare point placement across all methods",
                "Rows are test surfaces; columns are methods. Gray shading shows truth, teal marks the mode-switch boundary, and dots show sampled locations. These are final-cap snapshots of one paired seed, not averages. N includes the four shared corner evaluations; every method uses exactly the same number of points.")

    # 3: All-method convergence curves; split global and boundary scores.
    fig, axes = plt.subplots(2, len(cases), figsize=(4.6*len(cases), 8.5), squeeze=False)
    for col, case in enumerate(cases):
        case_rows = [r for r in ok if r["case"] == case]
        for i, metric in enumerate(("error", "band_error")):
            ax = axes[i, col]
            for arm in arms:
                curve(ax, [r for r in case_rows if r["arm"] == arm], metric, COLORS[arm], names[arm], expected_seeds=cfg["seeds"], lw=1.6)
            target = epsilon if i == 0 else band_epsilon
            ax.axhline(target, color="#333333", ls="--", lw=1)
            ax.set(xscale="log", yscale="log", xlim=(4, max(cfg["budgets"])*1.15), ylim=(1e-3, 1),
                   title=case if i == 0 else "", xlabel="Actual evaluations" if i == 1 else "")
            if col == 0:
                ax.set_ylabel(("Global RMS" if i == 0 else "Boundary-band RMS")+" / fixed truth range")
            ax.grid(alpha=.20, which="both")
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(arms)), frameon=False, fontsize=10)
    fig.suptitle(f"03  Error versus points | all {len(cfg['seeds'])} paired seeds | targets: global {epsilon:g}, boundary {band_epsilon:g}", fontsize=17)
    fig.tight_layout(rect=(0, .065, 1, .94))
    save_figure(fig, "03-error-versus-points.png", "3. Judge accuracy per evaluation",
                "Each panel overlays all methods. Top: global RMS; bottom: boundary-band RMS. Lines are medians and shading is the interquartile spread across paired seeds/geometries, not confidence intervals. Both axes use identical limits in all panels. Curves require the complete paired seed set at each N. Dashed lines show the qualification targets. Lower and further left is better; placement alone is not a score.")

    # 4: Matching residual sheet, one shared normalized error scale.
    fig, axes = plt.subplots(len(cases), len(arms), figsize=(3.25*len(arms), 3.15*len(cases)),
                             squeeze=False, layout="constrained")
    vmax = .5
    for i, case in enumerate(cases):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            row = chosen.get((case, arm))
            if row:
                residual = np.abs(reconstruct(row["x"], row["y"], query)-truth[case].ravel()).reshape(a.shape)/row["scale"]
                ax.pcolormesh(a, b, residual, cmap="magma", vmin=0, vmax=vmax, shading="auto")
            else:
                ax.text(.5, .5, "Unavailable / failed", transform=ax.transAxes, ha="center")
            if case != "smooth":
                ax.contour(a, b, surfaces[case].region(query).reshape(a.shape), levels=np.arange(len(surfaces[case].normals)-1)+.5, colors="cyan", linewidths=.7)
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(names[arm], color=COLORS[arm], fontsize=12, weight="bold")
            if j == 0:
                ax.set_ylabel(case+"\nParameter 2", fontsize=11)
            if i == len(cases)-1:
                ax.set_xlabel("Parameter 1")
    fig.suptitle(f"04  What each design misses | common low-poly reconstruction | seed {seed}, cap {cap}", fontsize=17)
    fig.colorbar(ScalarMappable(Normalize(0, vmax), "magma"), ax=axes.ravel().tolist(),
                 location="bottom", shrink=.4, aspect=70, extend="max",
                 label="Absolute reconstruction error / fixed truth range (brightest color includes values above 0.5)")
    save_figure(fig, "04-reconstruction-error-map.png", "4. Diagnose missed peaks and boundaries",
                "The same row/column layout as the placement sheet, using the same seed and cap. Dark is accurate; bright is inaccurate. A single normalized color scale makes methods comparable. Values above 0.5 saturate at the brightest color. Cyan marks the true boundary.")

    # 5: Compact performance table with the same predeclared qualification rule.
    cells, colors = [], []
    qualifies_everywhere = dict.fromkeys(arms, True)
    table_html = []
    for case in cases:
        line, colorline, hline = [], [], []
        for arm in arms:
            group = [r for r in rows if r["case"] == case and r["arm"] == arm]
            costs = [qualifying_cost([r for r in group if r["seed"] == s], epsilon, band_epsilon) for s in cfg["seeds"]]
            passing = [v for v in costs if v is not None]
            all_pass = len(passing) == len(cfg["seeds"])
            qualifies_everywhere[arm] &= all_pass
            cost = f"N = {np.median(passing):g}" if all_pass else "not qualified"
            label = f"{len(passing)}/{len(costs)} seeds\n{cost}"
            line.append(label)
            colorline.append("#dcefe4" if all_pass else "#f9e6df")
            hline.append(f"<td>{html.escape(label).replace(chr(10), '<br>')}</td>")
        cells.append(line)
        colors.append(colorline)
        table_html.append("<tr><th>"+html.escape(case)+"</th>"+"".join(hline)+"</tr>")
    aggregate = aggregate_errors(payload)
    aggregate_payload = dict(formula="sqrt(mean((per-test RMS / per-test truth range)^2))",
                             weighting="equal weight per case and seed; global metric only",
                             cases=cases, seeds=cfg["seeds"], curves=aggregate)
    (out/"aggregate-scores.json").write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12.5, 7.8), layout="constrained")
    ordered_arms = sorted(arms, key=lambda arm: aggregate[arm][-1]["error"] if aggregate[arm] else np.inf)
    for arm in ordered_arms:
        points = aggregate[arm]
        if not points:
            continue
        x = [p["n"] for p in points]
        y = [p["error"] for p in points]
        ax.plot(x, y, color=COLORS[arm], lw=2.5 if arm == "moe" else 1.8,
                label=f"{names[arm]}  ({y[-1]:.4f})")
        selected = [p for p in points if p["n"] in cfg["budgets"]]
        ax.scatter([p["n"] for p in selected], [p["error"] for p in selected],
                   color=COLORS[arm], s=24, zorder=3)
    ax.axhline(epsilon, color="#777777", ls=":", lw=1)
    ax.set(xscale="log", yscale="log", xlim=(4, cap*1.12),
           xlabel="Paid points per test (four common corners included)",
           ylabel="Combined normalized RMS across all test cases and seeds")
    ax.grid(alpha=.2, which="both")
    ax.spines[["right", "top"]].set_visible(False)
    if any(aggregate.values()):
        last_n = next(points[-1]["n"] for points in aggregate.values() if points)
        ax.legend(title=f"Methods ordered by error at N = {last_n}", loc="upper right", fontsize=10)
    else:
        ax.text(.5, .5, "No point count has a complete matched field", transform=ax.transAxes, ha="center")
    ax.set_title(f"05  Combined performance | {len(cases)} surfaces x {len(cfg['seeds'])} seeds | all methods", fontsize=16, pad=16)
    fig.supxlabel("Equal-weight pooled RMS of the per-test normalized errors; every integer N is measured.\n"
                  f"Dots mark diagnostic checkpoints. Dotted line: {epsilon:g} aggregate reference, not per-case qualification.", fontsize=10)
    save_figure(fig, "05-performance-scorecard.png", "5. Compare combined performance across all tests",
                "One curve per method pools global normalized squared errors across every surface and seed, then takes the square root. "
                "All tests receive equal weight at the same paid point count. Fold-band and peak errors remain separate diagnostics, "
                "avoiding double counting. Lower is better. An aggregate target does not guarantee that every test passes; "
                "the per-case qualification table remains below.")

    qualified_names = ", ".join(names[arm] for arm in arms if qualifies_everywhere[arm]) or "None within the tested budgets"
    guide = f"""# Plot guide â€” start here

Five overview images consolidate the saved results. All {len(arms)} methods and
all {len(cases)} test cases are shown together. **Every integer point count from 4 to {cap} is scored on one nested trajectory per method and seed.**

Read these in order:

1. [True 3-D manifolds](figures/01-manifold-reference.png) â€” what is being sampled.
2. [Point placement](figures/02-point-placement.png) â€” all methods side by side; rows are surfaces, columns are methods.
3. [Error versus points](figures/03-error-versus-points.png) â€” the main accuracy/cost comparison, all methods on each chart.
4. [Error maps](figures/04-reconstruction-error-map.png) â€” where the low-poly reconstruction misses a spike or boundary.
5. [Performance scorecard](figures/05-performance-scorecard.png) â€” one combined error-versus-points curve per method, pooling all surfaces and seeds.

Or open [the single scrolling report](index.html), which includes all five sheets.

## How to read the figures

- **Placement/reference/maps:** one representative paired seed ({seed}), at the largest point count ({cap}). N printed on placement panels is the actual number of paid samples, including four shared corners. Every method spends exactly this many points.
- **Dot colors:** dark = early, yellow = late within that design. Progressive-grid order is geometric, independent of observed values. Pale gray contours show the true surface; teal/cyan marks the true mode-switch boundary. The smooth control has no boundary.
- **Error curves:** medians across all {len(cfg['seeds'])} seeds/geometries; bands are the middle 50%, not confidence intervals. Top row is global error; bottom is error near the boundary. The smooth control has no fold; its lower panel repeats global error.
- **Axes and colors:** 3-D height scales, convergence axes and normalized residual colors are shared. Residuals above 0.5 use the brightest color. The reference height is a synthetic growth-rate proxy, not calibrated GENE output.
- **Winning:** smaller error with fewer actual evaluations is better. Qualification requires global RMS/range <= {epsilon:g} and boundary RMS/range <= {band_epsilon:g}, sustained through subsequent tested point counts. A dense-looking cluster of dots is not itself evidence of accuracy.
- **Snapshot versus curve:** the dots show one seed; curves summarize all seeds. Different seeds rotate the geometry. Do not expect the snapshot's individual error to equal the median curve.
- **Budgets:** one trajectory per case/seed/method, scored at every integer N. Every curve compares identical N across methods. Sparse-grid batches are evaluated in prescribed order; a prefix may end inside a batch before its native surrogate can be updated. This compares point placement through the common low-poly reconstruction, not native-model update frequency.

## Combined scorecard

For each N, scorecard 05 computes `sqrt(mean(error**2))` over all
{len(cases)*len(cfg['seeds'])} case/seed tests, where each error is global RMS
divided by that test's fixed truth range. Cases and seeds have equal weight.
This is a pooled normalized RMS, not an arithmetic mean of RMS values or a sum
of overlapping global/fold/peak metrics. N is points **per test**; total suite
cost per method is {len(cases)*len(cfg['seeds'])} times N. Both axes are logarithmic.
All configured tests and methods must be present at N for that point to appear.
Legend values are final aggregate errors. No uncertainty band is implied.

The dotted aggregate reference is not an all-tests qualification rule: easy
cases can offset difficult cases. Retain sheet 03 and the HTML qualification
table when diagnosing individual failures. [Combined data](aggregate-scores.json)
contains the formula, weights and every plotted value.

## Methods and geometry

Sobol acquisition is removed. A fixed scrambled Sobol **integration set** remains
independent of every sampler; it is only used to measure error fairly.

The mixture's frozen shortlist is triangles, grid and GP uncertainty from the
previous pilot (commit b4899ad). The grid predictive expert uses bilinear basis
regression on the shared observations. Local gates use prequential errors, with
uniform exploration and model disagreement guiding acquisition. It pays for one
shared sample per step, not three separate simulation runs. Ionut / sg_lib uses fixed-budget continuation: each direction is capped at level
20, while other admissible subspaces continue. Native surplus priorities remain
in effect. Deterministic node geometry is cached; observations are never shared
between trials. The common low-poly
score measures whether this design places points better; optional native error
measures the mixture predictor separately. This is a new hybrid, not an average
of three independent full-budget runs.

Cases: smooth + one peak; two planes + two peaks per region (four total);
three planes + one peak per region; two planes + two peaks in one region and
one in the other. Gaussian centers have a positive margin from every fold.
No jump or on-fold Gaussian case is included.

## Pilot takeaway

Methods qualifying on every seed of every case: **{qualified_names}**.
Read the HTML qualification table for per-case costs; qualifying everywhere does not mean
winning every case. These results select finalists for harder tests, not a
production GENE runner. See [the detailed results](../../RESULTS.md).

## Regenerate without new experiments

From the repository root, using the configured Python environment:

```bash
python -m benchmark2d --plots-only --output results/2d
```

All coordinates, sample values, costs, per-seed errors, peak-region metrics and
the common-RBF cross-check remain in [results.json](results.json). The new overview
does not repeat every intermediate 3-D view; the full learning curves retain all
integer point counts. RBF checks are saved only at configured checkpoints; coordinates are stored once at the final N and sliced for earlier prefixes. The previous experiments remain available in Git history.
"""
    (out/"README.md").write_text(guide, encoding="utf-8")
    pieces = ["<h1>2-D benchmark â€” five comparison sheets</h1>",
              "<p>Start with the reference surfaces, compare the point placement, then judge error per evaluation. "
              "The placement maps show one seed; the error curves summarize all paired seeds.</p>",
              '<p><a href="README.md">Plot-reading guide</a> Â· <a href="../../RESULTS.md">Detailed results</a> Â· <a href="results.json">Saved data</a></p>',
              "<nav>"+" Â· ".join(f'<a href="#sheet-{i}">{html.escape(title)}</a>' for i, (_, title, _) in enumerate(sheets, 1))+"</nav>"]
    for i, (filename, title, caption) in enumerate(sheets, 1):
        pieces.append(f'<section id="sheet-{i}"><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>'
                      f'<a href="figures/{filename}"><img src="figures/{filename}" alt="{html.escape(title)}"></a></section>')
    pieces.append("<h2>Qualification values</h2><table><tr><th>Case</th>"+"".join(f"<th>{names[a]}</th>" for a in arms)+"</tr>"+"".join(table_html)+"</table>")
    failures = [r for r in rows if r["status"] != "ok"]
    if failures:
        pieces.append("<p><strong>Incomplete field: no overall winner.</strong></p><pre>"+html.escape("\n".join(sorted({f"{r['arm']}: {r['status']} â€” {r['reason']}" for r in failures})))+"</pre>")
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2-D benchmark overview</title><style>body{max-width:1600px;margin:28px auto;padding:0 24px;font:16px/1.55 system-ui;color:#23313c;background:#f7f9fb}h1,h2{color:#12324b}img{width:100%;background:white}section{margin:48px 0}nav{padding:16px;background:#e8edf3}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #dae0e5}pre{white-space:pre-wrap}p{max-width:1100px}</style><body>'+"\n".join(pieces)+"</body></html>"
    (out/"index.html").write_text(document, encoding="utf-8")
    (out/"render-manifest.json").write_text(json.dumps(dict(
        renderer_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        experiment_source_hash=payload["source_hash"], matplotlib=matplotlib.__version__,
        figure_files=[s[0] for s in sheets], snapshot_seed=seed, snapshot_cap=cap), indent=2), encoding="utf-8")
    # Remove only filenames emitted by the superseded renderer, after success.
    # Never clear a directory or touch unknown/user-authored images.
    for case in cases:
        for suffix in ["comparison"]+list(arms):
            old = figures/f"{case}-{suffix}.png"
            if old.is_file():
                old.unlink()
