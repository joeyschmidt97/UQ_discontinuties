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

COLORS = dict(grid="#758398", sobol="#343a40", sglib="#b07813", sgpp="#9b59b6",
              **{"gpr-var": "#007c91", "gpr-grad": "#dd5f42", "triangles": "#318448"})


def curve(ax, rows, metric, color, label, **kw):
    # Group by requested cap; plot actual measured cost, not nominal budget.
    points = []
    for budget in sorted({r["budget"] for r in rows}):
        group = [r for r in rows if r["budget"] == budget]
        points.append((np.median([r["n"] for r in group]),
                       np.median([r[metric] for r in group]),
                       np.quantile([r[metric] for r in group], [.25, .75])))
    if not points:
        return
    points.sort(key=lambda p: p[0])
    x, y, spread = zip(*points)
    ax.plot(x, y, "o-", color=color, label=label, markersize=4, **kw)
    ax.fill_between(x, np.array(spread)[:, 0], np.array(spread)[:, 1], color=color, alpha=.10)


def format_axis(ax, epsilon):
    ax.axhline(epsilon, color="#666666", ls=":", lw=1, label=f"global target {epsilon:g}")
    ax.set(xlabel="Unique evaluations (corners included)", ylabel="RMS / fixed truth range", yscale="log", xscale="log")
    ax.grid(alpha=.2, which="both")
    ax.spines[["right", "top"]].set_visible(False)


def qualifying_cost(rows, epsilon, band_epsilon):
    # Measured checkpoint only; all subsequent tested checkpoints must pass.
    ordered = sorted(rows, key=lambda r: r["budget"])
    for i, row in enumerate(ordered):
        if all(r["status"] == "ok" and r["error"] <= epsilon and r["band_error"] <= band_epsilon for r in ordered[i:]):
            return row["n"]
    return None


def render(payload, out):
    out = pathlib.Path(out)
    rows, cfg = payload["rows"], payload["config"]
    ok = [r for r in rows if r["status"] == "ok"]
    figures = out/"figures"
    figures.mkdir(exist_ok=True)
    epsilon, band_epsilon = cfg["epsilon"], cfg["band_epsilon"]
    incomplete = any(r["status"] != "ok" for r in rows)
    sections = ["<h1>2-D sampling benchmark</h1>",
                "<p>Low-poly reconstruction from sampled values. Every evaluation is charged; four shared corners cover the box. "
                "Bands show the interquartile spread across paired seeds/geometries, not confidence intervals.</p>",
                f"<p><b>{'Incomplete field: no overall winner.' if incomplete else 'All requested arms completed.'}</b> "
                f"Targets: global error &le; {epsilon:g}; boundary-band error &le; {band_epsilon:g}. "
                "Qualification must persist through subsequent tested checkpoints. Results are synthetic pilot evidence.</p>"]
    for case in cfg["cases"]:
        case_rows = [r for r in rows if r["case"] == case]
        case_ok = [r for r in case_rows if r["status"] == "ok"]
        errors = [r[k] for r in case_ok for k in ("error", "band_error", "peak_error", "rbf_error")]
        limits = (max(1e-5, min(errors+[epsilon])*.7), max(errors+[band_epsilon])*1.3)
        sections.append(f"<h2>{html.escape(case)}</h2>")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        for name in cfg["arms"]:
            group = [r for r in ok if r["case"] == case and r["arm"] == name]
            for ax, metric in zip(axes, ("error", "band_error", "rbf_error")):
                curve(ax, group, metric, COLORS[name], name)
        for ax, title in zip(axes, ("Common linear: whole domain", "Common linear: boundary band", "Common RBF: ranking cross-check")):
            format_axis(ax, epsilon)
            ax.set_ylim(*limits)
            ax.set_title(title, fontsize=11)
        axes[1].axhline(band_epsilon, color="#666666", ls="--", lw=1)
        axes[-1].legend(fontsize=8)
        fig.suptitle(case)
        fig.tight_layout()
        overview = f"{case}-comparison.png"
        fig.savefig(figures/overview, dpi=150)
        plt.close(fig)
        sections.append(f'<img src="figures/{overview}" alt="Error versus cost across methods">')
        table = ["<table><tr><th>Strategy</th><th>Qualified seeds</th><th>Median cost if all qualify</th><th>Median final global / band error</th></tr>"]
        candidates = []
        for name in cfg["arms"]:
            group = [r for r in case_rows if r["arm"] == name]
            costs = [qualifying_cost([r for r in group if r["seed"] == seed], epsilon, band_epsilon) for seed in cfg["seeds"]]
            qualified = [v for v in costs if v is not None]
            final = [r for r in group if r["budget"] == max(cfg["budgets"]) and r["status"] == "ok"]
            median_cost = float(np.median(qualified)) if len(qualified) == len(cfg["seeds"]) else None
            if median_cost is not None:
                candidates.append((median_cost, name))
            final_text = (f"{np.median([r['error'] for r in final]):.4f} / {np.median([r['band_error'] for r in final]):.4f}" if final else "unavailable / failed")
            table.append(f"<tr><td>{name}</td><td>{len(qualified)}/{len(costs)}</td><td>{median_cost if median_cost is not None else 'not reached on every seed'}</td><td>{final_text}</td></tr>")
        table.append("</table>")
        sections.extend(table)
        if candidates:
            candidates.sort()
            tied = [name for cost, name in candidates if cost == candidates[0][0]]
            sections.append(f"<p>Lowest qualifying measured median cost among completed arms: <b>{', '.join(tied)}</b> ({candidates[0][0]:g} evaluations). "
                            "This is a per-case pilot result, not a universal winner.</p>")
        else:
            sections.append("<p><b>No arm reached both targets on every seed within the tested budgets.</b></p>")
        for name in cfg["arms"]:
            group = [r for r in ok if r["case"] == case and r["arm"] == name]
            if not group:
                continue
            seed = min(r["seed"] for r in group)
            snapshots = sorted([r for r in group if r["seed"] == seed], key=lambda r: r["budget"])
            # Every saved budget gets its own synchronized diagnostic row.
            surface = Surface(case, seed)
            a, b = np.meshgrid(np.linspace(0, 1, 70), np.linspace(0, 1, 70))
            xy = np.column_stack([a.ravel(), b.ravel()])
            truth = surface(xy).reshape(a.shape)
            residuals = [np.abs(reconstruct(r["x"], r["y"], xy)-truth.ravel()).reshape(a.shape) for r in snapshots]
            # A shared scale across all strategies/budgets in this geometry.
            vmax = float(np.ptp(truth))
            fig = plt.figure(figsize=(18, 5*len(snapshots)))
            for index, (row, residual) in enumerate(zip(snapshots, residuals)):
                ax = fig.add_subplot(len(snapshots), 3, index*3+1, projection="3d")
                x, y = np.array(row["x"]), np.array(row["y"])
                ax.plot_surface(a, b, truth, color="#a4b8c6", alpha=.18, linewidth=0)
                ax.plot_trisurf(x[:, 0], x[:, 1], y, color=COLORS[name], alpha=.30, edgecolor="#46515b", linewidth=.25)
                ax.scatter(x[:, 0], x[:, 1], y, c=np.arange(len(x)), cmap="plasma", s=12, depthshade=False)
                ax.set(xlabel="Parameter 1", ylabel="Parameter 2", zlabel="Growth-rate proxy", xlim=(0, 1), ylim=(0, 1), zlim=(0, 1.8))
                ax.view_init(elev=28, azim=-58)
                ax.set_title(f"{name} | cap {row['budget']}, actual {row['n']}", fontsize=10)
                err_ax = fig.add_subplot(len(snapshots), 3, index*3+2)
                for metric, color, label in (("error", COLORS[name], "global"), ("band_error", "#b44747", "boundary band"),
                                             ("peak_error", "#c88925", "peak region")):
                    curve(err_ax, group, metric, color, label)
                err_ax.scatter(row["n"], row["error"], s=90, facecolors="none", edgecolors="black", zorder=8, label=f"shown seed {seed}")
                format_axis(err_ax, epsilon)
                err_ax.set_ylim(*limits)
                err_ax.axhline(band_epsilon, color="#b44747", ls=":", lw=.8)
                err_ax.legend(fontsize=8)
                map_ax = fig.add_subplot(len(snapshots), 3, index*3+3)
                im = map_ax.pcolormesh(a, b, residual, cmap="magma", vmin=0, vmax=vmax, shading="auto")
                map_ax.contour(a, b, surface.distance(xy).reshape(a.shape), levels=[0], colors="cyan", linewidths=.8)
                map_ax.scatter(x[:, 0], x[:, 1], s=5, c="white")
                reference = "reference line" if case == "smooth" else "true boundary"
                map_ax.set(xlabel="Parameter 1", ylabel="Parameter 2", title=f"Absolute error; cyan = {reference}", aspect="equal")
                fig.colorbar(im, ax=map_ax, shrink=.75)
            fig.suptitle(f"{case} / {name} | placement colors: early dark to late yellow | geometry seed {seed}")
            fig.subplots_adjust(left=.02, right=.97, bottom=.07, top=.93,
                                wspace=.40, hspace=.40)
            filename = f"{case}-{name}.png"
            fig.savefig(figures/filename, dpi=130)
            plt.close(fig)
            sections.append(f'<details><summary>{name}: surface, cost curve and residuals at each budget</summary><img loading="lazy" src="figures/{filename}" alt="{name} diagnostic plots"></details>')
    failures = [r for r in rows if r["status"] != "ok"]
    if failures:
        sections.append("<h2>Unavailable or failed experiments</h2><pre>"+html.escape("\n".join(sorted({f"{r['arm']}: {r['status']} — {r['reason']}" for r in failures})))+"</pre>")
    sections.append("<h2>Protocol and interpretation</h2><p>The strategies see only evaluations they purchase. "
                    "Truth, boundaries and test points never enter acquisition. The surface is a synthetic scalar proxy, not a GENE solve. "
                    "The grid is regenerated at each budget; its point color indicates enumeration, not adaptive search. "
                    "Sobol uses complete power-of-two blocks plus four corners, and prescribed sparse grids may underspend. "
                    "All charts use actual costs. Independent budget runs share seeds but need not form nested trajectories. "
                    "The peak-region metric, secondary native errors, parameters and all samples are in results.json. "
                    "A true jump is smeared by continuous triangulation; boundary errors expose that limitation. "
                    "Noise, failed GENE runs, adaptive spectra and 4-D transfer remain follow-up experiments.</p>")
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2-D sampling benchmark</title><style>body{max-width:1250px;margin:32px auto;padding:0 24px;font:16px/1.55 system-ui;color:#23313c;background:#f7f9fb}h1,h2{color:#12324b}img{width:100%;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #dae0e5}details{margin:18px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap}p{max-width:1050px}</style><body>'+"\n".join(sections)+"</body></html>"
    (out/"index.html").write_text(document, encoding="utf-8")
    (out/"render-manifest.json").write_text(json.dumps(dict(
        renderer_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        experiment_source_hash=payload["source_hash"], matplotlib=matplotlib.__version__), indent=2), encoding="utf-8")
