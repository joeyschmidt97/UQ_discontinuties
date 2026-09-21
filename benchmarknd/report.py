"""Render the high-dimensional benchmark into inspectable sheets.

Built around the split the scoring enforces: the fit-free spine decides, the
reconstruction-based family is shown but labelled secondary. Every sheet keeps
the spine components separately visible, because one scalar cannot distinguish
a geometric hole from an unresolved peak -- which is the whole reason the spine
has three terms rather than one.

Runs on a partial `results.json`, so a run still in flight can be inspected
without stopping it.

    python -m benchmarknd.report --run results/5d-spine-2026-09-20
"""
import argparse
import json
import math
import pathlib
import statistics as st
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPINE = ("vwfd_p95", "nonlinear_p95", "fill_p95")
SPINE_LABEL = {"vwfd_p95": "VWFD P95", "nonlinear_p95": "nonlinear P95",
               "fill_p95": "fill distance P95"}
SECONDARY = ("error", "nmae", "p95_error")
FIGSIZE = (13, 8)


def load(run):
    payload = json.loads((pathlib.Path(run)/"results.json").read_text(encoding="utf-8"))
    ok = [r for r in payload["rows"] if r["status"] == "ok"]
    return payload, ok


def complete(rows):
    """Trajectories that reached their budget; the only rows a ranking may use."""
    return [r for r in rows if r["n"] == r["budget"]]


def pooled_rms(values):
    values = [v for v in values if v is not None]
    return math.sqrt(sum(v*v for v in values)/len(values)) if values else None


def arms_of(rows):
    return sorted({r["arm"] for r in rows})


def cases_of(rows):
    return sorted({r["case"] for r in rows})


def _style(index):
    colors = plt.get_cmap("tab20").colors
    return dict(color=colors[index % len(colors)],
                linestyle=["-", "--", ":", "-."][index // len(colors) % 4])


def convergence_sheet(rows, path, title):
    """One panel per spine term plus H, all arms overlaid, cases pooled."""
    arms = arms_of(rows)
    panels = [*SPINE, "holistic_error"]
    figure, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    for axis, key in zip(axes.ravel(), panels):
        for index, arm in enumerate(arms):
            series = defaultdict(list)
            for row in rows:
                if row["arm"] == arm and row.get(key) is not None:
                    series[row["n"]].append(row[key])
            if not series:
                continue
            counts = sorted(series)
            axis.plot(counts, [pooled_rms(series[n]) for n in counts],
                      label=arm, linewidth=1.4, **_style(index))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("paid evaluations $N$")
        axis.set_ylabel(SPINE_LABEL.get(key, "holistic $H$"))
        axis.grid(alpha=.25, which="both")
        if key == "holistic_error":
            axis.axhline(1., color="black", linewidth=1., linestyle="--")
            axis.set_title("holistic $H$ (dashed: qualification at $H=1$)")
        else:
            axis.set_title(SPINE_LABEL[key])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    figure.suptitle(f"{title} — fit-free spine, pooled over cases and seeds (RMS)")
    figure.tight_layout(rect=(0, .08, 1, 1))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def per_case_sheet(rows, path, title, key="holistic_error"):
    cases, arms = cases_of(rows), arms_of(rows)
    columns = min(4, len(cases)) or 1
    layout = (math.ceil(len(cases)/columns), columns)
    figure, axes = plt.subplots(*layout, figsize=FIGSIZE, squeeze=False)
    for axis, case in zip(axes.ravel(), cases):
        for index, arm in enumerate(arms):
            series = defaultdict(list)
            for row in rows:
                if row["case"] == case and row["arm"] == arm and row.get(key) is not None:
                    series[row["n"]].append(row[key])
            if not series:
                continue
            counts = sorted(series)
            axis.plot(counts, [pooled_rms(series[n]) for n in counts],
                      label=arm, linewidth=1.2, **_style(index))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(case.replace("ionut-", ""), fontsize=9)
        axis.grid(alpha=.25, which="both")
        if key == "holistic_error":
            axis.axhline(1., color="black", linewidth=.9, linestyle="--")
    for axis in axes.ravel()[len(cases):]:
        axis.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    figure.suptitle(f"{title} — {SPINE_LABEL.get(key, 'holistic $H$')} per case, seeds pooled")
    figure.tight_layout(rect=(0, .08, 1, 1))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def leaderboard(rows):
    """Final-budget summary, ordered by mean H. Ranking uses the spine only."""
    final = complete(rows)
    table = []
    for arm in arms_of(final):
        subset = [r for r in final if r["arm"] == arm]
        holistic = [r["holistic_error"] for r in subset if r["holistic_error"] is not None]
        ported = [r["ported_holistic_error"] for r in subset
                  if r.get("ported_holistic_error") is not None]
        table.append(dict(
            arm=arm, trajectories=len(subset),
            mean_h=st.mean(holistic) if holistic else None,
            max_h=max(holistic) if holistic else None,
            **{key: st.mean([r[key] for r in subset]) for key in SPINE},
            ported_mean_h=st.mean(ported) if ported else None,
            rbf_error=pooled_rms([r["error"] for r in subset]),
        ))
    table.sort(key=lambda row: (row["mean_h"] is None, row["mean_h"]))
    return table


def driver_census(rows):
    final = complete(rows)
    return (Counter(r["holistic_driver"] for r in final if r.get("holistic_driver")),
            Counter(r["ported_holistic_driver"] for r in final
                    if r.get("ported_holistic_driver")))


def _cell(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def html(payload, rows, figures, path):
    board = leaderboard(rows)
    spine_drivers, ported_drivers = driver_census(rows)
    config = payload["config"]
    expected = len(config["cases"])*len(config["seeds"])*len(config["arms"])
    finished = len({(r["case"], r["seed"], r["arm"]) for r in complete(rows)})
    failed = [r for r in payload["rows"] if r["status"] not in ("ok",)]

    head = "".join(f"<th>{name}</th>" for name in (
        "arm", "n", "mean H", "max H", "VWFD P95", "nonlinear P95",
        "fill P95", "ported H", "RBF RMS"))
    body = ""
    for index, row in enumerate(board):
        mark = " class='lead'" if index == 0 else ""
        body += (f"<tr{mark}><td>{row['arm']}</td><td>{row['trajectories']}</td>"
                 f"<td>{_cell(row['mean_h'], 3)}</td><td>{_cell(row['max_h'], 3)}</td>"
                 f"<td>{_cell(row['vwfd_p95'])}</td><td>{_cell(row['nonlinear_p95'], 3)}</td>"
                 f"<td>{_cell(row['fill_p95'])}</td>"
                 f"<td class='secondary'>{_cell(row['ported_mean_h'], 3)}</td>"
                 f"<td class='secondary'>{_cell(row['rbf_error'])}</td></tr>")

    drivers = "".join(f"<li><code>{name}</code> — {count}</li>"
                      for name, count in spine_drivers.most_common())
    ported = "".join(f"<li><code>{name}</code> — {count}</li>"
                     for name, count in ported_drivers.most_common())
    images = "".join(f"<h2>{caption}</h2><img src='figures/{name}'>"
                     for name, caption in figures)
    status = ("<p class='warn'>Partial run: this sheet reflects "
              f"{finished} of {expected} trajectories.</p>" if finished < expected else "")
    failures = (f"<p class='warn'>{len(failed)} non-ok rows recorded.</p>" if failed else "")

    path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>High-dimensional resolution benchmark</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1180px;color:#111}}
 table{{border-collapse:collapse;margin:1rem 0;font-variant-numeric:tabular-nums}}
 th,td{{border:1px solid #ccc;padding:.35rem .6rem;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 tr.lead{{background:#eef7ee;font-weight:600}}
 .secondary{{color:#777}}
 .warn{{background:#fff4e5;padding:.6rem .9rem;border-left:3px solid #e8a33d}}
 img{{max-width:100%;border:1px solid #ddd}}
 code{{background:#f4f4f4;padding:.1rem .3rem}}
</style>
<h1>High-dimensional resolution benchmark</h1>
{status}{failures}
<p>Commit <code>{payload.get('commit', '?')[:10]}</code>, budgets
<code>{config.get('budgets')}</code>, {config.get('test_size')} reference points,
seeds {config.get('seeds')}.</p>

<h2>Leaderboard at the final budget</h2>
<p>Ordered by mean holistic <em>H</em>, which above three dimensions is built from
fit-free terms only. The two greyed columns are the ported four-term <em>H</em> and
the thin-plate-spline RMS: both are shown for continuity with the 2D and 3D
scores and neither decides the ranking, because that evaluator's error rises
with budget on concentrated designs.</p>
<table><tr>{head}</tr>{body}</table>

<h2>Which term binds</h2>
<p>Spine <em>H</em> driver across completed trajectories:</p><ul>{drivers}</ul>
<p>Ported <em>H</em> driver, for comparison:</p><ul>{ported}</ul>

{images}
""", encoding="utf-8")


def render(run):
    run = pathlib.Path(run)
    payload, rows = load(run)
    if not rows:
        raise SystemExit(f"{run}: no completed rows to render yet")
    figures = run/"figures"
    figures.mkdir(exist_ok=True)
    dims = sorted({r["dim"] for r in rows})
    title = f"d={', '.join(str(d) for d in dims)}"

    convergence_sheet(rows, figures/"01-spine-convergence.png", title)
    per_case_sheet(rows, figures/"02-holistic-per-case.png", title)
    per_case_sheet(rows, figures/"03-vwfd-per-case.png", title, key="vwfd_p95")
    per_case_sheet(rows, figures/"04-rbf-secondary-per-case.png", title, key="error")
    sheets = [("01-spine-convergence.png", "Spine convergence, cases pooled"),
              ("02-holistic-per-case.png", "Holistic H per case"),
              ("03-vwfd-per-case.png", "VWFD P95 per case"),
              ("04-rbf-secondary-per-case.png",
               "Thin-plate-spline RMS per case (secondary, does not rank)")]
    html(payload, rows, sheets, run/"index.html")
    return run/"index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    print(f"wrote {render(parser.parse_args().run).resolve()}")


if __name__ == "__main__":
    main()
