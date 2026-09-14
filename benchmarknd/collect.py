"""Join the per-worker result files into one payload per dimension and report.

Workers run one case/seed each so they can be restarted independently. Rows are
copied verbatim; nothing is recomputed. A dimension is written out only when
every configured case, seed and arm completed, so a partial run cannot be
mistaken for a finished one.
"""
import argparse
import json
import pathlib
import numpy as np

from .cases import CASES


def load_workers(run_dir):
    payloads = []
    for path in sorted(pathlib.Path(run_dir).glob("*/results.json")):
        payloads.append((path, json.loads(path.read_text(encoding="utf-8"))))
    if not payloads:
        raise SystemExit(f"no worker results under {run_dir}")
    return payloads


def combine(payloads, dim):
    rows, seen, provenance = [], set(), []
    for path, payload in payloads:
        kept = [r for r in payload["rows"] if CASES[r["case"]]["dim"] == dim]
        if not kept:
            continue
        for row in kept:
            key = (row["case"], row["seed"], row["arm"], row["n"])
            if key in seen:
                raise ValueError(f"duplicate row {key} in {path}")
            seen.add(key)
        rows.extend(kept)
        provenance.append(dict(worker=path.parent.name, results=str(path),
                               source_hash=payload["source_hash"],
                               source_hashes=payload.get("source_hashes"),
                               commit=payload.get("commit"), python=payload.get("python"),
                               platform=payload.get("platform"), versions=payload.get("versions")))
    template = payloads[0][1]
    config = dict(template["config"])
    config["dim"] = dim
    config["cases"] = sorted({r["case"] for r in rows})
    config["seeds"] = sorted({r["seed"] for r in rows})
    config["arms"] = sorted({r["arm"] for r in rows})
    return dict(config=config, provenance=provenance, rows=rows)


def completeness(payload):
    cfg = payload["config"]
    expected = {(case, seed, arm) for case in cfg["cases"] for seed in cfg["seeds"] for arm in cfg["arms"]}
    finished = {(r["case"], r["seed"], r["arm"]) for r in payload["rows"]
                if r["status"] == "ok" and r["n"] == r["budget"]}
    return sorted(expected-finished)


def final_table(payload):
    """Median final error per arm and case, plus the pooled combined score."""
    cfg = payload["config"]
    table, combined = {}, {}
    for arm in cfg["arms"]:
        per_case = {}
        for case in cfg["cases"]:
            errors = [r["error"] for r in payload["rows"]
                      if r["arm"] == arm and r["case"] == case and r["n"] == r["budget"] and r["status"] == "ok"]
            if errors:
                per_case[case] = float(np.median(errors))
        table[arm] = per_case
        pooled = [r["error"] for r in payload["rows"]
                  if r["arm"] == arm and r["n"] == r["budget"] and r["status"] == "ok"]
        combined[arm] = float(np.sqrt(np.mean(np.square(pooled)))) if pooled else float("nan")
    return table, combined


def curves(payload):
    """Median error against N per arm, pooled over cases and seeds at matched N."""
    cfg = payload["config"]
    out = {}
    for arm in cfg["arms"]:
        points = {}
        for row in payload["rows"]:
            if row["arm"] != arm or row["status"] != "ok":
                continue
            points.setdefault(row["n"], []).append(row["error"])
        complete = {n: v for n, v in points.items() if len(v) == len(cfg["cases"])*len(cfg["seeds"])}
        out[arm] = [dict(n=n, error=float(np.sqrt(np.mean(np.square(complete[n])))), tests=len(complete[n]))
                    for n in sorted(complete)]
    return out


COLORS = {"space-filling": "#758398", "moe": "#343a40", "gpr-var": "#007c91", "gpr-grad": "#dd5f42",
          "gpr-u50-g50": "#e7298a", "gpr-u70-g30": "#6a3d9a", "gpr-u30-g70": "#fdae61"}
NAMES = {"space-filling": "Space-filling", "moe": "Mixture of experts", "gpr-var": "GP uncertainty",
         "gpr-grad": "GP gradient", "gpr-u50-g50": "GP unc/grad 50/50",
         "gpr-u70-g30": "GP unc/grad 70/30", "gpr-u30-g70": "GP unc/grad 30/70"}


def render_curves(payload, path):
    """Combined error against paid points, pooled over every case and seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = payload["config"]
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    order = sorted(payload["curves"], key=lambda a: payload["combined"][a])
    for arm in order:
        points = payload["curves"][arm]
        if not points:
            continue
        ax.plot([p["n"] for p in points], [p["error"] for p in points], lw=2,
                color=COLORS.get(arm, "#888888"),
                label=f"{NAMES.get(arm, arm)}  ({payload['combined'][arm]:.4f})")
    ax.set(xscale="log", yscale="log", xlabel="Paid evaluations per test (shared initialization included)",
           ylabel="Combined normalized RMS across all cases and seeds")
    ax.grid(alpha=.2, which="both")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(title=f"Ordered by error at N = {max(p['n'] for p in payload['curves'][order[0]])}", fontsize=10)
    ax.set_title(f"{cfg['dim']}-D combined performance | {len(cfg['cases'])} cases x {len(cfg['seeds'])} seeds "
                 f"| common RBF reconstruction", fontsize=14, pad=12)
    fig.supxlabel("Equal-weight pooled RMS of per-test normalized errors at matched N; ten log-spaced checkpoints.",
                  fontsize=9)
    fig.savefig(path, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 8])
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results"),
                        help="root holding one <dim>d directory per study")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    payloads = load_workers(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    for dim in args.dims:
        payload = combine(payloads, dim)
        missing = completeness(payload)
        payload["complete"] = not missing
        payload["missing"] = [list(m) for m in missing]
        if missing and not args.allow_partial:
            print(f"d={dim}: SKIPPED, {len(missing)} trajectories still missing "
                  f"(e.g. {missing[0]}); rerun with --allow-partial to write anyway")
            continue
        table, combined = final_table(payload)
        payload["final_errors"] = table
        payload["combined"] = combined
        payload["curves"] = curves(payload)
        # One directory per dimension, laid out exactly like results/2d.
        destination = args.output/f"{dim}d"
        figures = destination/"figures"
        figures.mkdir(parents=True, exist_ok=True)
        path = destination/"results.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        render_curves(payload, figures/"performance.png")
        budget = payload["config"]["budgets"][str(dim)] if isinstance(payload["config"]["budgets"], dict) else None
        print(f"\nd={dim}  budget {budget}  {len(payload['rows'])} rows  "
              f"{'complete' if not missing else str(len(missing))+' missing'}  -> {path}")
        width = max(len(a) for a in payload["config"]["arms"])
        cases = payload["config"]["cases"]
        print(f"{'arm':{width}}  {'combined':>9}  " + "  ".join(f"{c[3:]:>14}" for c in cases))
        for arm in sorted(payload["config"]["arms"], key=lambda a: combined[a]):
            cells = "  ".join(f"{table[arm].get(c, float('nan')):14.4f}" for c in cases)
            print(f"{arm:{width}}  {combined[arm]:9.4f}  {cells}")


if __name__ == "__main__":
    main()
