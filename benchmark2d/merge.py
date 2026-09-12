"""Join independently executed trajectories into one matched-point report.

No observations are generated here. Rows are copied verbatim from saved runs,
so every curve remains an equal-N comparison of the trajectories as executed.
The merged payload keeps one provenance record per contributing run.
"""
import argparse
import json
import pathlib
import numpy as np
from .report import aggregate_errors, render

SHARED_CONFIG = ("cases", "seeds", "budgets", "test_size", "epsilon", "band_epsilon", "protocol")


def load(path):
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not payload.get("rows"):
        raise ValueError(f"{path} contains no rows")
    return payload


def provenance(path, payload):
    return dict(results=str(path), source_hash=payload["source_hash"], commit=payload.get("commit"),
                python=payload.get("python"), platform=payload.get("platform"),
                versions=payload.get("versions"), arms=payload["config"]["arms"])


def ranking(payload, arms):
    """Rank arms by combined normalized RMS at the largest matched point count."""
    scoped = dict(payload, config=dict(payload["config"], arms=list(arms)))
    curves = aggregate_errors(scoped)
    if not all(curves[arm] for arm in arms):
        raise ValueError("no point count has a complete matched field for the added arms")
    return sorted(((arm, curves[arm][-1]["n"], curves[arm][-1]["error"]) for arm in arms), key=lambda item: item[2])


def merge(base_path, add_paths, top=None, keep=None):
    base = load(base_path)
    payloads = [(base_path, base)]
    rows = list(base["rows"])
    added = []
    for path in add_paths:
        extra = load(path)
        for field in SHARED_CONFIG:
            if extra["config"].get(field) != base["config"].get(field):
                raise ValueError(f"{path} disagrees with the base run on {field}")
        payloads.append((path, extra))
        rows.extend(extra["rows"])
        added.extend(arm for arm in extra["config"]["arms"] if arm not in added)
    if set(added) & set(base["config"]["arms"]):
        raise ValueError("added arms already exist in the base run")
    seen = set()
    for row in rows:
        key = (row["arm"], row["case"], row["seed"], row["budget"], row.get("n"))
        if key in seen:
            raise ValueError(f"duplicate row for {key}")
        seen.add(key)
    scored = ranking(dict(base, config=dict(base["config"], arms=added), rows=rows), added)
    selected = [arm for arm, _, _ in scored[:top]] if top else [a for a in added if keep is None or a in keep]
    merged = dict(base)
    merged["config"] = dict(base["config"], arms=list(base["config"]["arms"])+selected)
    merged["rows"] = [r for r in rows if r["arm"] in merged["config"]["arms"]]
    merged["source_hash"] = "+".join(sorted({p["source_hash"] for _, p in payloads}))
    merged["provenance"] = [provenance(path, p) for path, p in payloads]
    merged["selection"] = dict(candidates=[dict(arm=arm, n=n, error=error) for arm, n, error in scored],
                               selected=selected, rule="lowest combined normalized RMS at the largest matched N",
                               note="post-hoc display selection among the added arms; not held-out validation")
    return merged, scored, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--add", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--top", type=int, help="keep only the N best added arms")
    parser.add_argument("--keep", nargs="+", help="keep exactly these added arms")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.top is not None and args.keep:
        parser.error("use --top or --keep, not both")
    merged, scored, selected = merge(args.base, args.add, args.top, args.keep)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"results.json").write_text(json.dumps(merged, indent=2, allow_nan=False), encoding="utf-8")
    width = max(len(arm) for arm, _, _ in scored)
    for rank, (arm, n, error) in enumerate(scored, 1):
        print(f"{rank:2}. {arm:{width}}  combined RMS at N={n}: {error:.6f}"
              + ("  [kept]" if arm in selected else ""))
    render(merged, args.output)
    print(f"Report: {(args.output/'index.html').resolve()}")


if __name__ == "__main__":
    main()
