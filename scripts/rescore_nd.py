"""Rescore finished high-dimensional trajectories under the current scorer.

    python -m scripts.rescore_nd --run results/6d-ionut-spine-2026-09-20 [--calibrate]

No sampler is re-run. The final row of every trajectory stores its design in
acquisition order and the oracle is deterministic, so each scored checkpoint's
prefix is rebuilt exactly as the runner built it -- the shared initial design
first, then points in order -- and scored again.

Used when a scoring definition changes after results exist, here the 6D
transition mask. Faithfulness is checked, not assumed: every score the change
should not touch must reproduce its stored value, or the rescore stops. The
file records which fields changed, why, and at which commit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import json
import math
import pathlib
import subprocess

from threadpoolctl import threadpool_limits

from benchmarknd.core import Observations, calibrating, score, tolerances_for
from benchmarknd.worker import build_surface, case_dim, prepared

# Fields the transition-mask change is allowed to move. Everything else in a
# row must come back identical.
CHANGED = {"band_error", "band_nmae", "ported_holistic_error", "ported_holistic_driver",
           "ported_holistic_targets", "ported_holistic_uncalibrated"}
UNCHECKED = {"x", "y", "metadata", "seconds", "acquisition_seconds", "native_error", "status",
             "case", "dim", "seed", "arm", "budget", "n"}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def rescore_trajectory(task):
    case, seed, arm, rows, calibrate = task
    calibrating(calibrate)
    final = next(r for r in rows if r["n"] == r["budget"])
    dim = case_dim(case)
    surface, test, spine, ported = prepared(case, seed, final.get("test_size", 65536))
    if calibrate:
        spine = ported = None
    x, y = final["x"], final["y"]
    lookup = {Observations.key(p): v for p, v in zip(x, y)}
    prefix = Observations(lambda q: [lookup[Observations.key(p)] for p in q],
                          final["budget"], dim, seed, shared=x[:2*dim+1])
    out, mismatches = [], []
    with threadpool_limits(limits=1):
        for row in sorted(rows, key=lambda r: r["n"]):
            while len(prefix.x) < row["n"]:
                prefix(x[len(prefix.x)])
            fresh = score(surface, prefix, test, targets=spine, secondary_targets=ported)
            for key, value in fresh.items():
                if key in CHANGED or key in UNCHECKED:
                    continue
                if key in row and not same(row[key], value):
                    mismatches.append((row["n"], key, row[key], value))
            new = dict(row)
            new.update({k: fresh[k] for k in CHANGED if k in fresh})
            out.append(new)
    return case, seed, arm, out, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--calibrate", action="store_true",
                        help="the run is a reference-arm calibration: no tolerances apply")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    path = args.run/"results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = {}
    for row in payload["rows"]:
        if row["status"] == "ok":
            groups.setdefault((row["case"], row["seed"], row["arm"]), []).append(row)
    tasks = [(c, s, a, rows, args.calibrate) for (c, s, a), rows in groups.items()
             if any(r["n"] == r["budget"] and r.get("x") for r in rows)]
    print(f"rescoring {len(tasks)} trajectories on {args.workers} workers", flush=True)
    rescored, problems = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (case, seed, arm, rows, mismatches) in enumerate(pool.map(rescore_trajectory, tasks), 1):
            rescored[(case, seed, arm)] = rows
            problems.extend((case, seed, arm, *m) for m in mismatches)
            if done % 24 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
    if problems:
        for p in problems[:20]:
            print("MISMATCH", p)
        raise SystemExit(f"{len(problems)} unchanged-field mismatches: rescoring is not faithful; "
                         "nothing written")
    kept = [r for r in payload["rows"] if (r["case"], r["seed"], r["arm"]) not in rescored
            or r["status"] != "ok"]
    payload["rows"] = kept + [r for rows in rescored.values() for r in rows]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload.setdefault("rescored", []).append(dict(
        when=datetime.datetime.now().isoformat(timespec="seconds"), commit=commit,
        reason=args.reason, fields=sorted(CHANGED), trajectories=len(rescored),
        verified_unchanged="every other stored score reproduced exactly"))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
    print(f"rescored {len(rescored)} trajectories; all unchanged fields reproduced")


if __name__ == "__main__":
    main()
