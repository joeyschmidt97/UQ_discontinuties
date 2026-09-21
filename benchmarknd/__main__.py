"""Run the high-dimensional experiments and save provenance-stamped results.

One nested trajectory per case/seed/arm. Scoring happens at ten log-spaced
checkpoints rather than every integer N: at d=8 the common RBF scorer costs
about 32 minutes per trajectory when every N is scored and 19 seconds at ten
checkpoints, while the trajectory itself is unchanged either way.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import time
from threadpoolctl import threadpool_limits
from .cases import CASES
from .ionut import CASES as IONUT_CASES
from .core import calibrating
from .strategies import ARMS
from .worker import case_dim, execute

ALL_CASES = list(CASES) + list(IONUT_CASES)


# The high-dimensional field. Tetrahedral refinement and the dyadic grid do not
# appear here and cannot: both are Delaunay-based, so the comparable arm set is
# genuinely smaller above three dimensions. Declared rather than discovered in
# the figures.
DEFAULT_ARMS = ("space-filling", "gpr-var", "gpr-grad", "gpr-u50-g50", "gpr-u70-g30",
                "gpr-u30-g70", "gpr-m05-var", "gpr-m05-grad", "gpr-m05-blend",
                "vwrs", "vurs", "moe")


def _record(payload, result_path, rows):
    """Append one finished trajectory and checkpoint the file immediately."""
    payload["rows"].extend(rows)
    save(result_path, payload)
    last = rows[-1]
    if last["status"] == "ok":
        holistic = ("uncalibrated" if last["holistic_error"] is None
                    else f"{last['holistic_error']:.3f} ({last['holistic_driver']})")
        detail = (f"H {holistic} vwfd95 {last['vwfd_p95']:.4f}"
                  f" nonlin95 {last['nonlinear_p95']:.3f} in {last['seconds']/60:.1f} min")
    else:
        detail = f"{last['status']}: {last['reason']}"
    print(f"[{time.strftime('%H:%M:%S')}] {last['case']} seed={last['seed']} "
          f"{last['arm']} {detail}", flush=True)


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=ALL_CASES, default=list(CASES))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(DEFAULT_ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget-5d", type=int, default=512)
    parser.add_argument("--budget-8d", type=int, default=1024)
    parser.add_argument("--budget-6d", type=int, default=512)
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=65536)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel trajectories; 1 keeps the original serial path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="reference-arm run for a dimension with no tolerances yet; "
                             "reports every component and leaves H uncalibrated")
    parser.add_argument("--allow-source-change", action="store_true",
                        help="resume after an edit that provably left the finished trajectories valid")
    args = parser.parse_args()
    budgets = {5: args.budget_5d, 6: args.budget_6d, 8: args.budget_8d}
    if args.calibrate:
        calibrating(True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("output", "resume")}
    config["budgets"] = budgets
    config["protocol"] = "one nested trajectory per case/seed/arm; ten log-spaced scored checkpoints"
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = sorted(list((root/"benchmarknd").glob("*.py")) + list((root/"arms").glob("*.py")))
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-checkout"
    payload = dict(config=config, source_hash=source_hash, commit=commit,
                   python=platform.python_version(), platform=platform.platform(),
                   versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "matplotlib")},
                   rows=[])
    result_path = args.output/"results.json"
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["source_hash"] != source_hash:
            if not args.allow_source_change:
                parser.error("resume requires identical source code; pass --allow-source-change "
                             "only when the edit left the finished trajectories valid")
            # Keep every hash that produced rows in this file, so a reader can
            # see the trajectories were not all generated by one source state.
            history = previous.get("source_hashes") or [previous["source_hash"]]
            payload["source_hashes"] = history + [source_hash]
        payload = dict(previous, config=config, source_hash=source_hash,
                       source_hashes=payload.get("source_hashes"))
    elif result_path.exists():
        parser.error("output already exists; use --resume or a new --output")
    # Only a completed trajectory counts as done; a failed one is retried.
    done = {(r["case"], r["seed"], r["arm"]) for r in payload["rows"]
            if r["status"] == "ok" and r["budget"] == r["n"]}
    # Group by case so each worker's evaluation-set cache hits: building one
    # costs 2*dim oracle passes over the reference cloud.
    tasks = [(case, seed, arm, budgets[case_dim(case)], args.test_size, args.checkpoints)
             for case in args.cases for seed in args.seeds for arm in args.arms
             if (case, seed, arm) not in done]
    print(f"{len(tasks)} trajectories to run on {args.workers} workers "
          f"({len(done)} already complete)", flush=True)
    if args.workers <= 1:
        with threadpool_limits(limits=1):
            for task in tasks:
                _record(payload, result_path, execute(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, task) for task in tasks]
            for future in as_completed(futures):
                _record(payload, result_path, future.result())

    failures = [r for r in payload["rows"] if r["status"] != "ok"]
    print(f"saved {result_path} ({len(payload['rows'])} rows, {len(failures)} failures)", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
