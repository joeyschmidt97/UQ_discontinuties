"""Run the matched 3D Ionut benchmark and render its report."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess

from scripts.generate_ionut_slices import CASES
from .strategies import ARMS
from .worker import execute


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/3d"))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=16384)
    parser.add_argument("--secondary", nargs="+", type=int, default=[64, 128, 192, 256])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    from .report import render
    if args.plots_only:
        render(json.loads((args.output/"results.json").read_text()), args.data, args.output)
        return
    if args.budget < 16 or args.test_size < 1024 or args.test_size & (args.test_size-1):
        parser.error("budget >= 16 and a power-of-two test size >= 1024 required")
    args.output.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = sorted(list((root/"benchmark3d").glob("*.py")) + list((root/"arms").glob("*.py")))
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    config = dict(cases=args.cases, arms=args.arms, seeds=args.seeds, budget=args.budget,
                  test_size=args.test_size, secondary=args.secondary,
                  protocol="one nested trajectory; every integer N from eight shared cube corners; common tetrahedral reconstruction")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-checkout"
    payload = dict(config=config, source_hash=source_hash, commit=commit,
                   python=platform.python_version(), platform=platform.platform(),
                   versions={name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "scikit-learn", "matplotlib")}, rows=[])
    result_path = args.output/"results.json"
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["config"] != config or previous["source_hash"] != source_hash:
            parser.error("resume requires identical configuration and source")
        payload = previous
    elif result_path.exists():
        parser.error("output exists; use --resume, --plots-only, or a new directory")
    complete = {(row["case"], row["seed"], row["arm"]) for row in payload["rows"]
                if (row["status"] == "ok" and row["n"] == args.budget) or row["status"] == "unavailable"}
    tasks = []
    for case in args.cases:
        dataset = args.data/case/"seed-0"
        for seed in args.seeds:
            for arm in args.arms:
                if (case, seed, arm) not in complete:
                    tasks.append((str(dataset.resolve()), case, seed, arm, args.budget,
                                  args.test_size, tuple(args.secondary)))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            rows = future.result()
            payload["rows"].extend(rows)
            save(result_path, payload)
            last = rows[-1]
            message = (f"error={last['error']:.5f}" if last["status"] == "ok"
                       else last["reason"])
            print(f"{last['case']} seed={last['seed']} {last['arm']} {last['status']} {message}", flush=True)
    render(payload, args.data, args.output)
    failures = [row for row in payload["rows"] if row["status"] == "failed"]
    print(f"Report: {(args.output/'index.html').resolve()} ({len(failures)} failed)")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
