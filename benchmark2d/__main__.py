"""Run reproducible 2-D experiments and render the saved results."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import time
from threadpoolctl import threadpool_limits
from .core import CASES, Surface, Observations, evaluation_set, score, rmse
from .strategies import ARMS, run_arm


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test-size", type=int, default=16384)
    parser.add_argument("--native-test-size", type=int, default=0,
                        help="optional native-predictor diagnostic points; default 0 skips the slow secondary metric")
    parser.add_argument("--epsilon", type=float, default=.05)
    parser.add_argument("--band-epsilon", type=float, default=.10)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("outputs/benchmark2d"))
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true", help="one case/seed, budgets 32 and 64")
    args = parser.parse_args()
    from .report import render
    if args.plots_only:
        render(json.loads((args.output/"results.json").read_text()), args.output)
        return
    if args.quick:
        args.cases, args.seeds, args.budgets = ["two-plane-four-peaks"], [0], [32, 64]
    if min(args.budgets) < 9 or args.epsilon <= 0 or args.band_epsilon <= 0 or args.native_test_size < 0:
        parser.error("budgets >= 9 and positive tolerances required")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("output", "plots_only", "resume", "quick")}
    config["protocol"] = "single trajectory, every integer N from 4; four charged corners"
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = list((root/"benchmark2d").glob("*.py")) + list((root/"arms").glob("*.py"))
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(sources))).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-checkout"
    payload = dict(config=config, source_hash=source_hash, commit=commit,
                   python=platform.python_version(), platform=platform.platform(),
                   versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "matplotlib")}, rows=[])
    result_path = args.output/"results.json"
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["config"] != config or previous["source_hash"] != source_hash:
            parser.error("resume requires identical configuration and source code")
        payload = previous
    elif result_path.exists():
        parser.error("output already exists; use --resume, --plots-only, or a new --output")
    completed = {(r["case"], r["seed"], r["budget"], r["arm"]) for r in payload["rows"]}
    with threadpool_limits(limits=1):
        for case in args.cases:
            for seed in args.seeds:
                surface = Surface(case, seed)
                test = evaluation_set(surface, args.test_size)
                budget = max(args.budgets)
                for name in args.arms:
                    if (case, seed, budget, name) in completed:
                        continue
                    start = time.perf_counter()
                    obs = Observations(surface, budget)
                    trial = dict(case=case, seed=seed, arm=name)
                    trial_rows = []
                    try:
                        predict, metadata = run_arm(name, obs, seed)
                        if len(obs.x) != budget:
                            raise RuntimeError("strategy failed to spend the exact budget")
                        all_x, all_y = obs.x, obs.y
                        # Post-hoc prefix scoring is causal: strategies have already
                        # committed their sequence and never receive the test set.
                        lookup = {Observations.key(x): y for x, y in zip(all_x, all_y)}
                        prefix = Observations(lambda x: [lookup[Observations.key(p)] for p in x], budget)
                        for n in range(4, budget+1):
                            if n > 4:
                                prefix(all_x[n-1])
                            row = dict(**trial, budget=n, status="ok")
                            row.update(score(surface, prefix, test, secondary=n in args.budgets))
                            if n != budget:
                                row.pop("x"); row.pop("y")
                            else:
                                row["metadata"] = metadata
                                row["seconds"] = time.perf_counter()-start
                                if predict is not None and args.native_test_size:
                                    count = min(args.native_test_size, len(test[0]))
                                    row["native_test_size"] = count
                                    try:
                                        row["native_error"] = rmse(predict(test[0][:count]), test[1][:count], test[2])
                                    except Exception as exc:
                                        row["native_failure"] = f"{type(exc).__name__}: {exc}"
                            trial_rows.append(row)
                        payload["rows"].extend(trial_rows)
                    except Exception as exc:
                        row = dict(**trial, budget=budget, status="unavailable" if isinstance(exc, ImportError) else "failed",
                                   reason=f"{type(exc).__name__}: {exc}", n=len(obs.x))
                        payload["rows"].append(row)
                    save(result_path, payload)
                    print(f"{case:24} seed={seed} N={budget:4} {name:10} {row['status']:11} "
                          + (f"error={row['error']:.4f}" if row["status"] == "ok" else row["reason"]), flush=True)
    render(payload, args.output)
    print(f"Report: {(args.output/'index.html').resolve()}")
    if any(r["status"] == "failed" for r in payload["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
