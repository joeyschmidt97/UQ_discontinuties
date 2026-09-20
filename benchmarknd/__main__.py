"""Run the high-dimensional experiments and save provenance-stamped results.

One nested trajectory per case/seed/arm. Scoring happens at ten log-spaced
checkpoints rather than every integer N: at d=8 the common RBF scorer costs
about 32 minutes per trajectory when every N is scored and 19 seconds at ten
checkpoints, while the trajectory itself is unchanged either way.
"""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .cases import CASES
from .ionut import CASES as IONUT_CASES, IonutSurface
from .core import (SurfaceND, Observations, calibrating, evaluation_set, score,
                   tolerances_for)
from .strategies import ARMS, run_arm

ALL_CASES = list(CASES) + list(IONUT_CASES)


def case_dim(case):
    return CASES[case]["dim"] if case in CASES else IonutSurface.dim


def build_surface(case, seed):
    """The synthetic surfaces carry a geometry seed; the proxies do not.

    Ionut's formulas are fixed, so `seed` varies the acquisition only and the
    surface -- and therefore the whole evaluation set -- is shared across seeds.
    """
    return SurfaceND(case, seed) if case in CASES else IonutSurface(case)

# The high-dimensional field. Tetrahedral refinement and the dyadic grid do not
# appear here and cannot: both are Delaunay-based, so the comparable arm set is
# genuinely smaller above three dimensions. Declared rather than discovered in
# the figures.
DEFAULT_ARMS = ("space-filling", "gpr-var", "gpr-grad", "gpr-u50-g50", "gpr-u70-g30",
                "gpr-u30-g70", "gpr-m05-var", "gpr-m05-grad", "gpr-m05-blend",
                "vwrs", "vurs", "moe")


def checkpoints(start, budget, count=10):
    """Log-spaced integer point counts, always including the final budget."""
    raw = np.unique(np.round(np.geomspace(start, budget, count)).astype(int))
    return [int(n) for n in raw if start <= n <= budget]


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
    with threadpool_limits(limits=1):
        for case in args.cases:
            dim = case_dim(case)
            budget = budgets[dim]
            shared_test = None
            for seed in args.seeds:
                surface = build_surface(case, seed)
                if case in CASES:
                    test = evaluation_set(surface, args.test_size)
                else:
                    # Seed-independent surface: build the reference set once
                    # rather than paying 2*dim oracle passes per seed.
                    shared_test = shared_test or evaluation_set(surface, args.test_size)
                    test = shared_test
                # Spine tolerances are primary above three dimensions; the
                # ported four-term shape rides along labelled. Both are
                # preregistered per dimension, so an uncalibrated dimension
                # stops the run here rather than producing an unreadable H.
                spine, ported = tolerances_for(dim, band_available=bool(test["band"].any()))
                for arm in args.arms:
                    if (case, seed, arm) in done:
                        continue
                    start = time.perf_counter()
                    obs = Observations(surface, budget, dim, seed)
                    trial = dict(case=case, dim=dim, seed=seed, arm=arm)
                    print(f"[{time.strftime('%H:%M:%S')}] start {case} seed={seed} {arm} budget={budget}", flush=True)
                    try:
                        predict, metadata = run_arm(arm, obs, seed)
                        if len(obs.x) != budget:
                            raise RuntimeError("strategy failed to spend the exact budget")
                        acquisition = time.perf_counter()-start
                        all_x, all_y = obs.x, obs.y
                        lookup = {Observations.key(x): y for x, y in zip(all_x, all_y)}
                        shared = all_x[:2*dim+1]
                        prefix = Observations(lambda x: [lookup[Observations.key(p)] for p in x],
                                              budget, dim, seed, shared=shared)
                        rows = []
                        for n in checkpoints(len(prefix.x), budget, args.checkpoints):
                            while len(prefix.x) < n:
                                prefix(all_x[len(prefix.x)])
                            row = dict(**trial, budget=budget, status="ok")
                            row.update(score(surface, prefix, test,
                                             targets=spine, secondary_targets=ported))
                            row["x"] = prefix.x.tolist() if n == budget else None
                            row["y"] = prefix.y.tolist() if n == budget else None
                            if n == budget:
                                row["metadata"] = metadata
                                row["acquisition_seconds"] = acquisition
                                row["seconds"] = time.perf_counter()-start
                            rows.append(row)
                            holistic = ("uncalibrated" if row["holistic_error"] is None
                                        else f"{row['holistic_error']:.3f} ({row['holistic_driver']})")
                            print(f"    N={n:5} H {holistic} vwfd95 {row['vwfd_p95']:.4f}"
                                  f" nonlin95 {row['nonlinear_p95']:.3f}"
                                  f" fill95 {row['fill_p95']:.4f} rbf {row['error']:.4f}", flush=True)
                        payload["rows"].extend(rows)
                        final = rows[-1]
                        summary = ("uncalibrated" if final["holistic_error"] is None
                                   else f"{final['holistic_error']:.3f}")
                        print(f"[{time.strftime('%H:%M:%S')}] done  {case} seed={seed} {arm} "
                              f"final H {summary} in {final['seconds']/60:.1f} min", flush=True)
                    except Exception as exc:
                        payload["rows"].append(dict(**trial, budget=budget, n=len(obs.x), status="failed",
                                                    reason=f"{type(exc).__name__}: {exc}"))
                        print(f"[{time.strftime('%H:%M:%S')}] FAILED {case} seed={seed} {arm}: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                    save(result_path, payload)
    failures = [r for r in payload["rows"] if r["status"] != "ok"]
    print(f"saved {result_path} ({len(payload['rows'])} rows, {len(failures)} failures)", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
