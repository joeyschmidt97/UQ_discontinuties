"""Pool replay graded on a fixed held-out test set, against the full-pool reference.

The standard pool replay (`benchmarknd.pool`) grades each arm on the pool
points it did not pay for. That set differs per arm and, at large budgets, is
whatever an adaptive arm judged least informative, so it cannot be compared
with a "use every run" reference. Here a stratified random share of the pool
is withheld once (same for every arm and seed) and no arm may pay for it.
Arms sample from the rest, and every checkpoint -- plus the reference that
pays for the whole remaining pool -- is graded on the same withheld points
with the pool's classify-then-regress scorer.

    .venv/Scripts/python.exe -m scripts.pool_fixed_testset --pool data/pools/hatch_pscans_global.csv \
        --seed 0 --output results/pool-hatch-global-testset-2026-10-06/seed0
    .venv/Scripts/python.exe -m scripts.pool_fixed_testset --pool data/pools/hatch_pscans_global.csv \
        --reference --output results/pool-hatch-global-testset-2026-10-06/reference
"""
import argparse
import json
import pathlib
import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree

from benchmarknd.core import Observations, reconstruct
from benchmarknd.pool import (PoolOracle, checkpoints, classify_then_regress, initial_pool_design,
                              load_pool, per_mode_scores)
from benchmarknd.strategies import run_arm


def split(oracle, share, seed):
    """Stratified by mode: the same share of every label is withheld."""
    rng = np.random.default_rng(seed)
    test = []
    for k in np.unique(oracle.labels):
        rows = np.flatnonzero(oracle.labels == k)
        test.extend(rng.choice(rows, int(round(share*len(rows))), replace=False))
    test = np.sort(np.array(test, int))
    return np.setdiff1d(np.arange(len(oracle.pool)), test), test


def score(oracle, paid_index, test):
    paid = np.zeros(len(oracle.pool), bool)
    paid[paid_index] = True
    held = np.zeros(len(oracle.pool), bool)
    held[test] = True
    yhat = reconstruct(oracle.pool[paid], oracle.y[paid], oracle.pool[held])
    nearest = cKDTree(oracle.pool[paid]).query(oracle.pool[held])[1]
    predicted = oracle.labels[paid][nearest]
    ctr = classify_then_regress(oracle, paid, held, predicted, yhat)
    modes = per_mode_scores(oracle, held, ctr["yhat"])
    return dict(n=int(paid.sum()), label_accuracy=float(np.mean(predicted == oracle.labels[held])),
                ctr_macro_nrmse=modes["macro_nrmse"], ctr_mode_nrmse=modes["mode_nrmse"],
                ctr_fallback_points=ctr["fallback_points"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=pathlib.Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["vurs", "vwrs", "space-filling", "gpr-var"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="largest budget as a fraction of the sampleable pool")
    parser.add_argument("--checkpoints", type=int, default=12)
    parser.add_argument("--reference", action="store_true",
                        help="score only the pay-for-everything reference and exit")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    full, meta = load_pool(args.pool, "gamma")
    train, test = split(full, args.test_share, args.split_seed)
    root = pathlib.Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/"results.json"
    if path.exists():
        parser.error("output already exists; choose a new --output")
    payload = dict(pool=meta, commit=commit, test=test.tolist(),
                   config={k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
                   rows=[])
    if args.reference:
        payload["rows"].append(dict(arm="all-sampleable", seed=None, **score(full, train, test)))
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(payload["rows"][0])
        return

    sub = PoolOracle(full.pool[train], full.y[train], full.labels[train], full.label_names)
    budget = int(args.fraction*len(train))
    for arm in args.arms:
        start = time.perf_counter()
        obs = Observations(sub, budget, sub.dim, args.seed, shared=initial_pool_design(sub, args.seed))
        run_arm(arm, obs, args.seed)
        order = train[sub.indices(obs.x)]
        seconds = time.perf_counter()-start
        for n in checkpoints(len(initial_pool_design(sub, args.seed)), budget, args.checkpoints):
            payload["rows"].append(dict(arm=arm, seed=args.seed, budget=budget, **score(full, order[:n], test)))
        payload["rows"][-1].update(seconds=seconds, selected=order.tolist())
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        last = payload["rows"][-1]
        print(f"[{time.strftime('%H:%M:%S')}] seed={args.seed} {arm:14s} N={last['n']} "
              f"E1 {last['ctr_macro_nrmse']:.3f} acc {last['label_accuracy']:.3f} in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
