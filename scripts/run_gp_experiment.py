"""Pool-based GP pilot; predictions are scored on a frozen independent set."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from scripts.datasets import load_dataset


def run(dataset, budget, policy, nu, beta=2., seed=0):
    manifest, pool, evaluation = load_dataset(dataset)
    x, y = pool["x"], pool["y"]
    dim = x.shape[1]
    initial = 2*dim+1
    if not initial <= budget <= len(x):
        raise ValueError("budget must lie between 2*dimension+1 and pool size")
    if policy not in ("uncertainty", "ucb") or nu not in (.5, 1.5, 2.5) or beta < 0:
        raise ValueError("invalid GP policy configuration")
    selected = list(np.random.default_rng(seed).choice(len(x), initial, replace=False))
    acquired = np.zeros(len(x), dtype=bool)
    acquired[selected] = True
    checkpoints = set(np.unique(np.round(np.geomspace(initial, budget, 8)).astype(int)))
    rows, warning_count = [], 0
    with threadpool_limits(limits=1):
        while True:
            gp = GaussianProcessRegressor(kernel=ConstantKernel(1., (1e-3,1e3))*Matern(np.full(dim,.2), (1e-2,10.), nu=nu),
                                          alpha=1e-8, normalize_y=True, random_state=seed, n_restarts_optimizer=0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                gp.fit(x[selected], y[selected])
            warning_count += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            if len(selected) in checkpoints:
                error = gp.predict(evaluation["x"])-evaluation["y"]
                row = dict(n=len(selected), rmse=float(np.sqrt(np.mean(error**2))))
                row["normalized_rmse"] = row["rmse"]/manifest["scale"] if manifest["scale"] > 0 else None
                for name in ("band", "peak"):
                    mask = evaluation[name]
                    row[name+"_rmse"] = float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None
                rows.append(row)
            if len(selected) == budget:
                break
            candidates = np.flatnonzero(~acquired)
            mean, std = gp.predict(x[candidates], return_std=True)
            merit = std if policy == "uncertainty" else mean + beta*std
            index = int(candidates[np.argmax(merit)])
            selected.append(index)
            acquired[index] = True
    return dict(dataset=str(Path(dataset).resolve()), dataset_sha256=manifest["sha256"],
                policy=policy, nu=nu, beta=beta, seed=seed, budget=budget,
                selected_indices=selected, fit_warnings=warning_count, rows=rows,
                objective="uncertainty reduction" if policy == "uncertainty" else "high response via upper confidence bound",
                scorer="native GP on independent frozen evaluation set; not legacy common reconstruction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--policy", choices=["uncertainty","ucb"], default="uncertainty")
    parser.add_argument("--nu", type=float, choices=[.5,1.5,2.5], default=.5)
    parser.add_argument("--beta", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new result path")
    result = run(args.dataset, args.budget, args.policy, args.nu, args.beta, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
