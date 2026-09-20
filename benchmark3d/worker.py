"""Importable process worker for the matched 3D benchmark."""
import pathlib
import time

from threadpoolctl import threadpool_limits

from scripts.datasets import load_dataset, surface_for
from .core import CORNERS, Observations, evaluation_set, score, normalized_rmse
from .strategies import run_arm


def execute(task):
    dataset_string, case, seed, arm, budget, test_size, secondary = task
    dataset = pathlib.Path(dataset_string)
    manifest, _, _ = load_dataset(dataset)
    oracle = surface_for(3, case, 0)
    test = evaluation_set(str(dataset.resolve()), test_size)
    start = time.perf_counter()
    obs = Observations(oracle, budget)
    trial = dict(case=case, seed=seed, arm=arm, budget=budget)
    try:
        with threadpool_limits(limits=1):
            predict, metadata = run_arm(arm, obs, seed)
        if len(obs.x) != budget:
            raise RuntimeError("strategy failed to spend the exact budget")
        acquisition_seconds = time.perf_counter()-start
        all_x, all_y = obs.x, obs.y
        lookup = {Observations.key(x): y for x, y in zip(all_x, all_y)}
        prefix = Observations(lambda x: [lookup[Observations.key(p)] for p in x],
                              budget, shared=all_x[:len(CORNERS)])
        rows = []
        for n in range(len(CORNERS), budget+1):
            if len(prefix.x) < n:
                prefix(all_x[n-1])
            row = dict(**trial, status="ok")
            row.update(score(prefix, test, secondary=n in secondary))
            if n == budget:
                row["x"], row["y"] = prefix.x.tolist(), prefix.y.tolist()
                row["metadata"] = metadata
                row["acquisition_seconds"] = acquisition_seconds
                row["seconds"] = time.perf_counter()-start
                if predict is not None:
                    native = predict(test["x"])
                    row["native_error"] = normalized_rmse(native, test["y"], manifest["scale"])
            rows.append(row)
        return rows
    except Exception as exc:
        return [dict(**trial, n=len(obs.x),
                     status="unavailable" if isinstance(exc, ImportError) else "failed",
                     reason=f"{type(exc).__name__}: {exc}")]
