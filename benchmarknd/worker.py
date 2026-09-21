"""Importable process worker for the high-dimensional benchmark.

One trajectory per task. The evaluation set is the expensive shared object --
it costs 2*dim oracle passes over the reference cloud to build the truth
variation -- so each process caches its own by case, and the scheduler groups
tasks by case to make those caches hit.

Trajectory content is identical to the serial path: the same arm, seed, budget
and checkpoints produce the same rows. Only the order they are computed in
changes.
"""
from functools import lru_cache
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .cases import CASES
from .core import Observations, SurfaceND, evaluation_set, score, tolerances_for
from .ionut import IonutSurface
from .strategies import run_arm


def case_dim(case):
    return CASES[case]["dim"] if case in CASES else IonutSurface.dim


def build_surface(case, seed):
    """Synthetic surfaces carry a geometry seed; the proxies do not.

    Ionut's formulas are fixed, so `seed` varies the acquisition only and the
    evaluation set is shared across seeds.
    """
    return SurfaceND(case, seed) if case in CASES else IonutSurface(case)


@lru_cache(maxsize=4)
def _cached_test(case, surface_seed, test_size):
    return evaluation_set(build_surface(case, surface_seed), test_size)


def prepared(case, seed, test_size):
    """Evaluation set plus the declared tolerance pair for this case."""
    surface_seed = seed if case in CASES else 0
    test = _cached_test(case, surface_seed, test_size)
    spine, ported = tolerances_for(case_dim(case),
                                   band_available=bool(test["band"].any()))
    return build_surface(case, surface_seed), test, spine, ported


def checkpoints(start, budget, count=10):
    raw = np.unique(np.round(np.geomspace(start, budget, count)).astype(int))
    return [int(n) for n in raw if start <= n <= budget]


def execute(task):
    case, seed, arm, budget, test_size, checkpoint_count = task
    dim = case_dim(case)
    trial = dict(case=case, dim=dim, seed=seed, arm=arm)
    start = time.perf_counter()
    obs = None
    try:
        surface, test, spine, ported = prepared(case, seed, test_size)
        obs = Observations(surface, budget, dim, seed)
        with threadpool_limits(limits=1):
            predict, metadata = run_arm(arm, obs, seed)
        if len(obs.x) != budget:
            raise RuntimeError("strategy failed to spend the exact budget")
        acquisition = time.perf_counter()-start

        all_x, all_y = obs.x, obs.y
        lookup = {Observations.key(x): y for x, y in zip(all_x, all_y)}
        prefix = Observations(lambda x: [lookup[Observations.key(p)] for p in x],
                              budget, dim, seed, shared=all_x[:2*dim+1])
        rows = []
        for n in checkpoints(len(prefix.x), budget, checkpoint_count):
            while len(prefix.x) < n:
                prefix(all_x[len(prefix.x)])
            row = dict(**trial, budget=budget, status="ok")
            row.update(score(surface, prefix, test, targets=spine, secondary_targets=ported))
            row["x"] = prefix.x.tolist() if n == budget else None
            row["y"] = prefix.y.tolist() if n == budget else None
            if n == budget:
                row["metadata"] = metadata
                row["acquisition_seconds"] = acquisition
                row["seconds"] = time.perf_counter()-start
                if predict is not None:
                    native = np.asarray(predict(test["x"]), float)
                    row["native_error"] = float(np.sqrt(np.mean(
                        (native-test["y"])**2))/test["scale"])
            rows.append(row)
        return rows
    except Exception as exc:
        # An unavailable optional backend is a recorded absence, not a failure.
        status = "unavailable" if isinstance(exc, ImportError) else "failed"
        return [dict(**trial, budget=budget, n=0 if obs is None else len(obs.x),
                     status=status, reason=f"{type(exc).__name__}: {exc}")]
