"""2D noise sweep: how much does a noisy oracle degrade where a sampler looks?

The sampler sees gamma +/- %, with the percentage rising toward a mode fold,
because that is where competing branches make a GENE trace oscillate instead of
settle. Everything the acquisition rule decides is made on those noisy values.

Two scores come out of every trajectory, from the same design:

  placement    reconstruct from the TRUE response at the sampled points
  end_to_end   reconstruct from the NOISY values the sampler actually saw

Both are graded against the noiseless surface on a common Delaunay
reconstruction, the preregistered 2D evaluator. `placement` isolates the
question asked -- does noise send the sampler to worse locations -- because the
values feeding the reconstruction are clean either way. `end_to_end` is what a
real campaign would suffer. Their difference is the value-corruption penalty,
which is worth separating: a method can place well and still produce a bad
surface, and the fix for each is different.

    python -m scripts.run_noise_2d --output results/noise-2d-<date>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import pathlib
import platform
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from benchmark2d.core import CASES, Surface, evaluation_set, reconstruct, rmse
from benchmarknd.noisy import NoisyObservations
from benchmarknd.strategies import run_arm
from resolution import fit_free_scores
from resolution.noise import ABSOLUTE_FLOOR

CORNERS = np.array(list(itertools.product((0., 1.), repeat=2)))
FOLD_SCALE = .06                 # the fold-band width the 2D scorer already uses
DEFAULT_PEAKS = (0., .05, .10, .20)
DEFAULT_FLOOR = .01
ARMS = ("space-filling", "gpr-var", "gpr-u50-g50", "vwrs", "vurs",
        "vwrs-n", "vurs-n", "vurs-a", "gpr-n")
CLEAN_ONLY = ("space-filling", "gpr-var", "gpr-u50-g50", "vwrs", "vurs")


class Noisy2DSurface:
    """A 2D folded manifold whose spread rises as the fold is approached.

    Distance to the nearest fold plays the role the branch gap plays for the
    microinstability proxies: at the fold two modes are tied, so the reported
    value is the average of a trace that never settled.
    """
    dim = 2

    def __init__(self, case, seed=0, floor=DEFAULT_FLOOR, peak=.20, reducible=True):
        # peak=0 is the clean control: the floor collapses with it rather
        # than leaving a 1% spread on a run labelled noiseless.
        floor = min(floor, peak)
        if not 0 <= floor <= peak:
            raise ValueError("require 0 <= floor <= peak relative spread")
        self.truth = Surface(case, seed)
        self.case, self.seed = case, seed
        self.floor, self.peak, self.reducible = floor, peak, reducible
        self._rng = np.random.default_rng(20260921+1000*seed)
        # A no-fold control has no competition anywhere, so it carries the
        # floor only -- which is the right answer, not a special case.
        self.noise = self

    def __call__(self, x):
        return self.truth(x)

    def competition(self, x):
        distance = np.abs(self.truth.distance(np.atleast_2d(x)))
        return np.where(np.isfinite(distance), np.exp(-distance/FOLD_SCALE), 0.)

    def spread(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        relative = self.floor + (self.peak-self.floor)*self.competition(x)
        return relative*np.abs(self.truth(x)) + ABSOLUTE_FLOOR

    def observe(self, x, replicates=1):
        x = np.atleast_2d(np.asarray(x, float))
        single = self.spread(x)
        effective = single/np.sqrt(replicates) if self.reducible else single
        truth = self.truth(x)
        return truth + effective*self._rng.standard_normal(truth.shape), effective


def checkpoints(start, budget, count=8):
    raw = np.unique(np.round(np.geomspace(start, budget, count)).astype(int))
    return [int(n) for n in raw if start <= n <= budget]


def truth_curvature(surface, x, step=1e-4):
    """Exact second-difference magnitude, for the spine's nonlinear term.

    The 2D evaluation set ships a truth gradient but no curvature, so it is
    computed here rather than approximated from the sampler's own view.
    """
    x = np.atleast_2d(np.asarray(x, float))
    base = surface(x)
    total = np.zeros(len(x))
    for axis in range(2):
        plus, minus = x.copy(), x.copy()
        plus[:, axis] = np.minimum(1., plus[:, axis]+step)
        minus[:, axis] = np.maximum(0., minus[:, axis]-step)
        span = plus[:, axis]-minus[:, axis]
        total += (4*(surface(plus)-2*base+surface(minus))/span**2)**2
    return np.sqrt(total)


def score_design(surface, design, noisy_values, test):
    """Both reconstructions from one design, plus the fit-free spine."""
    truth_at_design = surface.truth(design)
    x, truth, scale = test[0], test[1], test[2]
    out = {}
    for label, values in (("placement", truth_at_design), ("end_to_end", noisy_values)):
        predicted = reconstruct(design, values, x)
        absolute = np.abs(predicted-truth)/scale
        out[f"{label}_error"] = rmse(predicted, truth, scale)
        out[f"{label}_nmae"] = float(np.mean(absolute))
        out[f"{label}_p95"] = float(np.quantile(absolute, .95))
    out["corruption_penalty"] = out["end_to_end_error"]-out["placement_error"]
    out.update(fit_free_scores(design, x, test[5], scale, curvature=test[6]))
    band = test[3]
    if band.any():
        for label, values in (("placement", truth_at_design), ("end_to_end", noisy_values)):
            predicted = reconstruct(design, values, x)
            out[f"{label}_band_nmae"] = float(np.mean(
                np.abs(predicted[band]-truth[band])/scale))
    return out


def execute(task):
    case, seed, arm, peak, budget, checkpoint_count = task
    trial = dict(case=case, seed=seed, arm=arm, noise_peak=peak, budget=budget)
    started = time.perf_counter()
    try:
        surface = Noisy2DSurface(case, seed, peak=peak)
        test = evaluation_set(surface.truth)
        test = (*test, truth_curvature(surface.truth, test[0]))
        obs = NoisyObservations(surface, budget, 2, seed, shared=CORNERS)
        # One BLAS thread per worker. Without this every worker spawns a thread
        # per core and several pools on one machine oversubscribe it badly:
        # measured at about one trajectory per hour instead of minutes.
        with threadpool_limits(limits=1):
            run_arm(arm, obs, seed)
        if obs.spent != budget:
            raise RuntimeError("strategy failed to spend the exact budget")
        design, noisy = obs.x, obs.y
        rows = []
        for n in checkpoints(len(CORNERS)+1, budget, checkpoint_count):
            rows.append(dict(**trial, n=n, status="ok",
                             **score_design(surface, design[:n], noisy[:n], test)))
        rows[-1]["seconds"] = time.perf_counter()-started
        rows[-1]["unique_points"] = int(len(design))
        rows[-1]["max_replicates"] = int(obs.replicates.max())
        # The final design, so placement can be drawn without replaying the run.
        rows[-1]["x"] = design.tolist()
        rows[-1]["y_observed"] = noisy.tolist()
        rows[-1]["replicates"] = obs.replicates.tolist()
        return rows
    except Exception as exc:
        return [dict(**trial, n=0, status="failed", reason=f"{type(exc).__name__}: {exc}")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES),
                        default=["two-plane-four-peaks", "three-plane-three-peaks",
                                 "two-plane-asymmetric"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--noise-peaks", type=float, nargs="+", default=list(DEFAULT_PEAKS))
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="keep completed trajectories from a previous run")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    result_path = args.output/"results.json"
    previous, done = [], set()
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))["rows"]
        done = {(r["case"], r["seed"], r["arm"], r["noise_peak"]) for r in previous
                if r["status"] == "ok" and r["n"] == r["budget"]}
        previous = [r for r in previous
                    if (r["case"], r["seed"], r["arm"], r["noise_peak"]) in done]
    elif result_path.exists():
        parser.error("output exists; pass --resume or choose a new --output")

    tasks = []
    for peak, case, seed, arm in itertools.product(
            args.noise_peaks, args.cases, args.seeds, args.arms):
        # At zero noise a noise-aware arm is the same algorithm as its clean
        # twin, so running both would double the cost for identical rows.
        if peak == 0. and arm not in CLEAN_ONLY:
            continue
        if (case, seed, arm, peak) in done:
            continue
        tasks.append((case, seed, arm, peak, args.budget, args.checkpoints))

    # Run the map case first, highest noise first, so the placement maps the
    # report draws are available long before the full grid finishes.
    tasks.sort(key=lambda t: (not (t[0] == args.cases[0] and t[1] == args.seeds[0]), -t[3]))

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = sorted((root/"resolution").glob("*.py")) + [
        root/"benchmarknd/noisy.py", root/"benchmarknd/strategies.py", pathlib.Path(__file__)]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-checkout"
    payload = dict(config=vars(args) | {"output": str(args.output)},
                   protocol="sampler sees noisy values; placement score reconstructs "
                            "from true values at the sampled design, end_to_end from "
                            "the noisy ones; both graded on the noiseless surface",
                   fold_scale=FOLD_SCALE, noise_floor=DEFAULT_FLOOR, commit=commit,
                   source_hash=hashlib.sha256(
                       b"".join(p.read_bytes() for p in sources)).hexdigest(),
                   python=platform.python_version(), rows=list(previous))
    print(f"{len(tasks)} trajectories on {args.workers} workers "
          f"({len(done)} already complete)", flush=True)
    # A dead worker breaks the whole pool; finished trajectories are already on
    # disk, so report the loss and let --resume pick up the rest.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for finished, future in enumerate(as_completed(futures), 1):
            try:
                rows = future.result()
            except Exception as exc:
                print(f"  worker lost: {type(exc).__name__}: {exc}", flush=True)
                continue
            payload["rows"].extend(rows)
            result_path.write_text(json.dumps(payload, indent=1, allow_nan=False),
                                   encoding="utf-8")
            last = rows[-1]
            note = (f"placement {last['placement_error']:.4f} "
                    f"end_to_end {last['end_to_end_error']:.4f}"
                    if last["status"] == "ok" else last["reason"])
            print(f"[{finished}/{len(tasks)}] {last['case']} s{last['seed']} "
                  f"{last['arm']} peak={last['noise_peak']:.2f} {note}", flush=True)
    failures = [r for r in payload["rows"] if r["status"] != "ok"]
    print(f"saved {result_path} ({len(payload['rows'])} rows, {len(failures)} failures)")


if __name__ == "__main__":
    main()
