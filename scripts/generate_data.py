"""Freeze the existing synthetic cases into independent pool/evaluation files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import scipy
from scipy.stats import qmc
from scripts.datasets import surface_for

ROOT = Path(__file__).resolve().parents[1]


def cases_for(dim):
    if dim == 2:
        from benchmark2d.core import CASES
        return list(CASES)
    from benchmarknd.cases import CASES
    return [name for name, spec in CASES.items() if spec["dim"] == dim]


def power_of_two(value):
    value = int(value)
    if value < 16 or value & (value - 1):
        raise argparse.ArgumentTypeError("size must be a power of two >= 16")
    return value


def generate(output, dim, case, seed, pool_size, test_size, pool_seed=1729, test_seed=91479):
    if pool_seed == test_seed:
        raise ValueError("pool and evaluation seeds must differ")
    for value in (pool_size, test_size):
        power_of_two(value)
    destination = Path(output) / f"{dim}d" / case / f"seed-{seed}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}; choose a new output directory")
    surface = surface_for(dim, case, seed)
    x = qmc.Sobol(dim, scramble=True, seed=pool_seed).random_base2(pool_size.bit_length()-1)
    q = qmc.Sobol(dim, scramble=True, seed=test_seed).random_base2(test_size.bit_length()-1)
    pool = dict(x=x, y=surface(x))
    truth = surface(q)
    if dim == 2:
        band = np.zeros(len(q), dtype=bool) if case == "smooth" else surface.distance(q) < .06
        peak = np.min(np.linalg.norm(q[:, None]-surface.centers()[None], axis=2), axis=1) < .1
        region = np.zeros(len(q), dtype=int) if case == "smooth" else surface.region(q)
    else:
        band, peak, region = surface.distance(q) < .06, surface.peak_distance(q) < 2., surface.region(q)
    evaluation = dict(x=q, y=truth, band=band, peak=peak, region=region)
    source_paths = [ROOT / "benchmark2d/core.py"] if dim == 2 else [ROOT / "benchmarknd/core.py", ROOT / "benchmarknd/cases.py"]
    source_paths += [Path(__file__), ROOT / "scripts/datasets.py"]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    destination.mkdir(parents=True)
    for name, values in (("pool", pool), ("evaluation", evaluation)):
        np.savez_compressed(destination / (name + ".npz"), **values)
    manifest = dict(schema_version=1, family="existing-affine-envelope-gaussian-peaks", dimension=dim,
                    case=case, surface_seed=seed, bounds=[[0., 1.]]*dim,
                    measure="uniform unit box", pool_size=pool_size, test_size=test_size,
                    pool_seed=pool_seed, test_seed=test_seed, scale=float(np.ptp(truth)),
                    created_utc=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                    numpy=np.__version__, scipy=scipy.__version__, source_sha256=source_hashes,
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob("*.npz")},
                    note="Synthetic continuous functions with possible kinks, not jump-discontinuous plasma truth. Evaluation labels are scorer-only.")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", nargs="+", type=int, choices=[2,5,8], default=[2,5,8])
    parser.add_argument("--cases", nargs="+", help="optional exact case names; must match selected dimensions")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--pool-size", type=power_of_two, default=4096)
    parser.add_argument("--test-size", type=power_of_two, default=65536)
    parser.add_argument("--pool-seed", type=int, default=1729)
    parser.add_argument("--test-seed", type=int, default=91479)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    available = {case for dim in args.dims for case in cases_for(dim)}
    if args.cases and not set(args.cases) <= available:
        parser.error("case names must belong to the selected dimensions")
    jobs = [(dim, case, seed) for dim in dict.fromkeys(args.dims) for case in cases_for(dim)
            if not args.cases or case in args.cases for seed in dict.fromkeys(args.seeds)]
    for dim, case, seed in jobs:
        if (args.output / f"{dim}d" / case / f"seed-{seed}").exists():
            parser.error("output already contains requested datasets; choose a new --output")
    for dim, case, seed in jobs:
        print(generate(args.output, dim, case, seed, args.pool_size, args.test_size, args.pool_seed, args.test_seed), flush=True)


if __name__ == "__main__":
    main()
