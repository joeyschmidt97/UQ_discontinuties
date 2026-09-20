"""Run the preregistered two-kernel GP pilot over all Ionut conditional 3D slices."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from scripts.generate_ionut_slices import CASES
from scripts.run_gp_experiment import run


CONFIGS = ((1.5, "m15"), (.5, "m05"))


def execute(task):
    dataset, output, budget, nu = task
    if output.exists():
        return str(output), "existing"
    payload = run(dataset, budget, "blend", nu, uncertainty_weight=.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return str(output), "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/3d"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    tasks = []
    for case in CASES:
        dataset = args.data / case / "seed-0"
        for nu, tag in CONFIGS:
            tasks.append((dataset, args.output / f"{case}-gpr-blend-{tag}.json", args.budget, nu))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            path, status = future.result()
            print(f"{status}: {path}", flush=True)


if __name__ == "__main__":
    main()
