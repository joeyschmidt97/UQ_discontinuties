"""Generate declared 3D slices of Ionut's native 6D transition proxies.

The formulas are unchanged. Each slice varies three physically interpretable
coordinates and fixes the other normalized coordinates at documented baseline
values. These are conditional slices, not dimension-reduced replacements for
the native 6D benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc

from scripts.generate_data import power_of_two
from scripts.generate_ionut_data import UPSTREAM, values as native_values

ROOT = Path(__file__).resolve().parents[1]

SLICE_SPECS = {
    "itg-tem": dict(active=(0, 1, 3), columns=("RLTi", "RLTe", "nu"),
                    fixed=(.5, .5, .5, 0., 0., .5)),
    "itg-kbm": dict(active=(0, 4, 5), columns=("RLTi", "beta", "ky_scale"),
                    fixed=(.5, .5, .5, 0., 0., .5)),
}
CASES = tuple(f"ionut-{kind}-3d-{mode}-{out}" for kind in SLICE_SPECS
              for mode in ("argmax", "softmax") for out in ("gamma", "omega"))


def expand(case, x):
    x = np.atleast_2d(np.asarray(x, float))
    if case not in CASES or x.shape[1] != 3 or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("known case and finite unit-box 3D inputs required")
    kind = "itg-tem" if "itg-tem" in case else "itg-kbm"
    spec = SLICE_SPECS[kind]
    full = np.tile(np.asarray(spec["fixed"], float), (len(x), 1))
    full[:, spec["active"]] = x
    return full


def values(case, x):
    full = expand(case, x)
    kind = "itg-tem" if "itg-tem" in case else "itg-kbm"
    mode = "argmax" if "argmax" in case else "softmax"
    out = "gamma" if case.endswith("gamma") else "omega"
    native_case = f"ionut-{kind}-{mode}-{out}"
    result = native_values(native_case, full)
    return {**result, "x": np.atleast_2d(np.asarray(x, float)), "native_x": full}


def generate(output, case, pool_size=4096, test_size=65536):
    power_of_two(pool_size); power_of_two(test_size)
    path = Path(output)/"3d"/case/"seed-0"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    pool = values(case, qmc.Sobol(3, scramble=True, seed=1729).random_base2(pool_size.bit_length()-1))
    test = values(case, qmc.Sobol(3, scramble=True, seed=91479).random_base2(test_size.bit_length()-1))
    test.update(band=np.zeros(test_size, dtype=bool), peak=np.zeros(test_size, dtype=bool),
                region=np.argmax(test["G"], axis=1))
    kind = "itg-tem" if "itg-tem" in case else "itg-kbm"
    mode = "argmax" if "argmax" in case else "softmax"
    out = "gamma" if case.endswith("gamma") else "omega"
    spec = SLICE_SPECS[kind]
    sources = [ROOT/"scripts/ionut_proxies.py", ROOT/"scripts/generate_ionut_data.py",
               Path(__file__), ROOT/"scripts/datasets.py"]
    manifest = dict(
        schema_version=1, family="ionut-phenomenological-microinstability-3d-slice",
        dimension=3, case=case, surface_seed=0, bounds=[[0., 1.]]*3,
        measure="uniform conditional 3D slice", pool_size=pool_size, test_size=test_size,
        pool_seed=1729, test_seed=91479, scale=float(np.ptp(test["y"])),
        numpy=np.__version__, scipy=scipy.__version__, created_utc=datetime.now(timezone.utc).isoformat(),
        upstream_commit=UPSTREAM, upstream_repository="https://github.com/ionutfarcas/UQ_discontinuties",
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        native_dimension=6, native_case=f"ionut-{kind}-{mode}-{out}", columns=list(spec["columns"]),
        active_native_indices=list(spec["active"]), fixed_native_coordinates=list(spec["fixed"]),
        mode=mode, output=out, branches=["ITG", kind.split("-")[1].upper()],
        diagnostics={"band": "undefined; empty mask", "peak": "undefined; empty mask"},
        note="Conditional 3D slice of the unchanged native 6D proxy; not a learned dimensional reduction.")
    path.mkdir(parents=True)
    for name, data in (("pool", pool), ("evaluation", test)):
        np.savez_compressed(path/(name+".npz"), **data)
    manifest["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.npz")}
    (path/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path, default=ROOT/"data")
    parser.add_argument("--pool-size", type=power_of_two, default=4096)
    parser.add_argument("--test-size", type=power_of_two, default=65536)
    args = parser.parse_args()
    for case in args.cases:
        if (args.output/"3d"/case/"seed-0").exists():
            parser.error("requested output exists; select a new output directory")
    for case in dict.fromkeys(args.cases):
        print(generate(args.output, case, args.pool_size, args.test_size), flush=True)


if __name__ == "__main__":
    main()
