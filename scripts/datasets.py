"""Load frozen datasets; continuous oracles retain arbitrary-node sampling."""
import hashlib
import json
from pathlib import Path
import numpy as np


def surface_for(dim, case, seed):
    if dim == 3:
        from scripts.generate_ionut_slices import CASES, values
        if case not in CASES or seed != 0:
            raise ValueError('3D Ionut slices require a known case and surface seed 0')
        return lambda x: values(case, x)['y']
    if dim == 6:
        from scripts.generate_ionut_data import CASES, values
        if case not in CASES or seed != 0:
            raise ValueError('Ionut proxies require a known case and surface seed 0')
        return lambda x: values(case, x)['y']
    if dim == 2:
        from benchmark2d.core import Surface
        return Surface(case, seed)
    from benchmarknd.core import SurfaceND
    surface = SurfaceND(case, seed)
    if surface.dim != dim:
        raise ValueError("case does not match dimension")
    return surface


def load_dataset(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported dataset schema")
    arrays = {}
    for name in ("pool", "evaluation"):
        target = path / (name + ".npz")
        if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["sha256"][target.name]:
            raise ValueError("dataset checksum mismatch: " + target.name)
        with np.load(target, allow_pickle=False) as data:
            arrays[name] = {key: data[key].copy() for key in data.files}
        x, y = arrays[name]["x"], arrays[name]["y"]
        if x.shape != (len(y), manifest["dimension"]) or y.ndim != 1:
            raise ValueError("invalid dataset shapes")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite dataset")
    return manifest, arrays["pool"], arrays["evaluation"]
