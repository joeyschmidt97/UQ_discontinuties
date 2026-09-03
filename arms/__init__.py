"""Benchmark arms. Heavy/optional backends are imported lazily."""

from .base import Arm, Score, score_arm, design_metrics, RandomNearestArm

__all__ = ["Arm", "Score", "score_arm", "design_metrics", "RandomNearestArm",
           "SGppArm", "SGppRegularArm", "HAVE_PYSGPP",
           "SgLibArm", "HAVE_SG_LIB", "GPRArm", "HAVE_SKLEARN"]

_LAZY = {
    "SGppArm": "sgpp_arm", "SGppRegularArm": "sgpp_arm", "HAVE_PYSGPP": "sgpp_arm",
    "SgLibArm": "sglib_arm", "HAVE_SG_LIB": "sglib_arm",
    "GPRArm": "gpr_arm", "HAVE_SKLEARN": "gpr_arm",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(name)
    import importlib
    return getattr(importlib.import_module(f".{mod}", __package__), name)
