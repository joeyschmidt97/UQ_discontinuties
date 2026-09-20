"""Dimension-independent variation estimation and fit-free resolution scores.

One implementation shared by the 2D, 3D and high-dimensional benchmarks so the
resolution spine means the same thing at every input dimension.
"""
from .variation import LocalVariation, knn_variation
from .scores import fit_free_scores, spine_targets

__all__ = ["LocalVariation", "knn_variation", "fit_free_scores", "spine_targets"]
