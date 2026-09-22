"""The native 6D transition mask counts live competition only."""
import numpy as np
import pytest

from benchmarknd.ionut import IonutSurface

TEM, KBM = "ionut-itg-tem-argmax-gamma", "ionut-itg-kbm-argmax-gamma"


def probes(n=20000, seed=0):
    return np.random.default_rng(seed).random((n, 6))


@pytest.mark.parametrize("case", (TEM, KBM))
def test_band_excludes_regions_where_no_branch_grows(case):
    surface = IonutSurface(case)
    x = probes()
    g = surface.branches(x)
    band = surface.distance(x) < surface.band_threshold
    dead = g.max(axis=1) <= surface.band_threshold*surface.branch_scale()
    assert not (band & dead).any()


def test_the_itg_kbm_band_is_small_once_the_dead_zone_is_removed():
    """93% of the gap-only ITG-KBM band was both branches near zero."""
    surface = IonutSurface(KBM)
    x = probes()
    g = surface.branches(x)
    gap_only = np.abs(g[:, 0]-g[:, 1])/surface.branch_scale() <= surface.band_threshold
    live = surface.distance(x) < surface.band_threshold
    assert gap_only.mean() > .10
    assert live.mean() < .03


def test_the_mask_does_not_depend_on_the_query_batch():
    """The scale used to be the range of whatever batch was passed in."""
    surface = IonutSurface(TEM)
    x = probes(4000)
    whole = surface.distance(x)
    pieces = np.concatenate([surface.distance(x[i:i+7]) for i in range(0, len(x), 7)])
    single = np.array([surface.distance(p[None, :])[0] for p in x[:50]])
    np.testing.assert_array_equal(whole, pieces)
    np.testing.assert_array_equal(whole[:50], single)
