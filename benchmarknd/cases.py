"""Declared case geometry: per-mode axis strengths, fold type and peak counts.

A strength pattern is one character per axis: S strong, M moderate, W weak,
'.' inert. Inert axes carry no envelope slope, no base slope and no peak
curvature, so the case is exactly non-responsive in them. Patterns are the
ground truth plotted in the strength table; the measured table is derived from
the realized surface and may differ where peaks add their own anisotropy.

`fold` selects the sign pattern of the second mode and therefore the fold
orientation, holding every per-mode strength magnitude fixed:
  "axis"  second normal flips sign on the leading strong axis only, so the
          fold hyperplane is exactly perpendicular to that coordinate axis.
  "dense" second normal is the negation of the first, so the fold normal is the
          full strength vector, spread over every active axis.
  "pair"  two modes with disjoint supports; the fold normal is their difference.

Every mode carries at least two strong axes. With a single strong axis the peak
is a slab covering ~40% of the cube and its diagnostic stops being distinct from
the global error. The one-mode cases share the fold cases' strength pattern, so
they are exact no-fold controls for them.
"""
LEVELS = {"S": 1., "M": .3, "W": .05, ".": 0.}

CASES = {
    "5d-m1-p1-anis": dict(dim=5, modes=["SSMW."], peaks=[1], fold=None),
    "5d-m2-aligned": dict(dim=5, modes=["SSMW.", "SSMW."], peaks=[1, 1], fold="axis"),
    "5d-m2-rotated": dict(dim=5, modes=["SSMW.", "SSMW."], peaks=[1, 1], fold="dense"),
    "5d-m2-disjoint": dict(dim=5, modes=["SSM..", "..MSS"], peaks=[1, 2], fold="pair"),
    "8d-m1-p1-anis": dict(dim=8, modes=["SSMW...."], peaks=[1], fold=None),
    "8d-m2-aligned": dict(dim=8, modes=["SSMW....", "SSMW...."], peaks=[1, 1], fold="axis"),
    "8d-m2-rotated": dict(dim=8, modes=["SSMW....", "SSMW...."], peaks=[1, 1], fold="dense"),
    "8d-m2-disjoint": dict(dim=8, modes=["SSM.....", "..MSS..."], peaks=[1, 2], fold="pair"),
}


def strengths(case):
    """Declared strength matrix, one row per mode and one column per axis."""
    spec = CASES[case]
    if any(len(p) != spec["dim"] for p in spec["modes"]):
        raise ValueError(f"{case}: strength pattern length must equal the dimension")
    if len(spec["peaks"]) != len(spec["modes"]):
        raise ValueError(f"{case}: one peak count per mode is required")
    return [[LEVELS[c] for c in pattern] for pattern in spec["modes"]]


def weak_axes(case):
    """Axes a mode ignores (weak or inert), per mode, and the globally inert set."""
    rows = strengths(case)
    per_mode = [[j for j, v in enumerate(row) if v <= LEVELS["W"]] for row in rows]
    inert = [j for j in range(CASES[case]["dim"]) if all(row[j] == 0. for row in rows)]
    return per_mode, inert
