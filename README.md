# UQ discontinuities: equal-point 2-D benchmark

Compare simulation-placement strategies by the RMS error of the same low-poly
reconstruction at **every integer number of paid evaluations**. Start with the
[plot guide](reports/pilot/README.md), [five-sheet report](reports/pilot/index.html)
or [results and interpretation](RESULTS.md).

## Surfaces

All inputs are in the unit square. Cases are continuous, with Gaussian peaks
inside mode regions, away from mode switches:

| CLI case | Geometry | Total peaks |
|---|---|---:|
| `smooth` | Smooth baseline with one peak | 1 |
| `two-plane-four-peaks` | Two affine regions, two peaks in each | 4 |
| `three-plane-three-peaks` | Three affine regions, one peak in each | 3 |
| `two-plane-asymmetric` | Two affine regions, two peaks in one and one in the other | 3 |

Folded baselines are maxima of affine planes. Smooth additive bumps preserve
the baseline switches. Seed rotates the geometry; peak centers remain at least
0.15 normalized units from folds. These are synthetic growth-rate proxies, not
validated mode physics. The previous on-fold Gaussian and jump cases are removed.

## Run

Use Python 3.11 and the dependencies in `requirements-sgpp.txt` for the full field.
Real SG++ 3.3.1 requires the compatible Linux/NumPy 1.x environment described there.
Ionut's library is an external checkout selected by `SG_LIB_PATH`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-sgpp.txt
export SG_LIB_PATH=/path/to/sensitivity-driven-sparse-grid-approx
.venv/bin/python -m benchmark2d --quick --output outputs/quick
.venv/bin/python -m benchmark2d --seeds 0 1 2 --budgets 32 64 128 256 --native-test-size 1024 --output outputs/full
.venv/bin/python -m benchmark2d --plots-only --output reports/pilot
```

`--budgets` sets diagnostic checkpoints and the largest run length. It does **not**
launch independent runs at each cap: all N from 4 to the maximum are scored on
one nested trajectory. `--resume` requires identical code/configuration. Results
are saved atomically after each complete trajectory. Use a new output directory
to change the experiment; existing data are not silently overwritten.

For built-in methods on Windows, install `requirements.txt` and explicitly select
`--arms grid moe gpr-var gpr-grad triangles`. Unavailable backends are reported;
mocks never enter rankings.

## Contenders

| Arm | Placement rule |
|---|---|
| `grid` | Progressive dyadic regular grid, maximin ordering within each level |
| `sglib` | Ionut Farcas's sensitivity-driven subspace refinement |
| `sgpp` | Real SG++ modified-linear surplus refinement |
| `gpr-var` | Matern-3/2 GP posterior standard deviation |
| `gpr-grad` | GP uncertainty times predicted gradient, with an exploration floor |
| `triangles` | Triangle area / neighboring-gradient disagreement; area exploration every fifth step |
| `moe` | Locally gated triangle, bilinear-grid and GP experts, sharing one observation budget |

Sobol is removed as a contender. The independent scoring integration set still
uses a fixed scrambled Sobol design; it never enters acquisition.

GP, triangle and mixture policies share five Latin-hypercube initialization
points plus four charged corners. GP policies now acquire one point per fit.
The progressive grid replaces the previous full-square-at-each-cap design so
it can spend every budget exactly. The triangle policy is a proposed heuristic,
not a published high-order simplex stochastic collocation implementation.

### Mixture of experts

The prior pilot's shortlist, excluding Sobol, is frozen before the new experiment:
triangles, grid and GP uncertainty (commit `b4899ad`, old `RESULTS.md`). This is
selection from the prior pilot, not a claim of equal-N ranking in that old data.

All experts fit the **same paid samples**. The grid is a sampling policy, so its
predictive expert is a regular bilinear basis fitted by ridge regression to those
shared observations. Local weights learn squared prediction errors recorded
before each new observation is revealed. Acquisition blends triangle discrepancy,
space filling, GP uncertainty and expert disagreement, with periodic exploration.
No reference test values, peak centers or fold locations reach the mixture.

The primary score tests whether the mixture **places points better**. Optional
`native_error` at final N separately tests its mixed predictor and GP predictors
on the same held-out subset. It does not replace the common score.

## Fair comparison and score

- One unique point is one paid evaluation. Four common corners count for every
  method and guarantee the common triangulation covers the square. Duplicate
  requests are cached. An incomplete trajectory fails rather than padding points.
- Native sparse grids still choose refinement batches. Their prescribed nodes
  are paid in order; a trajectory can finish partway through a batch. Every prefix
  is scored by the common reconstructor, even when a native grid update is not yet
  complete. Zero-tolerance sg_lib and negative stopping-threshold SG++ continuation
  disable convergence-based early exit. sg_lib caps each direction at level 20
  and continues other admissible subspaces, retaining its refinement priorities;
  reaching one axis limit no longer stops the whole fixed-budget experiment.
  Native sparse-grid predictions are omitted for these truncated grid states.
- Truth and geometry are evaluator-only. A fixed independent set of 16,384 points
  scores piecewise-linear Delaunay reconstruction. RMS is normalized by one fixed
  truth range per case/seed, shared by all methods and regions.
- Global, fold-band (distance < 0.06) and peak-region errors are saved at every N.
  Smooth has no fold: its band metric repeats global RMS. Exact common RBF scores
  are secondary cross-checks at the requested diagnostic checkpoints only.
- Curves use log-log axes and show medians/IQR over paired seeds. Every point on
  every curve is an equal-N comparison. Placement and residual sheets show the
  same final N and representative seed for all methods.
- Qualification requires normalized global RMS <= 0.05 and fold-band RMS <= 0.10,
  sustained at **every subsequent integer N through the tested maximum**. Report
  per-case qualifying costs and fixed-N errors; unreached targets remain unreached.
  Qualification is finite-horizon evidence, not guaranteed future monotonicity.
- The final row stores coordinates and observed values once. Earlier designs are
  prefixes of those arrays. Scoring replays cached values without new oracle calls.

## Tests and code

```bash
python -m pytest tests/test_benchmark2d.py -q
python -m pytest tests/test_exact_sparse.py tests/test_sgpp_real.py -q
python -m pytest tests/test_sgpp_arm_mock.py -q
```

Run mocks separately because they replace the compiled backend at import time.
`benchmark2d/core.py` defines geometry/scoring; `strategies.py` sampling;
`mixture.py` shared-budget experts and gates; `report.py` the five comparison
sheets; `__main__.py` trajectories, provenance and resume. Existing `arms/`
adapters remain. Earlier experiments and removed scripts are retained in Git.

Next, confirm the strongest designs on held-out geometries and peak widths,
then add failed/retried evaluations and real GENE node-hour costs before selecting
a production runner.

The saved pilot records both execution source hashes for reused trajectories and
rerun sg_lib trajectories. Exact source archives accompany the results; renderer
provenance is recorded separately. No new observations were generated when
merging the independently executed methods into the final comparison.
