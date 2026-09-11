# UQ discontinuities: 2-D sampling benchmark

Find which simulation-placement strategy reconstructs a kinked or discontinuous
growth-rate surface with the fewest evaluations. This is a synthetic benchmark,
not a GENE runner or a validated physics surrogate.

**Completed pilot:** [results and interpretation](RESULTS.md) / [plotted report](reports/pilot/index.html).
All 420 experiments ran with seven real strategies, including SG++. Triangle
refinement is the strongest overall finalist; cheaper baselines win some easy cases.

## Run

Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m benchmark2d --quick
```

Open `outputs/benchmark2d/index.html`. Each strategy has a 3-D truth/sample/triangle
plot, a paired error-versus-evaluations chart, and an overhead residual map at each
budget. The comparison overview shows median error and the interquartile spread
across paired geometry/seeds. Saved `results.json` contains all sampled coordinates,
values, actual costs, parameters, failures, package versions and a source hash.

The default run covers on-plane peaks, offset peaks and true jumps, three seeds,
and caps of 32, 64 and 128 evaluations:

```powershell
.\.venv\Scripts\python -m benchmark2d --output outputs/pilot
.\.venv\Scripts\python -m benchmark2d --resume --output outputs/pilot
.\.venv\Scripts\python -m benchmark2d --plots-only --output outputs/pilot
.\.venv\Scripts\python -m benchmark2d --cases smooth kink on-plane offset-peaks jump --seeds 0 1 2 3 4 --budgets 32 64 128 256 --output outputs/full
.\.venv\Scripts\python -m pytest tests/test_benchmark2d.py -q
```

Existing results are never silently overwritten. Resume requires matching code
and configuration. Every finished experiment is saved atomically. Failed arms
are recorded and give a nonzero exit status; unavailable optional backends are
listed explicitly and prevent a claim of an overall winner.

## Contenders

| Arm | Placement rule |
|---|---|
| `grid` | Largest complete regular square grid within the cap |
| `sobol` | Complete scrambled Sobol power-of-two block plus four corners |
| `sglib` | Ionut Farcas's sensitivity-driven subspace refinement |
| `sgpp` | Real SG++ modified-linear surplus refinement |
| `gpr-var` | Matern-3/2 GP posterior standard deviation |
| `gpr-grad` | GP uncertainty times predicted gradient, with an exploration floor |
| `triangles` | Proposed area/neighbor-gradient heuristic, largest-triangle exploration every fifth step |

The new GP policies share a fixed five-point Latin-hypercube initialization plus
the four common corners. They use four-point batches with a spatial exclusion
radius. The triangle method uses the same initialization. It is a simple proposed
heuristic, not the published high-order simplex-stochastic-collocation algorithm.
GP fitting warnings and fitted kernels are recorded instead of being hidden.
Use `--native-test-size 128` for an optional secondary comparison of each model's
own predictor. It is disabled by default because sg_lib's per-point prediction
can dominate runtime; both common reconstruction scores always run.

`sglib` uses the external checkout specified by `SG_LIB_PATH`; its Windows default
is `C:\Users\joesc\git\sensitivity-driven-sparse-grid-approx`. It is not vendored.
SG++ requires a real `pysgpp` installation. The PyPI 3.3.1 release provides a Linux
wheel, not a Windows wheel; use a compatible Linux environment or a source build.
The published wheel requires NumPy 1.x; the requirements cap NumPy below 2.
The real adapter was validated on Linux x86_64 with Python 3.11 and the versions
in `requirements-sgpp.txt`. To run the complete field in Linux/WSL:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-sgpp.txt
export SG_LIB_PATH=/path/to/sensitivity-driven-sparse-grid-approx
.venv/bin/python -m pytest tests/test_sgpp_real.py -q
.venv/bin/python -m benchmark2d --output outputs/pilot
```

Never use the mock backend for ranking. A missing backend is **unavailable**, not
an approximation score. Use `--arms grid sobol sglib gpr-var gpr-grad triangles`
to explicitly request only the available field.

## Fairness and winner rule

- Inputs are normalized to the unit square. Observations are clean and deterministic
  in this first pilot. All methods purchase four corner evaluations so the common
  triangulation covers the entire box. External sparse-grid native models still use
  their prescribed nodes; the extra corners support the common reconstruction.
- One unique point costs one evaluation. Repeated requests use the cache. Whole
  batches must fit the remaining budget before any observations are taken.
- Truth, boundary locations, peak masks and scoring points are evaluator-only.
  Search strategies receive observations they purchased, never true residuals.
- A fixed independent scrambled Sobol test set scores common piecewise-linear
  reconstruction. RMS is normalized by the same truth range for every arm and every
  region in a case/seed. Nonfinite predictions fail scoring; no silent fallback.
- Global, boundary-band and peak-region errors are stored. Boundary-band width is
  0.06 in normalized signed distance. A common exact RBF reconstruction cross-checks
  dependence on the triangle-based scorer. Native GP/sparse-grid error is secondary.
- Default qualification: global normalized RMS <= 0.05 and boundary-band RMS <= 0.10.
  Both must pass at that checkpoint and all later tested checkpoints. Report the
  lowest **measured** qualifying cost; do not interpolate an unobserved crossing.
- Per-case cost comparisons require qualification on every seed. Unreached targets
  remain unreached. An incomplete field cannot produce an overall winner. Even a
  complete synthetic pilot does not identify a universal or production GENE winner.
- Plot actual evaluations, including corners, not the nominal cap. Complete grids,
  Sobol blocks and atomic sparse-grid refinements can underspend. Each cap is an
  independent run with matched seeds; designs need not be nested. Point colors for
  the regular grid show enumeration, not acquisition history.

## Code and retained work

`benchmark2d/` is the only current benchmark entry point:

- `core.py`: analytic surfaces, evaluation accounting and common scoring.
- `strategies.py`: search policies and existing sparse-grid adapters.
- `report.py`: synchronized scientific figures and HTML scorecard.
- `__main__.py`: CLI, provenance, atomic saves and resume.

The older six-dimensional pilot, placement/plot scripts, notebook, visual-test
script, their tracked figures and original SG++ example scripts were removed from this branch; Git
history on `benchmark-harness` retains them. Existing `arms/` adapters and their
tests remain. `manifold.py` remains for the future physics-inspired stage.
Uncommitted spectrum/reconstruction work from other checkouts is not overwritten.

## Next gates

1. Real SG++ validation is complete: six compiled-backend tests cover grid/alpha
   consistency at budget stops and failed-value repair. Keep the matrix backing
   a SWIG evaluation alive until multiplication finishes.
2. Inspect the saved case/seed/budget matrix and reconstruction-dependent rankings.
3. Tune only on a separate development case set; confirm finalists on held-out geometries.
4. Add noise/failure accounting and validate inner k_y stopping against dense spectra.
5. Transfer finalists to the four campaign axes, then a small metered GENE pilot.

Credit: [SG++](https://github.com/SGpp/SGpp),
[Farcas sensitivity-driven grids](https://github.com/ionutfarcas/sensitivity-driven-sparse-grid-approx),
[SciPy interpolation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html).
