# Shared synthetic datasets

Data inputs live here; experiment outputs remain under `outputs/` or `results/`.

```
data/
  2d/<case>/seed-<seed>/
  5d/<case>/seed-<seed>/
  8d/<case>/seed-<seed>/
    manifest.json
    pool.npz
    evaluation.npz
scripts/
  generate_data.py
  datasets.py
  run_gp_experiment.py
```

Run from the repository root using its Python environment:

```sh
python -m scripts.generate_data
# A separate version with multiple surface seeds:
python -m scripts.generate_data --seeds 0 1 2 --output data/v2
# Generate just one case:
python -m scripts.generate_data --dims 2 --cases smooth --output data/pilot
```

Defaults generate all twelve existing cases (four each in 2D/5D/8D), surface
seed 0, a 4,096-point scrambled Sobol pool and a separate 65,536-point test set.
Seeds, unit-box measure, truth range, source hashes, versions and file checksums
are recorded in each manifest. Existing datasets are never overwritten.
Generated numerical files are ignored by Git; regenerate using the command above.

`pool.npz` contains `x` (N,d) and `y` (N,). `evaluation.npz` contains independent
`x`, `y`, and diagnostic `band`, `peak`, `region` arrays. Evaluation labels and
masks are scorer-only. Pool labels are revealed only when their index is acquired.
Initial acquisitions count toward the budget. Smooth cases have no fold band.
These surfaces are continuous with possible kinks, not true jump-discontinuous
plasma data. No new physics model is claimed.

```python
from scripts.datasets import load_dataset, surface_for
manifest, pool, test = load_dataset("data/5d/5d-m2-rotated/seed-0")
# For continuous-space methods, including sparse grids:
oracle = surface_for(manifest["dimension"], manifest["case"], manifest["surface_seed"])
# Wrap oracle in the benchmark's metered Observations before acquisition.
```

Frozen arrays remain usable after code changes. Recreating a continuous oracle
requires matching the source hashes in the manifest; do not silently mix changed
truth code with an older test set. Legacy benchmark CLIs remain unchanged and
retain their original reconstruction/scoring protocol.

## GP comparisons

```sh
python -m scripts.run_gp_experiment --dataset data/2d/smooth/seed-0 --policy uncertainty --nu 0.5 --budget 64 --output outputs/gp/smooth-uncertainty-nu05.json
python -m scripts.run_gp_experiment --dataset data/2d/smooth/seed-0 --policy ucb --beta 2 --nu 0.5 --budget 64 --output outputs/gp/smooth-ucb-nu05.json
# Repeat with --nu 1.5 and 2.5, keeping dataset, seed and budget identical.
```

Uncertainty acquisition maximizes posterior standard deviation. UCB maximizes
`mean + beta * std`, explicitly favoring high response as well as uncertainty.
This is a proposed exploration/exploitation baseline, not integrated-error
minimization. Both share an initial random pool subset of 2*d+1 points. The
runner scores the native GP on the frozen test set at checkpoints; these scores
must not be merged with the legacy Delaunay/RBF leaderboard. Test data never
participate in acquisition. This runner intentionally does not rerun the older
sparse-grid, mixture or gradient policies.

Matern nu=0.5 is a rough continuous kernel, not a jump model. Comparing it on
these kinked cases is a first test; actual discontinuity families can be added
as separately named generators with their own manifests.

For an additional family, add a dedicated generator in `scripts/` and retain
this array/manifest contract. Give new families separate directories or a new
output root. Keep empirical datasets labeled by source and fidelity.
