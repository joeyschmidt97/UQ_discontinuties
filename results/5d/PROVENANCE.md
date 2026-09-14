# results/5d — provenance

Five-dimensional folded manifolds with axis-weak modes: 4 cases
x 3 seeds x 7 arms, budget 512,
ten log-spaced scored checkpoints, common thin-plate-spline RBF reconstruction
against a 65536-point Sobol set.

Code: `benchmarknd/`.

## Commands

```bash
# one worker per case/seed, all arms, run in parallel
for case in 5d-m1-p1-anis 5d-m2-aligned 5d-m2-disjoint 5d-m2-rotated; do
  for seed in 0 1 2; do
    python -m benchmarknd --cases $case --seeds $seed --budget-5d 512       --output outputs/nd-run/$case-s$seed
  done
done

python -m benchmarknd.collect --run outputs/nd-run --dims 5 --output results
python -m benchmarknd.tables --cases 5d-m1-p1-anis 5d-m2-aligned 5d-m2-disjoint 5d-m2-rotated
python -m benchmarknd.checks          # geometry and scorer validation, printed not saved
```

## Contents

| File | Produced by |
|---|---|
| `results.json` | `benchmarknd.collect`, joining the per-worker files verbatim |
| `figures/performance.png` | `benchmarknd.collect.render_curves` |
| `figures/strength-*.png` | `python -m benchmarknd.tables` |

`collect` refuses to write a dimension until every case, seed and arm finished;
`results.json` carries `complete`, `missing` and one provenance record per worker.

## Environment

Source hash `7ff50f1e0a1c2e69f4585cce371f479a42cd6ee2fbe325fb42f8d1aa5e560cc0` (resumed across 2 source states).
Executed on Windows-10-10.0.26200-SP0, Python 3.11.15, {'numpy': '1.26.4', 'scipy': '1.17.1', 'scikit-learn': '1.9.0', 'matplotlib': '3.11.1'}.
