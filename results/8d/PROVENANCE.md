# results/8d — provenance

Eight-dimensional folded manifolds with axis-weak modes, budget 1024. Same
cases, arms, checkpoint scheme and RBF scorer as `results/5d`.

Code: `benchmarknd/`.

## Status

**Runs in flight.** Only the strength tables are published here so far; the
trajectories are still being collected, and `benchmarknd.collect` will refuse to
write `results.json` until every case, seed and arm has finished.

## Commands

```bash
for case in 8d-m1-p1-anis 8d-m2-aligned 8d-m2-rotated 8d-m2-disjoint; do
  for seed in 0 1 2; do
    python -m benchmarknd --cases $case --seeds $seed --budget-8d 1024       --output outputs/nd-run/$case-s$seed
  done
done

python -m benchmarknd.collect --run outputs/nd-run --dims 8 --output results
python -m benchmarknd.tables --cases 8d-m1-p1-anis 8d-m2-aligned 8d-m2-rotated 8d-m2-disjoint
```

## Contents

| File | Produced by |
|---|---|
| `figures/strength-*.png` | `python -m benchmarknd.tables` |
| `results.json` | `benchmarknd.collect` — pending |
