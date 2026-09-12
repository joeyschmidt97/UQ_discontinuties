# Equal-point 2-D benchmark results

**The mixture of experts gives the lowest median final reconstruction error on
all four surfaces at exactly 256 paid points.** It also wins the common-RBF
cross-check on all four. It does not reach every error target with the fewest
points: triangles win the smooth case and the progressive grid wins the
three-plane case at the declared targets.

Start with the [plot guide](reports/pilot/README.md) or [single-page report](reports/pilot/index.html).
The five sheets show 3-D truth, point placement, log-log error curves, residual
maps and the combined performance curve; qualifying costs remain in the HTML table. Every method is compared at the same integer N.

## Combined score across all tests

The scorecard pools all four surfaces and three seeds at every matched integer
N: `sqrt(mean(normalized_global_RMS ** 2))`, with equal weight for each of the
12 tests. N is the paid point count **per test**, including the four corners.
Fold and peak diagnostics are not added because they overlap the global metric.
The curve includes a point count only when every method has all 12 valid tests.

| Method | Combined RMS at N = 256 | First N staying at or below 0.05 through 256 |
|---|---:|---:|
| Mixture of experts | 0.008929 | 73 |
| Triangles | 0.012814 | 81 |
| GP unc/grad 50/50 | 0.013909 | 81 |
| GP unc/grad 70/30 | 0.015345 | 72 |
| GP unc/grad 30/70 | 0.016905 | 96 |
| Progressive grid | 0.018417 | 100 |
| GP gradient | 0.019386 | 103 |
| GP uncertainty | 0.019815 | 94 |
| SG++ | 0.034339 | 176 |
| Ionut / sg_lib | 0.042529 | 223 |

The three GP mixtures are the best three of a five-ratio sweep; the other two
are listed in the sweep section below. Their trajectories were executed
separately and joined to the original runs without generating new observations.

MoE reduces the final combined error by about 30% relative to triangles. It does
not win at every budget: triangles are slightly better at N = 32. The combined
0.05 reference is not a guarantee that every individual case qualifies. Use the
per-case curves and qualification table for that question. Machine-readable
curves are in [aggregate-scores.json](reports/pilot/aggregate-scores.json).

## Cost to reach the declared targets

Global normalized RMS <= 0.05 and fold-band normalized RMS <= 0.10 must both
hold at every subsequent integer point count through N = 256. All three seeds
must qualify. Costs below are medians of measured crossings, with no interpolation.

| Surface | Lowest median cost | Method | Per-seed costs |
|---|---:|---|---|
| Smooth, one peak | 21 | Triangles | [21, 21, 26] |
| Two planes, four peaks | 79 | Mixture of experts | [85, 79, 71] |
| Three planes, three peaks | 71 | Progressive grid | [67, 127, 71] |
| Two planes, asymmetric three peaks | 63 | Mixture of experts | [73, 59, 63] |

## Accuracy at exactly 256 points

Median global RMS divided by a fixed truth range, using the same independent
16,384-point integration set and common low-poly reconstructor for every method.
Lower is better. Each column aggregates the same three paired seeds.

| Method | Smooth | Two planes / four peaks | Three planes / three peaks | Asymmetric |
|---|---:|---:|---:|---:|
| Progressive grid | 0.00927 | 0.02169 | 0.01972 | 0.01945 |
| Mixture of experts | 0.00309 | 0.01108 | 0.00923 | 0.00886 |
| Ionut / sg_lib | 0.03586 | 0.03887 | 0.04423 | 0.04237 |
| SG++ | 0.00714 | 0.03880 | 0.03733 | 0.03200 |
| GP uncertainty | 0.00937 | 0.02288 | 0.02110 | 0.02271 |
| GP gradient | 0.00621 | 0.02566 | 0.02010 | 0.02130 |
| GP unc/grad 50/50 | 0.00465 | 0.01734 | 0.01440 | 0.01521 |
| GP unc/grad 70/30 | 0.00869 | 0.01731 | 0.01581 | 0.01640 |
| GP unc/grad 30/70 | 0.00554 | 0.02004 | 0.01802 | 0.01762 |
| Triangles | 0.00480 | 0.01550 | 0.01460 | 0.01289 |

## GP uncertainty / GP gradient ratio sweep

**Every mixture of the two GP acquisition rules beats both of its own
components, and the ranking is single-peaked at the even split.** Weights were
declared before running; the uncertainty share is the first number. The 0% and
100% ends of the sweep are the existing `gpr-grad` and `gpr-var` arms.

| Uncertainty / gradient | Combined RMS at N = 256 | Smooth | Two planes / four peaks | Three planes / three peaks | Asymmetric |
|---|---:|---:|---:|---:|---:|
| 0 / 100 (GP gradient) | 0.019386 | 0.00621 | 0.02566 | 0.02010 | 0.02130 |
| 20 / 80 | 0.018476 | 0.00513 | 0.02203 | 0.02159 | 0.02006 |
| 30 / 70 | 0.016905 | 0.00554 | 0.02004 | 0.01802 | 0.01762 |
| 50 / 50 | 0.013909 | 0.00465 | 0.01734 | 0.01440 | 0.01521 |
| 70 / 30 | 0.015345 | 0.00869 | 0.01731 | 0.01581 | 0.01640 |
| 80 / 20 | 0.017207 | 0.00827 | 0.01914 | 0.01840 | 0.01944 |
| 100 / 0 (GP uncertainty) | 0.019815 | 0.00937 | 0.02288 | 0.02110 | 0.02271 |

The 50/50 mixture reduces the combined error by about 28% relative to GP
gradient and about 30% relative to GP uncertainty, and moves the GP family from
below the progressive grid to just behind triangles. It does not reach the
mixture of experts, and it takes no per-case qualifying-cost crown: the best
blend cost on three planes is N = 72 against the progressive grid's 71, and on
the other three surfaces triangles or the mixture remain cheapest.

The 50/50 arm is the same policy as the earlier `gpr-blend` run and reproduces
it to 6e-17 in every one of its 3,036 rows, so the sweep is anchored to an
already-published trajectory. Only the best three appear in the figures; the
selection reads the benchmark it is plotted on, so it is a display choice rather
than held-out confirmation. The full five-arm ranking is stored under
`selection` in [results.json](reports/pilot/results.json).

## Placement and prediction are different tests

The primary question is which strategy buys the most useful points. On this
criterion the mixture is the strongest final-budget design, supported by the
independent common-RBF reconstruction cross-check.

Its **blended predictor does not beat the standalone uncertainty GP predictor**
on these cases. The optional native diagnostic uses the same first 1,024 held-out
points for both predictors at N = 256; it does not enter the placement ranking.

| Surface | Mixture native RMS | Uncertainty GP native RMS |
|---|---:|---:|
| Smooth, one peak | 0.00358 | 0.00043 |
| Two planes, four peaks | 0.01609 | 0.00788 |
| Three planes, three peaks | 0.01355 | 0.00550 |
| Two planes, asymmetric three peaks | 0.01325 | 0.00626 |

The mixture's experts are triangles, a fitted bilinear grid basis and a GP. They
share one paid sample set. Gates learn only prequential errors; the previous
pilot's shortlist was frozen before looking at these new test results. These
are not three independent 256-point runs hidden behind a 256-point label.

## What changed and what was verified

- Four continuous test surfaces with 1, 4, 3 and 3 interior peaks. Peak centers
  are more than 0.15 normalized units from folds. On-fold peaks and jumps are
  removed. Three seeds rotate the geometry.
- Sobol acquisition is removed. The fixed Sobol integration set remains an
  evaluator-only numerical integration tool.
- **84 trajectories and 21,252 pointwise scores:** seven methods, four surfaces,
  three paired seeds; every N from 4 through 256. All trajectories spend exactly
  256 unique evaluations, including the four shared corners. No failed or
  unavailable trajectories enter the final report.
- Sparse-grid prefixes can finish inside a prescribed batch. Their common
  reconstruction is scoreable at every N; native sparse-grid prediction is
  omitted when the native grid update is incomplete.
- sg_lib fixed-budget mode caps individual axes at level 20 and continues other
  admissible subspaces. This is an explicitly documented fixed-budget variant of
  the native convergence-stopped runner. Its normal convergence mode remains
  the adapter default. Deterministic node caching never shares observations.
- Audited all trajectory identities, budgets, coordinate uniqueness and saved
  truth values. Independently recomputed 504 prefix errors from stored samples.
  Real compiled SG++ and external sg_lib were used, never mocks for ranking.
- 39 regression tests passed: core/sampling/report contracts, compiled sparse-grid
  checks including fixed-budget continuation and cache equivalence, and separate
  SG++ mock controls.
- Exact execution source archives and hashes are saved with the results. Six
  unchanged methods retain their original run data; all twelve sg_lib sequences
  were rerun after correcting continuation. Rendering has separate provenance.
  sg_lib checkout: `d13bc4661cbd3901a70190b52c477e7606576fa3`.

## Next experiment

This ranks sample efficiency. Saved timings include acquisition and scoring,
with different cache warm-up and parallel execution conditions; they are not a
controlled wall-clock benchmark. The level-20 sg_lib configuration has at most
400 native 2-D nodes; larger budgets require a higher per-axis limit.

Confirm the mixture, triangle and progressive-grid designs on held-out rotations,
peak locations and widths. Compare GP/RBF reconstruction on those designs before
choosing the predictor. Then add failed runs, retries, inner-spectrum stopping
and real node-hour accounting for a small GENE pilot. Three synthetic seeds do
not establish a universal or production runner.

Reproduce with the Linux environment in the root README:

```bash
python -m benchmark2d --seeds 0 1 2 --budgets 32 64 128 256 \
  --native-test-size 1024 --test-size 16384 --output outputs/reproduction
```

The superseded pilot and its unequal-count comparisons remain in Git history at
`b4899ad`. These new results replace its rankings.
