# Initial 2-D benchmark results

**Triangle refinement is the strongest overall finalist in this pilot.** It is
the only arm that reaches the global and boundary targets on all three seeds of
all five cases within the tested budgets. This is synthetic evidence for a next
experiment, not a production GENE winner.

[Open the consolidated report](reports/pilot/index.html) or [the plot-reading guide](reports/pilot/README.md).
Five sheets put all methods and cases together: 3-D truth, final-cap point placement,
error-versus-cost curves, residual maps and a scorecard. Curves retain all four
budgets; placement/maps show seed 0 at the largest cap. Actual point counts are shown.

## Cost to reach the targets

Targets were fixed before running: global normalized RMS <= 0.05 and
boundary-band normalized RMS <= 0.10. Both must remain satisfied at later tested
checkpoints. Normalization uses one fixed truth range per case/geometry for all
methods and regions. The cost below is the median of measured qualifying costs
over three seeds; all three must qualify to enter the comparison.

| Case | Lowest median qualifying cost | Method |
|---|---:|---|
| Smooth control | 20 | Scrambled Sobol |
| Kink without peaks | 25 | Regular grid |
| Peaks on the crossing plane | 121 | Regular grid |
| Peaks offset from the plane | 64 | Regular grid and triangle refinement tie |
| Genuine jump plus peaks | 256 | Triangle refinement only |

On offset peaks, the grid needs [64, 121, 64] evaluations across the three seeds;
triangle refinement needs [64, 64, 64]. On the jump case, the grid and GPR
uncertainty arm each pass on only two seeds; every other arm except triangles
passes on none. A good median global error therefore does not suffice to qualify.

## Accuracy at the largest requested budget

Triangle refinement's median common-linear global / boundary errors:

| Case | Global error | Boundary error |
|---|---:|---:|
| Smooth | 0.0030 | 0.0038 |
| Kink | 0.0033 | 0.0090 |
| On-plane peaks | 0.0095 | 0.0213 |
| Offset peaks | 0.0094 | 0.0101 |
| Jump | 0.0293 | 0.0788 |

The ranking is not solely a consequence of scoring with triangles: at the largest
requested cap, triangle refinement also has the lowest median common-RBF error
on the kink, on-plane peaks, offset peaks and jump cases. These are final-cap
comparisons, not equal-actual-cost claims: Sobol and some sparse-grid arms underspend.

## What was run and checked

- **420 experiments:** 7 real strategies x 5 cases x 3 paired seeds/geometries x
  4 budgets. All completed; no unavailable or failed rows. Every row respects its
  evaluation cap, has unique case/seed/budget/arm identity and stores all samples.
- The independent scoring design has 16,384 points. Exact truth is never passed
  to acquisition. Four common corners are charged to every strategy.
- Real SG++ 3.3.1 on Linux/Python 3.11.16 with NumPy 1.26.4. The adapter's native
  matrix lifetime, grid rollback and repair-grid bugs were fixed and tested.
- `sg_lib` external revision: `d13bc4661cbd3901a70190b52c477e7606576fa3`.
- **25 passing regression tests:** 11 benchmark contracts, 6 mock SG++ controls,
  6 compiled SG++ checks, and 2 existing sg_lib budget/prediction regressions.
- Primary package versions, experiment source hash and starting Git revision are
  recorded in `reports/pilot/results.json`. The renderer hash is recorded separately
  in `reports/pilot/render-manifest.json`, since plot layout was refined after the run.
- Native surrogate prediction is optional and was disabled for this matrix; it is
  expensive for sg_lib. Common linear and common RBF scores are present for every row.

## Interpretation and next experiment

Keep **triangle refinement, GPR uncertainty, and the regular/Sobol controls** as
the immediate shortlist. SG++ remains useful as a local-basis comparator. Test
held-out orientations, narrower spikes and different jump sizes before promoting
any method to the four campaign axes. Three seeds are pilot-scale evidence, not
a statistical proof of general superiority.

The triangle policy is a simple observed-gradient disagreement heuristic with
periodic area-based exploration. It can still miss a peak between samples, and
its mesh costs in higher dimensions are untested. The score also deliberately
uses a continuous piecewise-linear reconstruction even across true jumps.

Future GENE selection must include spectrum resolution, failed/retried evaluations,
convergence checks and actual node-hours. Those costs are outside this clean 2-D pilot.

Reproduce the matrix in the Linux environment described in README:

```bash
python -m benchmark2d --cases smooth kink on-plane offset-peaks jump \
  --seeds 0 1 2 --budgets 32 64 128 256 --output outputs/reproduction
```
