# 5D holistic-tolerance calibration — 2026-09-20

The holistic score $H$ divides each component by a tolerance. Those tolerances
are dimension-local: the 2D set was calibrated against a regular grid at N=128
in two dimensions and has no meaning at another dimension. `benchmarknd.core`
now refuses to compute $H$ without an explicit tolerance set, so 5D needs its
own calibration before any scored 5D comparison.

## What was run

Reference arm `space-filling`, which is the high-dimensional substitute for the
dyadic grid used as the 2D reference (the grid and tetrahedral refinement are
both Delaunay-based and do not survive past three dimensions).

```bash
python -m benchmarknd \
  --cases 5d-m1-p1-anis 5d-m2-aligned 5d-m2-rotated 5d-m2-disjoint \
  --arms space-filling --seeds 0 1 2 \
  --budget-5d 512 --checkpoints 10 --test-size 65536 \
  --output results/nd-calibration-2026-09-20/5d
```

All four 5D cases, three seeds, budget 512, 65,536 independent Sobol reference
points. 120 rows, 12 complete trajectories, zero failures. Code state is
commit `bf6c8b0`; `results.json` carries the exact source hash and versions.

## Measured reference values at N=512

Across twelve case/seed trajectories.

| Component | Min | Median | Max | 2D tolerance | Reference ratio at 2D tolerance |
|---|---:|---:|---:|---:|---:|
| `nmae` | 0.0176 | 0.0265 | 0.0347 | 0.05 | 0.53 |
| `band_nmae` | 0.0190 | 0.0273 | 0.0323 | 0.10 | 0.27 |
| `p95_error` | 0.0740 | 0.1028 | 0.1274 | 0.15 | 0.69 |
| `vwfd_p95` | 0.2583 | 0.5117 | 0.6660 | 0.25 | **2.05** |

Per case, median over three seeds:

| Case | `nmae` | `band_nmae` | `p95_error` | `vwfd_p95` |
|---|---:|---:|---:|---:|
| `5d-m1-p1-anis` | 0.0177 | — (no fold) | 0.0777 | 0.2954 |
| `5d-m2-aligned` | 0.0264 | 0.0273 | 0.1087 | 0.5239 |
| `5d-m2-disjoint` | 0.0266 | 0.0231 | 0.0990 | 0.5072 |
| `5d-m2-rotated` | 0.0325 | 0.0309 | 0.1122 | 0.5822 |

**Porting the 2D tolerances would have been wrong in a specific, measurable
way.** The baseline arm already satisfies all three reconstruction-based terms
by a factor of 1.4 to 3.7 while missing the VWFD term by 2.05x. $H$ at 5D would
therefore have been a VWFD ranking with three inert terms attached -- the same
pathology measured in 3D, where VWFD P95 was the binding term in 225 of 264
final rows, but worse.

## Proposed 5D tolerances

Calibrated the way the 2D VWFD tolerance was: take the reference arm's worst
case at the declared budget and round up to the next round number, so the
reference sits at $H \approx 1$ and every term is live.

| Component | Reference max | Proposed 5D tolerance |
|---|---:|---:|
| `nmae` | 0.0347 | 0.035 |
| `band_nmae` | 0.0323 | 0.035 |
| `p95_error` | 0.1274 | 0.13 |
| `vwfd_p95` | 0.6660 | 0.70 |

Provisional normalized tolerances at budget 512, not fitted accuracy claims.
They are valid only for d=5 at this budget and must not be carried to 8D.

## The evaluator caveat, now quantified

Three of the four terms above (`nmae`, `band_nmae`, `p95_error`) are computed
through the common thin-plate-spline RBF, which this work demoted to a
secondary cross-check. This run sharpens why the demotion was right and bounds
how wrong it is.

Normalized RMS against budget, `space-filling`, seed 0:

| Case | N=11 | N=61 | N=218 | N=512 |
|---|---:|---:|---:|---:|
| `5d-m1-p1-anis` | 0.0690 | 0.0546 | 0.0510 | 0.0497 |
| `5d-m2-aligned` | 0.0849 | 0.0629 | 0.0606 | 0.0550 |
| `5d-m2-disjoint` | 0.0960 | 0.0532 | 0.0432 | 0.0383 |
| `5d-m2-rotated` | 0.1342 | 0.0793 | 0.0780 | 0.0675 |

The RBF evaluator behaves correctly here: error falls with budget on every case.
On the same `5d-m1-p1-anis` surface it *rose* from 0.1213 to 0.2887 between
N=218 and N=512 under `gpr-grad`, whose design concentrates 74% of its points
inside the peak mask.

So the failure is not the evaluator in isolation -- it is the evaluator's
interaction with concentrated designs. That is the worst possible property for
a comparator in this study, because concentration is precisely the behaviour
under test: the scorer rewards the baseline and penalizes the methods whose
merit is being measured. Treat `nmae`, `band_nmae` and `p95_error` as valid for
space-filling-like designs and unreliable for concentrating ones.

The spine term is clean by contrast. `vwfd_p95` falls monotonically with budget
on every case and seed, requires no reconstruction, and grades against exact
finite-difference variation of the analytic surface:

| Case | N=11 | N=61 | N=218 | N=512 |
|---|---:|---:|---:|---:|
| `5d-m1-p1-anis` | 0.578 | 0.442 | 0.316 | 0.258 |
| `5d-m2-aligned` | 1.124 | 0.875 | 0.665 | 0.524 |
| `5d-m2-disjoint` | 0.951 | 0.733 | 0.542 | 0.436 |
| `5d-m2-rotated` | 1.479 | 1.094 | 0.845 | 0.666 |

## Open decision

Two ways to build the 5D $H$, not resolved by this calibration:

1. **Ported shape** — keep the four 2D components with the recalibrated
   tolerances above. Comparable in form to the published 2D and 3D scores, but
   three of four terms ride an evaluator known to invert on concentrated
   designs.
2. **Spine-only** — build $H$ from fit-free components alone (`vwfd_p95`,
   `nonlinear_p95`, `fill_p95`), and report the reconstruction-based family
   beside it rather than inside it. Every term then survives to 8D unchanged
   and none can be inverted by concentration, at the cost of no longer
   resembling the 2D and 3D $H$.

Reference values for the spine-only option are available in the same file:
`nonlinear_p95` ranges 1.1260 to 2.4054 with median 1.9100, and `fill_p95` is
0.2689 to 0.2698 with median 0.2694 -- the fill-distance term is nearly
constant across cases because `space-filling` optimizes exactly it.
