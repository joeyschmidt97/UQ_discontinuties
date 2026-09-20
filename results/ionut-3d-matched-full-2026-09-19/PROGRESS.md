# Matched 3D benchmark progress — 2026-09-19 23:29 -06:00

This is an in-progress checkpoint. `results.json` and rendered figures are not
committed at this checkpoint because the benchmark process is still writing
them. The final result commit must replace this status with completed counts and
validated report artifacts.

## Protocol in flight

- Eight conditional Ionut proxy surfaces: ITG–TEM and ITG–KBM, each with
  argmax/softmax selection and gamma/omega outputs.
- Three acquisition seeds per surface.
- Twelve declared arms: grid, `sglib`, SG++, tetrahedral refinement, six GP
  policies, VWRS and VURS.
- Eleven arms are locally runnable. SG++ is explicitly recorded unavailable
  because `pysgpp` is not installed; no values are substituted.
- One nested trajectory per case/seed/arm, scored at every integer paid-point
  count from the eight shared cube corners through N=256.
- Common piecewise-linear tetrahedral reconstruction on 16,384 independent
  frozen reference points. RBF is retained only as a secondary cross-check at
  N=64, 128, 192 and 256.
- Metrics: global RMS/NMAE/P95; transition RMS/NMAE; high-response RMS/NMAE;
  per-branch RMS/NMAE; fill-distance P95/maximum; minimum separation; VWFD
  RMS/P95/coverage; holistic H; final native predictor error when available.

## Snapshot

- Completed runnable trajectories: **58 / 264**.
- Runnable trajectories remaining: **206**.
- SG++ unavailability records written: **6**.
- Runnable-arm failures: **0**.
- Cases started: hard ITG–TEM gamma and hard ITG–TEM omega.
- Hard ITG–TEM gamma is complete for all eleven runnable arms and all three seeds.
- The run remains active with four worker processes. `sglib` is much slower than
  pointwise policies in 3D because it constructs native sparse-grid batches.

## What the completed portion shows

On hard ITG–TEM gamma at N=256, the lowest global errors are around 0.015–0.016
for grid, GP uncertainty, Matérn-1/2 GP uncertainty and VURS. The holistic
ranking differs: Matérn-1/2 50/50 blend and GP gradient have the lowest H because
their VWFD scores are stronger, while `sglib` finishes at global NRMSE 0.0439
and H=1.82. This confirms that global RMS alone changes the apparent winner.

The partially completed hard ITG–TEM frequency case is much harder. At seed 0,
the Matérn-3/2 50/50 blend has global NRMSE 0.115 and H=2.19; VWRS and VURS are
near 0.12 global NRMSE, while the pure gradient policies rise to roughly
0.18–0.25. Every completed arm still has H well above one because the frequency
jump drives transition and tail error. Gradient targeting is therefore not
automatically beneficial on a discontinuous selected-frequency response.

These are interim observations, not final rankings. Six surfaces and most of the
second surface remain incomplete at this snapshot.

## Work remaining

1. Finish hard ITG–TEM omega for all seeds and arms.
2. Run soft ITG–TEM gamma/omega and all four ITG–KBM cases.
3. Finish the remaining native `sglib` trajectories and SG++ availability records.
4. Render reference, placement, per-case convergence, pooled error, branch/design
   diagnostic and final holistic-score sheets.
5. Validate expected trajectory/row counts, inspect the report figures, rerun the
   combined tests and commit the final raw results and report separately.
