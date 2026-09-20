# Ionut conditional 3D microinstability pilot

Open [the scrolling report](index.html). It contains:

1. ITG–TEM reference atlases for hard/soft growth rate and frequency.
2. ITG–KBM reference atlases for the same four outputs.
3. Normalized-RMSE convergence curves through 256 paid evaluations.
4. A final-budget scorecard with NRMSE, NMAE, normalized P95, transition NRMSE and high-response NRMSE.

A scalar response over three inputs has a three-dimensional graph embedded in
four dimensions, so there is no single 3D height surface equivalent to the 2D
benchmark view. Each reference atlas instead shows all three orthogonal planes
through the midpoint. White contours mark equal branch growth.

## Field

The eight cases are conditional slices of Ionut Farcas's unchanged native 6D
phenomenological formulas:

- ITG–TEM varies `(RLTi, RLTe, nu)` while fixing `RLn=2`, `beta=0`, `ky=0.30`.
- ITG–KBM varies `(RLTi, beta, ky_scale)` while fixing `RLTe=5`, `RLn=2`, `nu=0`.
- Both families include argmax and softmax selection and gamma and omega outputs.

Each case uses a 4,096-point frozen acquisition pool and 65,536 independent
reference points. The two algorithms are the 50/50 normalized
gradient–uncertainty GP with Matérn 3/2 and Matérn 1/2 kernels. All 16 runs
reached 256 paid evaluations.

## Main result

Matérn 3/2 had the lower mean final error across the eight cases: mean NRMSE
0.03065 versus 0.03504 for Matérn 1/2, mean normalized P95 0.05565 versus
0.07022, and mean high-response NRMSE 0.02735 versus 0.03663. Matérn 1/2 was
better on some hard-frequency scores, including final NRMSE for both argmax
omega cases. Kernel preference is therefore response-dependent.

The hard ITG–TEM frequency case is the clear unresolved case: final NRMSE stays
near 0.18 and normalized P95 near 0.30 for both kernels. Its discontinuous
branch-selected frequency is substantially harder than the growth-rate and
soft-selection cases, most of which finish near 0.003–0.026 NRMSE.

This report is separate from the synthetic 2D leaderboard. It evaluates native
GP predictions and does not yet compare VWRS, VURS, `sglib`, SG++, triangles or
grid sampling in three dimensions.
