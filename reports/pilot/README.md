# Plot guide â€” start here

Five overview images consolidate the saved results. All 7 methods and
all 4 test cases are shown together. **Every integer point count from 4 to 256 is scored on one nested trajectory per method and seed.**

Read these in order:

1. [True 3-D manifolds](figures/01-manifold-reference.png) â€” what is being sampled.
2. [Point placement](figures/02-point-placement.png) â€” all methods side by side; rows are surfaces, columns are methods.
3. [Error versus points](figures/03-error-versus-points.png) â€” the main accuracy/cost comparison, all methods on each chart.
4. [Error maps](figures/04-reconstruction-error-map.png) â€” where the low-poly reconstruction misses a spike or boundary.
5. [Performance scorecard](figures/05-performance-scorecard.png) â€” one combined error-versus-points curve per method, pooling all surfaces and seeds.

Or open [the single scrolling report](index.html), which includes all five sheets.

## How to read the figures

- **Placement/reference/maps:** one representative paired seed (0), at the largest point count (256). N printed on placement panels is the actual number of paid samples, including four shared corners. Every method spends exactly this many points.
- **Dot colors:** dark = early, yellow = late within that design. Progressive-grid order is geometric, independent of observed values. Pale gray contours show the true surface; teal/cyan marks the true mode-switch boundary. The smooth control has no boundary.
- **Error curves:** medians across all 3 seeds/geometries; bands are the middle 50%, not confidence intervals. Top row is global error; bottom is error near the boundary. The smooth control has no fold; its lower panel repeats global error.
- **Axes and colors:** 3-D height scales, convergence axes and normalized residual colors are shared. Residuals above 0.5 use the brightest color. The reference height is a synthetic growth-rate proxy, not calibrated GENE output.
- **Winning:** smaller error with fewer actual evaluations is better. Qualification requires global RMS/range <= 0.05 and boundary RMS/range <= 0.1, sustained through subsequent tested point counts. A dense-looking cluster of dots is not itself evidence of accuracy.
- **Snapshot versus curve:** the dots show one seed; curves summarize all seeds. Different seeds rotate the geometry. Do not expect the snapshot's individual error to equal the median curve.
- **Budgets:** one trajectory per case/seed/method, scored at every integer N. Every curve compares identical N across methods. Sparse-grid batches are evaluated in prescribed order; a prefix may end inside a batch before its native surrogate can be updated. This compares point placement through the common low-poly reconstruction, not native-model update frequency.

## Combined scorecard

For each N, scorecard 05 computes `sqrt(mean(error**2))` over all
12 case/seed tests, where each error is global RMS
divided by that test's fixed truth range. Cases and seeds have equal weight.
This is a pooled normalized RMS, not an arithmetic mean of RMS values or a sum
of overlapping global/fold/peak metrics. N is points **per test**; total suite
cost per method is 12 times N. Both axes are logarithmic.
All configured tests and methods must be present at N for that point to appear.
Legend values are final aggregate errors. No uncertainty band is implied.

The dotted aggregate reference is not an all-tests qualification rule: easy
cases can offset difficult cases. Retain sheet 03 and the HTML qualification
table when diagnosing individual failures. [Combined data](aggregate-scores.json)
contains the formula, weights and every plotted value.

## Methods and geometry

Sobol acquisition is removed. A fixed scrambled Sobol **integration set** remains
independent of every sampler; it is only used to measure error fairly.

The mixture's frozen shortlist is triangles, grid and GP uncertainty from the
previous pilot (commit b4899ad). The grid predictive expert uses bilinear basis
regression on the shared observations. Local gates use prequential errors, with
uniform exploration and model disagreement guiding acquisition. It pays for one
shared sample per step, not three separate simulation runs. Ionut / sg_lib uses fixed-budget continuation: each direction is capped at level
20, while other admissible subspaces continue. Native surplus priorities remain
in effect. Deterministic node geometry is cached; observations are never shared
between trials. The common low-poly
score measures whether this design places points better; optional native error
measures the mixture predictor separately. This is a new hybrid, not an average
of three independent full-budget runs.

Cases: smooth + one peak; two planes + two peaks per region (four total);
three planes + one peak per region; two planes + two peaks in one region and
one in the other. Gaussian centers have a positive margin from every fold.
No jump or on-fold Gaussian case is included.

## Pilot takeaway

Methods qualifying on every seed of every case: **Progressive grid, Mixture of experts, GP uncertainty, GP gradient, Triangles**.
Read the HTML qualification table for per-case costs; qualifying everywhere does not mean
winning every case. These results select finalists for harder tests, not a
production GENE runner. See [the detailed results](../../RESULTS.md).

## Regenerate without new experiments

From the repository root, using the configured Python environment:

```bash
python -m benchmark2d --plots-only --output reports/pilot
```

All coordinates, sample values, costs, per-seed errors, peak-region metrics and
the common-RBF cross-check remain in [results.json](results.json). The new overview
does not repeat every intermediate 3-D view; the full learning curves retain all
integer point counts. RBF checks are saved only at configured checkpoints; coordinates are stored once at the final N and sliced for earlier prefixes. The previous experiments remain available in Git history.
