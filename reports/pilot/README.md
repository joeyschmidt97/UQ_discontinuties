# Plot guide — start here

Five overview images consolidate the saved results. All 7 methods and
all 5 test cases are shown together. **The 420 saved experiment records are unchanged.**

Read these in order:

1. [True 3-D manifolds](figures/01-manifold-reference.png) — what is being sampled.
2. [Point placement](figures/02-point-placement.png) — all methods side by side; rows are surfaces, columns are methods.
3. [Error versus points](figures/03-error-versus-points.png) — the main accuracy/cost comparison, all methods on each chart.
4. [Error maps](figures/04-reconstruction-error-map.png) — where the low-poly reconstruction misses a spike or boundary.
5. [Performance scorecard](figures/05-performance-scorecard.png) — which methods meet both targets and at what cost.

Or open [the single scrolling report](index.html), which includes all five sheets.

## How to read the figures

- **Placement/reference/maps:** one representative paired seed (0), at the largest requested cap (256). N printed on placement panels is the actual number of paid samples, including four shared corners. The regular grid, Sobol blocks and sparse grids can underspend.
- **Dot colors:** dark = early, yellow = late within that design. Regular-grid colors show enumeration, not adaptive decisions. Pale gray contours show the true surface; teal/cyan marks the true crossing or jump boundary. The smooth control has no boundary.
- **Error curves:** medians across all 3 seeds/geometries; bands are the middle 50%, not confidence intervals. Top row is global error; bottom is error near the boundary. In the smooth control, this band is only a reference strip.
- **Axes and colors:** 3-D height scales, convergence axes and normalized residual colors are shared. Residuals above 0.5 use the brightest color. The reference height is a synthetic growth-rate proxy, not calibrated GENE output.
- **Winning:** smaller error with fewer actual evaluations is better. Qualification requires global RMS/range <= 0.05 and boundary RMS/range <= 0.1, sustained through subsequent tested checkpoints. A dense-looking cluster of dots is not itself evidence of accuracy.
- **Snapshot versus curve:** the dots show one seed; curves summarize all seeds. Different seeds rotate the geometry. Do not expect the snapshot's individual error to equal the median curve.
- **Budgets:** each requested cap is an independent run with matched seeds, not necessarily a prefix of the next design.

## Pilot takeaway

Methods qualifying on every seed of every case: **Triangles**.
Read the scorecard for per-case costs; qualifying everywhere does not mean
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
budget checkpoints. The previous figure layout remains available in Git history.
