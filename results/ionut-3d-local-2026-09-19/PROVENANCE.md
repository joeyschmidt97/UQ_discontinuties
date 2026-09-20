# Provenance

- **Run date:** 2026-09-19
- **Branch:** `codex/benchmark-2d`
- **Parent code commit:** `e6a9c22` (`Add full local 2D benchmark results`)
- **Upstream proxy commit:** `13f87b95f90be9dbb5942e317739bb15a1918e0e`
- **Platform:** local Windows Python environment
- **Status:** 16 of 16 configured runs completed at 256 evaluations

The dataset, expanded scoring, suite runner, renderer and results are included
in the same repository commit. Generate the eight frozen datasets with:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_ionut_slices --output data
```

Run and render the pilot with:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_ionut_3d_suite --output results\ionut-3d-local-2026-09-19\runs --budget 256 --workers 4
.\.venv\Scripts\python.exe -m scripts.render_ionut_3d_report --data data\3d --results results\ionut-3d-local-2026-09-19\runs --output results\ionut-3d-local-2026-09-19
```

All acquisition uses the 4,096-point pool. The 65,536-point frozen reference
set participates only in scoring. The transition mask is
`abs(G_ITG - G_other) <= 0.05 * pooled branch-growth range`; the high-response
mask is the top response decile of the frozen reference set. Scores introduce
no Delaunay, RBF or other common reconstruction surface.
