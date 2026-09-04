# Figures

Tracked on purpose. A fresh clone should be able to *see* the results before
deciding whether to spend ~20 min regenerating them.

| file | what it shows | regenerate with |
|---|---|---|
| `verification_sheet.png` | **every test assertion, rendered** — measured value, pass threshold, PASS/FAIL per panel | `python tests/test_visual.py` |
| `fig_manifold_slices.png` | γ, ω_r and the competing-mode indicator on a 2-D slice, for three difficulty settings | `python plot_manifold.py` |
| `fig_manifold_sweeps.png` | 1-D sweeps: the kink in γ, the ω_r sign flip, the band gap | `python plot_manifold.py` |
| `fig_manifold_stats.png` | branch occupancy, band volume fraction, band-gap footprint | `python plot_manifold.py` |
| `fig_placement_argmax_gap.png` | where each method spent its budget, coloured by acquisition order | `python plot_placement.py` |
| `fig_targeting_curve_argmax_gap.png` | when each method stops exploring and starts exploiting | `python plot_placement.py` |
| `fig_marginals_argmax_gap.png` | per-axis marginals — the dimension-adaptivity readout | `python plot_placement.py` |

`report.ipynb` carries its own embedded copies plus the accuracy-vs-budget
sweeps, so it is readable on GitHub without running anything.

Everything else written at runtime goes to `outputs/`, which is **not** tracked.
