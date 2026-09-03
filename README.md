# UQ_discontinuities — sparse-grid contender benchmark

Which sampler can steer a semi-autonomous NSTX linear-GENE scan when the target
is **discontinuous**: `sg_lib` (sensitivity-driven, Farcaș), **SG++** (spatially
adaptive, local bases), or **GPR**?

The blocking question is not "which is more accurate on a smooth function" — it
is what happens at mode crossings, at rational-surface band gaps, and when runs
come back noisy or not at all. This repo answers that on a free oracle, before
any GENE time is spent.

Status: **all four contenders implemented.** sg_lib and GPR run here today;
SG++ needs a `pysgpp` build that this machine does not have.

---

## Layout

```
report.ipynb                READ THIS FIRST — results, plots, findings, limits
manifold.py                 the test oracle (numpy only)
plot_manifold.py            sanity figures — look at these before trusting a run
arms/base.py                Arm interface + scoring + design geometry + floor arm
arms/sgpp_arm.py            SG++ adaptive and regular arms   (needs pysgpp)
arms/sglib_arm.py           sensitivity-driven dimension-adaptive sparse grid
arms/gpr_arm.py             GP regression + batch active learning (3 acquisitions)
run_pilot.py                sweep cases x budgets x arms -> results.json + figures
plot_placement.py           WHERE the budget goes and HOW next points are picked
tests/test_arms.py          tests for sg_lib + GPR (backends installed)
tests/test_sgpp_arm_mock.py control-flow tests against a mock pysgpp
sgpp_examples/              the two original exploratory scripts (superseded)
outputs/                    figures and results.json
```

## Quick start

```bash
jupyter lab report.ipynb                # the report (plots + findings)
python plot_manifold.py                 # eyeball the manifold
python tests/test_sgpp_arm_mock.py      # arm control-flow tests (no pysgpp needed)
python run_pilot.py --quick             # short sweep
python run_pilot.py                     # full pilot sweep
python run_pilot.py --figures-only      # re-plot from outputs/results.json
python plot_placement.py --case argmax+gap --budget 300   # placement figures
python tests/test_arms.py               # sg_lib + GPR tests
```

Everything except `arms/sgpp_arm.py` runs on numpy + matplotlib alone.

---

## The manifold

Six axes, the frozen first-campaign electron-channel knobs plus `tau` and a
`k_y` selector:

| axis | range | meaning |
|---|---|---|
| `Te_ped_scale` | 0.70–1.30 | pedestal-top $T_e$ scale factor |
| `ne_ped_scale` | 0.70–1.30 | pedestal-top $n_e$ scale factor |
| `w_Te_scale` | 0.70–1.30 | $T_e$ pedestal width scale factor |
| `w_ne_scale` | 0.70–1.30 | $n_e$ pedestal width scale factor |
| `Ti_Te` | 0.50–2.00 | $\tau = T_i/T_e$ |
| `ky_scale` | 0–1 | selects $k_y\rho_s \in [0.05, 1.20]$ |

Pedestal-top values and widths set the normalized gradients, $\beta_e$ and the
collisionality, so the axes are **coupled** the way the real scan's are — not
six independent knobs.

Four competing branches: **ITG**, **TEM**, **KBM**, **MTM**. Amplitudes were
grid-searched for near-uniform dominant-branch occupancy over the box:

```
ITG 0.284   TEM 0.304   KBM 0.330   MTM 0.082
competing-mode band volume fraction : 0.069
band-gap footprint (>1% of max gamma): 0.045
```

MTM's smaller share is structural, not a mistune — it is a low-$k_y$ mode, so it
can only win at the low-$k_y$ end of the $k_y$ axis.

### The four difficulty knobs

| knob | what it creates |
|---|---|
| `mode='softmax'`, `T` | hybridization width as a **fraction of local peak $\gamma$**. `T` is a continuous dial from trivially interpolable to worst case; `T -> 0` is exactly `mode='argmax'` |
| `mode='argmax'` | hard mode selection: **C0 kink** in $\gamma$, **jump** in $\omega_r$ |
| `align='gap'` | rational-surface alignment gate on MTM: a **true jump discontinuity**, standing in for the SLiM-style $\mu$ feature. `'smooth'` gives band structure without the jump; `'off'` disables it |
| `noise_rel`, `noise_abs`, `fail_rate` | convergence scatter and runs that come back as `NaN` |

### Two contracts the comparison depends on

**Clean truth vs. observations.** `evaluate()` never lies; `observe()` adds noise
and failures. Errors are always scored against clean truth, so an arm that fits
noise gets punished.

**Frozen noise field.** Noise and failures are a deterministic hash of the point
coordinates, not a random stream. Two arms that visit the same point see the same
value — otherwise a noisy comparison is measuring luck.

### Why the outputs are what they are

- `gamma` — only **kinked** under argmax.
- `omega` — genuinely **jumps** when the dominant branch flips diamagnetic
  direction. An arm that looks acceptable on `gamma` can be hopeless on `omega`,
  which is why both are swept. This bears directly on the open roadmap decision
  about the sparse grid's driving QoI.
- `mix` — competing-mode indicator in [0,1], computed from the **branch growth
  rates**, not the weights, so the same physical band is flagged under argmax
  (where weights are one-hot) as under softmax. Damped by how unstable the point
  actually is, so the quiescent corner — where every branch ties at
  $\gamma \approx 0$ — is not flagged as the most interesting region in the box.
  This is the mask "target mixed-mode regions" is scored against.
- `label`, `share`, `mu`, `weights`, `G`, `W` — per-branch detail.

One call returns all of them: one $k_y$-scan node yields every candidate QoI at
once, mirroring the real scan, so only the *steering* QoI has to be chosen.

---

## Scoring

Budget is metered in **function evaluations**, the only currency shared by a
point-adaptive grid, a dimension-adaptive grid and a free-placement acquisition
loop — one SG++ point, one Leja node and one GPR acquisition are all exactly one
GENE run.

| metric | question it answers |
|---|---|
| `nrmse` | overall accuracy, normalized by the QoI's range so $\gamma$ and $\omega_r$ compare |
| `rmse_mixed` | accuracy **inside the competing-mode band** — the number the exercise is about |
| `rmse_calm` | accuracy in the bulk, for contrast |
| `frac_design_mixed` | what fraction of the **budget** landed in the band — targeting, not accuracy. Compare against the band's volume fraction (~0.069) as the random-sampling baseline |
| `n_failed` | how many prescribed nodes came back `NaN` |

Design geometry, scored separately from accuracy (`design_metrics`) � an arm can
be accurate on average and still never place a point where the sensitivity
analysis needs one:

| metric | question |
|---|---|
| `band_lift` | in-band design fraction / the band's own volume fraction. **1.0 = no targeting at all**, >1 = genuinely seeking mode competition |
| `frac_peak` | fraction of the design in the top decile of \|QoI\| � is it finding the max-growth-rate region? 0.10 = no targeting |
| `min_dist` | smallest pairwise gap; small means the design clumped |
| `hole` | largest unsampled void; small means good coverage |
| `coverage` | std/mean of nearest-design-point distance (Gunzburger); 0 = perfectly regular |
| `axis_spread` | per-axis spread vs a uniform design; uneven = the method concentrated on some axes |

## Arms — and how each picks its next points

This is the part that separates them. All three are "adaptive"; none of them
adapt over the same thing.

| arm | what it refines | granularity | follows an oblique boundary? | failed run |
|---|---|---|---|---|
| `sglib` | multiindices (subspaces), scored by the variance their spectral coefficients carry | a whole subspace, atomic | **no** — refinement is axis-aligned | hole in the interpolant |
| `sgpp-*` | individual grid points, by hierarchical surplus | one point | partly — local bases, but nodes stay dyadic | hole in the interpolant |
| `gpr-*` | nothing prescribed — acquisition scores a free candidate pool | one point, anywhere | **yes** | dropped sample |

GPR acquisitions, because the right one depends on the objective:

- `gpr-var` — predictive standard deviation. Space-filling early, then drawn to
  whatever is hardest to predict.
- `gpr-ucb` — mean + kappa * sigma. **Targets max growth rate.**
- `gpr-grad` — sigma * |grad mean|. **Targets the transition / mixed-mode
  manifold** — uncertainty on a steep slope, not on a smooth peak.

Batches use a greedy distance penalty; without it every point in a batch lands
on the same argmax and the batch is wasted.

- `sgpp-surplus` / `sgpp-volume` — spatially adaptive, ModLinear, surplus vs.
  surplus-times-volume refinement
- `sgpp-regular` — non-adaptive regular grid at the largest level fitting the
  budget. Says how much of the adaptive score comes from adaptivity rather than
  from sparse grids as such
- `sgpp-bspline` — ModBspline basis (experimental; not all pysgpp builds
  hierarchise it)
- `random-nn` — random design + nearest neighbour. The floor: any contender that
  cannot beat it at equal budget is not a contender

`nan_policy` on the SG++ arms makes the failed-run contract explicit. A sparse
grid prescribes its nodes, so a `NaN` is a hole in the interpolant, not a dropped
sample — `'fill'` substitutes the current interpolant's own value (zero surplus,
hole papered over), `'zero'` is the worst case, `'error'` asserts a clean run.

---

## sg_lib: the empty-step trap

Worth writing down because the obvious port is wrong and fails quietly.

`do_one_adaption_step_preproc()` retires the highest-scoring multiindex from the
active set and returns **only its newly admissible successors** � which is often
none, because a successor still needs its other predecessors admitted first. The
reference script (`adapt_code_one_step_at_a_time.py`) hides this by looping a
fixed number of steps and ignoring the return length.

Reading an empty return as "converged" stops the method after ~3 steps and ~10
points **regardless of the budget** � a ~20x under-spend that looks like
legitimate early convergence, since nothing errors. Correct loop: count the empty
step, call `check_termination_criterion()`, and continue while the active set is
non-empty. `tests/test_arms.py::test_sglib_spends_a_real_budget` guards it.

In 6-D at a 250-point budget, roughly a third of adaption steps come back empty.

## Known limits

- **`pysgpp` is not installed on this machine and has no wheel for CPython 3.13
  on Windows.** The SG++ arm has therefore never been run against real SG++. Its
  control flow — budget cap, refinement loop, NaN repair, early stop, regular
  level selection — is covered by `tests/test_sgpp_arm_mock.py` against a mock;
  the **pysgpp API signatures are unverified**. Expect to fix call signatures on
  first contact with a real install.
- The manifold is **not a gyrokinetic solve**. Coefficients are plausible, not
  calibrated. It reproduces the *structure* of mode competition, not NSTX growth
  rates.
- No `sg_lib` arm and no GPR arm yet — so the pilot currently compares SG++
  variants against a floor, not the three contenders against each other.
- The alignment gate is a stand-in for the SLiM $\mu$ feature, driven by a
  rational-surface count. It is the right *shape* of discontinuity, not a
  reproduction of any published band structure.

## Next

1. `arms/gpr_arm.py` — GP regression with a nugget (the noise/failure arm).
2. `arms/sglib_arm.py` — wrap `C:\Users\joesc\git\sensitivity-driven-sparse-grid-approx`
   via the existing `SparseScanSession` ask/tell interface.
3. Sweep `T` continuously rather than at three points: error-vs-budget as a
   function of feature sharpness is the crux figure.
4. Sweep `fail_rate` — the property most likely to decide this in a
   semi-autonomous run.

## Credit

SG++ — <https://github.com/SGpp/SGpp>, free to use with credit.
Sensitivity-driven sparse grids — Farcaș et al., *J. Comput. Phys.* **410**
(2020) 109394, arXiv:1812.00080.
