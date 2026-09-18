# Datasets and generators

This directory contains versioned data arrays, manifests and data documentation.

- `2d/`, `5d/`, `8d/`: existing affine-envelope/Gaussian-peak synthetic cases.
- `6d/`: Ionut Farcas's phenomenological microinstability proxies at their native dimension.
- Each case contains `seed-0/pool.npz`, `evaluation.npz`, and `manifest.json`.
- All generated arrays and manifests are tracked in Git. Generators refuse overwrites.

## Existing synthetic data

```sh
python -m scripts.generate_data
python -m scripts.generate_data --seeds 0 1 2 --output data/v2
```

The twelve existing cases have 4,096 pool points and 65,536 independent
reference points each. They are continuous, with possible affine-envelope kinks.

## Ionut's microinstability proxies

Source: https://github.com/ionutfarcas/UQ_discontinuties/tree/13f87b95f90be9dbb5942e317739bb15a1918e0e

Pure numerical definitions are extracted into `scripts/ionut_proxies.py` from
`sgpp_examples/itg_transition_sg.py` and `sgpp_examples/itg_kbm_surrogate.py`.
SG++ and plotting code are not required to generate their values. These are
phenomenological formulas, not fitted experimental data or gyrokinetic solves.

```sh
python -m scripts.generate_ionut_data
```

Nine scalar datasets: ITG/TEM and ITG/KBM, each with argmax/softmax selection
and gamma/omega outputs, plus the separate stellarator-style ITG/KBM blend.
The first eight use normalized columns `[RLTi, RLTe, RLn, nu, beta, ky_scale]`,
mapped to `[3,9]`, `[2,8]`, `[0.5,3.5]`, `[0,0.8]`, `[0,1.5]`, and
`ky=0.30*(0.5+x5)` respectively. Softmax width is T=0.05.
The separate blend uses unit-box `[nref,Tref,aLTi,aLTe,aLn,tau]`.
Default parameters and fixed baseline quantities remain exactly as upstream.

Hard selection chooses the largest branch growth rate: gamma can have kinks
and the selected omega can jump. Softmax smooths the selection, but clipped
square-root branch onsets remain nonsmooth. The blend is a separate smooth
phenomenological target. These proxies contain ITG/TEM/KBM, not ETG or MTM.
No 2D/5D slices or 8D extensions are silently substituted for native 6D data.
The formulas do not use a random surface seed; seed-0 is a common layout label.

## Array and provenance format

`x` has shape (N,d), `y` shape (N,). The independent reference archive also
contains `band`, `peak`, and `region`. Existing synthetic cases have defined
peak and fold masks. For the Ionut cases these masks are false because no
geometric band/peak region is defined; do not interpret empty masks as zero error.
Branch cases additionally store `gamma`, `omega`, `G`, `W`, and `share` in both
archives; G and W have shape (N,2). `region` identifies the largest-growth branch,
including for softmax data; it is not a stable/unstable classification. Ties
follow NumPy's first-branch argmax convention. Branch names are in the manifest.

Manifests include seeds, bounds, probability measure, response range, file
checksums, source hashes and provenance. The frozen values are authoritative;
recreating values requires the matching generator version and parameters.

```python
from scripts.datasets import load_dataset, surface_for
manifest, pool, reference = load_dataset('data/6d/ionut-itg-tem-argmax-omega/seed-0')
f = surface_for(manifest['dimension'], manifest['case'], manifest['surface_seed'])
```

To add a new family, use a dedicated data generator in `scripts/`, document its
coordinate mapping and physical fidelity, and preserve the array/manifest contract.
