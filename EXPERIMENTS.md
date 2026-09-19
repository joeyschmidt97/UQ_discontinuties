# GP experiments

Run from the repository root. Data format and generation: [data/README.md](data/README.md).
NERSC environment, SG++ setup and full-run commands: [NERSC_RUNBOOK.md](NERSC_RUNBOOK.md).


```sh
python -m scripts.run_gp_experiment --dataset data/2d/smooth/seed-0 --policy uncertainty --nu 0.5 --budget 64 --output outputs/gp/smooth-uncertainty-nu05.json
python -m scripts.run_gp_experiment --dataset data/2d/smooth/seed-0 --policy ucb --beta 2 --nu 0.5 --budget 64 --output outputs/gp/smooth-ucb-nu05.json
python -m scripts.run_gp_experiment --dataset data/6d/ionut-itg-tem-argmax-gamma/seed-0 --policy blend --uncertainty-weight 0.5 --nu 1.5 --budget 256 --output outputs/gp/itg-tem-blend-m15.json
# Repeat with --nu 1.5 and 2.5, keeping dataset, seed and budget identical.
```

Uncertainty acquisition maximizes posterior standard deviation. UCB maximizes
`mean + beta * std`, explicitly favoring high response as well as uncertainty.
`gradient` uses posterior standard deviation times posterior-mean gradient;
`blend` combines independently normalized uncertainty and gradient merits, with
the uncertainty share set by `--uncertainty-weight`.
This is a proposed exploration/exploitation baseline, not integrated-error
minimization. Both share an initial random pool subset of 2*d+1 points. The
runner scores the native GP on the frozen test set at checkpoints; these scores
must not be merged with the legacy Delaunay/RBF leaderboard. Test data never
participate in acquisition. This frozen-pool runner does not yet run sparse-grid,
VWRS or VURS arms.

Matern nu=0.5 is a rough continuous kernel, not a jump model. Comparing it on
these kinked cases is a first test; actual discontinuity families can be added
as separately named generators with their own manifests.

