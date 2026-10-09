"""Bayesian surprise scores of a paid design on a pool.

A design is good if, after paying for it, the runs it did not pay for are no
longer surprising. Surprise of an unseen run is its negative log posterior
predictive probability under one common model fitted on the paid runs -- the
same model for every arm, so the score grades placement, not each arm's own
surrogate. Two parts, matching the two things a campaign must learn:

- label surprise, in bits: -log2 p(mode | x) under a Dirichlet-kNN classifier.
  The prior is a symmetric Dirichlet over the pool's modes (Jeffreys, alpha
  1/2), so with no paid neighbours every mode is equally likely and the
  surprise is log2 K; the posterior adds the label counts of the 2d+1 nearest
  paid runs.
- growth-rate surprise, in nats: -log p(gamma | x) under a mode mixture.
  Each mode's component is a GP fitted on that mode's paid runs (Matern-5/2,
  ARD, white noise); the mixture weights are the label posterior above. A mode
  with too few paid runs to fit its own GP uses the GP fitted on all paid
  runs. gamma is divided by its pool standard deviation first, a constant
  shared by every arm.

The scorer kernel is Matern-5/2 on purpose: it is not the kernel of any arm
(VURS uses Matern-1/2, the GP arms Matern-3/2), so no arm grades itself.
Scores are averaged per true mode and then across modes with equal weight,
as `pool.per_mode_scores` does for the reconstruction error.
"""
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

ALPHA = .5          # Jeffreys prior on each mode
LN2 = np.log(2.)
STD_FLOOR = .05     # predictive std floor, in pool-std units of gamma


def label_posterior(paid_x, paid_labels, query, n_modes, k=None, alpha=ALPHA):
    """Posterior predictive p(mode | x) of a Dirichlet-kNN classifier, one row per query."""
    k = min(k or 2*paid_x.shape[1]+1, len(paid_x))
    _, nearest = cKDTree(paid_x).query(query, k=k)
    nearest = np.asarray(nearest).reshape(len(query), k)
    counts = np.zeros((len(query), n_modes))
    for column in nearest.T:
        counts[np.arange(len(query)), paid_labels[column]] += 1
    return (counts+alpha)/(k+n_modes*alpha)


def gp_label_posterior(paid_x, paid_labels, query, n_modes, alpha=ALPHA):
    """Posterior predictive p(mode | x) of a GP classifier, shrunk toward the uniform prior.

    One-vs-rest Laplace GP classifiers, Matern-5/2 with ARD length scales, the
    scorer kernel of the growth-rate GPs. Modes absent from the paid runs get
    probability only through the shrinkage, which adds alpha pseudo-counts per
    mode against N paid runs: p = (N p_gpc + alpha) / (N + K alpha). It plays
    the part of the Jeffreys prior in the Dirichlet-kNN posterior, and unlike a
    k-neighbour vote the GP's confidence is not tied to the nearest-run
    distance through a fixed stencil.
    """
    n = len(paid_x)
    probs = np.zeros((len(query), n_modes))
    seen = np.unique(paid_labels)
    if len(seen) == 1:
        probs[:, seen[0]] = 1.
    else:
        gpc = GaussianProcessClassifier(
            kernel=ConstantKernel(1., (1e-3, 1e3))*Matern([.3]*paid_x.shape[1], (1e-2, 1e2), nu=2.5),
            random_state=0, multi_class="one_vs_rest")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gpc.fit(paid_x, paid_labels)
        probs[:, gpc.classes_] = gpc.predict_proba(query)
    return (n*probs+alpha)/(n+n_modes*alpha)


def fit_scorer_gp(x, y):
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (1e-3, 1e3))*Matern([.3]*x.shape[1], (1e-2, 1e2), nu=2.5)
        + WhiteKernel(1e-2, (1e-6, 1.)),
        normalize_y=True, random_state=0, n_restarts_optimizer=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x, y)
    return gp


def gaussian_log_density(y, mean, std):
    return -.5*np.log(2*np.pi*std**2)-.5*((y-mean)/std)**2


def macro(values, labels, n_modes):
    per_mode = [float(np.mean(values[labels == k])) if (labels == k).any() else None
                for k in range(n_modes)]
    live = [v for v in per_mode if v is not None]
    return float(np.mean(live)), per_mode


def surprise_scores(oracle, paid_index, held_index=None, k=None, alpha=ALPHA, std_floor=STD_FLOOR,
                    classifier="knn", return_points=False):
    """Label and growth-rate surprise of the held points after paying for `paid_index`.

    Held points default to every unpaid pool point, as in `pool.score_prefix`.
    Also returns the label model's own entropy (what it expects its surprise
    to be); held surprise above that entropy is overconfidence.

    `classifier` picks the label posterior: "knn" (Dirichlet-kNN, the
    registered default) or "gpc" (`gp_label_posterior`). With `return_points`
    the per-point surprises and the held pool indices are added, for scoring
    a region of the pool.
    """
    n_modes = len(oracle.label_names) or int(oracle.labels.max())+1
    paid = np.zeros(len(oracle.pool), bool)
    paid[paid_index] = True
    held = ~paid if held_index is None else np.isin(np.arange(len(oracle.pool)), held_index)
    px, pl = oracle.pool[paid], oracle.labels[paid]
    scale = float(np.std(oracle.y))
    py, hx, hy, hl = oracle.y[paid]/scale, oracle.pool[held], oracle.y[held]/scale, oracle.labels[held]

    if classifier == "knn":
        post = label_posterior(px, pl, hx, n_modes, k, alpha)
    elif classifier == "gpc":
        post = gp_label_posterior(px, pl, hx, n_modes, alpha)
    else:
        raise ValueError(f"unknown classifier {classifier!r}")
    label_bits = -np.log(post[np.arange(len(hl)), hl])/LN2
    entropy_bits = -np.sum(post*np.log(post), axis=1)/LN2

    overall = fit_scorer_gp(px, py)
    mean, std = overall.predict(hx, return_std=True)
    components = np.empty((len(hx), n_modes))
    own_gp = []
    for m in range(n_modes):
        source = pl == m
        tail = np.column_stack([np.ones(source.sum()), px[source]])
        if source.sum() >= px.shape[1]+2 and np.linalg.matrix_rank(tail) >= px.shape[1]+1:
            mode_mean, mode_std = fit_scorer_gp(px[source], py[source]).predict(hx, return_std=True)
            own_gp.append(m)
        else:
            mode_mean, mode_std = mean, std
        components[:, m] = gaussian_log_density(hy, mode_mean, np.maximum(mode_std, std_floor))
    gamma_nats = -logsumexp(components, axis=1, b=post)
    global_nats = -gaussian_log_density(hy, mean, np.maximum(std, std_floor))

    label_macro, label_modes = macro(label_bits, hl, n_modes)
    gamma_macro, gamma_modes = macro(gamma_nats, hl, n_modes)
    names = oracle.label_names or [str(m) for m in range(n_modes)]
    out = dict(
        n=int(paid.sum()),
        s_label_bits=label_macro,
        s_label_mode_bits=dict(zip(names, label_modes)),
        s_label_pooled_bits=float(np.mean(label_bits)),
        label_entropy_bits=macro(entropy_bits, hl, n_modes)[0],
        s_gamma_nats=gamma_macro,
        s_gamma_mode_nats=dict(zip(names, gamma_modes)),
        s_gamma_median_nats=float(np.median(gamma_nats)),
        s_gamma_global_nats=macro(global_nats, hl, n_modes)[0],
        modes_with_own_gp=[names[m] for m in own_gp],
        prior_label_bits=float(np.log2(n_modes)),
    )
    if return_points:
        out.update(held_index=np.flatnonzero(held), label_bits=label_bits, gamma_nats=gamma_nats)
    return out


def prequential_label_bits(oracle, order, start, k=None, alpha=ALPHA):
    """Surprise of each newly paid run's label, given only the runs paid before it.

    The sum over a trajectory is the label code length of what the arm bought:
    high early values mean the arm was finding runs its posterior did not expect.
    """
    n_modes = len(oracle.label_names) or int(oracle.labels.max())+1
    order = np.asarray(order)
    bits = []
    for n in range(start, len(order)):
        post = label_posterior(oracle.pool[order[:n]], oracle.labels[order[:n]],
                               oracle.pool[order[n:n+1]], n_modes, k, alpha)
        bits.append(float(-np.log2(post[0, oracle.labels[order[n]]])))
    return bits
