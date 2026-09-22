"""Embed the native 6D Ionut proxy and read sampler placement on it.

    python -m scripts.embed_6d --output results/embedding-6d-<date>

Ground rules, from the VWRS/VURS note's sampling-space display section:

* Every distance, neighbourhood, segment and score is computed in the scaled
  feature space, never in the embedding. UMAP and PaCMAP are display only.
* One fixed probe cloud, embedded once, so methods and budgets share a frame.
* Two projections side by side, each with its neighbourhood preservation
  (trustworthiness) reported, before any island is read as structure.

What is embedded. A uniform 6D input box embeds as a featureless blob, so the
inputs alone cannot produce islands. Islands appear only when the embedding
also sees the response. Features are therefore the scaled inputs plus the
standardized response block -- growth rate, frequency and both branch growth
rates -- multiplied by a response weight w:

    w = 0    pure parameter space
    w > 0    the response pulls points of one branch together

The branch label (argmax of the branch growth rates) is ground truth here,
which is the point of running this on a proxy before real GENE data: every
tool can be checked against a known answer.

Tools ported from the MGKDB fingerprint study (fmc.analysis.concordance):

* kNN branch assortativity -- mean fraction of each probe's k=15 feature-space
  neighbours sharing its branch;
* concordance coherence g -- the same fraction per probe; low g marks a probe
  living among the other branch. On MGKDB the low-g survivors were the
  mixed-mode worklist. Here they should sit on the branch transition, which is
  checked directly against the transition mask;
* segmentation -- HDBSCAN in feature space with a bootstrap adjusted-Rand
  stability gate, compared with the true branch by ARI.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.cluster import HDBSCAN
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from scripts.generate_ionut_data import values
from resolution.noise import competition

CASE = "ionut-itg-tem-argmax-gamma"
INPUTS = ("R/L_Ti", "R/L_Te", "R/L_n", "ν", "β", "k_y scale")
BRANCH_NAMES = ("ITG", "TEM")
K = 15
TRANSITION = .05                 # branch-gap fraction, as in the benchmark mask
PLACEMENT_ARMS = ("space-filling", "gpr-var", "gpr-u50-g50", "gpr-grad", "vwrs", "vurs")


def probe_cloud(count, seed=4242):
    return qmc.Sobol(6, scramble=True, seed=seed).random_base2(int(np.log2(count)))


# Feature views, after the MGKDB study's circularity control. The branch label
# is argmax of the branch growth rates, so any view containing them makes branch
# coherence true by construction: measured assortativity 1.000 and a transition
# AUC of exactly 0.5. "branches" is kept only to show that leak.
VIEWS = {"observables": ("gamma", "omega"), "branches": ("gamma", "omega", "G0", "G1")}


def response_block(case, x, view="observables"):
    v = values(case, x)
    columns = dict(gamma=v["gamma"], omega=v["omega"], G0=v["G"][:, 0], G1=v["G"][:, 1])
    return v, np.column_stack([columns[name] for name in VIEWS[view]])


def features(x, block, stats, weight):
    """Scaled inputs beside the standardized response, weighted by w."""
    mean, sd = stats
    return np.hstack([x, weight*(block-mean)/sd])


def coherence(feature, labels, k=K):
    index = NearestNeighbors(n_neighbors=k+1).fit(feature).kneighbors(feature)[1][:, 1:]
    return (labels[index] == labels[:, None]).mean(axis=1)


def stable_segments(feature, seed=0, rounds=10, fraction=.8, min_cluster=60):
    """HDBSCAN segments plus mean bootstrap ARI on the overlapping points."""
    base = HDBSCAN(min_cluster_size=min_cluster, copy=True).fit_predict(feature)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(rounds):
        take = rng.choice(len(feature), int(fraction*len(feature)), replace=False)
        again = HDBSCAN(min_cluster_size=min_cluster, copy=True).fit_predict(feature[take])
        scores.append(adjusted_rand_score(base[take], again))
    return base, float(np.mean(scores)), float(np.std(scores))


def embed(feature, method, seed=0):
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise SystemExit("umap-learn is not installed in this environment") from exc
        return umap.UMAP(n_neighbors=K, min_dist=.1, random_state=seed).fit_transform(feature)
    if method == "pacmap":
        try:
            import pacmap
        except ImportError as exc:
            raise SystemExit("pacmap is not installed in this environment") from exc
        return pacmap.PaCMAP(n_components=2, n_neighbors=K, random_state=seed).fit_transform(feature)
    raise ValueError(method)


def designs(run, case, seed=0):
    payload = json.loads((pathlib.Path(run)/"results.json").read_text(encoding="utf-8"))
    out = {}
    for row in payload["rows"]:
        if (row["status"] == "ok" and row["case"] == case and row["seed"] == seed
                and row["n"] == row["budget"] and row.get("x") is not None):
            out[row["arm"]] = np.asarray(row["x"])
    return out


def scatter(axis, xy, color, size, title, cmap=None, norm=None, **kw):
    order = np.argsort(size)
    image = axis.scatter(xy[order, 0], xy[order, 1], c=color[order] if np.ndim(color) else color,
                         s=size[order], cmap=cmap, norm=norm, linewidths=0, **kw)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([]); axis.set_yticks([])
    return image



# The two views the first sweep said to show side by side. Parameter space
# locates where the modes compete (low coherence finds the transition band);
# observable space separates the modes into islands, via the sign of omega.
DISPLAY = (("parameter space", "observables", 0.), ("observable space", "observables", 1.))


def sweep_views(x, weights):
    """Every view and weight, all measured in feature space."""
    rows = []
    for view in VIEWS:
        truth, block = response_block(CASE, x, view)
        stats = (block.mean(axis=0), block.std(axis=0))
        branch = np.argmax(truth["G"], axis=1)
        gap = np.abs(truth["G"][:, 0]-truth["G"][:, 1])/float(np.ptp(truth["G"]))
        transition = gap <= TRANSITION
        for w in weights:
            if view != "observables" and w == 0:
                continue                      # identical to observables at w = 0
            f = features(x, block, stats, w)
            g = coherence(f, branch)
            seg, stab, stab_sd = stable_segments(f)
            rows.append(dict(view=view if w else "inputs only", weight=w,
                             circular=view == "branches",
                             assortativity=float(g.mean()),
                             segments=int(len(set(seg)) - (1 if -1 in seg else 0)),
                             noise_fraction=float((seg == -1).mean()),
                             segment_branch_ari=float(adjusted_rand_score(branch, seg)),
                             stability_ari=stab, stability_sd=stab_sd,
                             transition_auc=float(roc_auc_score(transition, 1-g))))
            if rows[-1]["segments"] == 0:
                # All-unassigned agrees with itself trivially; not a stability.
                rows[-1]["stability_ari"] = rows[-1]["stability_sd"] = None
            stability = ("n/a" if rows[-1]["stability_ari"] is None
                         else f"{rows[-1]['stability_ari']:.2f}")
            print("{view:>16} w={weight:<4} assort {assortativity:.3f}  segments {segments}  "
                  "ARI {segment_branch_ari:.3f}  ".format(**rows[-1])
                  + f"stability {stability}  transition AUC {rows[-1]['transition_auc']:.3f}",
                  flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--probes", type=int, default=8192)
    parser.add_argument("--weights", type=float, nargs="+", default=[0., .25, .5, 1., 2.])
    parser.add_argument("--run", type=pathlib.Path,
                        default=pathlib.Path("results/6d-ionut-spine-2026-09-20"))
    args = parser.parse_args()
    figures = args.output/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    x = probe_cloud(args.probes)
    truth, block = response_block(CASE, x, "observables")
    stats = (block.mean(axis=0), block.std(axis=0))
    branch = np.argmax(truth["G"], axis=1)
    gap = np.abs(truth["G"][:, 0]-truth["G"][:, 1])/float(np.ptp(truth["G"]))
    transition = gap <= TRANSITION
    contest = competition(truth["G"])
    gamma, omega = truth["gamma"], truth["omega"]
    size = 2 + 40*(gamma-gamma.min())/max(float(np.ptp(gamma)), 1e-12)
    branch_colors = np.array(["#3b6fd6", "#e0702b"])

    sweep = sweep_views(x, args.weights)

    # Embed each display view once, with both projections.
    shown = []
    sub = np.random.default_rng(1).choice(len(x), min(3000, len(x)), replace=False)
    for label, view, w in DISPLAY:
        f = features(x, block, stats, w)
        g = coherence(f, branch)
        seg, stab, _ = stable_segments(f)
        for method in ("pacmap", "umap"):
            xy = embed(f, method)
            trust = float(trustworthiness(f[sub], xy[sub], n_neighbors=K))
            print(f"{label} / {method}: trustworthiness {trust:.3f}", flush=True)
            shown.append(dict(label=label, view=view, weight=w, method=method, xy=xy,
                              trust=trust, feature=f, coherence=g, segments=seg, stability=stab))

    # Sheet 1: overview. One row per view x projection.
    figure, axes = plt.subplots(len(shown), 4, figsize=(20, 4.7*len(shown)), layout="constrained")
    for i, item in enumerate(shown):
        xy = item["xy"]
        head = f"{item['label']} — {item['method'].upper()} (trust {item['trust']:.3f})"
        scatter(axes[i, 0], xy, branch_colors[branch], size, f"{head}\nbranch, size = γ", alpha=.8)
        image = scatter(axes[i, 1], xy, contest, size, "branch competition s(x)",
                        cmap="YlOrRd", alpha=.85)
        figure.colorbar(image, ax=axes[i, 1], shrink=.75)
        image = scatter(axes[i, 2], xy, omega, size, "frequency ω", cmap="coolwarm", alpha=.85)
        figure.colorbar(image, ax=axes[i, 2], shrink=.75)
        seg = item["segments"]
        colors = np.where(seg[:, None] == -1, np.array([[.82, .82, .82, 1.]]),
                          plt.get_cmap("tab10")(np.mod(seg, 10)))
        count = len(set(seg)) - (1 if -1 in seg else 0)
        # With no segments every bootstrap agrees trivially (all unassigned),
        # so a stability number there would be meaningless.
        scatter(axes[i, 3], xy, colors, size,
                (f"HDBSCAN in feature space: {count} segments, stability {item['stability']:.2f}"
                 if count else "HDBSCAN in feature space: no segments (stability n/a)\n"
                               "grey = unassigned: a uniform box has no density islands"),
                alpha=.85)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n)
               for c, n in zip(branch_colors, BRANCH_NAMES)]
    axes[0, 0].legend(handles=handles, loc="lower right", fontsize=8)
    figure.suptitle(f"{CASE}, {len(x)} fixed probes.  Parameter space = scaled inputs only; "
                    "observable space = inputs + 1 × standardized (γ, ω).  "
                    "Neighbourhoods and segments computed in feature space; projections are display only.",
                    fontsize=12)
    figure.savefig(figures/"01-embedding-overview.png", dpi=100)
    plt.close(figure)

    # Sheet 2: concordance against the true transition band, in both views.
    figure, axes = plt.subplots(2, 3, figsize=(17, 11), layout="constrained")
    bins = np.linspace(0, 1, 21)
    for i, item in enumerate(s for s in shown if s["method"] == "pacmap"):
        g, xy = item["coherence"], item["xy"]
        auc = roc_auc_score(transition, 1-g)
        image = scatter(axes[i, 0], xy, g, size,
                        f"{item['label']}: coherence g (own-branch share of {K} neighbours)",
                        cmap="viridis_r", alpha=.85)
        figure.colorbar(image, ax=axes[i, 0], shrink=.8)
        scatter(axes[i, 1], xy, np.where(transition, "#d62728", "#cfcfcf"), size,
                f"true transition band (gap ≤ {TRANSITION:.0%} of branch range)", alpha=.85)
        axes[i, 2].hist(g[~transition], bins=bins, alpha=.6, density=True, label="away from transition")
        axes[i, 2].hist(g[transition], bins=bins, alpha=.6, density=True, label="in transition band")
        axes[i, 2].set_xlabel("coherence g"); axes[i, 2].set_ylabel("density")
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_title(f"low coherence as a transition detector: AUC {auc:.3f}")
    figure.suptitle("Concordance lens on a known answer (PaCMAP display).  In parameter space the modes "
                    "interpenetrate at the boundary, so low coherence finds it.\nIn observable space ω's "
                    "sign splits the modes into separate islands, coherence is 1 everywhere, and the lens "
                    "has nothing to find.", fontsize=12)
    figure.savefig(figures/"02-concordance-vs-transition.png", dpi=100)
    plt.close(figure)

    # Sheet 3: placement. Each arm's final design, mapped to its nearest probe
    # in feature space, drawn on the fixed embeddings.
    found = designs(args.run, CASE)
    arms = [a for a in PLACEMENT_ARMS if a in found]
    placement = {}
    if arms:
        panels = [s for s in shown if s["method"] == "pacmap"]
        figure, axes = plt.subplots(len(panels), len(arms), figsize=(3.7*len(arms), 4.2*len(panels)),
                                    squeeze=False, layout="constrained")
        for j, arm in enumerate(arms):
            d = found[arm]
            dtruth, dblock = response_block(CASE, d, "observables")
            dgap = np.abs(dtruth["G"][:, 0]-dtruth["G"][:, 1])/float(np.ptp(truth["G"]))
            share = float((dgap <= TRANSITION).mean())
            placement[arm] = dict(points=int(len(d)), transition_share=share,
                                  tem_share=float((np.argmax(dtruth["G"], axis=1) == 1).mean()))
            for i, item in enumerate(panels):
                probe = NearestNeighbors(n_neighbors=1).fit(item["feature"]).kneighbors(
                    features(d, dblock, stats, item["weight"]), return_distance=False)[:, 0]
                xy = item["xy"]
                axis = axes[i, j]
                axis.scatter(xy[:, 0], xy[:, 1], c=branch_colors[branch], s=2, alpha=.12, linewidths=0)
                axis.scatter(xy[probe, 0], xy[probe, 1], c=np.arange(len(probe)), cmap="plasma",
                             s=10, edgecolors="black", linewidths=.2)
                axis.set_xticks([]); axis.set_yticks([])
                if i == 0:
                    axis.set_title(f"{arm}\n{share:.0%} in transition band  "
                                   f"(probes {transition.mean():.0%})", fontsize=9)
                if j == 0:
                    axis.set_ylabel(item["label"])
        figure.suptitle(f"Where each method's N={len(d)} design lands (seed 0, PaCMAP).  Points coloured "
                        "by acquisition order; faint background by true branch (blue ITG, orange TEM).",
                        fontsize=12)
        figure.savefig(figures/"03-placement-on-embedding.png", dpi=100)
        plt.close(figure)

    metrics = dict(case=CASE, probes=len(x), k=K, transition_fraction=float(transition.mean()),
                   tem_fraction=float(branch.mean()), sweep=sweep,
                   display=[dict(label=s["label"], method=s["method"], weight=s["weight"],
                                 trustworthiness=s["trust"], stability=s["stability"])
                            for s in shown],
                   placement=placement, seconds=time.perf_counter()-started)
    (args.output/"metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"done in {metrics['seconds']/60:.1f} min")


if __name__ == "__main__":
    main()
