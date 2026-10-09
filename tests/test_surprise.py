"""Surprise-score contracts: the prior is the uniform code, data lowers surprise."""
import numpy as np

from benchmarknd.pool import PoolOracle
from benchmarknd.surprise import gp_label_posterior, label_posterior, prequential_label_bits, surprise_scores
from tests.test_pool import grid_oracle


def test_label_posterior_is_a_distribution_and_follows_neighbours():
    x = np.array([[0., 0.], [.1, 0.], [0., .1], [1., 1.], [.9, 1.]])
    labels = np.array([0, 0, 0, 1, 1])
    post = label_posterior(x, labels, np.array([[0., 0.], [1., 1.]]), 3, k=3)
    assert np.allclose(post.sum(axis=1), 1)
    assert post[0].argmax() == 0 and post[1].argmax() == 1
    assert post[0, 2] == post.min()


def test_uninformative_neighbours_give_the_prior_surprise():
    # Four modes, each nearest-neighbour set holds one of each: posterior = prior.
    x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    oracle = PoolOracle(np.vstack([x, [[.5, .5]]]), np.arange(5.), [0, 1, 2, 3, 0], ("a", "b", "c", "d"))
    post = label_posterior(x, oracle.labels[:4], np.array([[.5, .5]]), 4, k=4)
    assert np.allclose(post, .25)


def test_more_runs_leave_less_surprise():
    oracle = grid_oracle()
    order = np.random.default_rng(0).permutation(len(oracle.pool))
    few, many = (surprise_scores(oracle, order[:n]) for n in (12, 120))
    assert many["s_label_bits"] < few["s_label_bits"] <= few["prior_label_bits"]+.5
    assert many["s_gamma_nats"] < few["s_gamma_nats"]
    assert set(many["modes_with_own_gp"]) == {"low", "high"}


def test_prequential_code_has_one_entry_per_paid_run_after_the_start():
    oracle = grid_oracle()
    order = np.random.default_rng(1).permutation(len(oracle.pool))[:40]
    bits = prequential_label_bits(oracle, order, 7)
    assert len(bits) == 33 and min(bits) > 0


def test_gp_label_posterior_is_shrunk_toward_the_prior_and_follows_the_data():
    oracle = grid_oracle()
    order = np.random.default_rng(2).permutation(len(oracle.pool))
    paid, held = order[:60], order[60:]
    post = gp_label_posterior(oracle.pool[paid], oracle.labels[paid], oracle.pool[held], 2)
    assert np.allclose(post.sum(axis=1), 1) and post.min() > 0
    assert np.mean(post.argmax(axis=1) == oracle.labels[held]) > .8
    single = gp_label_posterior(oracle.pool[:5], np.zeros(5, int), oracle.pool[5:8], 3)
    assert np.allclose(single[:, 0], (5+.5)/(5+1.5))


def test_gpc_scores_and_region_points_are_reported():
    oracle = grid_oracle()
    order = np.random.default_rng(3).permutation(len(oracle.pool))[:50]
    knn = surprise_scores(oracle, order, return_points=True)
    gpc = surprise_scores(oracle, order, classifier="gpc")
    assert len(knn["label_bits"]) == len(knn["held_index"]) == len(oracle.pool)-50
    assert gpc["s_label_bits"] > 0
