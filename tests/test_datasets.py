import json
import numpy as np
import pytest
from scripts.generate_data import generate, cases_for
from scripts.datasets import load_dataset, surface_for
from scripts.run_gp_experiment import run


@pytest.mark.parametrize("dim", [2,5,8])
def test_frozen_data_matches_oracle_and_is_repeatable(tmp_path, dim):
    case = cases_for(dim)[0]
    first = generate(tmp_path / "a", dim, case, 0, 32, 64)
    second = generate(tmp_path / "b", dim, case, 0, 32, 64)
    manifest, pool, evaluation = load_dataset(first)
    _, pool2, evaluation2 = load_dataset(second)
    assert np.array_equal(pool["x"], pool2["x"])
    assert np.array_equal(evaluation["y"], evaluation2["y"])
    assert not set(map(tuple,pool["x"])) & set(map(tuple,evaluation["x"]))
    assert np.allclose(surface_for(dim,case,0)(evaluation["x"]), evaluation["y"])
    assert len(pool["y"]) == 32 and len(evaluation["y"]) == 64
    with pytest.raises(FileExistsError):
        generate(tmp_path / "a", dim, case, 0, 32, 64)
    with (first / "pool.npz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(first)


def test_policies_share_initialization_and_spend_exact_budget(tmp_path):
    path=generate(tmp_path,2,"smooth",0,32,64)
    a=run(path,8,"uncertainty",.5)
    b=run(path,8,"ucb",.5)
    assert a["selected_indices"][:5] == b["selected_indices"][:5]
    for result in (a,b):
        assert len(set(result["selected_indices"])) == 8
        assert result["rows"][-1]["n"] == 8
        assert np.isfinite(result["rows"][-1]["rmse"])


def test_gp_gradient_blend_is_reproducible_and_records_weights(tmp_path):
    path=generate(tmp_path,2,"smooth",0,32,64)
    a=run(path,8,"blend",.5,uncertainty_weight=.5)
    b=run(path,8,"blend",.5,uncertainty_weight=.5)
    assert a["selected_indices"] == b["selected_indices"]
    assert a["uncertainty_weight"] == a["gradient_weight"] == .5
    assert a["rows"][-1]["n"] == 8
    json.dumps(a, allow_nan=False)


def test_reject_identical_design_seeds(tmp_path):
    with pytest.raises(ValueError,match="seeds"):
        generate(tmp_path,2,"smooth",0,32,64,7,7)
