import numpy as np
import pytest
from scripts.generate_ionut_data import CASES,generate,values
from scripts.datasets import load_dataset,surface_for
from scripts import ionut_proxies as proxy
from scripts.generate_ionut_slices import CASES as SLICE_CASES, expand, generate as generate_slice
from scripts.run_gp_experiment import run

@pytest.mark.parametrize('case',CASES)
def test_proxy_archives_match_continuous_values(tmp_path,case):
    path=generate(tmp_path,case,32,64)
    meta,pool,test=load_dataset(path)
    assert np.array_equal(surface_for(6,case,0)(test['x']),test['y'])
    assert not set(map(tuple,pool['x'])) & set(map(tuple,test['x']))
    if 'G' in test:
        if meta['mode']=='argmax':
            index=test['G'].argmax(axis=1)
            assert np.array_equal(test['gamma'],test['G'].max(axis=1))
            assert np.array_equal(test['omega'],test['W'][np.arange(64),index])
        else:
            assert np.all(test['gamma']>=test['G'].min(axis=1)-1e-15)
            assert np.all(test['gamma']<=test['G'].max(axis=1)+1e-15)
    with pytest.raises(FileExistsError): generate(tmp_path,case,32,64)

def test_reject_invalid_proxy_input():
    with pytest.raises(ValueError): values(CASES[0],np.zeros((2,5)))
    with pytest.raises(ValueError): surface_for(6,CASES[0],1)


@pytest.mark.parametrize('case', SLICE_CASES)
def test_3d_slices_match_the_declared_native_6d_condition(tmp_path, case):
    path = generate_slice(tmp_path, case, 16, 32)
    meta, pool, test = load_dataset(path)
    assert meta['native_dimension'] == 6 and meta['dimension'] == 3
    assert np.array_equal(test['native_x'], expand(case, test['x']))
    assert np.array_equal(surface_for(3, case, 0)(test['x']), test['y'])
    native = values(meta['native_case'], test['native_x'])
    assert np.array_equal(native['y'], test['y'])
    assert not set(map(tuple, pool['x'])) & set(map(tuple, test['x']))


def test_3d_gp_scores_transition_tail_and_branches(tmp_path):
    case = 'ionut-itg-tem-3d-argmax-gamma'
    path = generate_slice(tmp_path, case, 16, 64)
    row = run(path, 7, 'blend', .5)['rows'][-1]
    for key in ('normalized_rmse', 'nmae', 'normalized_p95',
                'transition_normalized_rmse', 'high_response_normalized_rmse'):
        assert np.isfinite(row[key])
    assert row['transition_count'] > 0
    assert len(row['branch_normalized_rmse']) == 2
