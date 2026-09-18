import numpy as np
import pytest
from scripts.generate_ionut_data import CASES,generate,values
from scripts.datasets import load_dataset,surface_for
from scripts import ionut_proxies as proxy

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
