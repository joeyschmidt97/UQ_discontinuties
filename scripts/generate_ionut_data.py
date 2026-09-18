"""Generate native 6D microinstability proxy data without fitting any models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import qmc
from scripts import ionut_proxies as proxy
from scripts.generate_data import power_of_two

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = '13f87b95f90be9dbb5942e317739bb15a1918e0e'
CASES = [f'ionut-{kind}-{mode}-{out}' for kind in ('itg-tem','itg-kbm')
         for mode in ('argmax','softmax') for out in ('gamma','omega')] + ['ionut-stellarator-itg-kbm']


def values(case, x):
    x = np.atleast_2d(np.asarray(x, float))
    if case not in CASES or x.shape[1] != 6 or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError('known case and finite unit-box 6D inputs required')
    if case == CASES[-1]:
        return dict(x=x, y=proxy.test_model(x))
    _, _, second, mode, out = case.split('-')
    params = {key: lo+(hi-lo)*x[:,i] for i,(key,(lo,hi)) in enumerate(proxy._RANGES.items())}
    result = proxy.combine_modes(params,.30*(.5+x[:,5]),('ITG',second.upper()),mode=mode,T=.05)
    return dict(x=x,y=result[out],gamma=result['gamma'],omega=result['omega'],
                G=result['G'].T,W=result['W'].T,share=result['share'])


def generate(output, case, pool_size=4096, test_size=65536):
    power_of_two(pool_size); power_of_two(test_size)
    path=Path(output)/'6d'/case/'seed-0'
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    pool=values(case,qmc.Sobol(6,scramble=True,seed=1729).random_base2(pool_size.bit_length()-1))
    test=values(case,qmc.Sobol(6,scramble=True,seed=91479).random_base2(test_size.bit_length()-1))
    test.update(band=np.zeros(test_size,dtype=bool),peak=np.zeros(test_size,dtype=bool),
                region=np.argmax(test['G'],axis=1) if 'G' in test else np.zeros(test_size,dtype=int))
    paths=[ROOT/'scripts/ionut_proxies.py',Path(__file__),ROOT/'scripts/datasets.py',ROOT/'scripts/generate_data.py']
    manifest=dict(schema_version=1,family='ionut-phenomenological-microinstability',dimension=6,
                  case=case,surface_seed=0,bounds=[[0.,1.]]*6,measure='uniform unit box',
                  pool_size=pool_size,test_size=test_size,pool_seed=1729,test_seed=91479,
                  scale=float(np.ptp(test['y'])),numpy=np.__version__,scipy=scipy.__version__,
                  created_utc=datetime.now(timezone.utc).isoformat(),upstream_commit=UPSTREAM,
                  upstream_repository='https://github.com/ionutfarcas/UQ_discontinuties',
                  source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  diagnostics={'band':'undefined; empty mask','peak':'undefined; empty mask'},
                  note='Phenomenological proxy, not gyrokinetic truth; native dimension retained.')
    if case == CASES[-1]:
        manifest.update(columns=['nref','Tref','aLTi','aLTe','aLn','tau'],parameters=dict(beta_c=.32,delta_beta=.04,eps=.025,alpha_n=.5,alpha_tau=.3,grad_shift=.05),upstream_file='sgpp_examples/itg_kbm_surrogate.py')
    else:
        _,_,second,mode,out=case.split('-')
        manifest.update(columns=list(proxy._COLS),physical_ranges=proxy._RANGES,ky_mapping='0.30*(0.5+x5)',
                        branches=['ITG',second.upper()],mode=mode,output=out,T=.05,baseline=proxy.BASELINE,
                        upstream_file='sgpp_examples/itg_transition_sg.py')
    path.mkdir(parents=True)
    for name,data in [('pool',pool),('evaluation',test)]:
        np.savez_compressed(path/(name+'.npz'),**data)
    manifest['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('*.npz')}
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    parser.add_argument('--output',type=Path,default=ROOT/'data')
    parser.add_argument('--pool-size',type=power_of_two,default=4096)
    parser.add_argument('--test-size',type=power_of_two,default=65536)
    args=parser.parse_args()
    for case in args.cases:
        if (args.output/'6d'/case/'seed-0').exists():
            parser.error('requested output exists; select a new output directory')
    for case in dict.fromkeys(args.cases):
        print(generate(args.output,case,args.pool_size,args.test_size),flush=True)


if __name__=='__main__':
    main()
