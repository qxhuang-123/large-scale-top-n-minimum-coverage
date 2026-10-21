"""Independent HiGHS LP oracle for four variants of the NEW TPCAR only."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import state_variants_progressive as native
import state_variants_bench as baseline
from bench_common import ROOT, write_json
from validate_comparison import make_graph, oracle
from bench_solvers import solve_tpcar

MODES = ['state_source', 'state_source_only', 'state_progress_v2', 'state_progress_v1']


def main():
    folder=ROOT/'validation/progressive_20261018';folder.mkdir(parents=True,exist_ok=True)
    stats=dict(instances=100, calls=0, feasible=0, infeasible=0, full_backend_equivalence_calls=0,
               max_integer_objective_error=0, modes=MODES)
    proxy=ModuleType('state_variants_bench');proxy.__file__=native.__file__
    mode_box=['state_source']
    def run(*args):
        result=native.run(*args[:-1],mode_box[0])
        if mode_box[0]=='state_source':
            original=baseline.run(*args)
            for key in ['naive_obj','tpcar_obj','naive_D','tpcar_D','target_feasible',
                        'augmentations','swaps','adjacency_scans','heap_queries','local_repairs']:
                assert result.get(key)==original.get(key),(key,result.get(key),original.get(key))
            assert result['selected_edges']==original['selected_edges']
            stats['full_backend_equivalence_calls']+=1
        return result
    proxy.run=run;sys.modules['state_variants_bench']=proxy
    for seed in range(stats['instances']):
        rng=np.random.default_rng(2026101800+seed)
        U=int(rng.integers(2,9));I=int(rng.integers(4,13));N=int(rng.integers(1,min(4,I)))
        if seed%20==19:
            N=[15,20,25,30][(seed//20)%4];I=N+int(rng.integers(1,8));U=3
        rows=[];values=[]
        for u in range(U):
            pool=np.arange(I-1 if seed%5==0 else I)
            size=int(rng.integers(N,len(pool)+1))
            j=rng.choice(pool,size,replace=False)
            if seed%3==0:r=rng.integers(-2,6,size).astype(float)
            else:r=rng.uniform(-2,6,size)
            if seed%7==0:r[:]=5.0
            rows.append(j);values.append(r)
        g=make_graph(folder/('graph_%03d'%seed),U,I,rows,values)
        K0,_=g.initial(N)
        for D in sorted(set([0,min(K0+1,I),I])):
            ref=oracle(g,N,D)
            assert ref.success or ref.status==2
            expected=int(round(-ref.fun)) if ref.success else None
            complete_result=None
            for mode in MODES:
                mode_box[0]=mode
                result=solve_tpcar(g,N,D,folder,lambda *a,**k:None)
                stats['calls']+=1
                if expected is None:
                    assert result['status']=='INFEASIBLE',(seed,N,D,mode,result['status'])
                    assert result['maximum_coverage_certified']
                    boundary=oracle(g,N,result['audit']['coverage'])
                    assert boundary.success and int(round(-boundary.fun))==result['audit']['integer_objective']
                    stats['infeasible']+=1
                else:
                    assert result['status']=='OPTIMAL',(seed,N,D,mode,result['status'])
                    assert result['audit']['integer_objective']==expected,(seed,N,D,mode,expected,result['audit'])
                    stats['feasible']+=1
                if mode=='state_source':complete_result=result
                if mode=='state_progress_v1':
                    counters=result['native_diagnostics']
                    assert counters.get('heap_queries',0)==0
                if complete_result and expected is None:
                    assert result['audit']['coverage']==complete_result['audit']['coverage']
        g.close()
        if (seed+1)%25==0:
            print('Validated',seed+1,'graphs;',stats['calls'],'calls',flush=True)
    stats.update(passed=True,native_extension_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        progressive_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'native_progressive').iterdir() if p.is_file()},
        scope='Ties, continuous/negative scores, N=1..3 and selected N=15/20/25/30, feasible/infeasible and zero/positive gaps. Closed-set boundary checks plus independent LP objective checks.')
    write_json(folder/'validation.json',stats)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__': main()
