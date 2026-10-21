"""Validate the R/A/B/C/D comparison without changing the supplied backend."""
import hashlib
import json
from pathlib import Path
import numpy as np
import bgcr_variants as supplied
import bgcr_d_bench as core
from validate_bgcr import arrays, greedy_oracle, audit

ROOT=Path(__file__).resolve().parent
MODES={'R':'random_source','A':'original','B':'fixed_source','C':'dynamic_item_heap','D':'dynamic_user_heap'}
SEEDS=list(range(2026101901,2026101911))


def per_item_minimum(a,N,result):
    up,items,scores,ip,_,_,_=a;U=len(up)-1;I=len(ip)-1
    sel=(up[:-1,None]+np.arange(N)).astype('int32')
    counts=np.bincount(items[sel.ravel()],minlength=I)
    for u,pos,old,edge in result['events']:
        j=int(items[edge]);best=None
        for v in range(U):
            targets=np.flatnonzero(items[up[v]:up[v+1]]==j)
            if not len(targets):continue
            new=int(up[v]+targets[0])
            for p,e in enumerate(sel[v]):
                if counts[items[e]]<2:continue
                choice=(float(scores[e]-scores[new]),j,v,p)
                if best is None or choice<best:best=choice
        assert best==(float(scores[old]-scores[edge]),j,int(u),int(pos))
        counts[items[old]]-=1;counts[j]+=1;sel[u,pos]=edge


def main():
    calls=0;random_cases=0;nontrivial=0
    for index in range(120):
        rng=np.random.default_rng(202610191000+index)
        U=int(rng.integers(2,8));I=int(rng.integers(4,13));N=int(rng.integers(1,min(4,I)))
        if index%30==29:U=3;N=[10,15,20,30][index//30];I=N+4
        rows=[]
        for u in range(U):
            pool=np.arange(I-1 if index%7==0 else I)
            js=rng.choice(pool,size=int(rng.integers(N,len(pool)+1)),replace=False)
            q=rng.integers(-3,7,len(js)).astype(float)*1e6
            if index%3==0:q=np.rint(rng.uniform(-2,6,len(js))*1e6)
            if index%11==0:q[:]=5e6
            rows.append(list(zip(map(int,js),map(float,q))))
        a=arrays(rows,I);K0=len(np.unique(a[1][a[0][:-1,None]+np.arange(N)]))
        for D in sorted(set([0,min(K0+1,I),I])):
            out={};events,_=greedy_oracle(a,N,D)
            for variant,mode in MODES.items():
                seed=SEEDS[0] if variant=='R' else 0
                r=core.solve(*a[:6],N,D,mode,seed)
                audit(a,N,D,r);old=supplied.solve(*a,N,D,mode,seed)
                assert np.array_equal(r['events'],old['events'])
                assert np.array_equal(r['selected'],old['selected'])
                out[variant]=r;calls+=2
                if variant in ['C','D']:assert np.array_equal(r['events'],events)
            assert np.array_equal(out['A']['events'],out['B']['events'])
            assert np.array_equal(out['C']['events'],out['D']['events'])
            signatures=set()
            for seed in SEEDS[:3]:
                r=core.solve(*a[:6],N,D,'random_source',seed)
                again=core.solve(*a[:6],N,D,'random_source',seed)
                old=supplied.solve(*a,N,D,'random_source',seed)
                assert np.array_equal(r['events'],again['events'])
                assert np.array_equal(r['events'],old['events'])
                audit(a,N,D,r);per_item_minimum(a,N,r)
                signatures.add(r['events'].tobytes());random_cases+=1;calls+=3
            nontrivial+=len(signatures)>1
    assert nontrivial>0
    result=dict(passed=True,graphs=120,native_calls=calls,random_seed_cases=random_cases,
        cases_with_distinct_random_outcomes=nontrivial,A_B_same_events=True,C_D_match_global_current_direct_swap_oracle=True,
        random_repeats_reproducible=True,random_move_uses_minimum_loss_for_visited_item=True,
        backend_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        native_source_sha256=hashlib.sha256((ROOT/'native/bgcr_d_bench.cpp').read_bytes()).hexdigest(),seeds=SEEDS)
    folder=ROOT/'validation/strategy_20261019';folder.mkdir(parents=True,exist_ok=True)
    (folder/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
