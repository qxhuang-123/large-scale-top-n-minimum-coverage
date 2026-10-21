"""Check supplied D, compact interface, module ablations and greedy decisions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import bgcr_variants as supplied
import bgcr_d_bench as core

ROOT=Path(__file__).resolve().parent
USER_MODES=['dynamic_user_heap','dynamic_user_heap_nosource','dynamic_user_heap_noterminal','dynamic_user_scan']


def arrays(rows,I):
    up=[0];items=[];scores=[]
    for row in rows:
        row=sorted(row,key=lambda a:(-a[1],a[0]))
        items.extend(j for j,q in row);scores.extend(q for j,q in row);up.append(len(items))
    up=np.array(up,dtype='int32');items=np.array(items,dtype='int32');scores=np.array(scores,dtype='float64')
    ie=np.argsort(items,kind='stable').astype('int32')
    eu=np.repeat(np.arange(len(rows),dtype='int32'),np.diff(up))
    ip=np.r_[0,np.bincount(items,minlength=I).cumsum()].astype('int32')
    return up,items,scores,ip,eu[ie],ie,scores[ie]


def greedy_oracle(a,N,D):
    up,items,scores,ip,_,_,_=a;U=len(up)-1;I=len(ip)-1
    selected=(up[:-1,None]+np.arange(N)).astype('int32')
    counts=np.bincount(items[selected.ravel()],minlength=I);K=int(np.count_nonzero(counts));events=[]
    while K<D:
        best=None
        for u in range(U):
            for pos,old in enumerate(selected[u]):
                if counts[items[old]]<2:continue
                for edge in range(up[u],up[u+1]):
                    j=int(items[edge])
                    if counts[j]:continue
                    c=(float(scores[old]-scores[edge]),j,u,pos,edge)
                    if best is None or c<best:best=c
        if best is None:break
        _,j,u,pos,edge=best;old=int(selected[u,pos]);counts[items[old]]-=1;counts[j]+=1
        selected[u,pos]=edge;events.append([u,pos,old,edge]);K+=1
    return np.array(events,dtype='int32').reshape(-1,4),selected


def audit(a,N,D,r):
    up,items,scores,ip,_,_,_=a;U=len(up)-1;I=len(ip)-1
    selected=(up[:-1,None]+np.arange(N)).astype('int32');counts=np.bincount(items[selected.ravel()],minlength=I)
    q=int(scores[selected].sum());assert r['initial_objective']==q
    for step,(u,p,old,e) in enumerate(r['events'],1):
        assert selected[u,p]==old and up[u]<=e<up[u+1]
        assert counts[items[old]]>=2 and counts[items[e]]==0
        counts[items[old]]-=1;counts[items[e]]+=1;selected[u,p]=e
        q+=int(scores[e]-scores[old]);assert r['history'][step,1]==q
    assert np.array_equal(selected,r['selected'])
    assert r['objective']==q==int(scores[selected].sum())
    K=int(np.count_nonzero(counts));assert K==r['coverage'] and bool(K>=D)==r['target_reached']
    if K<D:
        assert all(not np.any(counts[items[selected[u]]]>=2) or not np.any(counts[items[up[u]:up[u+1]]]==0) for u in range(U))


def max_coverage(a,N):
    up,j,_,ip,_,_,_=a;U=len(up)-1;I=len(ip)-1;E=len(j)
    users=np.repeat(np.arange(U),np.diff(up));eq=np.zeros((U,E+I));eq[users,np.arange(E)]=1
    ub=np.zeros((I,E+I));ub[j,np.arange(E)]=-1;ub[np.arange(I),E+np.arange(I)]=1
    r=linprog(np.r_[np.zeros(E),-np.ones(I)],A_eq=eq,b_eq=np.full(U,N),A_ub=ub,b_ub=np.zeros(I),bounds=(0,1),method='highs')
    assert r.success;return int(round(-r.fun))


def main():
    calls=0;supplied_matches=0;terminal_checks=0
    for seed in range(200):
        rng=np.random.default_rng(20261018000+seed)
        U=int(rng.integers(2,7));I=int(rng.integers(4,11));N=int(rng.integers(1,min(3,I)+1))
        if seed%25==24:N=[15,20,25,30][(seed//25)%4];I=N+3;U=3
        rows=[]
        for u in range(U):
            pool=np.arange(I-1 if seed%7==0 else I)
            js=rng.choice(pool,size=int(rng.integers(N,len(pool)+1)),replace=False)
            scores=rng.integers(-8,21,size=len(js)).astype('float64')*250000
            if seed%4==0:scores=np.rint(rng.uniform(-2,6,len(js))*1e6)
            if seed%11==0:scores[:]=5000000
            rows.append(list(zip(map(int,js),map(float,scores))))
        a=arrays(rows,I);initial=a[0][:-1,None]+np.arange(N);K0=len(np.unique(a[1][initial]));C=int(np.count_nonzero(np.diff(a[3])))
        for D in sorted(set([0,min(K0+1,I),I])):
            events,selected=greedy_oracle(a,N,D)
            full=None
            for mode in USER_MODES:
                r=core.solve_user(*a[:3],I,N,D,mode,0);audit(a,N,D,r);calls+=1
                assert np.array_equal(r['events'],events),(seed,D,mode,'greedy decision')
                assert np.array_equal(r['selected'],selected)
                if mode=='dynamic_user_heap':full=r
            for mode in ['dynamic_user_heap','dynamic_item_heap','fixed_source','id_source']:
                original=supplied.solve(*a,N,D,mode,0)
                derived=core.solve(*a[:6],N,D,mode,0)
                audit(a,N,D,derived);calls+=1
                assert np.array_equal(original['events'],derived['events'])
                assert np.array_equal(original['selected'],derived['selected'])
                if mode.startswith('dynamic_'):
                    assert np.array_equal(original['events'],events)
                    assert original['objective']==full['objective']
                supplied_matches+=1
            if D==I:
                Kmax=max_coverage(a,N)
                assert full['coverage']*2>=Kmax
                terminal_checks+=1
        if (seed+1)%25==0:print('Validated',seed+1,'graphs;',calls,'calls',flush=True)
    result=dict(passed=True,graphs=200,adapter_calls=calls,supplied_equivalence_checks=supplied_matches,
        terminal_global_half_bound_checks=terminal_checks,all_module_variants_match_current_minimum_direct_swap_oracle=True,
        coverage='Integer ties, rounded continuous/negative scores, N=1..3 and N=15/20/25/30, zero/positive gaps, terminal plans.',
        binary_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        derived_source_sha256=hashlib.sha256((ROOT/'native/bgcr_d_bench.cpp').read_bytes()).hexdigest())
    (ROOT/'validation').mkdir(exist_ok=True)
    (ROOT/'validation/small_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
