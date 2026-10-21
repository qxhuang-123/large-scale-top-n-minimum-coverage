"""One process per BGCR solve on the exact shared TPCAR candidate snapshot."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parent
BENCH=ROOT.parent/'TPCA'
sys.path.insert(0,str(BENCH))
from bench_common import SharedGraph, audit_selected, close_array, write_json
from resilient_json import write_json
from bench_memory import ProcessMemory
import bgcr_d_bench as core

USER_MODES=['dynamic_user_heap','dynamic_user_heap_nosource','dynamic_user_heap_noterminal','dynamic_user_scan']
BASELINES=['original','fixed_source','id_source','random_source','dynamic_item_heap']


def sources():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'strategy_worker.py',ROOT/'strategy_supervisor.py',ROOT/'native/bgcr_d_bench.cpp']}


def exact_reference(g,N,K,D):
    C=g.meta['candidate_items']
    base=(BENCH/'outputs/first_batch_final_20261018' if N==10 else
          BENCH/'outputs/second_batch_N_exact_20261018'/('N%d'%N))
    folder=base/('%s_TPCAR_N%d_D%d_r1'%(g.dataset,N,C))
    result_path=folder/'result.json'
    if not result_path.exists():return None
    ref=json.loads(result_path.read_text(encoding='utf8'))
    if ref.get('status') not in ['OPTIMAL','INFEASIBLE']:return None
    assert ref['job']['graph_id']==g.meta['graph_id']
    frontier={}
    with (folder/'trajectory.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            coverage=int(row.get('coverage') or row['diversity'])
            obj=int(row['integer_objective'])
            if coverage in frontier:assert frontier[coverage]==obj
            frontier[coverage]=obj
    K0=ref['job']['initial_coverage'];initial=ref['job']['initial_integer_objective']
    at_K=initial if K<=K0 else frontier.get(K)
    at_D=initial if D<=K0 else frontier.get(D)
    return dict(result_file=str(result_path),graph_id=g.meta['graph_id'],
        exact_boundary=ref.get('audit',{}).get('coverage'),integer_objective_at_BGCR_coverage=at_K,
        integer_objective_at_requested_D=at_D)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);args=parser.parse_args()
    folder=args.job.parent;job=json.loads(args.job.read_text(encoding='utf8'))
    source_hashes=sources();started=time.perf_counter();optstart=[None];timings={};g=None
    def phase(name,**kwargs):
        timings.update(kwargs)
        if name=='build' and optstart[0] is None:optstart[0]=time.time()
        write_json(folder/'phase.json',dict(phase=name,pid=os.getpid(),
            optimization_started_unix=optstart[0],timings=timings,updated_unix=time.time()))
    try:
        phase('import_and_load');mode=job['native_mode'];seed=int(job.get('seed',0))
        assert mode in USER_MODES+BASELINES
        assert hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()==job['backend_sha256']
        g=SharedGraph(job['graph_folder']);g.dataset=job['dataset'];N=job['N'];D=job['D']
        assert g.meta['graph_id']==job['graph_id']
        K0,initial=g.initial(N)
        assert (K0,initial)==(job['initial_coverage'],job['initial_integer_objective'])
        load_seconds=time.perf_counter()-started
        phase('build');build_start=time.perf_counter()
        scores=np.empty(g.E,dtype=np.float64)
        for lo in range(0,g.E,1_000_000):scores[lo:lo+1_000_000]=g.q[lo:lo+1_000_000]
        close_array(g.q)
        reverse=None
        if mode in BASELINES:
            count=np.zeros(g.I,dtype=np.int64)
            for lo in range(0,g.E,1_000_000):count+=np.bincount(g.items[lo:lo+1_000_000],minlength=g.I)
            ip=np.r_[0,np.cumsum(count)].astype(np.int32);cursor=ip[:-1].astype(np.int64)
            ie=np.empty(g.E,dtype=np.int32);iu=np.empty(g.E,dtype=np.int32)
            for u in range(g.U):
                lo,hi=int(g.up[u]),int(g.up[u+1]);j=g.items[lo:hi];pos=cursor[j]
                ie[pos]=np.arange(lo,hi,dtype=np.int32);cursor[j]+=1
            for lo in range(0,g.E,500_000):
                hi=min(g.E,lo+500_000);iu[lo:hi]=np.searchsorted(g.up[1:],ie[lo:hi],side='right')
            reverse=(ip,iu,ie)
            del ip,iu,ie,cursor,count
        assembly=time.perf_counter()-build_start
        phase('build',input_assembly_seconds=assembly,construction_stage='forward_input_validation')
        validation_start=time.perf_counter()
        for lo in range(0,g.E,1_000_000):assert np.isfinite(scores[lo:lo+1_000_000]).all()
        for u in range(g.U):
            lo,hi=int(g.up[u]),int(g.up[u+1]);j=g.items[lo:hi];s=scores[lo:hi]
            assert hi-lo>=N and len(np.unique(j))==hi-lo
            assert not np.any(s[1:]>s[:-1])
            assert not np.any((s[1:]==s[:-1])&(j[1:]<j[:-1]))
        del s,j
        validation_seconds=time.perf_counter()-validation_start
        build_seconds=time.perf_counter()-build_start
        phase('solve',build_seconds=build_seconds)
        t=time.perf_counter()
        if mode in USER_MODES:r=dict(core.solve_user(g.up,g.items,scores,g.I,N,D,mode,seed))
        else:r=dict(core.solve(g.up,g.items,scores,*reverse,N,D,mode,seed))
        solve_seconds=time.perf_counter()-t
        phase('audit',build_seconds=build_seconds,solve_seconds=solve_seconds)
        selected=np.asarray(r.pop('selected'),dtype=np.int32).reshape(-1)
        events=np.asarray(r.pop('events'),dtype=np.int32)
        history=np.asarray(r.pop('history'),dtype=np.float64)
        assert initial==r['initial_objective'] and K0==r['initial_coverage']
        # Release dedicated input before original-score auditing, as in the TPCAR adapter.
        del scores,reverse
        g.q=np.load(g.folder/'scores_integer.npy',mmap_mode='r')
        audit=audit_selected(g,selected,N,D)
        assert r['objective']==audit['integer_objective'] and r['coverage']==audit['coverage']
        assert len(events)==audit['coverage']-K0
        assert bool(r['target_reached'])==audit['target_feasible']
        assert history.shape==(len(events)+1,3)
        assert np.all(np.diff(history[:,0])==1) and np.all(np.diff(history[:,1])<=0)
        assert int(history[-1,1])==audit['integer_objective']
        exhausted=None
        if not audit['target_feasible']:
            counts=np.bincount(g.items[selected],minlength=g.I);sel=selected.reshape(g.U,N)
            for u in range(g.U):
                if np.any(counts[g.items[sel[u]]]>=2):
                    assert not np.any(counts[g.items[g.up[u]:g.up[u+1]]]==0),'Direct swap remains at terminal state'
            exhausted=True
        elif audit['coverage']==g.meta['candidate_items']:exhausted=True
        ref=exact_reference(g,N,audit['coverage'],D)
        if ref:
            exact=ref['integer_objective_at_BGCR_coverage']
            assert exact is not None and audit['integer_objective']<=exact
            ref['score_gap_at_same_coverage_integer']=exact-audit['integer_objective']
            ref['score_gap_at_same_coverage_pct']=(exact-audit['integer_objective'])/exact*100 if exact else None
        audit_end=time.perf_counter();phase('export')
        np.save(folder/'selected_edges.npy',selected);np.save(folder/'events.npy',events)
        # Save readable recommendations using the same user/item IDs and ranking rule.
        users=np.repeat(np.arange(g.U),N);order=np.lexsort((g.items[selected],-g.q[selected],users))
        with (folder/'recommendations.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['user_id_0based','rank','item_id_0based','integer_score','original_score'])
            last=-1;rank=0
            for p in order:
                u=int(users[p]);e=int(selected[p]);rank=rank+1 if u==last else 1;last=u
                w.writerow([u,rank,int(g.items[e]),int(g.q[e]),float(g.raw[e])])
        with (folder/'trajectory.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['coverage','integer_objective','rounded_APS','time_sec'])
            for K,q,elapsed in history:w.writerow([int(K),int(q),q/(1e6*g.U*N),elapsed])
        monitor=ProcessMemory(os.getpid())
        try:memory=monitor.read()
        finally:monitor.close()
        result=dict(status='FEASIBLE' if audit['target_feasible'] else 'TARGET_UNREACHED',
            job=job,audit=audit,initial_coverage=K0,initial_integer_objective=initial,
            native_mode=mode,seed=seed,native_diagnostics=r,build_seconds=build_seconds,solve_seconds=solve_seconds,
            optimization_total_seconds=build_seconds+solve_seconds,input_assembly_seconds=assembly,
            input_validation_seconds=validation_seconds,load_and_shared_checks_seconds=load_seconds,
            audit_and_extraction_seconds=audit_end-started-load_seconds-build_seconds-solve_seconds,
            export_seconds=time.perf_counter()-audit_end,worker_elapsed_seconds=time.perf_counter()-started,
            direct_neighborhood_exhausted=exhausted,global_infeasibility_claim=False,
            tpcar_reference=ref,worker_self_memory=memory,scale=1_000_000,
            native_extension_sha256=job['backend_sha256'],bgcr_source_sha256=source_hashes,
            python=sys.version,platform=platform.platform(),numpy=np.__version__)
        write_json(folder/'worker_result.json',result);phase('done')
    except BaseException as exc:
        traceback.print_exc()
        write_json(folder/'worker_result.json',dict(status='MEMORY_ERROR' if isinstance(exc,MemoryError) else 'ERROR',
            error=str(exc),traceback=traceback.format_exc(),job=job,partial_timings=timings))
        raise
    finally:
        if g is not None:g.close()


if __name__=='__main__':main()
