"""Original shared input pipeline + original Top-N + optimized DGCA stage two.

This deliberately retains the original score copy and Python input checks.
Do not replace them for only one algorithm in a TPCA/DGCA comparison.
"""
import argparse
import os
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import dgca_stage2

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent/'TPCA'))
from bench_common import SharedGraph, close_array


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--N',type=int,default=10)
    target=p.add_mutually_exclusive_group()
    target.add_argument('--D',type=int)
    target.add_argument('--percentage',type=int,default=100)
    p.add_argument('--mode',choices=('grouped',),default='grouped')
    p.add_argument('--reference',type=Path)
    args=p.parse_args()
    if args.D is None and not 0<=args.percentage<=100:p.error('percentage must be in 0..100')
    required=('graph.json','user_indptr.npy','user_items.npy','scores_integer.npy','scores_original.npy')
    missing=[f for f in required if not (args.folder/f).is_file()]
    if missing:p.error('Missing original candidate graph files: '+', '.join(missing))
    args.out.mkdir(parents=True,exist_ok=False)

    # Same loader and initial audit as the supplied experiment.py.
    started=time.perf_counter();g=SharedGraph(args.folder)
    k0,q0=g.initial(args.N)
    load_s=time.perf_counter()-started
    D=args.D if args.D is not None else (int(g.meta['candidate_items'])*args.percentage+99)//100

    # UNCHANGED common build: float64 assembly, full finite check, row uniqueness,
    # descending scores and ascending item ties. All modes use this same branch.
    started=time.perf_counter()
    scores=np.empty(g.E,dtype=np.float64)
    for lo in range(0,g.E,1_000_000):
        scores[lo:lo+1_000_000]=g.q[lo:lo+1_000_000]
    close_array(g.q)
    assembly=time.perf_counter()-started
    check_start=time.perf_counter()
    for lo in range(0,g.E,1_000_000):
        assert np.isfinite(scores[lo:lo+1_000_000]).all()
    for u in range(g.U):
        lo,hi=int(g.up[u]),int(g.up[u+1])
        j=g.items[lo:hi];s=scores[lo:hi]
        assert hi-lo>=args.N and len(np.unique(j))==hi-lo
        assert not np.any(s[1:]>s[:-1])
        assert not np.any((s[1:]==s[:-1])&(j[1:]<j[:-1]))
    del j,s
    validation=time.perf_counter()-check_start
    build=time.perf_counter()-started

    # The only algorithm-dependent call. All constructors execute the original
    # Top-N code in vendor/reference_core.cpp. Logs remain enabled in every mode.
    started=time.perf_counter();cpu_start=time.process_time()
    r=dict(dgca_stage2.solve_user(g.up,g.items,scores,g.I,args.N,D,'dynamic_user_grouped_cached',0))
    core=time.perf_counter()-started;cpu=time.process_time()-cpu_start
    from bench_memory import ProcessMemory
    monitor=ProcessMemory(os.getpid()); memory=monitor.read(); monitor.close()
    assert r['initial_coverage']==k0 and r['initial_objective']==q0
    selected=np.asarray(r.pop('selected'));events=np.asarray(r.pop('events'));history=np.asarray(r.pop('history'))
    assert selected.shape==(g.U,args.N)
    for u,es in enumerate(selected):
        assert np.all((es>=g.up[u])&(es<g.up[u+1])) and len(np.unique(es))==args.N
    assert int(np.count_nonzero(np.bincount(g.items[selected.ravel()],minlength=g.I)))==r['coverage']
    assert int(scores[selected.ravel()].sum())==r['objective']
    match=None
    if args.reference:
        ref=json.loads((args.reference/'result.json').read_text(encoding='utf-8-sig'))
        assert ref['job']['graph_id']==g.meta['graph_id']
        assert int(ref['job']['N'])==args.N and int(ref['job']['D'])==D
        assert np.array_equal(events,np.load(args.reference/'events.npy',allow_pickle=False))
        assert np.array_equal(selected.ravel(),np.load(args.reference/'selected_edges.npy',allow_pickle=False).ravel())
        assert ref['audit']['coverage']==r['coverage'] and ref['audit']['integer_objective']==r['objective']
        match=True
    np.save(args.out/'selected_edges.npy',selected)
    np.save(args.out/'events.npy',events)
    np.save(args.out/'history.npy',history)
    result=dict(status='OK',job=dict(graph_id=g.meta['graph_id'],N=args.N,D=D,mode=args.mode),
                coverage=r['coverage'],integer_objective=int(r['objective']),
                build_s=build,core_s=core,total_s=build+core,cpu_core_s=cpu,
                peak_working_set_gib=memory['peak_working_set_os']/1024**3,
                assembly_s=assembly,input_validation_s=validation,load_s=load_s,native=r,
                exact_historical_match=match,
                timing='Same shared graph loader, float64 assembly, row checks and original Top-N for every mode. total=build+native call. Load/initial audit/final audit/export excluded, as in supplied experiment.py.',
                stage2_source_sha256=hashlib.sha256((HERE/'stage2.cpp').read_bytes()).hexdigest())
    (args.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    g.close()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
