"""Serial, guarded NEW-TPCAR ablation; wait for the N-result queue if requested."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import state_variants_ablation as native
from bench_common import ROOT, SharedGraph, prepare_shared, write_json
from prepare_cached_datasets import preparation_lock
from second_supervisor import run_job, read_json

MODES = {'state_source':'Full new TPCAR',
         'state_terminal':'Without source aggregation',
         'state_source_only':'Without terminal aggregation',
         'state_source_noheap':'Without boundary heap (exact column scan)'}
SESSION = ROOT/'outputs/second_batch_ablation_20261018'


def summarize(results):
    rows=[]
    for r in results:
        job=r['job'];audit=r.get('audit',{});diag=r.get('native_diagnostics',{})
        row=dict(dataset=job['dataset'],mode=job['native_mode'],N=job['N'],D=job['D'],repeat=job['repeat'],
            graph_id=job['graph_id'],status=r['status'],coverage=audit.get('coverage'),
            integer_objective=audit.get('integer_objective'),original_APS=audit.get('original_APS'),
            build_seconds=r.get('build_seconds'),solve_seconds=r.get('solve_seconds'),
            total_seconds=r.get('optimization_total_seconds'),
            peak_working_set_gib=r.get('peak_working_set_bytes',0)/1024**3,
            peak_private_sampled_gib=r.get('sampled_peak_private_bytes',0)/1024**3,
            result_folder=r.get('result_folder'))
        for key in ['augmentations','swaps','adjacency_scans','local_repairs','affected_total','affected_max',
            'heap_build_edges','heap_queries','heap_top_checks','heap_refreshes','heap_exclusions',
            'boundary_scan_checks','internal_scan_checks','terminal_queries','terminal_checks',
            'terminal_refreshes','terminal_pointer_advances','source_refresh_calls','source_pool_checks',
            'source_pointer_advances','source_class_changes','source_equal_witness_changes']:
            row[key]=diag.get(key)
        rows.append(row)
    checks=[]
    for ds in sorted(set(r['dataset'] for r in rows)):
        completed=[r for r in rows if r['dataset']==ds and r['status'] in ('OPTIMAL','INFEASIBLE')]
        objectives=sorted(set(r['integer_objective'] for r in completed))
        statuses=set(r['status'] for r in completed)
        boundary_coverage=set(r['coverage'] for r in completed if r['status']=='INFEASIBLE')
        assert len(objectives)<=1 and len(statuses)<=1 and len(boundary_coverage)<=1, (ds,completed)
        checks.append(dict(dataset=ds,integer_objectives=objectives,
            completed_modes=sorted(set(r['mode'] for r in completed)),consistent=True))
    write_json(SESSION/'all_results.json',results);write_json(SESSION/'checks.json',checks)
    if rows:
        with (SESSION/'ablation_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-N',action='store_true')
    args=parser.parse_args();SESSION.mkdir(parents=True,exist_ok=True)
    while args.wait_for_N:
        if (ROOT/'execution/stop.request').exists():
            write_json(SESSION/'progress.json',dict(complete=False,state='STOP_REQUESTED'));return
        progress=read_json(ROOT/'outputs/second_batch_N_exact_20261018/queue_progress.json')
        if progress.get('complete'):break
        if progress.get('state')=='STOP_REQUESTED':
            write_json(SESSION/'progress.json',dict(complete=False,state='N_QUEUE_STOPPED'));return
        write_json(SESSION/'progress.json',dict(complete=False,state='WAITING_FOR_N_EXACT',updated_unix=time.time()))
        time.sleep(5)
    validation=read_json(ROOT/'validation/ablation_20261018/validation.json')
    digest=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    assert validation.get('passed') and validation.get('native_extension_sha256')==digest
    assert validation.get('full_backend_equivalence_calls',0)>0
    for name,sha in validation['ablation_source_sha256'].items():
        assert hashlib.sha256((ROOT/'native_ablation'/name).read_bytes()).hexdigest()==sha
    results=[]
    with preparation_lock(ROOT/'execution'):
        stop=ROOT/'execution/stop.request'
        if stop.exists():stop.unlink()
        write_json(SESSION/'protocol.json',dict(datasets=['OP','YELP','SO','VG','TG'],N=10,target='candidate union C',
            repeats=3,variants=MODES,integer_scale=1000000,timeout_seconds=1800,
            private_limit_gib=12,reserve_gib=1,native_extension_sha256=digest,
            full_runs='Rerun full variant with the same separate binary/build flags as the ablations; no timing mixed from first batch.',
            preserved='Exact augmentation and local tree repair in all variants; no old TPCAR or BGCR.',
            validation=validation))
        for ds in ['OP','YELP','SO','VG','TG']:
            graph_folder=prepare_shared(ds);g=SharedGraph(graph_folder)
            K0,initial=g.initial(10);C=g.meta['candidate_items'];graph_id=g.meta['graph_id'];g.close()
            for mode in MODES:
                for repeat in [1,2,3]:
                    if stop.exists():
                        write_json(SESSION/'progress.json',dict(complete=False,state='STOP_REQUESTED',recorded_jobs=len(results)));return
                    job=dict(dataset=ds,method='TPCAR',native_mode=mode,N=10,D=C,percentage=100,repeat=repeat,
                        timeout_seconds=1800.0,private_limit_gib=12.0,reserve_gib=1.0,
                        graph_folder=str(graph_folder),graph_id=graph_id,
                        initial_coverage=K0,initial_integer_objective=initial)
                    write_json(SESSION/'current_job.json',job)
                    result=run_job(job,SESSION/('%s_%s_N10_D%d_r%d'%(ds,mode,C,repeat)))
                    if result.get('native_extension_sha256') and result['native_extension_sha256']!=digest:
                        raise RuntimeError('Ablation backend changed')
                    results.append(result);summarize(results)
                    write_json(SESSION/'progress.json',dict(complete=False,recorded_jobs=len(results),
                        current_dataset=ds,current_mode=mode,repeat=repeat,last_status=result['status'],updated_unix=time.time()))
                    if result['status']=='STOP_REQUESTED':return
                    if result['status'] in ['ERROR','WORKER_CRASH','SOLVER_ERROR']:
                        raise RuntimeError('Ablation worker failed; inspect '+str(result))
                    if result['status'] in ['TIME_LIMIT','PRIVATE_MEMORY_LIMIT','SYSTEM_MEMORY_GUARD','MEMORY_ERROR']:
                        break
    write_json(SESSION/'progress.json',dict(complete=True,recorded_jobs=len(results),updated_unix=time.time()))
    print('Ablation results:',SESSION,flush=True)


if __name__=='__main__':main()
