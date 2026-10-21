"""Guarded DGCA runs on TPCA's unchanged graph and integer objective."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent
BENCH=ROOT.parent/'TPCA'
sys.path.insert(0,str(BENCH))
from bench_common import SharedGraph, prepare_shared, write_json
from resilient_json import write_json
from ablation_completion import complete_for_bgcr as tpcar_ablation_complete
from prepare_cached_datasets import preparation_lock
from bgcr_supervisor import run_job, read_json
import bgcr_d_bench as core

DATASETS=['OP','YELP','TG','VG','SO']
MODES=['dynamic_user_heap','dynamic_user_heap_nosource','dynamic_user_heap_noterminal','dynamic_user_scan','fixed_source','id_source']
RESOURCE=['TIME_LIMIT','PRIVATE_MEMORY_LIMIT','SYSTEM_MEMORY_GUARD','MEMORY_ERROR','PROCESS_WALL_GUARD']
FULL_SESSION=ROOT/'outputs/BGCR_matched_TPCAR_20261018'


def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'run_bgcr.py',ROOT/'bgcr_worker.py',ROOT/'bgcr_supervisor.py',ROOT/'native/bgcr_d_bench.cpp']}


def summarize(results,session):
    rows=[]
    for result in results:
        j=result['job'];a=result.get('audit',{});ref=result.get('tpcar_reference') or {};d=result.get('native_diagnostics',{})
        row=dict(dataset=j['dataset'],method='BGCR-D',native_mode=j['native_mode'],N=j['N'],D=j['D'],
            percentage=j['percentage'],repeat=j['repeat'],graph_id=j['graph_id'],
            initial_coverage=j['initial_coverage'],initial_integer_objective=j['initial_integer_objective'],
            status=result['status'],coverage=a.get('coverage'),integer_objective=a.get('integer_objective'),
            rounded_APS=a.get('rounded_APS'),original_APS=a.get('original_APS'),
            build_seconds=result.get('build_seconds'),solve_seconds=result.get('solve_seconds'),
            optimization_total_seconds=result.get('optimization_total_seconds'),
            peak_working_set_gib=result.get('peak_working_set_bytes',0)/1024**3,
            sampled_peak_private_gib=result.get('sampled_peak_private_bytes',0)/1024**3,
            direct_neighborhood_exhausted=result.get('direct_neighborhood_exhausted'),
            exact_integer_objective_same_coverage=ref.get('integer_objective_at_BGCR_coverage'),
            exact_gap_same_coverage_integer=ref.get('score_gap_at_same_coverage_integer'),
            exact_gap_same_coverage_pct=ref.get('score_gap_at_same_coverage_pct'),
            result_folder=result.get('result_folder'))
        for key in ['swaps','core_seconds','initial_seconds','strategy_setup_seconds']:row[key]=d.get(key)
        row.update(d.get('stats',{}));rows.append(row)
    write_json(session/'all_results.json',results)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        pending=session/'comparison.pending.csv'
        with pending.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        try:pending.replace(session/'comparison.csv')
        except PermissionError:pending.replace(session/('comparison_snapshot_%d.csv'%time.time_ns()))
    checks=[]
    for ds in DATASETS:
        good=[r for r in rows if r['dataset']==ds and r['N']==10 and r['percentage']==100
              and r['native_mode'].startswith('dynamic_user') and r['status'] in ['FEASIBLE','TARGET_UNREACHED']]
        q={r['integer_objective'] for r in good};K={r['coverage'] for r in good}
        assert len(q)<=1 and len(K)<=1,(ds,good)
        checks.append(dict(dataset=ds,full_and_module_ablation_same_objective=len(q)<=1,
            same_terminal_coverage=len(K)<=1,completed_modes=sorted(set(r['native_mode'] for r in good))))
    write_json(session/'checks.json',checks)
    return rows


def build_plan(args):
    digest=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest();cache={}
    def job(ds,N,p,repeat,mode):
        key=(ds,N)
        if key not in cache:
            folder=prepare_shared(ds);g=SharedGraph(folder);K0,obj=g.initial(N)
            cache[key]=dict(dataset=ds,N=N,graph_folder=str(folder),graph_id=g.meta['graph_id'],
                C=g.meta['candidate_items'],initial_coverage=K0,initial_integer_objective=obj)
            g.close()
        info=dict(cache[key]);C=info.pop('C')
        return dict(**info,method='BGCR-D',native_mode=mode,D=(C*p+99)//100,percentage=p,repeat=repeat,
            backend_sha256=digest,timeout_seconds=1800.0,private_limit_gib=12.0,reserve_gib=1.0)
    plan=[]
    if args.batch:
        for ds in DATASETS:
            for p in [100,80,60,40,20,0]:
                for r in [1,2,3]:plan.append(('main',job(ds,10,p,r,'dynamic_user_heap')))
        for N in [15,20,25,30]:
            for ds in DATASETS:plan.append(('N_results',job(ds,N,100,1,'dynamic_user_heap')))
        for ds in DATASETS:
            for mode in ['dynamic_user_heap_nosource','dynamic_user_heap_noterminal','dynamic_user_scan']:
                for r in [1,2,3]:plan.append(('module_ablation',job(ds,10,100,r,mode)))
        # These same-neighborhood controls address the original COR ordering/scanning concern.
        for ds in ['OP','YELP','SO']:
            for mode in ['fixed_source','id_source']:
                for r in [1,2,3]:plan.append(('direct_baselines',job(ds,10,100,r,mode)))
    else:
        for ds in args.datasets:
            for p in args.percentages:
                for r in range(1,args.repeats+1):plan.append(('single_dataset',job(ds,args.n,p,r,args.mode)))
    return plan


def main(dataset=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',action='store_true');parser.add_argument('--wait-for-TPCAR',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=[dataset] if dataset else DATASETS)
    parser.add_argument('--n',type=int,default=10);parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--percentages',nargs='+',type=int,default=[100])
    parser.add_argument('--mode',choices=MODES,default='dynamic_user_heap')
    parser.add_argument('--session',type=Path)
    args=parser.parse_args()
    if args.n<1 or args.n>30 or args.repeats<1 or any(p<0 or p>100 for p in args.percentages):parser.error('Invalid N/repeats/percentage')
    session=args.session or (FULL_SESSION if args.batch else ROOT/'outputs'/('DGCA_'+(dataset or 'custom')+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:5]))
    session.mkdir(parents=True,exist_ok=True)
    if args.resume and (BENCH/'execution/stop.request').exists():
        (BENCH/'execution/stop.request').unlink()
    while args.wait_for_TPCAR:
        if (BENCH/'execution/stop.request').exists():
            write_json(session/'progress.json',dict(complete=False,state='STOP_REQUESTED'));return
        if tpcar_ablation_complete():break
        write_json(session/'progress.json',dict(complete=False,state='WAITING_FOR_TPCAR_ABLATION',updated_unix=time.time()))
        time.sleep(5)
    validation=read_json(ROOT/'validation/small_validation.json')
    assert validation.get('passed')
    assert validation['binary_sha256']==hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    assert validation['derived_source_sha256']==hashlib.sha256((ROOT/'native/bgcr_d_bench.cpp').read_bytes()).hexdigest()
    results=[];limited=set()
    with preparation_lock(BENCH/'execution'):
        stop=BENCH/'execution/stop.request'
        if stop.exists():stop.unlink()
        plan=build_plan(args)
        protocol=dict(planned_max_jobs=len(plan),main_jobs=90 if args.batch else len(plan),
            N_result_jobs=20 if args.batch else 0,new_module_ablation_jobs=45 if args.batch else 0,
            full_ablation_runs_reused_from_main=15 if args.batch else 0,direct_baseline_jobs=18 if args.batch else 0,
            source_sha256=source_hashes(),json_writer_sha256=hashlib.sha256((BENCH/'resilient_json.py').read_bytes()).hexdigest(),candidate_inputs='Same immutable graph_id and rounded integer scores as TPCAR',
            stopping='Target reached or direct-neighborhood exhaustion; target failure is NOT global infeasibility.',
            input_construction='Forward score conversion/validation for user modes; exact reverse CSR additionally built for item-scan controls.',
            timing='Independent processes; build+solve; OS disk cache not cleared; guards match TPCAR.',
            validation=validation)
        previous=read_json(session/'protocol.json')
        if previous and previous!=protocol:raise RuntimeError('Protocol/source changed; choose a new output session')
        write_json(session/'protocol.json',protocol)
        for index,(phase,job) in enumerate(plan,1):
            key=(job['dataset'],job['N'],job['D'],job['native_mode'])
            if key in limited:continue
            if stop.exists():
                write_json(session/'progress.json',dict(complete=False,state='STOP_REQUESTED',recorded_jobs=len(results)));return
            write_json(session/'current_job.json',dict(stage=phase,plan_index=index,**job))
            folder=session/('%s_%s_N%d_D%d_r%d'%(job['dataset'],job['native_mode'],job['N'],job['D'],job['repeat']))
            result=run_job(job,folder);results.append(result);summarize(results,session)
            write_json(session/'progress.json',dict(complete=False,recorded_jobs=len(results),plan_index=index,
                planned_max_jobs=len(plan),current_stage=phase,current_dataset=job['dataset'],
                current_mode=job['native_mode'],last_status=result['status'],updated_unix=time.time()))
            if result['status']=='STOP_REQUESTED':return
            if result['status'] in ['ERROR','WORKER_CRASH','SOLVER_ERROR']:raise RuntimeError('BGCR worker error: '+str(folder))
            if result['status'] in RESOURCE:limited.add(key)
    write_json(session/'progress.json',dict(complete=True,recorded_jobs=len(results),planned_max_jobs=len(plan),updated_unix=time.time()))
    print('DGCA matched-input results:',session,flush=True)


if __name__=='__main__':main()
