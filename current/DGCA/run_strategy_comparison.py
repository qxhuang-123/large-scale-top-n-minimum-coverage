"""Fresh matched-input R/A/B/C/D runs, guarded and serial."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent;BENCH=ROOT.parent/'TPCA'
sys.path.insert(0,str(BENCH))
from bench_common import SharedGraph,prepare_shared
from resilient_json import write_json
from prepare_cached_datasets import preparation_lock
from strategy_supervisor import run_job,read_json
import bgcr_d_bench as core

SESSION=ROOT/'outputs/BGCR_strategy_comparison_20261019'
MODES={'R':'random_source','A':'original','B':'fixed_source','C':'dynamic_item_heap','D':'dynamic_user_heap'}
SEEDS=list(range(2026101901,2026101911))
RESOURCE={'TIME_LIMIT','PRIVATE_MEMORY_LIMIT','SYSTEM_MEMORY_GUARD','MEMORY_ERROR','PROCESS_WALL_GUARD'}


def hashes():
    result={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'run_strategy_comparison.py',ROOT/'strategy_worker.py',ROOT/'strategy_supervisor.py',BENCH/'resilient_json.py']}
    result['native/bgcr_d_bench.cpp']=hashlib.sha256((ROOT/'native/bgcr_d_bench.cpp').read_bytes()).hexdigest()
    return result


def build_plan(datasets):
    digest=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest();plan=[]
    schedule=random.Random(20261019)
    for ds in datasets:
        folder=prepare_shared(ds);g=SharedGraph(folder);K0,q=g.initial(10)
        base=dict(dataset=ds,method='BGCR',N=10,D=g.meta['candidate_items'],percentage=100,
            graph_folder=str(folder),graph_id=g.meta['graph_id'],initial_coverage=K0,initial_integer_objective=q,
            backend_sha256=digest,timeout_seconds=1800.,private_limit_gib=12.,reserve_gib=1.)
        g.close()
        for block in range(10):
            variants=['A','B','C','D','R'] if block<3 else ['R'];schedule.shuffle(variants)
            for variant in variants:
                plan.append(dict(**base,variant=variant,native_mode=MODES[variant],repeat=block+1,
                    seed=SEEDS[block] if variant=='R' else 0))
    return plan


def summarize(results):
    rows=[];checks=[]
    for r in results:
        j=r['job'];a=r.get('audit',{});ref=r.get('tpcar_reference') or {};diag=r.get('native_diagnostics',{})
        row=dict(dataset=j['dataset'],variant=j['variant'],mode=j['native_mode'],repeat=j['repeat'],seed=j['seed'],
            N=j['N'],D=j['D'],graph_id=j['graph_id'],initial_coverage=j['initial_coverage'],initial_q=j['initial_integer_objective'],
            status=r['status'],K=a.get('coverage'),q=a.get('integer_objective'),APS=a.get('rounded_APS'),
            Kmax=ref.get('exact_boundary'),gap_same_K_pct=ref.get('score_gap_at_same_coverage_pct'),
            build=r.get('build_seconds'),core=r.get('solve_seconds'),total=r.get('optimization_total_seconds'),
            peak_gib=r.get('peak_working_set_bytes',0)/1024**3,private_gib=r.get('sampled_peak_private_bytes',0)/1024**3,
            result_folder=r.get('result_folder'),native_seed=r.get('seed'))
        row.update(diag.get('stats',{}));rows.append(row)
    for ds in sorted({r['dataset'] for r in rows}):
        good=[r for r in rows if r['dataset']==ds and r['status'] in ['FEASIBLE','TARGET_UNREACHED']]
        for variant in ['A','B','C','D']:
            g=[r for r in good if r['variant']==variant]
            assert len({r['q'] for r in g})<=1 and len({r['K'] for r in g})<=1
        pairs=[]
        for first,second in [('A','B'),('C','D')]:
            a=next((r for r in good if r['variant']==first),None);b=next((r for r in good if r['variant']==second),None)
            if a and b:
                import numpy as np
                assert (a['q'],a['K'])==(b['q'],b['K'])
                assert np.array_equal(np.load(Path(a['result_folder'])/'events.npy'),np.load(Path(b['result_folder'])/'events.npy'))
                pairs.append(first+'='+second+' events verified')
        assert len({(r['graph_id'],r['D'],r['initial_coverage'],r['initial_q']) for r in rows if r['dataset']==ds})==1
        checks.append(dict(dataset=ds,same_inputs=True,equivalent_event_pairs=pairs,
            random_seeds_completed=[r['seed'] for r in good if r['variant']=='R']))
    write_json(SESSION/'all_results.json',results);write_json(SESSION/'checks.json',checks)
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        pending=SESSION/'comparison.csv.pending'
        with pending.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        pending.replace(SESSION/'comparison.csv')
    return rows


def main(dataset=None):
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',choices=['OP','YELP','SO'],default=[dataset] if dataset else ['OP','YELP','SO'])
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();SESSION.mkdir(parents=True,exist_ok=True)
    try:
        validation=read_json(ROOT/'validation/strategy_20261019/validation.json');assert validation.get('passed')
        assert validation['backend_sha256']==hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
        assert validation['native_source_sha256']==hashlib.sha256((ROOT/'native/bgcr_d_bench.cpp').read_bytes()).hexdigest()
        plan=build_plan(args.datasets);write_json(SESSION/('plan_'+('_'.join(args.datasets))+'.json'),plan)
        protocol=dict(variants=MODES,datasets=['OP','YELP','SO'],N=10,target='candidate item union C',
            deterministic_repeats=3,random_seeds=SEEDS,planned_max_jobs=66,
            randomization='One std::mt19937_64 + std::shuffle permutation of initially uncovered nonempty columns. Randomize item order only; each visited item uses its current minimum-loss feasible replacement.',
            seed_reproducibility='Tied to the recorded native binary and compiler standard library; seed and event sequence saved.',
            timing='Fresh processes; new A/B/C/D timed within this batch; common input, rounding, Top-N and feasibility audit. Real reverse CSR construction for R/A/B/C; D uses forward CSR.',
            interpretation='Random seeds change solutions and timing; deterministic repeats measure timing variation only. A/B and C/D event sequences must match.',
            validation=validation,source_sha256=hashes())
        previous=read_json(SESSION/'protocol.json')
        if previous and previous!=protocol:raise RuntimeError('Protocol/source changed; do not mix results')
        write_json(SESSION/'protocol.json',protocol)
        if args.plan_only:print('Validated plan:',len(plan),'jobs; no solvers started');return
        if args.resume and (BENCH/'execution/stop.request').exists():(BENCH/'execution/stop.request').unlink()
        with preparation_lock(BENCH/'execution'):
            # Preserve results from other dataset-specific entries, when present.
            results=[read_json(p) for p in SESSION.glob('*/result.json')];limited=set()
            for index,job in enumerate(plan,1):
                key=(job['dataset'],job['variant'])
                if key in limited:continue
                if (BENCH/'execution/stop.request').exists():
                    write_json(SESSION/'progress.json',dict(complete=False,state='STOP_REQUESTED',recorded_jobs=len(results)));return
                folder=SESSION/('%s_%s_seed%d_r%d'%(job['dataset'],job['variant'],job['seed'],job['repeat']))
                write_json(SESSION/'current_job.json',dict(plan_index=index,**job))
                result=run_job(job,folder)
                results=[r for r in results if r.get('job')!=job]+[result]
                summarize(results)
                write_json(SESSION/'progress.json',dict(complete=False,recorded_jobs=len(results),planned_max_jobs=66,
                    dataset=job['dataset'],variant=job['variant'],seed=job['seed'],last_status=result['status'],updated_unix=time.time()))
                if result['status']=='STOP_REQUESTED':return
                if result['status'] in RESOURCE:limited.add(key)
                elif result['status'] not in ['FEASIBLE','TARGET_UNREACHED']:raise RuntimeError('Strategy worker failed: '+str(folder))
        attempted={(r['job']['dataset'],r['job']['variant']) for r in results}
        all_cases=all((ds,v) in attempted for ds in ['OP','YELP','SO'] for v in MODES)
        write_json(SESSION/'progress.json',dict(complete=all_cases,state='COMPLETE' if all_cases else 'PARTIAL_DATASETS',
            recorded_jobs=len(results),planned_max_jobs=66,updated_unix=time.time()))
        if all_cases:print('All strategy configurations attempted; report waiter can continue.',flush=True)
    except BaseException as exc:
        write_json(SESSION/'progress.json',dict(complete=False,state='ERROR',error=str(exc),updated_unix=time.time()));raise


if __name__=='__main__':main()
