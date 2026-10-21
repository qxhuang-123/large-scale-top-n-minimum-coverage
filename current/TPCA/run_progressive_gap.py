"""Independent OP/Yelp positive-gap runs, after the current five-graph ablation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import state_variants_progressive as native
from bench_common import ROOT, SharedGraph, prepare_shared, write_json
from resilient_json import write_json
from prepare_cached_datasets import preparation_lock
from progressive_supervisor import run_job, read_json
from ablation_completion import complete as previous_ablation_complete

SESSION = ROOT/'outputs/progressive_gap_20261018'
MODES = {'V1':'state_progress_v1', 'V2':'state_progress_v2',
         'V3':'state_source_only', 'V4':'state_source'}
PERCENTAGES = [1, 10, 50, 100]
RESOURCE = {'TIME_LIMIT','PRIVATE_MEMORY_LIMIT','SYSTEM_MEMORY_GUARD','MEMORY_ERROR','PROCESS_WALL_GUARD'}


def validated_backend():
    validation = read_json(ROOT/'validation/progressive_20261018/validation.json')
    digest = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    assert validation.get('passed') and validation['native_extension_sha256'] == digest
    for name, expected in validation['progressive_source_sha256'].items():
        assert hashlib.sha256((ROOT/'native_progressive'/name).read_bytes()).hexdigest() == expected
    return digest, validation


def build_plan():
    digest, _ = validated_backend()
    plan = []
    for ds in ['OP','YELP']:
        folder = prepare_shared(ds)
        graph = SharedGraph(folder)
        K0, initial = graph.initial(10)
        C = graph.meta['candidate_items']; graph_id = graph.meta['graph_id']; graph.close()
        reference_folder = ROOT/'outputs/first_batch_final_20261018'/('%s_TPCAR_N10_D%d_r1'%(ds,C))
        ref = read_json(reference_folder/'result.json')
        assert ref['job']['graph_id'] == graph_id and ref['job']['N'] == 10
        assert ref['status'] in ['OPTIMAL','INFEASIBLE']
        assert ref['status'] != 'INFEASIBLE' or ref.get('maximum_coverage_certified')
        Kmax = ref['audit']['coverage']
        assert Kmax >= K0
        trace = read_json(reference_folder/'trajectory.json')
        q_at = {int(row.get('coverage',row.get('diversity',-1))):int(row['integer_objective'])
                for row in trace if 'integer_objective' in row}
        q_at[K0] = initial
        q_at[Kmax] = ref['audit']['integer_objective']
        for rho in PERCENTAGES:
            D = K0 + (rho*(Kmax-K0)+99)//100
            assert D in q_at
            # Fast configurations finish first; table and figure order remains V1--V4.
            for version in ['V4','V3','V2','V1']:
                plan.append(dict(dataset=ds, method='TPCAR', native_mode=MODES[version], version=version,
                    N=10, D=D, percentage=rho, gap_percentage=rho, repeat=1,
                    graph_folder=str(folder), graph_id=graph_id,
                    initial_coverage=K0, initial_integer_objective=initial, Kmax=Kmax,
                    expected_integer_objective=q_at[D], reference_folder=str(reference_folder),
                    timeout_seconds=1800.0, private_limit_gib=12.0, reserve_gib=1.0,
                    backend_sha256=digest))
    return plan


def summarize(results, session=SESSION):
    rows=[]; checks=[]
    for r in results:
        j=r['job']; a=r.get('audit',{}); d=r.get('native_diagnostics',{})
        if r['status']=='OPTIMAL':
            assert a['coverage']==j['D'] and a['integer_objective']==j['expected_integer_objective']
        elif r['status']=='INFEASIBLE':
            raise AssertionError('A target no greater than verified Kmax was reported infeasible')
        row=dict(dataset=j['dataset'],version=j['version'],native_mode=j['native_mode'],
            gap_percentage=j['gap_percentage'],N=j['N'],D=j['D'],K0=j['initial_coverage'],
            Kmax=j['Kmax'],gap=j['D']-j['initial_coverage'],repeat=j['repeat'],graph_id=j['graph_id'],
            status=r['status'],coverage=a.get('coverage'),integer_objective=a.get('integer_objective'),
            expected_integer_objective=j['expected_integer_objective'],
            build_seconds=r.get('build_seconds'),solve_seconds=r.get('solve_seconds'),
            total_seconds=r.get('optimization_total_seconds'),
            peak_working_set_gib=r.get('peak_working_set_bytes',0)/1024**3,
            sampled_peak_private_gib=r.get('sampled_peak_private_bytes',0)/1024**3,
            result_folder=r.get('result_folder'))
        for key in ['augmentations','full_searches','local_repairs','adjacency_scans','affected_total',
                    'heap_queries','boundary_scan_checks','source_pool_checks','terminal_checks']:
            row[key]=d.get(key)
        rows.append(row)
    for ds in ['OP','YELP']:
        for rho in PERCENTAGES:
            subset=[r for r in rows if r['dataset']==ds and r['gap_percentage']==rho and r['status']=='OPTIMAL']
            qs={r['integer_objective'] for r in subset}
            assert len(qs)<=1
            checks.append(dict(dataset=ds,gap_percentage=rho,completed_versions=[r['version'] for r in subset],
                agrees_with_independent_existing_exact_trace=True,integer_objectives=sorted(qs)))
    write_json(session/'all_results.json',results);write_json(session/'checks.json',checks)
    if rows:
        pending=session/'progressive_results.pending.csv'
        with pending.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        pending.replace(session/'progressive_results.csv')
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();SESSION.mkdir(parents=True,exist_ok=True)
    try:
        if not args.plan_only:
            while not previous_ablation_complete():
                if (ROOT/'execution/stop.request').exists():
                    write_json(SESSION/'progress.json',dict(complete=False,state='STOP_REQUESTED'));return
                write_json(SESSION/'progress.json',dict(complete=False,state='WAITING_FOR_FIVE_DATASET_ABLATION',updated_unix=time.time()))
                time.sleep(5)
        plan=build_plan();digest,validation=validated_backend()
        write_json(SESSION/'plan.json',plan)
        protocol=dict(datasets=['OP','YELP'],versions=MODES,N=10,gap_percentages=PERCENTAGES,
            formula='D=K0+ceil(rho*(Kmax-K0)/100)',repeats=1,planned_max_jobs=len(plan),
            V0='Deferred; no new repeated-Dijkstra runs',
            timing='Fresh process and independent solve per target; build and solve recorded separately; OS cache not cleared.',
            interpretation='Single runs: no repeat-based error bars or significance claims; no assumed monotone speedup.',
            budget=dict(timeout_seconds=1800,private_limit_gib=12,reserve_gib=1),
            backend_sha256=digest,validation=validation,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'run_progressive_gap.py',ROOT/'progressive_worker.py',ROOT/'progressive_supervisor.py',ROOT/'resilient_json.py']})
        old=read_json(SESSION/'protocol.json')
        if old and old!=protocol:raise RuntimeError('Progressive protocol changed; use a new session')
        write_json(SESSION/'protocol.json',protocol)
        if args.plan_only:
            print('Validated plan:',len(plan),'jobs; no experiments started');return
        results=[]
        with preparation_lock(ROOT/'execution'):
            for index,job in enumerate(plan,1):
                if (ROOT/'execution/stop.request').exists():
                    write_json(SESSION/'progress.json',dict(complete=False,state='STOP_REQUESTED',recorded_jobs=len(results)));return
                write_json(SESSION/'current_job.json',dict(plan_index=index,**job))
                folder=SESSION/('%s_%s_gap%03d_N10_D%d_r1'%(job['dataset'],job['version'],job['gap_percentage'],job['D']))
                result=run_job(job,folder)
                if result.get('native_extension_sha256'):assert result['native_extension_sha256']==digest
                results.append(result);summarize(results)
                write_json(SESSION/'progress.json',dict(complete=False,recorded_jobs=len(results),planned_max_jobs=len(plan),
                    current_dataset=job['dataset'],current_version=job['version'],gap_percentage=job['gap_percentage'],
                    last_status=result['status'],updated_unix=time.time()))
                if result['status']=='STOP_REQUESTED':return
                if result['status'] not in RESOURCE|{'OPTIMAL'}:raise RuntimeError('Unexpected worker status: '+str(result))
        write_json(SESSION/'progress.json',dict(complete=True,recorded_jobs=len(results),planned_max_jobs=len(plan),updated_unix=time.time()))
        print('Progressive-gap results:',SESSION,flush=True)
    except BaseException as exc:
        write_json(SESSION/'progress.json',dict(complete=False,state='ERROR',error=str(exc),updated_unix=time.time()));raise


if __name__=='__main__':main()
