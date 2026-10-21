"""Matched R/A/B/C/D comparisons; one run per strategy, same native backend."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'DGCA'))
import run_strategy_comparison as strategy
from bench_common import SharedGraph,prepare_shared
from prepare_cached_datasets import preparation_lock
from resilient_json import write_json

def main(dataset=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=['OP','YELP','VG','TG','SO'],default=[dataset] if dataset else ['OP','YELP','VG','TG','SO'])
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--session',type=Path)
    args=p.parse_args()
    validation=strategy.read_json(ROOT/'DGCA/validation/small_validation.json')
    assert validation['passed']
    assert validation['binary_sha256']==hashlib.sha256(Path(strategy.core.__file__).read_bytes()).hexdigest()
    session=args.session or ROOT/'DGCA/outputs'/('DGCA_strategies_'+time.strftime('%Y%m%d_%H%M%S'))
    strategy.SESSION=session
    session.mkdir(parents=True,exist_ok=True)
    plan=[j for j in strategy.build_plan(args.datasets) if j['repeat']==1]
    assert len(plan)==5*len(args.datasets)
    write_json(session/'plan.json',plan)
    write_json(session/'protocol.json',dict(datasets=args.datasets,variants=strategy.MODES,
        repeats=1,random_seed=2026101901,N=10,target='candidate union',
        backend_sha256=validation['binary_sha256'],source_sha256=strategy.hashes()))
    if args.plan_only:
        print('Validated',len(plan),'jobs; no solver started.');return
    results=[]
    with preparation_lock(ROOT/'TPCA/execution'):
        stop=ROOT/'TPCA/execution/stop.request'
        for i,job in enumerate(plan,1):
            if stop.exists():raise RuntimeError('Stop requested; completed records retained.')
            folder=session/('%s_%s_r1'%(job['dataset'],job['variant']))
            print('[%d/%d] %s %s'%(i,len(plan),job['dataset'],job['variant']),flush=True)
            result=strategy.run_job(job,folder)
            results.append(result);strategy.summarize(results)
            write_json(session/'progress.json',dict(complete=False,completed=len(results),planned=len(plan),last_status=result['status']))
            if result['status'] not in ['FEASIBLE','TARGET_UNREACHED']:
                raise RuntimeError('Stopped attempt retained: '+str(folder))
    write_json(session/'progress.json',dict(complete=True,completed=len(results),planned=len(plan)))
    print('Results:',session)

if __name__=='__main__':main()
