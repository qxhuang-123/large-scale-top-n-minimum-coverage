"""Post-run audit of all formal records, without invoking a solver."""
from pathlib import Path
import argparse
import hashlib
import json
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parent
SESSION=ROOT/'outputs/BGCR_strategy_comparison_20261019'


def main(session=SESSION):
    session=Path(session);protocol=json.loads((session/'protocol.json').read_text(encoding='utf8'))
    records=json.loads((session/'all_results.json').read_text(encoding='utf8'))
    assert len(records)==66
    groups={};signatures={};statuses=Counter()
    for r in records:
        j=r['job'];a=r['audit'];key=(j['dataset'],j['variant'])
        groups.setdefault(key,[]).append(r)
        assert r['status'] in ['FEASIBLE','TARGET_UNREACHED'];statuses[r['status']]+=1
        assert r['seed']==j['seed']
        assert r['native_extension_sha256']==j['backend_sha256']==protocol['validation']['backend_sha256']
        for name,digest in r['bgcr_source_sha256'].items():assert protocol['source_sha256'][name.replace('\\','/')]==digest
        assert a['integer_objective']==int(r['native_diagnostics']['objective'])
        assert a['coverage']==r['native_diagnostics']['coverage']
        assert a['total_recommendations']%j['N']==0
        assert a['target_feasible']==(a['coverage']>=j['D'])
        assert not r['global_infeasibility_claim']
        if not a['target_feasible']:assert r['direct_neighborhood_exhausted']
        assert r['tpcar_reference']['graph_id']==j['graph_id']
        assert r['tpcar_reference']['score_gap_at_same_coverage_integer']>=0
        folder=Path(r['result_folder']);events=np.load(folder/'events.npy')
        assert len(events)==a['coverage']-j['initial_coverage']
        signatures.setdefault(key,[]).append(hashlib.sha256(events.tobytes()).hexdigest())
        assert (folder/'memory_trace.csv').stat().st_size>100
        assert (folder/'trajectory.csv').stat().st_size>100
    per_dataset=[]
    for ds in ['OP','YELP','SO']:
        all_ds=[r for r in records if r['job']['dataset']==ds]
        assert len({(r['job']['graph_id'],r['job']['D'],r['job']['N'],r['job']['initial_coverage'],r['job']['initial_integer_objective']) for r in all_ds})==1
        for v in ['A','B','C','D']:
            assert len(groups[ds,v])==3
            assert len(set(signatures[ds,v]))==1
        assert signatures[ds,'A'][0]==signatures[ds,'B'][0]
        assert signatures[ds,'C'][0]==signatures[ds,'D'][0]
        seeds=[r['job']['seed'] for r in groups[ds,'R']]
        assert sorted(seeds)==protocol['random_seeds']
        per_dataset.append(dict(dataset=ds,formal_runs=len(all_ds),random_seeds=sorted(seeds),
            unique_random_event_sequences=len(set(signatures[ds,'R'])),
            all_deterministic_repeat_event_sequences_identical=True,A_B_events_identical=True,C_D_events_identical=True))
    audit=dict(passed=True,formal_runs=len(records),statuses=dict(statuses),datasets=per_dataset,
        input_identity=True,source_identity=True,native_seed_passed=True,integer_objective_and_coverage_audited=True,
        direct_neighborhood_exhausted_when_target_unreached=True,exact_reference_at_same_coverage=True)
    output=session/'paper_report/data/full_audit.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit,indent=2))
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session',type=Path,default=SESSION);args=parser.parse_args();main(args.session)
