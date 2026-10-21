"""VG/TG first-run extension using the unchanged validated strategy backend."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'TPCA'))
from bench_common import SharedGraph
from prepare_cached_datasets import preparation_lock
from resilient_json import write_json
import run_strategy_comparison as original

SESSION = ROOT / 'outputs/BGCR_strategy_extension_VG_TG_20261021'


def verify_sources():
    previous = original.read_json(original.SESSION / 'protocol.json')
    assert previous and previous['validation']['passed']
    hashes = original.hashes()
    assert hashes == previous['source_sha256'], 'Original strategy sources changed'
    binary = hashlib.sha256(Path(original.core.__file__).read_bytes()).hexdigest()
    assert binary == previous['validation']['backend_sha256']
    return previous, hashes, binary


def audit_results(results, planned):
    checks = []
    for ds in sorted({r['job']['dataset'] for r in results}):
        group = [r for r in results if r['job']['dataset'] == ds]
        assert len({(r['job']['graph_id'], r['job']['initial_coverage'],
                     r['job']['initial_integer_objective'], r['job']['D'],
                     r['job']['N']) for r in group}) == 1
        pairs = []
        for r in group:
            assert r['status'] in ['FEASIBLE', 'TARGET_UNREACHED']
            assert r['job']['repeat'] == 1
            assert r['seed'] == (2026101901 if r['job']['variant'] == 'R' else 0)
            assert r['tpcar_reference']['graph_id'] == r['job']['graph_id']
            assert r['tpcar_reference']['score_gap_at_same_coverage_integer'] >= 0
            if not r['audit']['target_feasible']:
                assert r['direct_neighborhood_exhausted']
        for a, b in [('A', 'B'), ('C', 'D')]:
            first = next((r for r in group if r['job']['variant'] == a), None)
            second = next((r for r in group if r['job']['variant'] == b), None)
            if first and second:
                ea = np.load(Path(first['result_folder']) / 'events.npy')
                eb = np.load(Path(second['result_folder']) / 'events.npy')
                assert np.array_equal(ea, eb), (ds, a, b)
                pairs.append(a + '=' + b)
        checks.append(dict(dataset=ds, configurations=len(group), same_inputs=True,
                           event_pairs=pairs, exact_reference_verified=True))
    complete = {(r['job']['dataset'], r['job']['variant']) for r in results} == planned
    write_json(SESSION / 'extension_audit.json', dict(passed=True, complete=complete,
        runs=len(results), checks=checks, repeats=1, random_seed=2026101901))
    return complete


def main(dataset=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=['VG', 'TG'],
                        default=[dataset] if dataset else ['VG', 'TG'])
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    previous, hashes, binary = verify_sources()
    SESSION.mkdir(parents=True, exist_ok=True)
    # Reuse the existing preparation, initialization, seeded scheduling and guards.
    plan = [job for job in original.build_plan(args.datasets) if job['repeat'] == 1]
    assert len(plan) == 5 * len(args.datasets)
    protocol = dict(datasets=['VG', 'TG'], variants=original.MODES, N=10,
        target='candidate item union C', repeats=1, random_seed=2026101901,
        source_sha256=hashes, backend_sha256=binary,
        extension_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_protocol=str(original.SESSION / 'protocol.json'),
        scope='Ten additional independently started first runs; no old records replaced',
        validation=previous['validation'])
    old = original.read_json(SESSION / 'protocol.json')
    if old and old != protocol:
        raise RuntimeError('Extension protocol changed; preserve completed records')
    write_json(SESSION / 'protocol.json', protocol)
    write_json(SESSION / ('plan_' + '_'.join(args.datasets) + '.json'), plan)
    print('Extension plan:', len(plan), 'jobs; same validated backend', binary, flush=True)
    if args.plan_only:
        return
    stop = ROOT.parent / 'TRMNL_BENCH/execution/stop.request'
    if stop.exists():
        raise RuntimeError('Existing stop request; no jobs started')
    original.SESSION = SESSION
    planned = {(ds, v) for ds in ['VG', 'TG'] for v in original.MODES}
    with preparation_lock(ROOT.parent / 'TRMNL_BENCH/execution'):
        results = [original.read_json(p) for p in SESSION.glob('*/result.json')]
        for index, job in enumerate(plan, 1):
            if stop.exists():
                write_json(SESSION / 'progress.json', dict(state='STOP_REQUESTED', complete=False))
                return
            folder = SESSION / ('%s_%s_seed%d_r1' % (job['dataset'], job['variant'], job['seed']))
            write_json(SESSION / 'current_job.json', dict(plan_index=index, **job))
            print('[%d/%d] %s variant %s' % (index, len(plan), job['dataset'], job['variant']), flush=True)
            result = original.run_job(job, folder)
            results = [r for r in results if r.get('job') != job] + [result]
            original.summarize(results)
            write_json(SESSION / 'progress.json', dict(complete=False, recorded_jobs=len(results),
                dataset=job['dataset'], variant=job['variant'], status=result['status'], updated_unix=time.time()))
            if result['status'] not in ['FEASIBLE', 'TARGET_UNREACHED']:
                raise RuntimeError('Stopped or failed attempt retained: ' + str(folder))
            audit_results(results, planned)
        complete = audit_results(results, planned)
        write_json(SESSION / 'progress.json', dict(complete=complete,
            state='COMPLETE' if complete else 'PARTIAL_DATASETS', recorded_jobs=len(results),
            planned_jobs=10, updated_unix=time.time()))
    print('Extension finished:', len(results), 'validated records', flush=True)


if __name__ == '__main__':
    main()
