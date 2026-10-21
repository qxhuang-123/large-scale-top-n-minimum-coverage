"""Serial, process-isolated comparison with actual time and memory traces."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

from bench_common import ROOT, DATASETS, METHODS, SharedGraph, prepare_shared, write_json
from bench_memory import ProcessMemory, available_memory


def read_json(path):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError): return {}


def cleanup_scratch(folder):
    """Delete only this finished worker's five explicitly named generated arrays."""
    folder = Path(folder).resolve(); scratch = (folder / 'scratch').resolve()
    if scratch.parent != folder or scratch.name != 'scratch':
        raise RuntimeError('Unexpected scratch location')
    for name in ('user_scores', 'edge_users', 'item_users', 'item_edges', 'item_scores'):
        path = scratch / (name + '.npy')
        if path.exists():
            if path.is_symlink() or path.resolve().parent != scratch:
                raise RuntimeError('Refusing to remove a non-worker path')
            path.unlink()
    if scratch.is_dir() and not any(scratch.iterdir()): scratch.rmdir()


def run_job(job, folder, poll=.2):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    old = read_json(folder / 'result.json')
    if old and old.get('status') == 'STOP_REQUESTED':
        write_json(folder / 'previous_stopped_result.json', old)
        (folder / 'result.json').unlink()
        old = {}
    if old:
        if old.get('job') != job:
            raise RuntimeError('Resume job protocol changed; choose a new --session directory')
        print('Reusing completed job', job['dataset'], job['method'], job['D'], job['repeat'], old['status'], flush=True)
        return old
    write_json(folder / 'job.json', job)
    phase_path = folder / 'phase.json'
    if phase_path.exists(): phase_path.unlink()
    worker_result_path = folder / 'worker_result.json'
    if worker_result_path.exists(): worker_result_path.unlink()
    print('Starting %s %s N=%d D=%d gap=%d repeat=%d' % (
        job['dataset'], job['method'], job['N'], job['D'],
        max(0, job['D'] - job['initial_coverage']), job['repeat']), flush=True)
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.perf_counter(); stop_reason = None
    peak_ws = 0; peak_private = 0; peak_commit = 0; minimum_available = None
    reserve_hits = 0; samples = 0
    with (folder / 'worker.log').open('w', encoding='utf-8') as log, \
         (folder / 'memory_trace.csv').open('w', encoding='utf-8-sig', newline='') as memory:
        proc = subprocess.Popen([sys.executable, str(ROOT / 'bench_worker.py'), str(folder / 'job.json')],
                cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor = ProcessMemory(proc.pid)
        writer = csv.DictWriter(memory, fieldnames=['elapsed_seconds', 'phase', 'working_set',
            'peak_working_set_os', 'private_bytes', 'peak_commit_os', 'system_available_bytes'])
        writer.writeheader()
        try:
            while True:
                phase = read_json(folder / 'phase.json')
                # Windows venv python.exe is a redirector. Monitor the actual
                # worker PID reported by this fresh run, rather than its tiny launcher.
                actual_pid = phase.get('pid')
                if actual_pid and actual_pid != monitor.pid:
                    try: actual = ProcessMemory(actual_pid)
                    except OSError: actual = None
                    if actual is not None:
                        monitor.close(); monitor = actual
                current = monitor.read()
                available, total = available_memory()
                minimum_available = available if minimum_available is None else min(minimum_available, available)
                if current:
                    peak_ws = max(peak_ws, current['peak_working_set_os'])
                    peak_private = max(peak_private, current['private_bytes'])
                    peak_commit = max(peak_commit, current['peak_commit_os'])
                    writer.writerow(dict(elapsed_seconds=time.perf_counter() - started,
                        phase=phase.get('phase', 'starting'), system_available_bytes=available, **current))
                    samples += 1
                    if samples % 10 == 0: memory.flush()
                if proc.poll() is not None: break
                if (ROOT / 'execution' / 'stop.request').exists() and actual_pid == monitor.pid:
                    stop_reason = 'STOP_REQUESTED'
                opt_start = phase.get('optimization_started_unix')
                if phase.get('phase') in ('build', 'solve') and opt_start and time.time() - opt_start > job['timeout_seconds']:
                    stop_reason = 'TIME_LIMIT'
                elif current and actual_pid == monitor.pid and current['private_bytes'] > job['private_limit_gib'] * 1024**3:
                    stop_reason = 'PRIVATE_MEMORY_LIMIT'
                reserve_hits = reserve_hits + 1 if available < job['reserve_gib'] * 1024**3 else 0
                if reserve_hits >= 5: stop_reason = 'SYSTEM_MEMORY_GUARD'
                if time.perf_counter() - started > job['timeout_seconds'] + 1200:
                    stop_reason = 'PROCESS_WALL_GUARD'
                if stop_reason:
                    monitor.terminate()
                    if proc.poll() is None: proc.terminate()
                    proc.wait()
                    break
                time.sleep(poll)
        except BaseException:
            monitor.terminate()
            if proc.poll() is None: proc.terminate(); proc.wait()
            raise
        finally: monitor.close()
    result = read_json(folder / 'worker_result.json')
    self_memory = result.get('worker_self_memory') or {}
    peak_ws = max(peak_ws, self_memory.get('peak_working_set_os', 0))
    peak_commit = max(peak_commit, self_memory.get('peak_commit_os', 0))
    phase = read_json(folder / 'phase.json')
    if stop_reason or not result:
        result = dict(status=stop_reason or 'WORKER_CRASH', job=job,
            last_phase=phase, exit_code=proc.returncode)
    result.update(process_wall_seconds=time.perf_counter() - started, exit_code=proc.returncode,
        peak_working_set_bytes=peak_ws, sampled_peak_private_bytes=peak_private,
        peak_commit_os_bytes=peak_commit, minimum_system_available_bytes=minimum_available,
        memory_sampling_seconds=poll, memory_samples=samples, result_folder=str(folder))
    write_json(folder / 'result.json', result)
    try: cleanup_scratch(folder)
    except OSError as exc:
        result['scratch_cleanup_note'] = str(exc); write_json(folder / 'result.json', result)
    print('%s %s D=%d r=%d: %s, build=%s solve=%s, peak WS=%.2f GiB' % (
        job['dataset'], job['method'], job['D'], job['repeat'], result['status'],
        format(result['build_seconds'], '.3f') if 'build_seconds' in result else '-',
        format(result['solve_seconds'], '.3f') if 'solve_seconds' in result else '-', peak_ws / 1024**3), flush=True)
    return result


def summarize(results, session):
    rows = []
    for r in results:
        j = r['job']; a = r.get('audit', {})
        rows.append(dict(dataset=j['dataset'], method=j['method'], N=j['N'], D=j['D'],
            percentage=j.get('percentage'), repeat=j['repeat'], graph_id=j['graph_id'],
            initial_coverage=j['initial_coverage'], requested_gap=max(0, j['D'] - j['initial_coverage']),
            status=r['status'], coverage=a.get('coverage'), integer_objective=a.get('integer_objective'),
            rounded_score=a.get('rounded_score'), original_score=a.get('original_score'),
            rounded_APS=a.get('rounded_APS'), original_APS=a.get('original_APS'),
            build_seconds=r.get('build_seconds'), solve_seconds=r.get('solve_seconds'),
            index_assembly_seconds=r.get('index_assembly_seconds'),
            index_validation_seconds=r.get('index_validation_seconds'),
            optimization_total_seconds=r.get('optimization_total_seconds'), process_wall_seconds=r.get('process_wall_seconds'),
            peak_working_set_gib=r.get('peak_working_set_bytes', 0) / 1024**3,
            sampled_peak_private_gib=r.get('sampled_peak_private_bytes', 0) / 1024**3,
            maximum_coverage_certified=r.get('maximum_coverage_certified'), result_folder=r.get('result_folder')))
    checks = []; groups = {}
    for row in rows:
        key = (row['graph_id'], row['N'], row['D'])
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        optimal = [r for r in group if r['status'] == 'OPTIMAL']
        values = sorted(set(r['integer_objective'] for r in optimal))
        statuses = {r['status'] for r in group}
        contradiction = bool(optimal) and 'INFEASIBLE' in statuses
        checks.append(dict(dataset=group[0]['dataset'], D=key[2], graph_id=key[0],
            methods_with_optimal=sorted(set(r['method'] for r in optimal)),
            exact_integer_objectives=values,
            consistency='MISMATCH' if len(values) > 1 or contradiction else 'MATCH' if len(set(r['method'] for r in optimal)) >= 2 else 'NO_CROSS_METHOD_RESULT',
            feasibility_contradiction=contradiction))
    write_json(Path(session) / 'comparison_checks.json', checks)
    write_json(Path(session) / 'all_results.json', results)
    if rows:
        csv_pending = Path(session) / 'comparison.pending.csv'
        with csv_pending.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        try: os.replace(csv_pending, Path(session) / 'comparison.csv')
        except PermissionError:
            snapshot = Path(session) / ('comparison_snapshot_%d.csv' % time.time_ns())
            os.replace(csv_pending, snapshot)
            print('Summary CSV is open in another application; saved updated snapshot:', snapshot, flush=True)
        lines = ['# Unified exact-method comparison', '',
            'Same candidate snapshot, rounded integer objective, N and requested D. '
            'Only OPTIMAL results enter objective-equality checks. '
            'TPCAR infeasible results may additionally contain a boundary plan.', '',
            '| Dataset | Method | D | Status | Recorded runs | Median build s | Median solve s | Median build+solve s | Max peak working set GiB | Integer objectives |',
            '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |']
        grouped = {}
        for row in rows: grouped.setdefault((row['dataset'], row['method'], row['D'], row['status']), []).append(row)
        for key, group in grouped.items():
            def median(field):
                values = [r[field] for r in group if r[field] is not None]
                return '%.6f' % statistics.median(values) if values else '-'
            objectives = sorted(set(r['integer_objective'] for r in group if r['integer_objective'] is not None))
            lines.append('| %s | %s | %d | %s | %d | %s | %s | %s | %.3f | %s |' % (
                *key, len(group), median('build_seconds'), median('solve_seconds'),
                median('optimization_total_seconds'), max(r['peak_working_set_gib'] for r in group), str(objectives)))
        lines.extend(['', '## Cross-method checks', '', *[
            '- %s D=%d: %s (%s)' % (c['dataset'], c['D'], c['consistency'], ', '.join(c['methods_with_optimal'])) for c in checks], '',
            'TIME_LIMIT and memory guards are resource outcomes, not model infeasibility. '
            'Peak working set includes file-backed pages. Private bytes are sampled; OS lifetime peak commitment is in each result. '
            'MCF provides final endpoints; its internal augmentation trajectory is not exposed.'])
        md_pending = Path(session) / 'comparison_report.pending.md'
        md_pending.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        try: os.replace(md_pending, Path(session) / 'comparison_report.md')
        except PermissionError:
            os.replace(md_pending, Path(session) / ('comparison_report_snapshot_%d.md' % time.time_ns()))
    return rows, checks


def run_suite(args, defaults=None):
    from prepare_cached_datasets import preparation_lock
    session = Path(args.session) if args.session else ROOT / 'outputs' / (
        time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    session.mkdir(parents=True, exist_ok=True)
    results = []
    with preparation_lock(ROOT / 'execution'):
        stop_file = ROOT / 'execution' / 'stop.request'
        if stop_file.exists(): stop_file.unlink()
        write_json(session / 'protocol.json', dict(datasets=args.datasets, methods=args.methods,
            repeats=args.repeats, percentages=args.percentages, N=args.n, timeout_seconds=args.timeout,
            private_limit_gib=args.private_limit_gib, reserve_gib=args.reserve_gib,
            independent_processes=True, original_F_unchanged=True,
            tpcar_mode='state_source', integer_scale=1_000_000,
            rounding='numpy.rint; ties-to-even; same canonical input for all methods',
            benchmark_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in ROOT.glob('bench_*.py')},
            method_inputs_rebuilt_each_run=True, OS_disk_cache_not_flushed=True,
            solver_baseline_scope='Gurobi LP/MIP only OP in --first-batch; all datasets available in individual entries',
            initial_gap_note='Same shared Top-N baseline defines D-K0; MCF/LP/MIP solve the model from scratch, not forced to use Top-N as a warm start.'))
        for dataset in args.datasets:
            graph_folder = prepare_shared(dataset); g = SharedGraph(graph_folder)
            K0, initial = g.initial(args.n)
            targets = [(None, args.target)] if args.target is not None else [
                (p, (g.meta['candidate_items'] * p + 99) // 100) for p in args.percentages]
            info = dict(graph_folder=str(graph_folder), graph_id=g.meta['graph_id'],
                        initial_coverage=K0, initial_integer_objective=initial)
            g.close()
            for method in args.methods:
                if args.first_batch and method.startswith('GUROBI') and dataset != 'OP': continue
                for percentage, target in targets:
                    for repeat in range(1, args.repeats + 1):
                        if stop_file.exists():
                            write_json(session / 'progress.json', dict(complete=False, state='STOP_REQUESTED', recorded_jobs=len(results)))
                            print('Comparison stopped; completed results retained. Resume with the same --session.', flush=True)
                            return session
                        job = dict(dataset=dataset, method=method, N=args.n, D=target,
                            percentage=percentage, repeat=repeat, timeout_seconds=args.timeout,
                            private_limit_gib=args.private_limit_gib, reserve_gib=args.reserve_gib, **info)
                        name = '%s_%s_N%d_D%d_r%d' % (dataset, method, args.n, target, repeat)
                        write_json(session / 'current_job.json', job)
                        result = run_job(job, session / name)
                        results.append(result); _, checks = summarize(results, session)
                        write_json(session / 'progress.json', dict(complete=False, recorded_jobs=len(results),
                            current_dataset=dataset, current_method=method, last_status=result['status'],
                            updated_unix=time.time()))
                        if result['status'] == 'STOP_REQUESTED':
                            write_json(session / 'progress.json', dict(complete=False, state='STOP_REQUESTED', recorded_jobs=len(results)))
                            print('Comparison stopped; completed results retained. Resume with the same --session.', flush=True)
                            return session
                        if any(c['consistency'] == 'MISMATCH' for c in checks):
                            raise RuntimeError('Exact cross-method mismatch: see comparison_checks.json; stopping before further runs')
                        if result['status'] in ('ERROR', 'SOLVER_ERROR', 'NONINTEGRAL_LP_SOLUTION', 'WORKER_CRASH'):
                            raise RuntimeError('Worker error: see %s; stopping before further runs' % (session / name))
                        if result['status'] in ('TIME_LIMIT', 'PRIVATE_MEMORY_LIMIT', 'SYSTEM_MEMORY_GUARD', 'MEMORY_ERROR'):
                            print('Resource-limited setting: remaining repeats skipped and not counted as completed.', flush=True)
                            break
    summarize(results, session)
    write_json(session / 'progress.json', dict(complete=True, recorded_jobs=len(results), updated_unix=time.time()))
    print('Comparison results:', session, flush=True)
    return session


def main(dataset=None, method=None):
    p = argparse.ArgumentParser(description='TPCA and exact baselines on the same rounded candidate graph')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=[dataset] if dataset else list(DATASETS))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=[method] if method else list(METHODS))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--percentages', nargs='+', type=int, default=[0, 20, 40, 60, 80, 100])
    p.add_argument('--target', type=int); p.add_argument('--n', type=int, default=10)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--private-limit-gib', type=float, default=12)
    p.add_argument('--reserve-gib', type=float, default=1)
    p.add_argument('--session', type=Path)
    p.add_argument('--first-batch', action='store_true', default=dataset is None)
    args = p.parse_args()
    if args.repeats < 1 or args.n < 1 or args.timeout <= 0 or args.private_limit_gib <= 0 or args.reserve_gib < 0:
        p.error('Invalid repetition, N, time, or memory limit')
    if any(pct < 0 or pct > 100 for pct in args.percentages) or (args.target is not None and args.target < 0):
        p.error('Invalid coverage target')
    return run_suite(args)


if __name__ == '__main__': main()
