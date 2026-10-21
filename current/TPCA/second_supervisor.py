"""Derived run-job supervisor; identical guards, with an explicit ablation worker selector. Original benchmark untouched."""
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
        proc = subprocess.Popen([sys.executable, str(ROOT / ('ablation_worker.py' if job.get('native_mode') else 'bench_worker.py')), str(folder / 'job.json')],
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

