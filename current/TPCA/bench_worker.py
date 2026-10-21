"""One method/target/repetition per process; supervised by run_comparison.py."""
import argparse
import hashlib
import csv
import json
from pathlib import Path
import platform
import os
import sys
import time
import traceback

import numpy as np
from bench_common import SharedGraph, SCALE, write_json
from bench_solvers import solve_tpcar, solve_mcf, solve_gurobi
from bench_memory import ProcessMemory

SOURCE_HASHES_AT_START = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in Path(__file__).parent.glob('bench_*.py')}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('job', type=Path)
    args = parser.parse_args(); job = json.loads(args.job.read_text(encoding='utf-8'))
    folder = args.job.parent
    started = time.perf_counter(); optimization_started = [None]
    timings = {}

    def phase(name, **fields):
        timings.update(fields)
        if name == 'build' and optimization_started[0] is None:
            optimization_started[0] = time.time()
        write_json(folder / 'phase.json', dict(phase=name,
            pid=os.getpid(),
            optimization_started_unix=optimization_started[0], timings=timings, updated_unix=time.time()))

    graph = None
    try:
        phase('import_and_load')
        graph = SharedGraph(job['graph_folder'])
        if graph.meta['graph_id'] != job['graph_id']:
            raise AssertionError('Worker received a different candidate graph')
        K0, initial_objective = graph.initial(job['N'])
        if K0 != job['initial_coverage'] or initial_objective != job['initial_integer_objective']:
            raise AssertionError('Shared Top-N initialization differs from supervisor')
        load_seconds = time.perf_counter() - started
        kwargs = dict(g=graph, N=job['N'], D=job['D'], folder=folder, phase=phase)
        if job['method'] == 'TPCAR': result = solve_tpcar(**kwargs)
        elif job['method'] == 'MCF': result = solve_mcf(**kwargs)
        else: result = solve_gurobi(**kwargs, mip=job['method'] == 'GUROBI-MIP', budget_seconds=job['timeout_seconds'])
        before_export = time.perf_counter()
        phase('export', build_seconds=result['build_seconds'], solve_seconds=result['solve_seconds'])
        selected = result.pop('selected', None)
        trajectory = result.pop('trajectory', [])
        if selected is not None:
            np.save(folder / 'selected_edges.npy', selected)
            users = np.searchsorted(graph.up[1:], selected, side='right')
            order = np.lexsort((graph.items[selected], -graph.q[selected], users))
            with (folder / 'recommendations.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f); writer.writerow(['user_id_0based', 'rank', 'item_id_0based', 'integer_score', 'original_score'])
                last = -1; rank = 0
                for p in order:
                    u, e = int(users[p]), int(selected[p])
                    rank = rank + 1 if u == last else 1; last = u
                    writer.writerow([u, rank, int(graph.items[e]), int(graph.q[e]), float(graph.raw[e])])
        if result.get('audit'):
            trajectory.append(dict(time_sec=result['solve_seconds'],
                coverage=result['audit']['coverage'], integer_objective=result['audit']['integer_objective'],
                trajectory_kind='final_audited_endpoint'))
        write_json(folder / 'trajectory.json', trajectory)
        if trajectory:
            keys = list(dict.fromkeys(k for row in trajectory for k in row))
            with (folder / 'trajectory.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(trajectory)
        result.update(job=job, initial_coverage=K0, initial_integer_objective=initial_objective,
            requested_gap=max(0, job['D'] - K0), scale=SCALE, load_and_shared_checks_seconds=load_seconds,
            audit_and_extraction_seconds=before_export - started - load_seconds - result['build_seconds'] - result['solve_seconds'],
            export_seconds=time.perf_counter() - before_export, worker_elapsed_seconds=time.perf_counter() - started,
            optimization_total_seconds=result['build_seconds'] + result['solve_seconds'],
            python=sys.version, interpreter=sys.executable, platform=platform.platform(), numpy=np.__version__)
        result['benchmark_source_sha256'] = SOURCE_HASHES_AT_START
        if result.get('backend_file'):
            result['native_extension_sha256'] = hashlib.sha256(Path(result['backend_file']).read_bytes()).hexdigest()
        monitor = ProcessMemory(os.getpid())
        try: result['worker_self_memory'] = monitor.read()
        finally: monitor.close()
        write_json(folder / 'worker_result.json', result)
        phase('done')
    except BaseException as exc:
        traceback.print_exc()
        write_json(folder / 'worker_result.json', dict(status='MEMORY_ERROR' if isinstance(exc, MemoryError) else 'ERROR',
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), job=job,
            worker_elapsed_seconds=time.perf_counter() - started, partial_timings=timings))
        raise
    finally:
        if graph is not None: graph.close()


if __name__ == '__main__': main()
