"""After the full-size validation, continue the formal suite automatically."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from bench_common import ROOT, write_json
from run_comparison import read_json


def main():
    validation = ROOT / 'outputs/SO_compact_validation_20261018'
    session = ROOT / 'outputs/first_batch_final_20261018'
    print('Waiting for the full-size compact-input SO validation; the suite will continue automatically.', flush=True)
    while not read_json(validation / 'progress.json').get('complete'):
        state = read_json(validation / 'progress.json')
        if state.get('state') == 'STOP_REQUESTED':
            print('Stop requested; formal suite not launched.', flush=True); return
        time.sleep(1)
    if (ROOT / 'execution/stop.request').exists():
        print('Stop requested; formal suite not launched.', flush=True); return
    source = validation / 'SO_TPCAR_N10_D17951_r1'
    result = read_json(source / 'result.json')
    if result.get('status') != 'OPTIMAL':
        raise RuntimeError('Full-size validation did not finish optimally; inspect %s' % source)
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('bench_*.py')}
    if result.get('benchmark_source_sha256') != current:
        raise RuntimeError('Benchmark code changed after validation; use a fresh validation run')
    extension = ROOT / 'state_variants_bench.cp39-win_amd64.pyd'
    if result.get('native_extension_sha256') != hashlib.sha256(extension.read_bytes()).hexdigest():
        raise RuntimeError('Native backend changed after validation')
    session.mkdir(parents=True, exist_ok=True)
    dest = session / source.name
    if not dest.exists():
        shutil.copytree(source, dest)
        result.update(reused_from_full_size_validation=str(source), result_folder=str(dest))
        write_json(dest / 'result.json', result)
    write_json(session / 'validation_reuse.json', dict(source=str(source), target=str(dest),
        rule='First independent run reused only with identical graph, integer scores, N, D, resources and actual code hashes. No source inputs or measured times changed.'))
    print('Full-size validation passed. Starting the full independent-process comparison suite.', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'run_comparison.py'),
        '--datasets', 'SO', 'YELP', 'TG', 'VG', 'OP',
        '--percentages', '100', '80', '60', '40', '20', '0',
        '--repeats', '3', '--first-batch', '--session', str(session)], check=True)


if __name__ == '__main__': main()
