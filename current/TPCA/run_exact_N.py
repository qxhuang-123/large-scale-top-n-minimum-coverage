"""Twenty deterministic exact-result settings; one run each, no old TPCAR."""
import json
from pathlib import Path
import subprocess
import sys
import time
from bench_common import ROOT, write_json

SESSION = ROOT/'outputs/second_batch_N_exact_20261018'


def main():
    SESSION.mkdir(parents=True, exist_ok=True)
    for N in [15, 20, 25, 30]:
        stage = SESSION/('N%d' % N)
        write_json(SESSION/'queue_progress.json', dict(complete=False, current_N=N,
            note='One deterministic quality/boundary run per graph and N; not a repeated performance comparison.',
            updated_unix=time.time()))
        subprocess.run([sys.executable, str(ROOT/'run_comparison.py'),
            '--datasets', 'OP', 'YELP', 'TG', 'VG', 'SO', '--methods', 'TPCAR',
            '--percentages', '100', '--n', str(N), '--repeats', '1', '--session', str(stage)], check=True)
        progress = json.loads((stage/'progress.json').read_text(encoding='utf8'))
        if not progress.get('complete'):
            write_json(SESSION/'queue_progress.json', dict(complete=False, state='STOP_REQUESTED', current_N=N))
            return
    rows = []
    for path in SESSION.glob('N*/*/result.json'):
        r = json.loads(path.read_text(encoding='utf8'))
        rows.append(dict(dataset=r['job']['dataset'], N=r['job']['N'],
            D=r['job']['D'], status=r['status'], initial_coverage=r['job']['initial_coverage'],
            **r.get('audit', {}), result_file=str(path)))
    write_json(SESSION/'exact_N_results.json', rows)
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (SESSION/'exact_N_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    write_json(SESSION/'queue_progress.json', dict(complete=True, recorded_jobs=len(rows), updated_unix=time.time()))
    print('Exact N results:', SESSION, flush=True)


if __name__ == '__main__': main()
