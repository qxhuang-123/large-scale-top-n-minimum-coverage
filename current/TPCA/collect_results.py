"""Merge explicitly selected completed sessions, preserving exact-match checks."""
import argparse
import json
from pathlib import Path
from bench_common import write_json
from run_comparison import read_json, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sessions', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for folder in args.sessions:
        for result in sorted(folder.glob('*/result.json')):
            data = read_json(result)
            if data: records.append(data)
    # Shared protocol fields must agree; never silently mix resource settings.
    protocols = {(r['job']['N'], r['job']['timeout_seconds'], r['job']['private_limit_gib'],
                  r['job']['reserve_gib']) for r in records}
    if len(protocols) != 1: raise ValueError('Select sessions with the same N and resource protocol')
    versions = {json.dumps(r.get('benchmark_source_sha256'), sort_keys=True) for r in records}
    if len(versions) > 1: raise ValueError('Select sessions built with the same benchmark source version')
    if not records: raise ValueError('No finished results')
    _, checks = summarize(records, args.output)
    write_json(args.output / 'merged_sessions.json', [str(p) for p in args.sessions])
    if any(c['consistency'] == 'MISMATCH' for c in checks): raise RuntimeError('Integer objective/feasibility mismatch')
    print('Merged report:', args.output / 'comparison_report.md')


if __name__ == '__main__': main()
