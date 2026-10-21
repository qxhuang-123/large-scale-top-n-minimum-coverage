"""Reuse the audited compact input/solver adapter with an explicit new-F mode.

No original benchmark source or binary is edited. The adapter imports this
process-local proxy; its native backend file/hash refers to the ablation binary.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import state_variants_ablation as native


def main():
    path = Path(sys.argv[1]); job = json.loads(path.read_text(encoding='utf8'))
    mode = job['native_mode']
    if mode not in ['state_source', 'state_terminal', 'state_source_only', 'state_source_noheap']:
        raise ValueError('Unknown ablation mode')
    proxy = ModuleType('state_variants_bench'); proxy.__file__ = native.__file__
    def run(*args):
        assert args[-1] == 'state_source'
        return native.run(*args[:-1], mode)
    proxy.run = run
    sys.modules['state_variants_bench'] = proxy
    import bench_worker
    bench_worker.main()
    from bench_common import ROOT, write_json
    result_path = path.parent/'worker_result.json'
    result = json.loads(result_path.read_text(encoding='utf8'))
    result['native_mode'] = mode
    result['ablation_source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'ablation_worker.py', ROOT/'second_supervisor.py', *sorted((ROOT/'native_ablation').glob('*'))]
        if p.is_file()}
    write_json(result_path, result)


if __name__ == '__main__': main()
