"""One isolated TPCA component-comparison job using the common audited adapter."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tpca_dis_index_core as native

if __name__ == '__main__':
    path = Path(sys.argv[1])
    job = json.loads(path.read_text(encoding='utf-8'))
    mode = job['native_mode']
    if mode not in ('state_dis1', 'state_dis2', 'state_source'):
        raise ValueError(mode)
    proxy = ModuleType('state_variants_bench')
    proxy.__file__ = native.__file__
    def run(*args):
        assert args[-1] == 'state_source'
        return native.run(*args[:-1], mode)
    proxy.run = run
    sys.modules['state_variants_bench'] = proxy
    import bench_common
    from resilient_json import write_json
    bench_common.write_json = write_json
    import bench_worker
    bench_worker.main()
    result_path = path.parent / 'worker_result.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    result['native_mode'] = mode
    result['component_backend_sha256'] = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    write_json(result_path, result)
