"""Adapt existing candidate caches to F without changing candidate membership."""
import json
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import shutil
import time
import uuid
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = Path(os.environ.get('TOPN_CANDIDATE_CACHE_ROOT', str(ROOT / 'data/candidate_caches'))).expanduser().resolve()
SPECS = {
    'SO': dict(users=35598, items=18357, edges=261282620, candidate_items=17951,
               initial_coverage=2964, initial_score=1768054.7759962082),
    'VG': dict(users=24442, items=10350, edges=101111454, candidate_items=10014,
               initial_coverage=1251, initial_score=1184136.1799118519),
    'TG': dict(users=19412, items=11924, edges=92529265, candidate_items=11709,
               initial_coverage=2426, initial_score=849787.793035984),
}
KEYS = ('user_indptr', 'user_items', 'user_scores', 'edge_users',
        'item_indptr', 'item_users', 'item_edges', 'item_scores')


def source_path(dataset):
    return CACHE_ROOT / (dataset + '贪心算法_cache') / (dataset + '_cand_pos_top40_seed20260704_v2.npz')


def read_header(stream):
    version = np.lib.format.read_magic(stream)
    return np.lib.format._read_array_header(stream, version)


def check_inputs(dataset):
    spec = SPECS[dataset]
    path = source_path(dataset)
    if not path.is_file():
        raise FileNotFoundError('Original candidate cache is missing: %s' % path)
    with ZipFile(path) as archive:
        for key, shape in [('user_items', (spec['edges'],)),
                           ('user_scores', (spec['edges'],)),
                           ('user_indptr', (spec['users'] + 1,)),
                           ('item_indptr', (spec['items'] + 1,))]:
            with archive.open(key + '.npy') as stream:
                actual, fortran, dtype = read_header(stream)
            if actual != shape or fortran or dtype.kind != ('f' if key.endswith('scores') else 'i'):
                raise ValueError('Unexpected cache header: %s %s %s' % (key, actual, dtype))
    with np.load(path, allow_pickle=False) as archive:
        up, ip = archive['user_indptr'], archive['item_indptr']
    if any(p[0] != 0 or p[-1] != spec['edges'] or np.any(np.diff(p) < 0) for p in (up, ip)):
        raise ValueError('Invalid original CSR pointers')
    if np.any(np.diff(up) < 10) or np.count_nonzero(np.diff(ip)) != spec['candidate_items']:
        raise ValueError('Original cache does not match the archived graph')
    return dict(dataset=dataset, source=str(path), source_bytes=path.stat().st_size,
                source_mtime_ns=path.stat().st_mtime_ns, **spec)


@contextmanager
def preparation_lock(folder):
    """Prevent concurrent writers; the OS releases the lock after a crash."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'preparation.lock').open('a+b') as lock:
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError(
                'This dataset is already being prepared by another process. '
                'Wait for that run to finish, or stop the earlier run in PyCharm before restarting. '
                'Do not delete preparation.lock; its existence alone does not mean it is locked.'
            ) from exc
        try:
            yield
        finally:
            lock.seek(0)
            release()


def close_mapping(array):
    """Close Windows file mappings explicitly, including on failed preparation."""
    try:
        array.flush()
    finally:
        array._mmap.close()


def prepare(dataset):
    folder = ROOT / 'data' / dataset
    with preparation_lock(folder), ExitStack() as mappings:
        return _prepare(dataset, mappings)


def _prepare(dataset, mappings):
    folder = ROOT / 'data' / dataset
    ready = folder / 'metadata.json'
    if ready.exists():
        meta = json.loads(ready.read_text(encoding='utf-8'))
        array_folder = Path(meta.get('array_directory', str(folder)))
        if all((array_folder / (k + '.npy')).exists() for k in KEYS):
            print('Using prepared dataset', dataset, flush=True)
            return meta
    info = check_inputs(dataset)
    path = Path(info['source'])
    U, I, E = info['users'], info['items'], info['edges']
    folder.mkdir(parents=True, exist_ok=True)
    required = E * 32 + (U + I + 2) * 4
    if shutil.disk_usage(folder).free < required + 512 * 1024**2:
        raise OSError('F cache preparation needs approximately %.2f GiB free disk space' % (required / 1024**3))
    # Never truncate files from an interrupted run: Windows may retain a mapping
    # in an IDE console even when the old run has ended. Publish the new folder
    # only after all arrays have passed the preparation checks.
    folder = folder / ('arrays_' + uuid.uuid4().hex)
    folder.mkdir()
    print('%s: first-time format conversion of %s candidate edges; solving has not started.' % (dataset, format(E, ',')), flush=True)
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        up = archive['user_indptr'].astype(np.int32)
        ip = archive['item_indptr'].astype(np.int32)
    np.save(folder / 'user_indptr.npy', up)
    np.save(folder / 'item_indptr.npy', ip)
    arrays = {}
    for key in KEYS:
        if key not in ('user_indptr', 'item_indptr'):
            try:
                array = np.lib.format.open_memmap(folder / (key + '.npy'), mode='w+',
                    dtype=np.float64 if key.endswith('scores') else np.int32, shape=(E,))
            except OSError as exc:
                if os.name == 'nt' and exc.errno in (13, 22):
                    raise RuntimeError(
                        'Windows cannot recreate %s because it may still be mapped by an earlier run. '
                        'Stop the previous dataset run (including any Python Console holding its arrays), '
                        'then run this entry point once. The original candidate cache is unchanged.'
                        % (folder / (key + '.npy'))
                    ) from exc
                raise
            arrays[key] = array
            mappings.callback(close_mapping, array)
    cursor = ip[:-1].astype(np.int64)
    covered = np.zeros(I, dtype=bool)
    initial_score = 0.0
    with ZipFile(path) as archive, archive.open('user_items.npy') as item_stream, archive.open('user_scores.npy') as score_stream:
        _, _, item_dtype = read_header(item_stream)
        _, _, score_dtype = read_header(score_stream)
        for u in range(U):
            lo, hi = int(up[u]), int(up[u + 1])
            count = hi - lo
            row = np.frombuffer(item_stream.read(count * item_dtype.itemsize), dtype=item_dtype)
            scores = np.frombuffer(score_stream.read(count * score_dtype.itemsize), dtype=score_dtype)
            if len(row) != count or len(scores) != count:
                raise ValueError('Truncated original cache')
            if np.any(row < 0) or np.any(row >= I) or len(np.unique(row)) != count or not np.all(np.isfinite(scores)):
                raise ValueError('Invalid candidate row %d' % u)
            order = np.lexsort((row, -scores))
            row, scores = row[order], scores[order].astype(np.float64)
            arrays['user_items'][lo:hi] = row
            arrays['user_scores'][lo:hi] = scores
            arrays['edge_users'][lo:hi] = u
            positions = cursor[row]
            if np.any(positions >= ip[row + 1]):
                raise ValueError('Original item degrees disagree with candidate rows')
            arrays['item_edges'][positions] = np.arange(lo, hi, dtype=np.int32)
            arrays['item_users'][positions] = u
            arrays['item_scores'][positions] = scores
            cursor[row] += 1
            covered[row[:10]] = True
            initial_score += float(scores[:10].sum())
            if (u + 1) % 1000 == 0:
                print('%s converted %d/%d users (%.1f s)' % (dataset, u + 1, U, time.perf_counter() - started), flush=True)
    if not np.array_equal(cursor, ip[1:]):
        raise ValueError('Incomplete item columns')
    if int(covered.sum()) != info['initial_coverage'] or abs(initial_score - info['initial_score']) > 1e-4:
        raise ValueError('Initial solution does not match the archived graph')
    for a in arrays.values():
        a.flush()
    meta = dict(dataset=dataset, source=str(path), source_bytes=info['source_bytes'],
                array_directory=str(folder),
                source_mtime_ns=info['source_mtime_ns'], n_users=U, n_items=I,
                candidate_edges=E, candidate_items=info['candidate_items'], N=10, alpha=0.4,
                candidate_selection='reuse_original_pos_top40_seed20260704_v2_cache_no_rescreening',
                ordering='score_desc_item_id_asc', initial_coverage=int(covered.sum()),
                initial_score=initial_score, reference_initial_score=info['initial_score'],
                initial_score_reference_difference=initial_score - info['initial_score'],
                preparation_seconds=time.perf_counter() - started)
    pending = ready.with_suffix('.pending.json')
    pending.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    pending.replace(ready)
    print('%s format conversion finished in %.2f s' % (dataset, meta['preparation_seconds']), flush=True)
    return meta
