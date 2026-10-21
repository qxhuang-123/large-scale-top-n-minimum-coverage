"""One immutable, rounded candidate input shared by every benchmark method."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
F_ROOT = ROOT.parent / 'CommonInput'
sys.path.insert(0, str(F_ROOT))
SCALE = 1_000_000
DATASETS = ('OP', 'YELP', 'TG', 'VG', 'SO')
METHODS = ('TPCAR', 'MCF', 'GUROBI-LP', 'GUROBI-MIP')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    def encode(v):
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, Path): return str(v)
        raise TypeError('Unsupported JSON value: %s' % type(v).__name__)
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode), encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(pending, path)
            break
        except PermissionError:
            # Readers can briefly deny rename/delete sharing on Windows.
            if attempt == 9: raise
            time.sleep(.01 * (attempt + 1))


def close_array(a):
    if isinstance(a, np.memmap) and not a._mmap.closed:
        if a.flags.writeable:
            a.flush()
        a._mmap.close()


class SharedGraph:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / 'graph.json').read_text(encoding='utf-8'))
        self.U = self.meta['n_users']; self.I = self.meta['n_items']; self.E = self.meta['edges']
        self.up = np.load(self.folder / 'user_indptr.npy', mmap_mode='r', allow_pickle=False)
        self.items = np.load(self.folder / 'user_items.npy', mmap_mode='r', allow_pickle=False)
        self.q = np.load(self.folder / 'scores_integer.npy', mmap_mode='r', allow_pickle=False)
        self.raw = np.load(self.folder / 'scores_original.npy', mmap_mode='r', allow_pickle=False)
        assert self.up.dtype == np.int32 and self.items.dtype == np.int32
        assert self.q.dtype == np.int64 and self.raw.dtype == np.float64
        assert self.up.shape == (self.U + 1,) and self.up[0] == 0 and self.up[-1] == self.E
        assert all(a.shape == (self.E,) for a in (self.items, self.q, self.raw))
        if self.meta['max_abs_integer_score'] * self.U * 30 >= 2**52:
            raise OverflowError('Integer objectives must remain exactly representable in float64')

    def initial(self, N):
        chosen = []
        for u in range(self.U):
            lo, hi = int(self.up[u]), int(self.up[u + 1])
            if hi - lo < N:
                raise ValueError('User %d has fewer than N candidates' % u)
            chosen.extend(range(lo, lo + N))
        e = np.asarray(chosen, dtype=np.int32)
        return len(np.unique(self.items[e])), int(self.q[e].sum(dtype=np.int64))

    def close(self):
        for a in (self.up, self.items, self.q, self.raw):
            close_array(a)


def prepare_shared(dataset):
    """Preserve every original candidate, then round and canonicalize row ties."""
    from prepare_cached_datasets import preparation_lock
    base = Path(os.environ.get('TOPN_SHARED_GRAPH_ROOT', str(ROOT / 'data'))).expanduser().resolve() / dataset
    with preparation_lock(base):
        ready = base / 'ready.json'
        if ready.exists():
            info = json.loads(ready.read_text(encoding='utf-8'))
            folder = base / Path(info['folder']).name
            if not (folder / 'graph.json').exists():
                folder = Path(info['folder'])
            if (folder / 'graph.json').exists():
                return folder
        if dataset in ('SO', 'VG', 'TG'):
            from prepare_cached_datasets import prepare
        else:
            from prepare_datasets import prepare
        source_meta = prepare(dataset)
        source = Path(source_meta.get('array_directory', F_ROOT / 'data' / dataset))
        up = np.load(source / 'user_indptr.npy', mmap_mode='r')
        old_items = np.load(source / 'user_items.npy', mmap_mode='r')
        old_scores = np.load(source / 'user_scores.npy', mmap_mode='r')
        U, I, E = source_meta['n_users'], source_meta['n_items'], len(old_items)
        folder = base / ('graph_' + uuid.uuid4().hex)
        folder.mkdir()
        arrays = []
        try:
            np.save(folder / 'user_indptr.npy', np.asarray(up, dtype=np.int32))
            items = np.lib.format.open_memmap(folder / 'user_items.npy', mode='w+', dtype=np.int32, shape=(E,))
            q = np.lib.format.open_memmap(folder / 'scores_integer.npy', mode='w+', dtype=np.int64, shape=(E,))
            raw = np.lib.format.open_memmap(folder / 'scores_original.npy', mode='w+', dtype=np.float64, shape=(E,))
            arrays.extend((items, q, raw))
            digests = {k: hashlib.sha256() for k in ('items', 'integer', 'original')}
            counts = np.zeros(I, dtype=np.int64)
            initial_items = set(); initial_score = 0; maxabs = 0
            started = time.perf_counter()
            for u in range(U):
                lo, hi = int(up[u]), int(up[u + 1])
                j = np.asarray(old_items[lo:hi]); r = np.asarray(old_scores[lo:hi])
                if not np.all(np.isfinite(r)) or len(np.unique(j)) != len(j):
                    raise ValueError('Invalid original row %d' % u)
                scaled = np.rint(r * SCALE)
                if np.max(np.abs(scaled), initial=0) >= 2**52:
                    raise OverflowError('Score out of range')
                rounded = scaled.astype(np.int64)
                order = np.lexsort((j, -rounded))
                j, rounded, r = j[order], rounded[order], r[order]
                items[lo:hi] = j; q[lo:hi] = rounded; raw[lo:hi] = r
                digests['items'].update(j.astype(np.int32).tobytes())
                digests['integer'].update(rounded.tobytes())
                digests['original'].update(r.astype(np.float64).tobytes())
                counts += np.bincount(j, minlength=I)
                initial_items.update(j[:10].tolist())
                initial_score += int(rounded[:10].sum(dtype=np.int64))
                maxabs = max(maxabs, int(np.max(np.abs(rounded), initial=0)))
                if (u + 1) % 5000 == 0:
                    print('%s shared rounding: %d/%d users' % (dataset, u + 1, U), flush=True)
            meta = dict(dataset=dataset, n_users=U, n_items=I, edges=E,
                        candidate_items=int(np.count_nonzero(counts)), scale=SCALE,
                        rounding='numpy.rint(float64(original_score)*1000000); ties-to-even',
                        ordering='integer_score_DESC_item_ID_ASC',
                        candidate_membership='unchanged from original prepared graph',
                        source=str(source), source_metadata=source_meta,
                        initial_N10_coverage=len(initial_items), initial_N10_integer_score=initial_score,
                        max_abs_integer_score=maxabs,
                        pointer_sha256=hashlib.sha256(np.asarray(up).tobytes()).hexdigest(),
                        content_sha256={k: v.hexdigest() for k, v in digests.items()},
                        common_preparation_seconds=time.perf_counter() - started)
            meta['graph_id'] = hashlib.sha256(json.dumps({k: meta[k] for k in (
                'n_users', 'n_items', 'edges', 'scale', 'rounding', 'ordering',
                'pointer_sha256', 'content_sha256')}, sort_keys=True).encode()).hexdigest()
            for a in arrays: a.flush()
            write_json(folder / 'graph.json', meta)
            write_json(ready, dict(folder=str(folder), graph_id=meta['graph_id']))
            print('%s shared graph ready: K0=%d C=%d, integer score=%d' % (
                dataset, len(initial_items), meta['candidate_items'], initial_score), flush=True)
            return folder
        finally:
            for a in arrays + [up, old_items, old_scores]: close_array(a)


def audit_selected(graph, selected, N, D):
    e = np.asarray(selected, dtype=np.int64)
    if e.shape != (graph.U * N,) or len(np.unique(e)) != len(e):
        raise AssertionError('Wrong number of selected edges or duplicate selections')
    if e.size and (e.min() < 0 or e.max() >= graph.E):
        raise AssertionError('Invalid edge ID')
    users = np.searchsorted(graph.up[1:], e, side='right')
    if not np.all(np.bincount(users, minlength=graph.U) == N):
        raise AssertionError('Per-user recommendation count is not N')
    K = len(np.unique(graph.items[e]))
    return dict(coverage=K, integer_objective=int(graph.q[e].sum(dtype=np.int64)),
                rounded_score=float(graph.q[e].sum(dtype=np.int64)) / SCALE,
                original_score=float(graph.raw[e].sum(dtype=np.float64)),
                rounded_APS=float(graph.q[e].sum(dtype=np.int64)) / SCALE / (graph.U * N),
                original_APS=float(graph.raw[e].sum(dtype=np.float64)) / (graph.U * N),
                target_feasible=K >= D, total_recommendations=len(e))
