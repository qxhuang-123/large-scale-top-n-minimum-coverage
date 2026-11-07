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
    """Read the explicitly configured immutable graph; never rebuild candidates implicitly."""
    config=ROOT.parent.parent/'数据路径.json'
    locations=json.loads(config.read_text(encoding='utf8'))
    folder=Path(locations[dataset])
    if not folder.is_absolute():folder=(config.parent/folder).resolve()
    if not (folder/'graph.json').is_file():
        raise FileNotFoundError('Configure 数据路径.json for '+dataset+': '+str(folder))
    return folder


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
