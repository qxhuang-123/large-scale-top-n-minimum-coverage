import numpy as np
from src.common.initial_topn import initial_topn

def test_score_descending_item_id_ascending_ties():
    indptr = np.array([0, 4, 7], dtype=np.int64)
    items = np.array([3, 1, 2, 0, 2, 0, 1], dtype=np.int32)
    scores = np.array([5, 5, 4, 5, 3, 4, 4], dtype=np.float32)
    selected, _, counts, objective, total, diversity = initial_topn(2, 4, indptr, items, scores, 2)
    assert selected.tolist() == [[0, 1], [0, 1]]
    assert counts.tolist() == [2, 2, 0, 0]
    assert objective == 18.0
    assert total == 4
    assert diversity == 2