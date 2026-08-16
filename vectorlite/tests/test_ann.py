import numpy as np

from vectorlite.index import ANNIndex, FlatIndex


def _dataset(n=2000, dim=32, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def test_ann_recall_above_threshold():
    dim, k = 32, 10
    data = _dataset(dim=dim)
    flat = FlatIndex(dim=dim, metric="cosine")
    ann = ANNIndex(dim=dim, metric="cosine", M=16, ef_construction=100, ef_search=100)
    for i, v in enumerate(data):
        flat.insert(str(i), v)
        ann.insert(str(i), v)

    rng = np.random.default_rng(123)
    queries = rng.normal(size=(30, dim)).astype(np.float32)
    hits = total = 0
    for q in queries:
        truth = {r.id for r in flat.query(q, k)}
        approx = {r.id for r in ann.query(q, k)}
        hits += len(truth & approx)
        total += len(truth)

    recall = hits / total
    assert recall > 0.85, f"recall too low: {recall:.3f}"


def test_ann_interface_matches_flat():
    dim = 8
    ann = ANNIndex(dim=dim)
    ann.insert("a", np.ones(dim), {"m": 1}, text="t")
    ann.insert("b", -np.ones(dim))
    res = ann.query(np.ones(dim), 1)
    assert res[0].id == "a" and res[0].metadata == {"m": 1} and res[0].text == "t"

    assert ann.delete("a") is True
    assert ann.delete("a") is False
    assert len(ann) == 1
    assert [r.id for r in ann.query(np.ones(dim), 5)] == ["b"]


def test_ann_handles_deleted_entry_point():
    dim = 6
    ann = ANNIndex(dim=dim, ef_construction=20, ef_search=20)
    data = _dataset(n=50, dim=dim, seed=3)
    for i, v in enumerate(data):
        ann.insert(str(i), v)
    ann.delete("0")  # entry point
    assert len(ann.query(data[10], 5)) == 5


def test_empty_index_returns_nothing():
    assert ANNIndex(dim=4).query(np.zeros(4), 5) == []
