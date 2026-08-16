import numpy as np

from vectorlite.index import ANNIndex, FlatIndex
from vectorlite.persistence import load_index, save_index


def _fill(idx, n=150, dim=24, seed=1):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, dim)).astype(np.float32)
    for i, v in enumerate(data):
        idx.insert(f"v{i}", v, {"i": i}, text=f"doc {i}")
    return data


def test_flat_roundtrip(tmp_path):
    idx = FlatIndex(dim=24, metric="cosine")
    data = _fill(idx, dim=24)
    idx.delete("v3")

    path = save_index(idx, tmp_path / "flat.npz")
    reloaded = load_index(path)

    assert isinstance(reloaded, FlatIndex)
    assert len(reloaded) == len(idx)
    assert reloaded.dim == idx.dim and reloaded.metric == idx.metric

    for q in data[:10]:
        before = [(r.id, round(r.score, 6), r.metadata, r.text) for r in idx.query(q, 5)]
        after = [(r.id, round(r.score, 6), r.metadata, r.text) for r in reloaded.query(q, 5)]
        assert before == after


def test_ann_roundtrip_preserves_graph(tmp_path):
    idx = ANNIndex(dim=24, metric="euclidean", M=8, ef_construction=50, ef_search=32)
    data = _fill(idx, dim=24, seed=2)
    idx.delete("v10")

    path = save_index(idx, tmp_path / "ann.npz")
    reloaded = load_index(path)

    assert isinstance(reloaded, ANNIndex)
    assert reloaded.stats()["edges"] == idx.stats()["edges"]
    for q in data[:10]:
        assert [r.id for r in idx.query(q, 5)] == [r.id for r in reloaded.query(q, 5)]


def test_writes_are_usable_after_reload(tmp_path):
    idx = FlatIndex(dim=8)
    _fill(idx, n=20, dim=8)
    path = save_index(idx, tmp_path / "i.npz")
    reloaded = load_index(path)
    reloaded.insert("new", np.ones(8, dtype=np.float32), {"fresh": True})
    assert reloaded.query(np.ones(8), 1)[0].id == "new"
