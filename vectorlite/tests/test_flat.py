import numpy as np
import pytest

from vectorlite.index import FlatIndex


def test_insert_and_exact_query_cosine():
    idx = FlatIndex(dim=3, metric="cosine")
    idx.insert("a", [1, 0, 0], {"tag": "x"}, text="alpha")
    idx.insert("b", [0, 1, 0], {"tag": "y"})
    idx.insert("c", [0.9, 0.1, 0], {"tag": "z"})

    res = idx.query([1, 0, 0], k=2)
    assert [r.id for r in res] == ["a", "c"]
    assert res[0].score == pytest.approx(1.0, abs=1e-5)
    assert res[0].metadata == {"tag": "x"}
    assert res[0].text == "alpha"


def test_euclidean_ranking():
    idx = FlatIndex(dim=2, metric="euclidean")
    idx.insert("near", [1.0, 1.0])
    idx.insert("far", [10.0, 10.0])
    res = idx.query([1.1, 1.1], k=2)
    assert [r.id for r in res] == ["near", "far"]
    assert res[0].score > res[1].score  # higher == better for both metrics


def test_delete_removes_from_results():
    idx = FlatIndex(dim=2)
    idx.insert("a", [1, 0])
    idx.insert("b", [0, 1])
    assert idx.delete("a") is True
    assert idx.delete("a") is False
    assert len(idx) == 1
    assert [r.id for r in idx.query([1, 0], k=5)] == ["b"]


def test_upsert_replaces_vector():
    idx = FlatIndex(dim=2)
    idx.insert("a", [1, 0], {"v": 1})
    idx.insert("a", [0, 1], {"v": 2})
    assert len(idx) == 1
    res = idx.query([0, 1], k=1)
    assert res[0].id == "a" and res[0].metadata == {"v": 2}


def test_matches_numpy_bruteforce():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 16)).astype(np.float32)
    idx = FlatIndex(dim=16, metric="cosine")
    for i, v in enumerate(data):
        idx.insert(str(i), v)

    q = rng.normal(size=16).astype(np.float32)
    norm = data / np.linalg.norm(data, axis=1, keepdims=True)
    expected = np.argsort(-(norm @ (q / np.linalg.norm(q))))[:5]
    assert [r.id for r in idx.query(q, 5)] == [str(i) for i in expected]


def test_dimension_validation():
    idx = FlatIndex(dim=4)
    with pytest.raises(ValueError):
        idx.insert("a", [1, 2, 3])
