import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTORLITE_INDEX", "flat")
    monkeypatch.setenv("VECTORLITE_DIM", "4")
    monkeypatch.setenv("VECTORLITE_METRIC", "cosine")
    monkeypatch.setenv("VECTORLITE_PERSIST_PATH", str(tmp_path / "index.npz"))

    import importlib

    import vectorlite.api as api

    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


def test_insert_query_delete_stats(client):
    r = client.post("/insert", json={"id": "a", "vector": [1, 0, 0, 0], "metadata": {"t": "x"}})
    assert r.status_code == 201
    client.post("/insert", json={"id": "b", "vector": [0, 1, 0, 0]})

    r = client.post("/query", json={"vector": [1, 0, 0, 0], "k": 2})
    hits = r.json()["results"]
    assert hits[0]["id"] == "a" and hits[0]["metadata"] == {"t": "x"}
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)

    assert client.delete("/vectors/a").status_code == 200
    assert client.delete("/vectors/a").status_code == 404

    stats = client.get("/stats").json()
    assert stats["size"] == 1 and stats["dim"] == 4 and stats["type"] == "flat"


def test_bad_dimension_returns_400(client):
    r = client.post("/insert", json={"id": "z", "vector": [1, 2]})
    assert r.status_code == 400


def test_state_persists_across_restart(client, tmp_path):
    client.post("/insert", json={"id": "keep", "vector": [0, 0, 1, 0]})
    client.post("/save")

    import importlib

    import vectorlite.api as api

    importlib.reload(api)
    with TestClient(api.app) as c2:
        assert c2.get("/stats").json()["size"] == 1
        assert c2.post("/query", json={"vector": [0, 0, 1, 0], "k": 1}).json()["results"][0]["id"] == "keep"
