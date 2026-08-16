"""FastAPI network layer for VectorLite.

The index is a single in-process object guarded by a lock (the numpy structures
are not thread-safe for concurrent writes). Reads are also taken under the lock
for simplicity; swap for an RWLock if read throughput matters.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Config, build_index, load_config
from .index.base import BaseIndex
from .persistence import load_index, save_index

_lock = threading.Lock()
_state: Dict[str, Any] = {"index": None, "config": None}


def get_index() -> BaseIndex:
    index = _state["index"]
    if index is None:
        raise HTTPException(status_code=503, detail="index not initialized")
    return index


def get_config() -> Config:
    return _state["config"]


# ------------------------------------------------------------------ schemas


class InsertRequest(BaseModel):
    id: str = Field(..., description="Unique id; re-inserting an id upserts it.")
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text: Optional[str] = None


class QueryRequest(BaseModel):
    vector: List[float]
    k: int = 10


class QueryHit(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None


class QueryResponse(BaseModel):
    results: List[QueryHit]


# ---------------------------------------------------------------- lifecycle


def _maybe_autosave() -> None:
    cfg = get_config()
    if cfg and cfg.autosave:
        save_index(_state["index"], cfg.persist_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state["config"] = cfg

    path = Path(cfg.persist_path)
    if path.exists():
        try:
            index = load_index(path)
            if index.dim != cfg.dim or index.metric != cfg.metric:
                raise ValueError("persisted index does not match configured dim/metric")
        except Exception as exc:  # corrupt or mismatched file -> start clean
            print(f"[vectorlite] could not load {path}: {exc}; starting empty index")
            index = build_index(cfg)
    else:
        index = build_index(cfg)

    _state["index"] = index
    print(f"[vectorlite] ready: {index.stats()}")
    yield
    if cfg.autosave:
        save_index(index, cfg.persist_path)


app = FastAPI(
    title="VectorLite",
    version="0.1.0",
    description="Lightweight embeddable vector search engine (flat + HNSW-inspired ANN).",
    lifespan=lifespan,
)


# ------------------------------------------------------------------- routes


@app.post("/insert", status_code=201)
def insert(req: InsertRequest) -> Dict[str, Any]:
    index = get_index()
    with _lock:
        try:
            index.insert(req.id, req.vector, req.metadata, req.text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _maybe_autosave()
    return {"ok": True, "id": req.id, "size": len(index)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    index = get_index()
    with _lock:
        try:
            hits = index.query(req.vector, req.k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return QueryResponse(results=[QueryHit(**h.to_dict()) for h in hits])


@app.delete("/vectors/{vector_id}")
def delete_vector(vector_id: str) -> Dict[str, Any]:
    index = get_index()
    with _lock:
        removed = index.delete(vector_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"id {vector_id!r} not found")
        _maybe_autosave()
    return {"ok": True, "id": vector_id, "size": len(index)}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    index = get_index()
    cfg = get_config()
    with _lock:
        payload = dict(index.stats())
    payload["persist_path"] = cfg.persist_path
    payload["persisted"] = os.path.exists(cfg.persist_path)
    return payload


@app.post("/save")
def save() -> Dict[str, Any]:
    """Force a snapshot to disk (useful when autosave is off)."""
    cfg = get_config()
    with _lock:
        path = save_index(get_index(), cfg.persist_path)
    return {"ok": True, "path": str(path)}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
