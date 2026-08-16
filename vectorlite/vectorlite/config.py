"""Runtime configuration, driven by environment variables.

All options can be set at startup, e.g.:
    VECTORLITE_INDEX=ann VECTORLITE_DIM=384 uvicorn vectorlite.api:app
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    index_type: str = field(default_factory=lambda: _env("VECTORLITE_INDEX", "flat"))
    dim: int = field(default_factory=lambda: int(_env("VECTORLITE_DIM", "128")))
    metric: str = field(default_factory=lambda: _env("VECTORLITE_METRIC", "cosine"))
    persist_path: str = field(
        default_factory=lambda: _env("VECTORLITE_PERSIST_PATH", "data/index.npz")
    )
    autosave: bool = field(
        default_factory=lambda: _env("VECTORLITE_AUTOSAVE", "true").lower() == "true"
    )
    # ANN tuning
    M: int = field(default_factory=lambda: int(_env("VECTORLITE_M", "16")))
    ef_construction: int = field(
        default_factory=lambda: int(_env("VECTORLITE_EF_CONSTRUCTION", "100"))
    )
    ef_search: int = field(default_factory=lambda: int(_env("VECTORLITE_EF_SEARCH", "100")))


def load_config() -> Config:
    cfg = Config()
    if cfg.index_type not in ("flat", "ann"):
        raise ValueError("VECTORLITE_INDEX must be 'flat' or 'ann'")
    if cfg.metric not in ("cosine", "euclidean"):
        raise ValueError("VECTORLITE_METRIC must be 'cosine' or 'euclidean'")
    return cfg


def build_index(cfg: Config):
    """Create a fresh index matching the config."""
    from .index import ANNIndex, FlatIndex

    if cfg.index_type == "flat":
        return FlatIndex(dim=cfg.dim, metric=cfg.metric)
    return ANNIndex(
        dim=cfg.dim,
        metric=cfg.metric,
        M=cfg.M,
        ef_construction=cfg.ef_construction,
        ef_search=cfg.ef_search,
    )
