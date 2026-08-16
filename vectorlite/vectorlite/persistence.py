"""Persistence for VectorLite indexes.

Format tradeoff
---------------
We use a hybrid: a single `.npz` archive holding
  - `vectors`  : the raw float32 matrix, stored natively by numpy (fast, compact,
                 zero parsing cost, no float precision loss)
  - `payload`  : one JSON blob with ids, metadata, texts, graph adjacency and
                 index hyper-parameters.

Why not pure pickle? Pickle is the easiest option but it executes arbitrary code
on load, so an index file becomes a remote-code-execution vector — unacceptable
for something meant to be shipped/downloaded. Why not pure JSON? Encoding
millions of floats as text is ~4x larger and orders of magnitude slower to
parse. npz + JSON gets numpy's binary speed for the bulk data and a
human-inspectable, safe format for the small structured part.

Writes are atomic: we write to a temp file and `os.replace` it, so a crash
mid-save can never corrupt an existing index.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from .index.ann import ANNIndex
from .index.base import BaseIndex
from .index.flat import FlatIndex

FORMAT_VERSION = 1
_KINDS = {"flat": FlatIndex, "ann": ANNIndex}


def save_index(index: BaseIndex, path: Union[str, Path]) -> Path:
    """Serialize `index` to `path` (a .npz file). Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = index.state()  # type: ignore[attr-defined]
    vectors = np.asarray(state.pop("vectors"), dtype=np.float32)
    payload = dict(state)
    payload["format_version"] = FORMAT_VERSION

    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, vectors=vectors, payload=np.array(json.dumps(payload)))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load_index(path: Union[str, Path]) -> BaseIndex:
    """Rebuild an index previously written with `save_index`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        vectors = archive["vectors"]
        payload = json.loads(str(archive["payload"]))

    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported index format version {payload.get('format_version')}")

    kind = payload["kind"]
    if kind not in _KINDS:
        raise ValueError(f"unknown index kind {kind!r}")

    payload["vectors"] = vectors
    return _KINDS[kind].from_state(payload)
