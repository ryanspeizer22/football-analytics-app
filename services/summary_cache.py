"""
summary_cache.py
----------------
Tiny two-level cache for generated AI summaries.

Level 1: in-process dict (fast path within a server run).
Level 2: JSON files under <cache root>/summaries/ (survives restarts and
         --reload). The root defaults to .cache and is set by storage.py.

Finished matches never change, so entries have no expiry — delete the
.cache directory (or use the route's ?refresh=true) to force regeneration.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from services import storage

CACHE_DIR = storage.subdir("summaries")

logger = logging.getLogger("pitchsense")

_memory: dict[str, Any] = {}


def _path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def get(key: str) -> Optional[Any]:
    """Return the cached value for key, or None on a miss."""
    if key in _memory:
        return _memory[key]

    file = _path(key)
    if file.exists():
        try:
            value = json.loads(file.read_text())
        except (OSError, json.JSONDecodeError):
            return None  # corrupt/unreadable entry — treat as a miss
        _memory[key] = value
        return value
    return None


def set(key: str, value: Any) -> None:
    """Store a JSON-serializable value in both cache levels.

    The disk half is best-effort, matching get() above: an unreadable entry is
    already treated as a miss, so an unwritable one is treated as a skipped
    write rather than an error. A cache root that cannot be written — a
    persistent volume that never mounted, one owned by another user, or one
    that has filled up — used to raise out of whichever handler was storing its
    result, which turned every generated section of the page into a 500. The
    value is still held in memory for the life of the process, so the request
    that produced it is answered either way.
    """
    _memory[key] = value
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _path(key).write_text(json.dumps(value, ensure_ascii=False))
    except OSError as exc:
        logger.warning("Could not persist cache entry %s (%s); keeping it in "
                       "memory only", key, exc)
