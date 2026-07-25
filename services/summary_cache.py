"""
summary_cache.py
----------------
Tiny two-level cache for generated AI summaries.

Level 1: in-process dict (fast path within a server run).
Level 2: JSON files under .cache/summaries/ (survives restarts and --reload).

Finished matches never change, so entries have no expiry — delete the
.cache directory (or use the route's ?refresh=true) to force regeneration.
"""

import json
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(".cache/summaries")

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
    """Store a JSON-serializable value in both cache levels."""
    _memory[key] = value
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _path(key).write_text(json.dumps(value, ensure_ascii=False))
