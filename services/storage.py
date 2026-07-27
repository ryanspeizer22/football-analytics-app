"""
storage.py
----------
Where everything the app writes lives, and whether that place is real.

One root, resolved once. Every cache in the app hangs off it — generated
analyses, match contexts, crests, headshots, competition logos, OG cards and
YouTube thumbnails — so there is a single answer to "is this persisting?"
rather than one per subdirectory.

The root was three separate hardcoded relative paths, which made it impossible
to point the app at a volume mounted anywhere other than `<cwd>/.cache`. It is
now one setting. The default is unchanged, so a service that mounts its volume
at /app/.cache (as LAUNCH.md instructs) behaves exactly as before.

`state()` exists because the failure this module guards against is silent. An
unmounted volume is not an error at any single moment — writes succeed, reads
succeed, and the data is simply gone after the next deploy. Nothing in a
request can observe that, so the check has to be explicit and reported.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("pitchsense")

# PITCHSENSE_CACHE_DIR matches the app's own convention for its settings;
# CACHE_DIR is accepted as well since that is the obvious name to reach for.
# Relative values stay relative to the working directory, which is what the
# default has always been — WORKDIR /app makes ".cache" mean "/app/.cache".
_CONFIGURED = (os.environ.get("PITCHSENSE_CACHE_DIR")
               or os.environ.get("CACHE_DIR") or ".cache").strip()

CACHE_ROOT = Path(_CONFIGURED or ".cache")


def subdir(name: str) -> Path:
    """One cache directory under the root. Not created here — writers do that."""
    return CACHE_ROOT / name


def _writable() -> bool:
    """Whether the root actually accepts a write, right now.

    Probed rather than inferred: the directory existing, and even being a mount
    point, says nothing about whether this process can write to it. A volume
    owned by another user is the case that looks correct from the outside.
    """
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        probe = CACHE_ROOT / ".write-probe"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def state() -> dict[str, Any]:
    """Cache health, for /health.

    `persistent` is the one field worth alerting on. It is deliberately the
    conjunction of "this is a mount point" and "we can write to it": either
    alone is satisfiable by a service that loses everything on redeploy.
    """
    resolved = CACHE_ROOT.resolve()
    try:
        is_mount = os.path.ismount(str(CACHE_ROOT))
    except OSError:
        is_mount = False
    writable = _writable()

    entries: dict[str, int] = {}
    for child in ("summaries", "photos", "crests", "competitions", "og", "youtube"):
        path = CACHE_ROOT / child
        try:
            entries[child] = sum(1 for _ in path.iterdir()) if path.is_dir() else 0
        except OSError:
            entries[child] = -1   # present but unreadable

    return {
        "path": str(resolved),
        "configured": _CONFIGURED,
        "is_mount_point": is_mount,
        "writable": writable,
        "persistent": is_mount and writable,
        "entries": entries,
    }
