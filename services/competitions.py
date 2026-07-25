"""
competitions.py
---------------
Registry of the competitions PitchSense ingests: the top-5 European leagues,
their domestic cups, the three UEFA club competitions, and the major
international tournaments.

Two different notions of "available season" are tracked, and conflating them
is the main source of dead-end searches:

  * ``seasons``  — what the competition actually has, per API-Football.
  * plan window  — what THIS account's subscription may request.

The effective set is the intersection. A Free plan can only read seasons
2022-2024, so asking for 2025 returns a plan error rather than data; the
window is configurable so upgrading the plan needs no code change:

    PITCHSENSE_MIN_SEASON=2022   # oldest season the plan allows
    PITCHSENSE_MAX_SEASON=2024   # newest season the plan allows
    PITCHSENSE_SEASON=2024       # season the UI targets by default

After an API-Football upgrade, set MAX_SEASON=2026 and SEASON=2025 to switch
the whole app to the 25/26 campaign.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

_DATA_FILE = Path(__file__).with_name("competitions_data.json")

# --- Plan access window -----------------------------------------------------
# Defaults describe the API-Football Pro tier (verified: seasons 2024-2026
# reachable, 7500 requests/day). On the Free tier these must be 2022/2024.
PLAN_MIN_SEASON = int(os.environ.get("PITCHSENSE_MIN_SEASON", "2021"))
PLAN_MAX_SEASON = int(os.environ.get("PITCHSENSE_MAX_SEASON", "2026"))

# The season the UI opens on. Deliberately the most recent *completed* season
# rather than the newest reachable one: 2026/27 exists in the provider's data
# but hasn't kicked off, so it has no played matches to analyse.
DEFAULT_COMPLETED_SEASON = int(os.environ.get("PITCHSENSE_SEASON", "2025"))

COMPETITIONS: list[dict[str, Any]] = json.loads(_DATA_FILE.read_text())
_BY_ID: dict[int, dict[str, Any]] = {c["id"]: c for c in COMPETITIONS}

KIND_LABELS = {
    "domestic_league": "Top Leagues",
    "domestic_cup": "Domestic Cups",
    "european": "European",
    "international": "International",
}


def plan_window() -> tuple[int, int]:
    return PLAN_MIN_SEASON, PLAN_MAX_SEASON


def accessible_seasons(competition_id: int) -> list[int]:
    """Seasons that both exist for this competition and the plan can request."""
    comp = _BY_ID.get(competition_id)
    if not comp:
        return []
    return [s for s in comp["seasons"] if PLAN_MIN_SEASON <= s <= PLAN_MAX_SEASON]


def default_season(competition_id: Optional[int] = None) -> int:
    """The season the UI targets.

    Honours PITCHSENSE_SEASON when set and reachable; otherwise picks the
    newest season this account can actually read.
    """
    if competition_id is not None:
        options = accessible_seasons(competition_id)
        if DEFAULT_COMPLETED_SEASON in options:
            return DEFAULT_COMPLETED_SEASON
        return options[-1] if options else PLAN_MAX_SEASON
    return DEFAULT_COMPLETED_SEASON


def get_competition(competition_id: int) -> Optional[dict[str, Any]]:
    return _BY_ID.get(competition_id)


def competition_name(competition_id: int, fallback: str = "") -> str:
    comp = _BY_ID.get(competition_id)
    return comp["name"] if comp else fallback


def is_tracked(competition_id: int) -> bool:
    return competition_id in _BY_ID


def public_list() -> list[dict[str, Any]]:
    """Competitions grouped for the UI, annotated with reachable seasons."""
    out = []
    for comp in COMPETITIONS:
        seasons = accessible_seasons(comp["id"])
        out.append(
            {
                "id": comp["id"],
                "name": comp["name"],
                "short": comp["short"],
                "kind": comp["kind"],
                "kind_label": KIND_LABELS.get(comp["kind"], comp["kind"]),
                "country": comp["country"],
                "emoji": comp["emoji"],
                "seasons": seasons,
                # default_season(), not seasons[-1]: the newest reachable
                # season may not have kicked off yet, and opening a competition
                # on a season with no played matches shows an empty grid.
                "default_season": default_season(comp["id"]) if seasons else None,
                "available": bool(seasons),
            }
        )
    return out


def search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Prefix/substring match on competition name, short code, or country."""
    q = query.lower().strip()
    if not q:
        return []
    hits = []
    for comp in public_list():
        haystack = [comp["name"].lower(), comp["short"].lower(), comp["country"].lower()]
        if any(h.startswith(q) or f" {q}" in f" {h}" for h in haystack):
            hits.append(comp)
    return hits[:limit]
