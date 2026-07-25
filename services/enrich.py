"""
enrich.py
---------
Attaches real player metadata (headshot, match rating, position, shirt number)
to the AI's player notes.

The model is asked to copy each player's numeric id out of the supplied stats,
but an id echoed by a language model is never trusted on its own: it is checked
against the actual match roster, and anything that doesn't resolve falls back
to name matching. A note that still can't be matched keeps its text and simply
renders without a photo, so enrichment can never drop analysis.
"""

import logging
import re
import unicodedata
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Casefold and strip accents/punctuation for tolerant comparison."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", ascii_only.lower()).strip()


def build_roster(match_context: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index every player in the match by id, with their headshot and stats."""
    roster: dict[int, dict[str, Any]] = {}
    for team_block in match_context.get("player_statistics") or []:
        team = team_block.get("team") or {}
        for entry in team_block.get("players") or []:
            info = entry.get("player") or {}
            pid = info.get("id")
            if pid is None:
                continue
            stats = (entry.get("statistics") or [{}])[0]
            games = stats.get("games") or {}
            roster[pid] = {
                "id": pid,
                "name": info.get("name") or "",
                "photo": info.get("photo"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "rating": games.get("rating"),
                "position": games.get("position"),
                "number": games.get("number"),
                "minutes": games.get("minutes"),
            }
    return roster


def _match_by_name(name: str, roster: dict[int, dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Resolve a player name against the roster with progressively looser rules.

    Handles the common mismatch where events abbreviate a forename
    ("A. Isak") while the roster spells it out ("Alexander Isak").
    """
    target = _normalize(name)
    if not target:
        return None

    candidates = [(p, _normalize(p["name"])) for p in roster.values()]

    for player, norm in candidates:
        if norm == target:
            return player

    target_parts = target.split()
    if not target_parts:
        return None
    target_surname = target_parts[-1]
    target_initial = target_parts[0][0] if len(target_parts) > 1 else None

    surname_hits = [
        player for player, norm in candidates
        if norm.split() and norm.split()[-1] == target_surname
    ]
    if len(surname_hits) == 1:
        return surname_hits[0]
    if target_initial:
        for player in surname_hits:
            norm = _normalize(player["name"]).split()
            if norm and norm[0][:1] == target_initial:
                return player
    return None


def enrich_player_notes(
    notes: list[dict[str, Any]], match_context: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return notes with photo/rating/position attached where resolvable."""
    roster = build_roster(match_context)
    if not roster:
        return notes

    unresolved = 0
    enriched = []
    for note in notes:
        note = dict(note)
        pid = note.get("player_id")
        player = roster.get(pid) if isinstance(pid, int) else None

        if player is None:  # id absent or hallucinated — fall back to the name
            player = _match_by_name(note.get("player_name", ""), roster)

        if player is None:
            unresolved += 1
            note.setdefault("photo", None)
            enriched.append(note)
            continue

        note["player_id"] = player["id"]
        note["photo"] = player["photo"]
        note["position"] = player["position"]
        note["number"] = player["number"]
        note["minutes"] = player["minutes"]
        # The provider's own rating is authoritative over anything parsed
        # out of the model's prose.
        if player["rating"] is not None:
            note["rating"] = float(player["rating"])
        enriched.append(note)

    if unresolved:
        logger.info("%d/%d player notes could not be matched to a roster entry",
                    unresolved, len(notes))
    return enriched
