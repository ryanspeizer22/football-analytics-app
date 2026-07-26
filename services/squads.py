"""
squads.py
---------
Resolves which club a player is at *now*, as distinct from the club they were
playing for in whatever match you happen to be looking at.

Those are different questions and the app was only ever answering the second.
A player's row in a match or a leaderboard carries the club they turned out for
that season, which is correct for that context and wrong as a statement about
today. After a summer window it is badly wrong.

The answer comes from the provider's own transfer ledger rather than from
anything hardcoded here. That matters: a club move is a fact with a date, and
the moment the provider records one this reflects it without a code change.
Equally, if the provider has not recorded a move, this will not invent one —
checked against Morgan Rogers, whose ledger ends at Middlesbrough to Aston
Villa in February 2024 with no Chelsea move on file.

Transfers are cached hard. A completed move never changes, and the lookup costs
a request against a metered quota.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "transfers-"


def _ledger(player_id: int) -> list[dict[str, Any]]:
    """Every recorded move for a player, newest first."""
    from services import api_service, summary_cache

    key = f"{_CACHE_PREFIX}{player_id}"
    cached = summary_cache.get(key)
    if cached is not None:
        return cached

    try:
        payload = api_service._get("transfers", {"player": player_id})
    except Exception as exc:
        logger.info("Transfer lookup failed for player %s: %s", player_id, exc)
        return []

    moves = []
    for block in payload.get("response") or []:
        for move in block.get("transfers") or []:
            teams = move.get("teams") or {}
            joined = (teams.get("in") or {})
            if not joined.get("id"):
                continue
            moves.append({
                "date": move.get("date") or "",
                "type": move.get("type"),
                "from": (teams.get("out") or {}).get("name"),
                "to": joined.get("name"),
                "to_id": joined.get("id"),
            })
    moves.sort(key=lambda m: m["date"], reverse=True)
    summary_cache.set(key, moves)
    return moves


def current_club(player_id: int) -> Optional[dict[str, Any]]:
    """The club a player most recently joined, per the provider's records.

    Returns None when nothing is on file — which is a real answer, not a
    failure. Callers should fall back to the club shown on the player's season
    row and describe it as such rather than presenting it as current.
    """
    moves = _ledger(player_id)
    if not moves:
        return None
    latest = moves[0]
    return {
        "id": latest["to_id"],
        "name": latest["to"],
        "since": latest["date"],
        "type": latest["type"],
        "from": latest["from"],
    }


def history(player_id: int, limit: int = 8) -> list[dict[str, Any]]:
    """The player's career moves, newest first."""
    return _ledger(player_id)[:limit]
