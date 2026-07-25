"""
leaderboard.py
--------------
Season-wide player rankings.

Built from the provider's full paginated player list rather than its
`topscorers`/`topassists` shortcuts. Those cost two calls instead of ~34, but
they return only attackers and midfielders — no goalkeepers and almost no
defenders — so a "best average rating" board built on them would silently
exclude half the pitch. On the Pro tier the extra calls are cheap and the
result is cached, so the honest version wins.

Note on clean sheets: the provider exposes season totals for goals conceded
and saves, but not clean sheets, and they can't be derived from totals without
per-match data. The keeper board therefore ranks on saves and goals conceded.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_PAGES = 40  # safety stop; a big-5 league season is ~34 pages

# Ranking a rate metric (rating, accuracy) without a floor lets a player with
# one substitute appearance top the table.
MIN_APPEARANCES = {"rating": 10, "goals": 1, "assists": 1, "key_passes": 1,
                   "defending": 5, "saves": 5, "conceded": 15}


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_league_players(league_id: int, season: int) -> list[dict[str, Any]]:
    """Every player with a stat line for one league-season, flattened."""
    from services import api_service

    players: list[dict[str, Any]] = []
    page = 1
    total_pages = 1

    while page <= min(total_pages, MAX_PAGES):
        payload = api_service._get(
            "players", {"league": league_id, "season": season, "page": page}
        )
        total_pages = (payload.get("paging") or {}).get("total", 1)
        for entry in payload.get("response") or []:
            flat = _flatten(entry, league_id, season)
            if flat:
                players.append(flat)
        page += 1

    logger.info(
        "Leaderboard pool: %d players from league %s season %s (%d pages)",
        len(players), league_id, season, min(total_pages, MAX_PAGES),
    )
    return players


def _flatten(entry: dict[str, Any], league_id: int, season: int) -> Optional[dict[str, Any]]:
    """Reduce a player entry to the one competition block we're ranking on."""
    info = entry.get("player") or {}
    pid = info.get("id")
    if pid is None:
        return None

    # A player can carry blocks for several competitions; keep this league's.
    block = None
    for candidate in entry.get("statistics") or []:
        if (candidate.get("league") or {}).get("id") == league_id:
            block = candidate
            break
    if block is None:
        return None

    games = block.get("games") or {}
    apps = _num(games.get("appearences")) or 0
    if apps <= 0:
        return None  # named in squads but never played

    goals = block.get("goals") or {}
    passes = block.get("passes") or {}
    tackles = block.get("tackles") or {}
    duels = block.get("duels") or {}
    team = block.get("team") or {}

    tackle_n = _num(tackles.get("total")) or 0
    intercept_n = _num(tackles.get("interceptions")) or 0
    block_n = _num(tackles.get("blocks")) or 0

    return {
        "id": pid,
        "name": info.get("name"),
        "photo": f"/api/player-photo/{pid}",
        "nationality": info.get("nationality"),
        "age": info.get("age"),
        "team": team.get("name"),
        "team_id": team.get("id"),
        "position": games.get("position"),
        "season": season,
        "appearances": int(apps),
        "minutes": int(_num(games.get("minutes")) or 0),
        "rating": round(_num(games.get("rating")), 2) if _num(games.get("rating")) else None,
        "goals": int(_num(goals.get("total")) or 0),
        "assists": int(_num(goals.get("assists")) or 0),
        "saves": int(_num(goals.get("saves")) or 0),
        "conceded": int(_num(goals.get("conceded")) or 0),
        "key_passes": int(_num(passes.get("key")) or 0),
        "passes": int(_num(passes.get("total")) or 0),
        "defending": int(tackle_n + intercept_n + block_n),
        "duels_won": int(_num(duels.get("won")) or 0),
        # Rate, not total: a raw "fewest conceded" board just ranks whoever
        # played least, so a backup keeper with five games tops a first choice
        # who played the full season.
        "conceded_per90": _per90(_num(goals.get("conceded")), _num(games.get("minutes"))),
    }


def _per90(total: Optional[float], minutes: Optional[float]) -> Optional[float]:
    if total is None or not minutes:
        return None
    return round(total / (minutes / 90.0), 2)


# metric key -> (label, sort field, descending, position filter)
METRICS: dict[str, dict[str, Any]] = {
    "rating":     {"label": "Average rating", "field": "rating",     "desc": True,  "positions": None},
    "goals":      {"label": "Goals",          "field": "goals",      "desc": True,  "positions": None},
    "assists":    {"label": "Assists",        "field": "assists",    "desc": True,  "positions": None},
    "key_passes": {"label": "Key passes",     "field": "key_passes", "desc": True,  "positions": None},
    "defending":  {"label": "Defensive actions", "field": "defending", "desc": True,
                   "positions": ("Defender", "Midfielder")},
    "saves":      {"label": "Saves",          "field": "saves",      "desc": True,
                   "positions": ("Goalkeeper",)},
    "conceded":   {"label": "Conceded per 90", "field": "conceded_per90", "desc": False,
                   "positions": ("Goalkeeper",)},
}


def rank(players: list[dict[str, Any]], metric: str, limit: int = 25) -> list[dict[str, Any]]:
    """Sort the pool by one metric, applying its appearance floor and filter."""
    spec = METRICS.get(metric) or METRICS["rating"]
    field = spec["field"]
    floor = MIN_APPEARANCES.get(metric, 1)

    pool = [
        p for p in players
        if p.get(field) is not None
        and p["appearances"] >= floor
        and (spec["positions"] is None or p.get("position") in spec["positions"])
    ]
    # Appearances break ties so a fuller season outranks an equal short one.
    pool.sort(key=lambda p: (p[field], p["appearances"]), reverse=spec["desc"])
    if not spec["desc"]:
        pool.sort(key=lambda p: (p[field], -p["appearances"]))

    return [{**p, "rank": i + 1, "metric_value": p[field]} for i, p in enumerate(pool[:limit])]


def metric_options() -> list[dict[str, str]]:
    return [{"key": k, "label": v["label"]} for k, v in METRICS.items()]
