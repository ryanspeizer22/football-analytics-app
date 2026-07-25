"""
player_profile.py
-----------------
Aggregates a player's season into the shape the card drawer renders.

The provider splits a season across one block per competition (league, cups,
internationals, plus zero-appearance friendly entries), with ratings as
strings and nulls scattered through every section. This module flattens that
into season totals, an appearance-weighted average rating, and a
position-aware set of highlight metrics.

No model call is involved: the drawer opens on a click, so it has to be fast
and free.
"""

from typing import Any, Optional

# Sections we sum across competitions: (section, field, output key)
_SUM_FIELDS: list[tuple[str, str, str]] = [
    ("games", "appearences", "appearances"),  # provider's spelling
    ("games", "lineups", "lineups"),
    ("games", "minutes", "minutes"),
    ("goals", "total", "goals"),
    ("goals", "assists", "assists"),
    ("goals", "saves", "saves"),
    ("goals", "conceded", "conceded"),
    ("shots", "total", "shots"),
    ("shots", "on", "shots_on"),
    ("passes", "total", "passes"),
    ("passes", "key", "key_passes"),
    ("tackles", "total", "tackles"),
    ("tackles", "interceptions", "interceptions"),
    ("tackles", "blocks", "blocks"),
    ("duels", "total", "duels"),
    ("duels", "won", "duels_won"),
    ("dribbles", "attempts", "dribbles"),
    ("dribbles", "success", "dribbles_won"),
    ("cards", "yellow", "yellow_cards"),
    ("cards", "red", "red_cards"),
]


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Optional[float]) -> Optional[int]:
    return int(value) if value is not None else None


def build_profile(raw: dict[str, Any], season: int) -> dict[str, Any]:
    """Flatten one `/players` response into a season profile."""
    info = raw.get("player") or {}
    blocks = raw.get("statistics") or []

    # Competitions the player actually featured in. Zero-appearance blocks
    # (unused friendly squads) would otherwise drag the rating average down
    # and pad the competition list with rows that say nothing.
    played = [
        b for b in blocks
        if (_num((b.get("games") or {}).get("appearences")) or 0) > 0
    ]

    totals: dict[str, Optional[int]] = {}
    for section, field, key in _SUM_FIELDS:
        values = [
            _num((b.get(section) or {}).get(field)) for b in played
        ]
        present = [v for v in values if v is not None]
        totals[key] = _int(sum(present)) if present else None

    # Rating must be weighted by appearances — a 9.0 from one cup tie should
    # not offset 38 league games at 7.2.
    weighted_sum = 0.0
    weight = 0.0
    for block in played:
        games = block.get("games") or {}
        rating = _num(games.get("rating"))
        apps = _num(games.get("appearences")) or 0
        if rating is not None and apps > 0:
            weighted_sum += rating * apps
            weight += apps
    average_rating = round(weighted_sum / weight, 2) if weight else None

    position = next(
        ((b.get("games") or {}).get("position") for b in played
         if (b.get("games") or {}).get("position")),
        None,
    )

    competitions = [
        {
            "name": (b.get("league") or {}).get("name"),
            "country": (b.get("league") or {}).get("country"),
            "logo": (b.get("league") or {}).get("logo"),
            "appearances": _int(_num((b.get("games") or {}).get("appearences"))),
            "minutes": _int(_num((b.get("games") or {}).get("minutes"))),
            "rating": (lambda r: round(r, 2) if r is not None else None)(
                _num((b.get("games") or {}).get("rating"))
            ),
            "goals": _int(_num((b.get("goals") or {}).get("total"))),
            "assists": _int(_num((b.get("goals") or {}).get("assists"))),
        }
        for b in played
    ]
    competitions.sort(key=lambda c: c["appearances"] or 0, reverse=True)

    team = next(((b.get("team") or {}) for b in played), {})

    return {
        "player": {
            "id": info.get("id"),
            "name": info.get("name"),
            "firstname": info.get("firstname"),
            "lastname": info.get("lastname"),
            "age": info.get("age"),
            "nationality": info.get("nationality"),
            "height": info.get("height"),
            "weight": info.get("weight"),
            "photo": f"/api/player-photo/{info.get('id')}" if info.get("id") else None,
            "position": position,
            "team": team.get("name"),
            "team_logo": team.get("logo"),
        },
        "season": season,
        "average_rating": average_rating,
        "totals": totals,
        "highlights": _highlights(totals, position),
        "competitions": competitions,
    }


def _pct(part: Optional[int], whole: Optional[int]) -> Optional[str]:
    if not part or not whole:
        return None
    return f"{round(part / whole * 100)}%"


def _highlights(totals: dict[str, Optional[int]], position: Optional[str]) -> list[dict[str, str]]:
    """Position-aware headline metrics — a keeper's season isn't a striker's."""
    out: list[dict[str, str]] = []

    def add(label: str, value: Any) -> None:
        if value not in (None, "", 0) or (value == 0 and label in ("Goals", "Assists")):
            out.append({"label": label, "value": str(value)})

    if position == "Goalkeeper":
        add("Saves", totals.get("saves"))
        add("Conceded", totals.get("conceded"))
        add("Passes", totals.get("passes"))
        add("Duels won", _pct(totals.get("duels_won"), totals.get("duels")))
    elif position == "Defender":
        add("Tackles", totals.get("tackles"))
        add("Interceptions", totals.get("interceptions"))
        add("Duels won", _pct(totals.get("duels_won"), totals.get("duels")))
        add("Blocks", totals.get("blocks"))
    elif position == "Midfielder":
        add("Key passes", totals.get("key_passes"))
        add("Passes", totals.get("passes"))
        add("Tackles", totals.get("tackles"))
        add("Duels won", _pct(totals.get("duels_won"), totals.get("duels")))
    else:  # Attacker, or unknown
        add("Shots", totals.get("shots"))
        add("On target", totals.get("shots_on"))
        add("Key passes", totals.get("key_passes"))
        add("Dribbles won", _pct(totals.get("dribbles_won"), totals.get("dribbles")))

    return [h for h in out if h["value"] not in ("None", "")]
