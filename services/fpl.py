"""
fpl.py
------
Joins Fantasy Premier League price/ownership to our per-90 performance data.

Why a join is unavoidable: a "differential" is by definition low ownership plus
high output. API-Football has the output and no ownership; FPL has ownership,
price and points but only shallow per-match stats. Neither source alone can
answer the question.

The matching problem is real. API-Football abbreviates forenames ("V. van Dijk",
"C. Gakpo"); FPL splits names into first/second and shows a display name
("web_name": "Raya"). Fourteen FPL surnames are shared by two or three players.
So matching is scored, not keyed: surname agreement is required, and forename
initial, club and position break ties. Anything that stays ambiguous is dropped
rather than guessed — a wrongly joined player would show someone else's price
and ownership, which is worse than an absent row.

Seasons deliberately differ. Pre-season FPL carries the *upcoming* season's
prices and ownership alongside the *previous* season's points and minutes,
which is exactly the comparison a pre-season differential hunt wants: what a
player did last year against what he costs and how many people own him now.
Because squads change between those seasons, club is never a hard requirement.
"""

import logging
import re
import unicodedata
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
REQUEST_TIMEOUT = 20

# FPL's short club names against API-Football's, keyed on the *normalised*
# form. `_norm` strips punctuation, so "Nott'm Forest" arrives as
# "nott m forest" — keying on the raw string means the alias never fires and
# Forest players quietly lose their club-match bonus.
TEAM_ALIASES = {
    "man utd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham",
    "nott m forest": "nottingham forest",
    "nottm forest": "nottingham forest",
    "ipswich town": "ipswich",
    "coventry city": "coventry",
}

# Particles that belong to the surname, not the forename.
_PARTICLES = {"van", "de", "der", "den", "da", "dos", "del", "di", "la", "le", "el", "al", "bin"}


def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", " ", ascii_only.lower()).strip()


def _norm_team(name: str) -> str:
    n = _norm(name)
    return TEAM_ALIASES.get(n, n)


def _surname(full: str) -> str:
    """Trailing surname including any particles ('van dijk', 'dos santos')."""
    tokens = _norm(full).split()
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    # Walk back over particles so 'Virgil van Dijk' -> 'van dijk'.
    start = len(tokens) - 1
    while start > 0 and tokens[start - 1] in _PARTICLES:
        start -= 1
    return " ".join(tokens[start:])


def _initial(full: str) -> str:
    tokens = _norm(full).split()
    return tokens[0][0] if tokens and tokens[0] not in _PARTICLES else ""


def fetch_bootstrap() -> dict[str, Any]:
    resp = requests.get(FPL_BOOTSTRAP, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def build_fpl_index(bootstrap: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten FPL players into comparable records."""
    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    positions = {p["id"]: p["singular_name"] for p in bootstrap.get("element_types", [])}

    out = []
    for e in bootstrap.get("elements", []):
        full = f"{e.get('first_name','')} {e.get('second_name','')}".strip()
        out.append({
            "fpl_id": e["id"],
            "web_name": e.get("web_name"),
            "full_name": full,
            "surname": _surname(e.get("second_name") or full),
            "initial": _initial(e.get("first_name") or full),
            "tokens": _name_tokens(full) | _name_tokens(e.get("web_name") or ""),
            "team": teams.get(e["team"], ""),
            "team_norm": _norm_team(teams.get(e["team"], "")),
            "position": positions.get(e["element_type"], ""),
            "price": e.get("now_cost", 0) / 10.0,
            "ownership": float(e.get("selected_by_percent") or 0),
            "total_points": e.get("total_points") or 0,
            "points_per_game": float(e.get("points_per_game") or 0),
            "minutes": e.get("minutes") or 0,
            "form": float(e.get("form") or 0),
            "status": e.get("status"),
        })
    return out


# FPL calls them Forwards; API-Football calls them Attackers.
_POSITION_EQUIV = {"forward": "attacker", "attacker": "attacker",
                   "goalkeeper": "goalkeeper", "defender": "defender",
                   "midfielder": "midfielder"}


def _same_position(a: Optional[str], b: Optional[str]) -> bool:
    return _POSITION_EQUIV.get(_norm(a or "")) == _POSITION_EQUIV.get(_norm(b or ""))


def _name_tokens(full: str) -> set[str]:
    """Every meaningful token in a name, for overlap scoring.

    Iberian and Brazilian names carry the family name in the middle as often
    as at the end ('Diego Gómez Amarilla'), so a trailing-token surname rule
    alone misses them. Comparing token sets catches those without loosening
    the surname rule for everyone else.
    """
    return {t for t in _norm(full).split() if len(t) > 2 and t not in _PARTICLES}


def _surname_agrees(af_surname: str, af_tokens: set[str], candidate: dict[str, Any]) -> int:
    """0 = no agreement, 2 = partial, 3 = exact. Drives the base score."""
    cand_surname = candidate["surname"]
    if not cand_surname:
        return 0        # an FPL entry with no second name can't anchor a match
    if cand_surname == af_surname:
        return 3
    # Compound vs short form: 'dos santos' vs 'santos'. Require the shared part
    # to be a whole token, or 'endo' would match 'kalimuendo'.
    cand_tokens = set(cand_surname.split())
    if af_surname in cand_tokens or cand_surname in af_surname.split():
        return 3
    # Family name sitting mid-name on either side.
    if af_surname in candidate["tokens"] or (cand_tokens & af_tokens):
        return 2
    return 0


def match_player(player: dict[str, Any], fpl_index: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Best-scoring FPL record for one API-Football player, or None.

    Surname agreement is mandatory; forename initial, club, position and
    whole-name token overlap separate candidates that already share one. A tie
    between two equally good candidates returns None — silently picking one
    would attach another player's price and ownership to a real name, which is
    worse than leaving the row out.
    """
    name = player.get("name") or ""
    surname = _surname(name)
    if not surname:
        return None

    af_tokens = _name_tokens(name)
    initial = _initial(name)
    team_norm = _norm_team(player.get("team") or "")
    position = player.get("position")

    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in fpl_index:
        agreement = _surname_agrees(surname, af_tokens, candidate)
        if not agreement:
            continue

        score = float(agreement)
        if initial and candidate["initial"] == initial:
            score += 3
        elif initial and candidate["initial"]:
            score -= 2            # both known and different — probably not him
        if team_norm and candidate["team_norm"] == team_norm:
            score += 3
        if _same_position(position, candidate["position"]):
            score += 1
        # Shared forenames/middle names break ties the initial can't
        # ('Andrey ... dos Santos' vs 'Alysson ... dos Santos').
        overlap = len(af_tokens & candidate["tokens"])
        score += min(overlap, 3) * 1.5
        if _norm(candidate["web_name"] or "") == _norm(name):
            score += 2
        scored.append((score, candidate))

    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    best_score, best = scored[0]
    if best_score < 4:
        return None                      # surname alone isn't enough
    if len(scored) > 1 and abs(scored[1][0] - best_score) < 0.5:
        return None                      # genuinely ambiguous
    return best


def join(players: list[dict[str, Any]], fpl_index: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach FPL data to a performance pool. Returns rows plus match stats."""
    rows, unmatched = [], []
    for player in players:
        fpl = match_player(player, fpl_index)
        if fpl is None:
            unmatched.append(player.get("name"))
            continue
        minutes = player.get("minutes") or 0
        rows.append({
            **player,
            "fpl_id": fpl["fpl_id"],
            "fpl_name": fpl["web_name"],
            "fpl_team": fpl["team"],
            "price": fpl["price"],
            "ownership": fpl["ownership"],
            "total_points": fpl["total_points"],
            "points_per_game": fpl["points_per_game"],
            "fpl_position": fpl["position"],
            "status": fpl["status"],
            "points_per_million": round(fpl["total_points"] / fpl["price"], 1) if fpl["price"] else 0,
            "per90": {
                "goals": _p90(player.get("goals"), minutes),
                "assists": _p90(player.get("assists"), minutes),
                "key_passes": _p90(player.get("key_passes"), minutes),
                "defending": _p90(player.get("defending"), minutes),
            },
        })
    return {
        "rows": rows,
        "matched": len(rows),
        "total": len(players),
        "match_rate": round(len(rows) / len(players) * 100, 1) if players else 0,
        "unmatched_sample": unmatched[:15],
    }


def _p90(total: Optional[float], minutes: Optional[float]) -> Optional[float]:
    if total is None or not minutes:
        return None
    return round(total / (minutes / 90.0), 2)


def differentials(rows: list[dict[str, Any]], max_ownership: float = 10.0,
                  min_minutes: int = 900, limit: int = 20) -> list[dict[str, Any]]:
    """Under-owned players whose underlying numbers outrun their price.

    Scored on value (points per million) and per-90 attacking involvement,
    then divided by ownership so genuinely under-picked players rise. The
    minutes floor keeps out cameo players whose per-90 rates are noise.
    """
    pool = [
        r for r in rows
        if r["ownership"] <= max_ownership
        and (r.get("minutes") or 0) >= min_minutes
        and r["price"] > 0
    ]
    for r in pool:
        involvement = (r["per90"].get("goals") or 0) + (r["per90"].get("assists") or 0) \
                      + 0.3 * (r["per90"].get("key_passes") or 0)
        # +1 keeps a 0.1%-owned player from producing a meaningless spike.
        r["differential_score"] = round(
            (r["points_per_million"] * 0.6 + involvement * 8) / (r["ownership"] + 1), 2
        )
    pool.sort(key=lambda r: r["differential_score"], reverse=True)
    return pool[:limit]
