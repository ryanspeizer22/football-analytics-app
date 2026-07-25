"""
teams.py
--------
Local team registry: API-Football team IDs, display names, brand colors,
and search aliases. Autocomplete matches against this list locally, so
typing in the search bar never burns API quota.

Coverage: 2023-24 Premier League plus the European giants and the two
World Cup finalists (everything the trending grid and free-tier data reach).
"""

import re
from typing import Any, Optional

# Compact names for the card UI — full names overflow narrow score cards.
DISPLAY_OVERRIDES: dict[int, str] = {
    33: "Man United",
    50: "Man City",
    34: "Newcastle",
    65: "Nott'm Forest",
    62: "Sheffield Utd",
    47: "Spurs",
    85: "PSG",
    530: "Atletico",
    165: "Dortmund",
    157: "Bayern",
    51: "Brighton",
    1359: "Luton",
}

# id, name, short code, primary color, secondary color, aliases
_RAW: list[tuple[int, str, str, str, str, list[str]]] = [
    # Premier League 2023-24
    (42,   "Arsenal",            "ARS", "#EF0107", "#FFFFFF", ["gunners"]),
    (66,   "Aston Villa",        "AVL", "#67002F", "#95BFE5", ["villa"]),
    (35,   "Bournemouth",        "BOU", "#DA020E", "#000000", ["cherries", "afc bournemouth"]),
    (55,   "Brentford",          "BRE", "#D20000", "#FFFFFF", ["bees"]),
    (51,   "Brighton",           "BHA", "#0057B8", "#FFFFFF", ["seagulls", "brighton and hove", "brighton & hove albion"]),
    (44,   "Burnley",            "BUR", "#6C1D45", "#99D6EA", ["clarets"]),
    (49,   "Chelsea",            "CHE", "#034694", "#FFFFFF", ["blues", "cfc"]),
    (52,   "Crystal Palace",     "CRY", "#1B458F", "#C4122E", ["palace", "eagles"]),
    (45,   "Everton",            "EVE", "#003399", "#FFFFFF", ["toffees"]),
    (36,   "Fulham",             "FUL", "#000000", "#CC0000", ["cottagers"]),
    (40,   "Liverpool",          "LIV", "#C8102E", "#00B2A9", ["reds", "lfc", "pool"]),
    (1359, "Luton",              "LUT", "#F78F1E", "#002D62", ["luton town", "hatters"]),
    (50,   "Manchester City",    "MCI", "#6CABDD", "#1C2C5B", ["man city", "city", "mancity", "mcfc"]),
    (33,   "Manchester United",  "MUN", "#DA020E", "#FBE122", ["man united", "man utd", "united", "mufc", "man u"]),
    (34,   "Newcastle",          "NEW", "#241F20", "#F1F1F1", ["newcastle united", "magpies", "toon"]),
    (65,   "Nottingham Forest",  "NFO", "#DD0000", "#FFFFFF", ["forest", "nottm forest"]),
    (62,   "Sheffield United",   "SHU", "#EE2737", "#000000", ["sheffield utd", "blades"]),
    (47,   "Tottenham",          "TOT", "#132257", "#FFFFFF", ["spurs", "tottenham hotspur"]),
    (48,   "West Ham",           "WHU", "#7A263A", "#1BB1E7", ["hammers", "west ham united"]),
    (39,   "Wolves",             "WOL", "#FDB913", "#231F20", ["wolverhampton", "wolverhampton wanderers"]),
    # European giants
    (541,  "Real Madrid",        "RMA", "#FEBE10", "#00529F", ["madrid", "real"]),
    (529,  "Barcelona",          "BAR", "#A50044", "#004D98", ["barca", "fc barcelona"]),
    (157,  "Bayern Munich",      "BAY", "#DC052D", "#0066B2", ["bayern", "fc bayern"]),
    (85,   "Paris Saint-Germain","PSG", "#004170", "#DA291C", ["psg", "paris"]),
    (496,  "Juventus",           "JUV", "#000000", "#FFFFFF", ["juve"]),
    (505,  "Inter",              "INT", "#0068A8", "#000000", ["inter milan", "internazionale"]),
    (489,  "AC Milan",           "MIL", "#FB090B", "#000000", ["milan"]),
    (530,  "Atletico Madrid",    "ATM", "#CB3524", "#262E62", ["atletico", "atleti"]),
    (165,  "Borussia Dortmund",  "BVB", "#FDE100", "#000000", ["dortmund", "bvb"]),
    (492,  "Napoli",             "NAP", "#12A0D7", "#003C82", []),
    # World Cup 2022 finalists
    (26,   "Argentina",          "ARG", "#75AADB", "#FFFFFF", []),
    (2,    "France",             "FRA", "#21304D", "#EF4135", ["les bleus"]),
]

TEAMS: list[dict[str, Any]] = [
    {
        "id": tid,
        "name": name,
        "short": short,
        "color": primary,
        "color2": secondary,
        "aliases": aliases,
    }
    for tid, name, short, primary, secondary, aliases in _RAW
]

_BY_ID = {t["id"]: t for t in TEAMS}


def get_team(team_id: int) -> Optional[dict[str, Any]]:
    return _BY_ID.get(team_id)


def team_colors(team_id: int) -> tuple[str, str]:
    """Primary/secondary colors for a team, with a neutral fallback."""
    team = _BY_ID.get(team_id)
    return (team["color"], team["color2"]) if team else ("#4a5568", "#a0aec0")


def display_name(team_id: int, api_name: str) -> str:
    """Short, card-friendly name. Falls back to trimming the API's name."""
    if team_id in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[team_id]
    team = _BY_ID.get(team_id)
    if team:
        return team["name"]
    # Unknown team: strip common club suffixes/prefixes so it fits a card.
    name = api_name
    for noise in (" Football Club", " FC", "FC ", " CF", " AFC", "AFC ", " United"):
        if name.endswith(noise) and len(name) > 14:
            name = name[: -len(noise)]
    return name[:16]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _luminance(hex_color: str) -> float:
    """Perceived brightness, 0 (black) to 255 (white)."""
    r, g, b = _hex_to_rgb(hex_color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _lighten(hex_color: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


# Below this brightness a color disappears against the dark UI background.
_MIN_LUMINANCE = 60.0
# Below this channel spread a color reads as neutral gray/white rather than
# as a team's identity — flat and washed out in a chart fill.
_MIN_CHROMA = 26


def _chroma(hex_color: str) -> int:
    r, g, b = _hex_to_rgb(hex_color)
    return max(r, g, b) - min(r, g, b)


def _tint_neutral(hex_color: str) -> str:
    """Give a near-neutral color a slight cool cast so it reads as chosen.

    Monochrome kits (Newcastle, Juventus, Fulham) resolve to white here; a
    pure white area fill looks like missing data, whereas a faint ice-blue
    still reads as "the white team" while holding its own as a color.
    """
    if _chroma(hex_color) >= _MIN_CHROMA:
        return hex_color
    r, g, b = _hex_to_rgb(hex_color)
    r = max(0, r - 26)
    g = max(0, g - 10)
    return f"#{r:02X}{g:02X}{b:02X}"


def readable_color(hex_color: str, fallback: str) -> str:
    """Ensure an accent color is visible, and reads as a color, on dark.

    Near-black kits (Newcastle, Juventus) would otherwise render as an
    invisible swatch — prefer the club's secondary color, else lighten —
    and pure monochrome results get a faint tint so they aren't mistaken
    for an empty chart region.
    """
    if _luminance(hex_color) >= _MIN_LUMINANCE:
        return _tint_neutral(hex_color)
    if _luminance(fallback) >= _MIN_LUMINANCE:
        return _tint_neutral(fallback)
    return _tint_neutral(_lighten(hex_color, 0.55))


def distinct_colors(home_id: int, away_id: int) -> tuple[str, str]:
    """Pick per-match accent colors that are visible and tell the sides apart.

    Two corrections are applied: near-black kits are lifted so they render at
    all, and same-colored derbies (Liverpool vs Man United, both red) fall back
    to the away team's secondary color so the coding stays meaningful.
    """
    home_c1, home_c2 = team_colors(home_id)
    away_c1, away_c2 = team_colors(away_id)

    home = readable_color(home_c1, home_c2)
    away = readable_color(away_c1, away_c2)

    hr, hg, hb = _hex_to_rgb(home)
    ar, ag, ab = _hex_to_rgb(away)
    # Manhattan distance in RGB is crude but sufficient to catch near-clashes.
    if abs(hr - ar) + abs(hg - ag) + abs(hb - ab) < 150:
        alt = readable_color(away_c2, away_c1)
        if alt != away:
            return home, alt
        return home, _lighten(away, 0.45)
    return home, away


def _matches(team: dict[str, Any], q: str) -> bool:
    q = q.lower().strip()
    if not q:
        return False
    candidates = [team["name"].lower(), team["short"].lower(), *team["aliases"]]
    return any(c.startswith(q) or f" {q}" in f" {c}" for c in candidates)


def search_teams(query: str, limit: int = 6) -> list[dict[str, Any]]:
    """Local prefix/word match against names, short codes, and aliases."""
    hits = [t for t in TEAMS if _matches(t, query)]
    return [_public(t) for t in hits[:limit]]


_VS_SEPARATORS = [" vs ", " v ", " versus ", " - ", " x "]


# --- Dynamic resolution for teams outside the curated registry --------------

_GENERATED_PALETTE = [
    "#E4572E", "#17BEBB", "#FFC914", "#76B041", "#9B5DE5",
    "#F15BB5", "#00BBF9", "#F79824", "#4CB944", "#EF6461",
]


def _generated_colors(team_id: int, name: str) -> tuple[str, str]:
    """Deterministic brand colors for a team we have no palette for.

    Keyed on the team id so a club always renders the same color, and drawn
    from a pre-vetted palette so every result stays readable on the dark UI.
    """
    primary = _GENERATED_PALETTE[team_id % len(_GENERATED_PALETTE)]
    secondary = _GENERATED_PALETTE[(team_id // 7 + 3) % len(_GENERATED_PALETTE)]
    if secondary == primary:
        secondary = _GENERATED_PALETTE[(team_id + 5) % len(_GENERATED_PALETTE)]
    return primary, secondary


def ensure_team(team_id: int, name: str, logo: Optional[str] = None) -> dict[str, Any]:
    """Return the registry entry for a team, creating one if it's unknown.

    Universal ingestion surfaces clubs far outside the curated set; without
    this they'd all render in the same neutral gray, so every discovered team
    gets a stable generated palette on first sight.
    """
    existing = _BY_ID.get(team_id)
    if existing:
        return existing
    return register_api_team({"team": {"id": team_id, "name": name, "logo": logo}})


def register_api_team(raw: dict[str, Any]) -> dict[str, Any]:
    """Adopt a team from an API `/teams` result into the in-memory registry.

    Lets a club we've never seen behave exactly like a curated one for the
    rest of the process's life — colors, display name, and future lookups.
    """
    team_info = raw.get("team", raw)
    tid = team_info.get("id")
    name = team_info.get("name") or ""
    if tid is None or not name:
        raise ValueError("API team payload missing id/name")

    existing = _BY_ID.get(tid)
    if existing:
        return existing

    color, color2 = _generated_colors(tid, name)
    entry = {
        "id": tid,
        "name": name,
        "short": (team_info.get("code") or name[:3]).upper(),
        "color": color,
        "color2": color2,
        "aliases": [],
        "logo": team_info.get("logo"),
        "generated": True,
    }
    TEAMS.append(entry)
    _BY_ID[tid] = entry
    return entry


def _public(team: dict[str, Any]) -> dict[str, Any]:
    out = {k: team[k] for k in ("id", "name", "short", "color", "color2")}
    out["generated"] = team.get("generated", False)
    return out


# Youth, women's, and reserve sides share their club's name and would other-
# wise crowd out the senior team. Every competition PitchSense tracks is a
# senior men's competition, so these are filtered from discovery results.
_NON_SENIOR = re.compile(
    r"(\bwomen\b|\bfeminin\w*\b|\bladies\b|\s+(w|u\d{2}|ii|iii|b)$)",
    re.IGNORECASE,
)


def _is_senior_side(name: str) -> bool:
    return not _NON_SENIOR.search(name or "")


def _relevance(name: str, query: str) -> int:
    """Rank: exact match, then prefix, then substring. Lower sorts first."""
    n, q = name.lower(), query.lower().strip()
    if n == q:
        return 0
    if n.startswith(q):
        return 1
    return 2


def resolve_teams(query: str, limit: int = 6) -> list[dict[str, Any]]:
    """Find teams for a query: curated registry first, live API as fallback.

    Keeping the local pass first means common searches cost no API quota; the
    remote lookup only runs for clubs outside the curated set.
    """
    local = search_teams(query, limit=limit)
    if local:
        return local

    # Import here to avoid a circular import at module load.
    from services import api_service

    try:
        remote = api_service.search_teams_api(query)
    except api_service.FootballAPIError:
        return []  # discovery is best-effort; never break the search box

    candidates = [
        raw for raw in remote
        if _is_senior_side((raw.get("team") or {}).get("name", ""))
    ]
    candidates.sort(
        key=lambda raw: _relevance((raw["team"]).get("name", ""), query)
    )

    resolved = []
    for raw in candidates[:limit]:
        try:
            resolved.append(_public(register_api_team(raw)))
        except ValueError:
            continue
    return resolved


def parse_query(query: str) -> dict[str, Any]:
    """Interpret a search query, understanding 'TeamA vs TeamB' matchups.

    Returns {"teams": [...matches...], "matchup": {"team1": .., "team2": ..} | None}
    """
    q = query.lower().strip()
    for sep in _VS_SEPARATORS:
        if sep in q:
            left, right = q.split(sep, 1)
            team1 = resolve_teams(left, limit=1)
            team2 = resolve_teams(right, limit=1)
            if team1 and team2 and team1[0]["id"] != team2[0]["id"]:
                return {"teams": [], "matchup": {"team1": team1[0], "team2": team2[0]}}
            break  # separator present but couldn't resolve both sides
    return {"teams": resolve_teams(q), "matchup": None}
