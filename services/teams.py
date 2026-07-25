"""
teams.py
--------
Local team registry: API-Football team IDs, display names, brand colors,
and search aliases. Autocomplete matches against this list locally, so
typing in the search bar never burns API quota.

Coverage: 2023-24 Premier League plus the European giants and the two
World Cup finalists (everything the trending grid and free-tier data reach).
"""

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


def readable_color(hex_color: str, fallback: str) -> str:
    """Ensure an accent color is visible on the dark background.

    Near-black kits (Newcastle, Juventus) would otherwise render as an
    invisible swatch — prefer the club's secondary color, else lighten.
    """
    if _luminance(hex_color) >= _MIN_LUMINANCE:
        return hex_color
    if _luminance(fallback) >= _MIN_LUMINANCE:
        return fallback
    return _lighten(hex_color, 0.55)


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
    return [{k: t[k] for k in ("id", "name", "short", "color", "color2")} for t in hits[:limit]]


_VS_SEPARATORS = [" vs ", " v ", " versus ", " - ", " x "]


def parse_query(query: str) -> dict[str, Any]:
    """Interpret a search query, understanding 'TeamA vs TeamB' matchups.

    Returns {"teams": [...matches...], "matchup": {"team1": .., "team2": ..} | None}
    """
    q = query.lower().strip()
    for sep in _VS_SEPARATORS:
        if sep in q:
            left, right = q.split(sep, 1)
            team1 = search_teams(left, limit=1)
            team2 = search_teams(right, limit=1)
            if team1 and team2 and team1[0]["id"] != team2[0]["id"]:
                return {"teams": [], "matchup": {"team1": team1[0], "team2": team2[0]}}
            break  # separator present but couldn't resolve both sides
    return {"teams": search_teams(q), "matchup": None}
