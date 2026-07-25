"""
slugs.py
--------
Human-readable URLs that still resolve deterministically.

The id is kept as the final segment of every slug (`…-868201`) rather than
looked up from a name index. Names are ambiguous and change — two "Bruno"s in
one match, a club renaming, a typo'd share — and a lookup table would need
invalidating. Trailing-id slugs are what large content sites use for exactly
this reason: the readable part is decoration, the id is the source of truth,
so a link stays valid even if the prose part rots.
"""

import re
import unicodedata
from typing import Optional

_SEPARATOR = "-vs-"


def slugify(text: str) -> str:
    """Lowercase ASCII slug: 'Bruno Guimarães' -> 'bruno-guimaraes'."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return cleaned or "item"


def match_slug(home: str, away: str, fixture_id: int, date: Optional[str] = None) -> str:
    """e.g. liverpool-vs-man-united-2023-03-05-868201"""
    parts = [slugify(home), "vs", slugify(away)]
    if date:
        parts.append(date[:10])
    parts.append(str(fixture_id))
    return "-".join(parts)


def compare_slug(name_a: str, id_a: int, name_b: str, id_b: int) -> str:
    """e.g. bruno-fernandes-1234-vs-jeremy-doku-5678"""
    return f"{slugify(name_a)}-{id_a}{_SEPARATOR}{slugify(name_b)}-{id_b}"


def player_slug(name: str, player_id: int) -> str:
    return f"{slugify(name)}-{player_id}"


def leaderboard_slug(competition: str, season: int) -> str:
    return f"{slugify(competition)}-{season}"


def trailing_id(slug: str) -> Optional[int]:
    """Pull the id off the end of a slug; None if it isn't there."""
    if not slug:
        return None
    match = re.search(r"(\d+)$", slug.strip("/"))
    return int(match.group(1)) if match else None


def parse_compare(slug: str) -> Optional[tuple[int, int]]:
    """Split a compare slug into its two player ids."""
    if not slug or _SEPARATOR not in slug:
        return None
    left, right = slug.split(_SEPARATOR, 1)
    a, b = trailing_id(left), trailing_id(right)
    return (a, b) if a and b else None


def parse_leaderboard(slug: str) -> Optional[tuple[int, int]]:
    """Split 'premier-league-2025' into (competition_id, season).

    The season is the trailing number, so the competition has to be resolved
    by name — leaderboard slugs are generated from a known competition list,
    so an unresolvable name means a hand-edited URL, not a stale link.
    """
    from services import competitions

    season = trailing_id(slug)
    if season is None:
        return None
    name_part = slug[: slug.rfind(str(season))].strip("-")
    for comp in competitions.COMPETITIONS:
        if slugify(comp["name"]) == name_part:
            return comp["id"], season
    return None
