"""
cache_migrate.py
----------------
Brings cached analyses forward to the current shape when they are read.

The schema has moved twice: the original single blob gained analytical fields
(what decided it, which stat misleads, what changed after the bench), and was
then split into an opening and an analysis so the page could paint before the
whole report was written. Entries on disk predate one or both, and a finished
match's analysis is expensive to produce and never changes — throwing them away
would spend real money to recover text already held.

So the rule here is: reshape, never invent. A missing list becomes an empty
list and a missing string becomes an empty string, which the UI already omits
rather than rendering blank. Prose is never fabricated to fill a gap — an
absent headline stays absent, because the alternative is putting words in the
model's mouth and presenting them as its analysis.

Regeneration is reserved for entries with no usable content at all.
"""

import re
import unicodedata
from typing import Any, Optional

# The shape each half is expected to have, with the default for a missing key.
# Defaults are type-correct so the client can index them without guarding.
OPENING_SHAPE: dict[str, Any] = {
    "headline": "",
    "tldr": "",
    "momentum_takeaways": [],
}

ANALYSIS_SHAPE: dict[str, Any] = {
    "decisive_factor": "",
    "misleading_stat": "",
    "tactical_shifts": [],
    "team_breakdowns": [],
}


def _shaped(entry: dict[str, Any], shape: dict[str, Any]) -> dict[str, Any]:
    """Copy the keys this shape defines, defaulting anything absent.

    A present-but-wrong-typed value is replaced too: an early entry that stored
    `momentum_takeaways` as a string would otherwise reach the client and be
    rendered a character at a time.
    """
    out = {}
    for key, default in shape.items():
        value = entry.get(key, default)
        if value is None or not isinstance(value, type(default)):
            value = default
        out[key] = value
    return out


def _has_content(entry: dict[str, Any], shape: dict[str, Any]) -> bool:
    """Whether anything in this half is worth serving."""
    return any(entry.get(key) for key in shape)


def opening_from(entry: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Normalise any cached narrative into the opening half.

    Accepts the current `open-` entry, a pre-split `narr-` blob, or the older
    `match-` three-layer summary — they all carry these keys under the same
    names, so one path handles all three.
    """
    if not entry or not _has_content(entry, OPENING_SHAPE):
        return None
    return _shaped(entry, OPENING_SHAPE)


def analysis_from(entry: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Normalise any cached narrative into the analysis half."""
    if not entry or not _has_content(entry, ANALYSIS_SHAPE):
        return None
    return _shaped(entry, ANALYSIS_SHAPE)


def _fold(name: str) -> str:
    """Case- and accent-insensitive form, for matching names across sources."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", " ", stripped.lower()).strip()


def _surname_initial(folded: str) -> Optional[tuple[str, str]]:
    parts = folded.split()
    return (parts[-1], parts[0][0]) if len(parts) >= 2 else None


def backfill_player_photos(notes: list[dict[str, Any]],
                           context: dict[str, Any]) -> int:
    """Give legacy player notes their portraits back, by name.

    Notes written before the schema carried `player_id` have only a name, so
    the photo proxy has no id to build a URL from and the card falls back to
    initials. The ids are not lost — they are in the match context on disk, so
    the name is resolved against that rather than regenerating the notes.

    Matching is deliberately conservative. An exact fold (case and accents
    removed) covers almost everything: on the 2022 World Cup final it resolves
    34 of 35 outright, the miss being "Rodrigo de Paul" against "Rodrigo De
    Paul". Surname-plus-initial catches the rest. Anything still ambiguous is
    left alone — a card showing initials is a small blemish, a card showing
    another player's face is a lie.

    Returns how many notes were repaired.
    """
    by_exact: dict[str, int] = {}
    by_surname: dict[tuple[str, str], list[int]] = {}
    for block in context.get("player_statistics") or []:
        for row in block.get("players") or []:
            info = row.get("player") or {}
            pid, name = info.get("id"), info.get("name")
            if not (isinstance(pid, int) and pid > 0 and name):
                continue
            folded = _fold(name)
            by_exact.setdefault(folded, pid)
            key = _surname_initial(folded)
            if key:
                by_surname.setdefault(key, []).append(pid)

    repaired = 0
    for note in notes:
        if isinstance(note.get("player_id"), int) and note["player_id"] > 0:
            continue
        folded = _fold(note.get("player_name") or "")
        if not folded:
            continue
        pid = by_exact.get(folded)
        if pid is None:
            key = _surname_initial(folded)
            candidates = by_surname.get(key, []) if key else []
            # Only when it is unambiguous; two players sharing a surname and an
            # initial cannot be told apart from a name alone.
            pid = candidates[0] if len(candidates) == 1 else None
        if pid is None:
            continue
        note["player_id"] = pid
        note["photo"] = f"/api/player-photo/{pid}"
        repaired += 1
    return repaired


def first_usable(candidates: list[Optional[dict[str, Any]]],
                 half: str) -> Optional[dict[str, Any]]:
    """First candidate that yields usable content for the named half.

    Callers pass caches newest-format first, so a current entry always wins
    over a legacy one holding the same match.
    """
    migrate = opening_from if half == "opening" else analysis_from
    for candidate in candidates:
        migrated = migrate(candidate)
        if migrated:
            return migrated
    return None
