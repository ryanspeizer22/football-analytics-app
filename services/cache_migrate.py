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
