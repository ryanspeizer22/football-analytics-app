"""
stats.py
--------
Builds head-to-head stat comparisons from the provider's own match totals.

These power the stat-delta bars in the UI. They are computed here rather than
asked of the model on purpose: the numbers are the one part of a match report
that must be exactly right, and copying them out of the API removes any chance
of a transcription slip in generated prose.
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# (provider key, display label, higher-is-better, unit)
TRACKED: list[tuple[str, str, bool, str]] = [
    ("Ball Possession", "Possession", True, "%"),
    ("expected_goals", "Expected goals", True, ""),
    ("Total Shots", "Shots", True, ""),
    ("Shots on Goal", "On target", True, ""),
    ("Total passes", "Passes", True, ""),
    ("Passes %", "Pass accuracy", True, "%"),
    ("Corner Kicks", "Corners", True, ""),
    ("Fouls", "Fouls", False, ""),
    ("Yellow Cards", "Yellow cards", False, ""),
]


def _to_number(raw: Any) -> Optional[float]:
    """Coerce the provider's mixed value types ('62%', '3.43', 12, None)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"-?\d+(?:\.\d+)?", str(raw))
    return float(match.group()) if match else None


def build_comparisons(
    team_statistics: list[dict[str, Any]],
    home_team_id: int,
    away_team_id: int,
) -> list[dict[str, Any]]:
    """Return per-metric home/away comparisons with bar proportions.

    `share` is the home side's fraction of the pair, so the UI can draw one
    split bar per metric. Metrics missing for either side are skipped rather
    than shown as zero, which would read as a real (and wrong) value.
    """
    by_team: dict[int, dict[str, Any]] = {}
    for block in team_statistics or []:
        tid = (block.get("team") or {}).get("id")
        if tid is None:
            continue
        by_team[tid] = {
            s.get("type"): s.get("value") for s in block.get("statistics") or []
        }

    home_stats = by_team.get(home_team_id, {})
    away_stats = by_team.get(away_team_id, {})
    if not home_stats or not away_stats:
        return []

    comparisons = []
    for key, label, higher_better, unit in TRACKED:
        home_value = _to_number(home_stats.get(key))
        away_value = _to_number(away_stats.get(key))
        if home_value is None or away_value is None:
            continue

        total = home_value + away_value
        share = 0.5 if total == 0 else home_value / total

        if home_value == away_value:
            leader = "even"
        elif (home_value > away_value) == higher_better:
            leader = "home"
        else:
            leader = "away"

        comparisons.append(
            {
                "label": label,
                "home": _format(home_value, unit),
                "away": _format(away_value, unit),
                "home_raw": home_value,
                "away_raw": away_value,
                "share": round(share, 4),
                "leader": leader,
                "higher_is_better": higher_better,
            }
        )
    return comparisons


def _format(value: float, unit: str) -> str:
    if value == int(value):
        return f"{int(value)}{unit}"
    return f"{value:.2f}{unit}"


def headline_metrics(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the few metrics worth showing as large callout badges.

    Chosen by how lopsided the split is, so the badges surface where the match
    was actually decided rather than a fixed list.
    """
    priority = {"Expected goals": 0, "Possession": 1, "On target": 2, "Shots": 3}
    scored = sorted(
        comparisons,
        key=lambda c: (priority.get(c["label"], 9), -abs(c["share"] - 0.5)),
    )
    return scored[:4]
