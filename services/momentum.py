"""
momentum.py
-----------
Derives a match "momentum" timeline from real, timestamped match events.

Honest scope: API-Football's free tier gives per-event minutes (goals, cards,
VAR decisions, substitutions) but only *match totals* for shots and possession
— there is no minute-by-minute pressure feed. So momentum here is an explicit
model over events that actually happened, not a measured pressure metric, and
the UI labels it as such. Nothing is invented: every inflection traces to an
event with a real minute stamp.

Model: each event applies a signed impulse toward one team, which then decays
exponentially. Momentum at minute t is the sum of surviving impulses, so a goal
dominates briefly and fades unless reinforced.
"""

import math
from typing import Any

# Impulse weights, positive = credit to the event's own team.
IMPULSE = {
    "goal": 100.0,
    "own_goal": -100.0,        # credited against the scoring player's team
    "penalty_missed": -40.0,
    "red_card": -75.0,
    "yellow_card": -12.0,
    "goal_cancelled": -45.0,   # VAR overturn
    "substitution": 6.0,
}

DECAY_MINUTES = 14.0  # time constant; a goal's surge is mostly spent after ~30'
FULL_TIME_DEFAULT = 90


def _classify(event: dict[str, Any]) -> tuple[str, bool]:
    """Map a raw event to (impulse key, credited_to_event_team).

    The bool is False when the impulse should be credited to the *opposing*
    team — an own goal helps the other side.
    """
    etype = (event.get("type") or "").lower()
    detail = (event.get("detail") or "").lower()

    if etype == "goal":
        if "own" in detail:
            return "own_goal", True  # negative weight already flips it
        if "missed" in detail:
            return "penalty_missed", True
        return "goal", True
    if etype == "card":
        if "red" in detail:
            return "red_card", True
        return "yellow_card", True
    if etype == "var":
        if "cancelled" in detail or "disallowed" in detail:
            return "goal_cancelled", True
        return "", True
    if etype == "subst":
        return "substitution", True
    return "", True


def _minute(event: dict[str, Any]) -> int:
    time = event.get("time") or {}
    return int(time.get("elapsed") or 0) + int(time.get("extra") or 0)


def build_timeline(
    events: list[dict[str, Any]],
    home_team_id: int,
    away_team_id: int,
) -> dict[str, Any]:
    """Return a per-minute momentum series plus the key moments to annotate.

    Series values are clamped to [-100, 100]; positive favours the home team.
    """
    impulses: list[tuple[int, float]] = []   # (minute, signed magnitude)
    moments: list[dict[str, Any]] = []

    for event in events:
        key, _ = _classify(event)
        if not key:
            continue
        weight = IMPULSE[key]
        minute = _minute(event)
        team_id = (event.get("team") or {}).get("id")

        # Orient the impulse: positive = toward home.
        sign = 1.0 if team_id == home_team_id else -1.0
        impulses.append((minute, weight * sign))

        if key in ("goal", "own_goal", "red_card", "goal_cancelled", "penalty_missed"):
            player = (event.get("player") or {}).get("name")
            moments.append(
                {
                    "minute": minute,
                    "kind": key,
                    "team_id": team_id,
                    "is_home": team_id == home_team_id,
                    "player": player,
                    "label": event.get("detail") or event.get("type"),
                }
            )

    last_minute = max([m for m, _ in impulses] + [FULL_TIME_DEFAULT])
    full_time = max(FULL_TIME_DEFAULT, last_minute)

    series = []
    for minute in range(0, full_time + 1):
        total = 0.0
        for at, magnitude in impulses:
            if at <= minute:
                total += magnitude * math.exp(-(minute - at) / DECAY_MINUTES)
        series.append(round(max(-100.0, min(100.0, total)), 2))

    moments.sort(key=lambda m: m["minute"])
    return {
        "series": series,
        "full_time": full_time,
        "moments": moments,
        "peak_home": max(series) if series else 0,
        "peak_away": min(series) if series else 0,
        "model": {
            "decay_minutes": DECAY_MINUTES,
            "derived_from": "timestamped match events",
        },
    }


def summarize_shift(timeline: dict[str, Any]) -> str:
    """One-line read of who controlled the match, for the chart caption."""
    series = timeline.get("series") or []
    if not series:
        return "No timed events recorded for this match."
    home_minutes = sum(1 for v in series if v > 5)
    away_minutes = sum(1 for v in series if v < -5)
    if home_minutes > away_minutes * 1.5:
        return "Home side held the momentum for most of the match."
    if away_minutes > home_minutes * 1.5:
        return "Away side held the momentum for most of the match."
    return "Momentum swung repeatedly — neither side settled into control."
