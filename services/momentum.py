"""
momentum.py
-----------
Builds the Match State Index: a continuous read of which side was on top,
minute by minute.

What the data actually supports
-------------------------------
The provider gives two different kinds of input, and the difference matters:

  * Timestamped events — goals, cards, VAR decisions, substitutions. Real
    minutes, so anything derived from them is anchored to something observed.
  * Match totals — shots, shots on target, shots in the box, corners, xG,
    possession, saves. These have no minute stamps at all.

There is no spatial tracking feed and no per-minute pressure feed behind this.
So the index is an explicit model, and it is built to keep the modelled part
separable from the measured part:

  index(t) = event impulses decaying over time  +  territorial pressure(t)

The event half is anchored: every inflection traces to an event with a real
minute. The territorial half distributes each side's *observed* match totals
across the 90 minutes using the score state, on the well-established basis
that a trailing team commits more players forward and a leading team drops
deeper. The totals are preserved exactly — only their distribution in time is
inferred, and it is inferred from the scoreline, which we do know minute by
minute.

That distinction is what the earlier version got wrong. It used events alone,
so a side that camped in the opposition half for 30 minutes without scoring
registered as flat — the chart showed nothing happening during the exact
passage a viewer would describe as relentless pressure. Totals fix that: a
side with 21 shots to 7 now sits visibly on top even in a goalless spell.

What this still cannot show is the rhythm *within* a spell of pressure. Real
attacking comes in waves; distributing a total smoothly cannot reproduce
individual waves, and manufacturing an oscillation would be inventing detail
the feed does not contain. The curve shows sustained pressure and where it
shifted, not each surge inside it.
"""

import math
import re
from typing import Any, Optional

# --- Event impulses (anchored to real minutes) ------------------------------

# Positive = credit toward the event's own team.
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

# --- Territorial pressure (distributed from match totals) -------------------

# What each unit of a statistic contributes to a side's share of the pressure.
# Weighted by how directly the statistic implies a threatening attack: xG and
# shots on target say a chance was actually created, possession barely says
# anything on its own, which is why it is worth a fraction of a point.
PRESSURE_WEIGHTS: dict[str, float] = {
    "expected_goals": 30.0,
    "Shots on Goal": 9.0,
    "Shots insidebox": 3.5,
    "Total Shots": 2.0,
    "Corner Kicks": 2.5,
    "Ball Possession": 0.30,   # per percentage point
}

# A defensive action by one side is evidence of pressure from the other, so it
# is credited to the opponent's pressure score rather than to the team's own.
OPPONENT_PRESSURE_WEIGHTS: dict[str, float] = {
    "Goalkeeper Saves": 7.0,   # every save is a shot on target faced
    "Blocked Shots": 1.5,      # the provider counts these as shots blocked BY
                               # the opponent, i.e. pressure this side absorbed
}

# How much harder a side pushes per goal it is behind, and the ceiling on that
# effect — a three-goal deficit does not make a side attack three times as hard.
CHASE_GAIN = 0.45
CHASE_CAP = 2.0
# A side does not reshape the instant a goal goes in; the new posture takes a
# few minutes to show up on the pitch. Also stops the curve stepping squarely.
POSTURE_RAMP_MINUTES = 6.0
# At kick-off no pressure has been exerted yet, so the territorial half starts
# at nothing and builds. Without this the curve opens at a side's match-average
# dominance, asserting a gap before a ball has been kicked.
WARMUP_MINUTES = 9.0

# Split of the final index between the two halves of the model. Events lead,
# because they are the part with real timestamps.
EVENT_SHARE = 0.62
PRESSURE_SHARE = 0.38
PRESSURE_FULL_SCALE = 100.0


def _to_number(raw: Any) -> Optional[float]:
    """Coerce the provider's mixed value types ('62%', '3.43', 12, None)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"-?\d+(?:\.\d+)?", str(raw))
    return float(match.group()) if match else None


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


def _stats_by_team(team_statistics: Optional[list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for block in team_statistics or []:
        team_id = (block.get("team") or {}).get("id")
        if team_id is None:
            continue
        out[team_id] = {s.get("type"): s.get("value") for s in block.get("statistics") or []}
    return out


def _pressure_score(own: dict[str, Any], opponent: dict[str, Any]) -> float:
    """How much attacking pressure one side generated across the whole match."""
    score = 0.0
    for key, weight in PRESSURE_WEIGHTS.items():
        value = _to_number(own.get(key))
        if value:
            score += value * weight
    for key, weight in OPPONENT_PRESSURE_WEIGHTS.items():
        value = _to_number(opponent.get(key))
        if value:
            score += value * weight
    return score


def _score_state(events: list[dict[str, Any]], home_team_id: int,
                 full_time: int) -> list[tuple[int, int]]:
    """Goals for (home, away) as they stood at each minute."""
    timeline = [(0, 0)] * (full_time + 1)
    home_goals = away_goals = 0
    goals: list[tuple[int, bool]] = []
    for event in events:
        key, _ = _classify(event)
        if key not in ("goal", "own_goal"):
            continue
        team_id = (event.get("team") or {}).get("id")
        scored_for_home = (team_id == home_team_id)
        if key == "own_goal":
            scored_for_home = not scored_for_home
        goals.append((_minute(event), scored_for_home))
    goals.sort()

    index = 0
    for minute in range(full_time + 1):
        while index < len(goals) and goals[index][0] <= minute:
            if goals[index][1]:
                home_goals += 1
            else:
                away_goals += 1
            index += 1
        timeline[minute] = (home_goals, away_goals)
    return timeline


def _posture(deficit: int) -> float:
    """How hard a side pushes when it is `deficit` goals behind.

    Behind, a side commits more players forward; ahead, it protects the lead.
    This is the mechanism that makes the curve show a side camped in the
    opponent's half after conceding, rather than flattening once the goal's
    own impulse has decayed away.
    """
    return max(1.0 / CHASE_CAP, min(CHASE_CAP, 1.0 + CHASE_GAIN * deficit))


def _distribute(pressure: float, postures: list[float]) -> list[float]:
    """Spread one side's match-total pressure across the minutes.

    Normalised so the area under the curve equals the observed total: the
    scoreline decides *when* that pressure was applied, never how much of it
    there was.
    """
    total_weight = sum(postures)
    if total_weight <= 0:
        return [0.0] * len(postures)
    return [pressure * w / total_weight for w in postures]


def build_timeline(
    events: list[dict[str, Any]],
    home_team_id: int,
    away_team_id: int,
    team_statistics: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Return the per-minute Match State Index plus the moments to annotate.

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
            moments.append(
                {
                    "minute": minute,
                    "kind": key,
                    "team_id": team_id,
                    "is_home": team_id == home_team_id,
                    "player": (event.get("player") or {}).get("name"),
                    "label": event.get("detail") or event.get("type"),
                }
            )

    last_minute = max([m for m, _ in impulses] + [FULL_TIME_DEFAULT])
    full_time = max(FULL_TIME_DEFAULT, last_minute)

    # --- territorial half -------------------------------------------------
    by_team = _stats_by_team(team_statistics)
    home_stats, away_stats = by_team.get(home_team_id, {}), by_team.get(away_team_id, {})
    have_totals = bool(home_stats and away_stats)

    pressure_series = [0.0] * (full_time + 1)
    home_pressure = away_pressure = 0.0
    if have_totals:
        home_pressure = _pressure_score(home_stats, away_stats)
        away_pressure = _pressure_score(away_stats, home_stats)

        state = _score_state(events, home_team_id, full_time)
        # Ease toward each new posture instead of stepping, so a goal shifts
        # the shape of the game over the following minutes rather than instantly.
        ease = 1.0 - math.exp(-1.0 / POSTURE_RAMP_MINUTES)
        home_postures, away_postures = [], []
        home_now = away_now = 1.0
        for minute in range(full_time + 1):
            home_goals, away_goals = state[minute]
            home_now += (_posture(away_goals - home_goals) - home_now) * ease
            away_now += (_posture(home_goals - away_goals) - away_now) * ease
            home_postures.append(home_now)
            away_postures.append(away_now)

        home_curve = _distribute(home_pressure, home_postures)
        away_curve = _distribute(away_pressure, away_postures)

        # Scale the differential against the busier side so a one-sided match
        # reaches the top of the chart and an even one hovers near the middle.
        peak = max(max(home_curve, default=0.0), max(away_curve, default=0.0)) or 1.0
        pressure_series = [
            (home_curve[i] - away_curve[i]) / peak * PRESSURE_FULL_SCALE
            * (1.0 - math.exp(-i / WARMUP_MINUTES))
            for i in range(full_time + 1)
        ]

    # --- combine ----------------------------------------------------------
    event_weight = EVENT_SHARE if have_totals else 1.0
    pressure_weight = PRESSURE_SHARE if have_totals else 0.0

    series = []
    for minute in range(0, full_time + 1):
        impulse_total = 0.0
        for at, magnitude in impulses:
            if at <= minute:
                impulse_total += magnitude * math.exp(-(minute - at) / DECAY_MINUTES)
        combined = impulse_total * event_weight + pressure_series[minute] * pressure_weight
        series.append(round(max(-100.0, min(100.0, combined)), 2))

    moments.sort(key=lambda m: m["minute"])
    total_pressure = home_pressure + away_pressure
    return {
        "series": series,
        "full_time": full_time,
        "moments": moments,
        "peak_home": max(series) if series else 0,
        "peak_away": min(series) if series else 0,
        "pressure_share": (
            round(home_pressure / total_pressure, 3) if total_pressure else None
        ),
        "model": {
            "name": "Match State Index",
            "derived_from": (
                "timestamped events (goals, cards, VAR, substitutions) plus each "
                "side's match totals for shots, shots on target, shots in the box, "
                "corners, expected goals, possession and saves"
            ),
            "method": (
                "Events apply a signed impulse that decays exponentially. Match "
                "totals are distributed across the 90 minutes by score state, on "
                "the basis that a trailing side commits forward and a leading side "
                "drops deeper — the totals are preserved exactly, only their timing "
                "is modelled."
            ),
            "not_derived_from": "spatial tracking or a live per-minute pressure feed",
            "limitation": (
                "Shows sustained pressure and where it shifted, not the individual "
                "waves inside a spell — the feed carries no minute-level shot data."
            ),
            "decay_minutes": DECAY_MINUTES,
            "event_share": event_weight,
            "pressure_share": pressure_weight,
            "has_match_totals": have_totals,
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
        return "Home side held the upper hand for most of the match."
    if away_minutes > home_minutes * 1.5:
        return "Away side held the upper hand for most of the match."
    return "The match state swung repeatedly — neither side settled into control."
