"""
highlights.py
-------------
Builds the Highlights module for a match: an ordered reel of the moments that
decided it, and the video clips attached to them.

On the video, plainly: match footage is owned by the broadcasters and
federations who filmed it, so none ships with this app and none is fetched
from anywhere it hasn't been licensed from. What the player renders is
whatever is declared in `highlights_data.json` — files you host, or clips you
hold the rights to. Absent that, the reel still stands on its own, because the
moments themselves come from the provider's timestamped event feed and are
real: the minute, the scorer, the assist, every kick of a shootout.

So the module has two layers, and the second is optional:

  1. The moment reel — always present for any match with events.
  2. Video attached to those moments — present when configured, and the UI
     says so honestly when it isn't rather than showing a dead player.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CLIP_DATA = Path(__file__).resolve().parent / "highlights_data.json"

# The provider tags every shootout kick with this comment. Using it beats any
# minute-based heuristic: an in-play penalty in the 118th minute and a shootout
# kick logged at 120+1 are otherwise indistinguishable.
_SHOOTOUT_COMMENT = "penalty shootout"

# Which events earn a place in the reel, and how prominently.
_GOAL_KINDS = {
    "normal goal": ("goal", "Goal"),
    "penalty": ("penalty", "Penalty"),
    "own goal": ("own_goal", "Own goal"),
    "missed penalty": ("miss", "Penalty missed"),
}


def _minute_label(time: dict[str, Any]) -> str:
    elapsed = time.get("elapsed") or 0
    extra = time.get("extra")
    return f"{elapsed}+{extra}'" if extra else f"{elapsed}'"


def _is_shootout(event: dict[str, Any]) -> bool:
    return (event.get("comments") or "").strip().lower() == _SHOOTOUT_COMMENT


def _sort_key(event: dict[str, Any]) -> tuple[int, int]:
    t = event.get("time") or {}
    return (t.get("elapsed") or 0, t.get("extra") or 0)


def build_moments(context: dict[str, Any]) -> list[dict[str, Any]]:
    """The decisive moments in play, in order, from the event feed."""
    fixture = context.get("fixture") or {}
    teams = (fixture.get("teams") or {})
    home_id = (teams.get("home") or {}).get("id")

    moments = []
    running_home = running_away = 0
    for event in sorted(context.get("events") or [], key=_sort_key):
        if _is_shootout(event):
            continue                      # the shootout is its own block
        etype = (event.get("type") or "").lower()
        detail = (event.get("detail") or "").lower()
        team_id = (event.get("team") or {}).get("id")
        is_home = team_id == home_id
        player = (event.get("player") or {}).get("name")
        assist = (event.get("assist") or {}).get("name")

        kind = title = None
        if etype == "goal":
            kind, title = _GOAL_KINDS.get(detail, ("goal", "Goal"))
            if kind != "miss":
                # An own goal counts for the other side.
                scored_home = not is_home if kind == "own_goal" else is_home
                if scored_home:
                    running_home += 1
                else:
                    running_away += 1
        elif etype == "card" and "red" in detail:
            kind, title = "red_card", "Red card"
        elif etype == "var":
            kind, title = "var", (event.get("detail") or "VAR decision")

        if not kind:
            continue

        moments.append({
            "id": f"m{len(moments)}",
            "minute": (event.get("time") or {}).get("elapsed") or 0,
            "minute_label": _minute_label(event.get("time") or {}),
            "kind": kind,
            "title": title,
            "player": player,
            "assist": assist if kind in ("goal", "penalty") else None,
            "team_id": team_id,
            "team": (event.get("team") or {}).get("name"),
            "is_home": is_home,
            "score": f"{running_home}–{running_away}" if kind != "miss" else None,
        })
    return moments


def build_shootout(context: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The penalty shootout as an ordered sequence with a running tally."""
    fixture = context.get("fixture") or {}
    home_id = ((fixture.get("teams") or {}).get("home") or {}).get("id")
    kicks_raw = [e for e in context.get("events") or [] if _is_shootout(e)]
    if not kicks_raw:
        return None

    home_score = away_score = 0
    kicks = []
    for event in sorted(kicks_raw, key=_sort_key):
        scored = (event.get("detail") or "").lower() != "missed penalty"
        is_home = (event.get("team") or {}).get("id") == home_id
        if scored:
            if is_home:
                home_score += 1
            else:
                away_score += 1
        kicks.append({
            "order": len(kicks) + 1,
            "player": (event.get("player") or {}).get("name"),
            "team": (event.get("team") or {}).get("name"),
            "team_id": (event.get("team") or {}).get("id"),
            "is_home": is_home,
            "scored": scored,
            "running": f"{home_score}–{away_score}",
        })

    final = (fixture.get("score") or {}).get("penalty") or {}
    return {
        "kicks": kicks,
        "home_score": final.get("home", home_score),
        "away_score": final.get("away", away_score),
    }


def _clip_config() -> dict[str, Any]:
    if not _CLIP_DATA.exists():
        return {}
    try:
        return json.loads(_CLIP_DATA.read_text())
    except (OSError, json.JSONDecodeError):
        logger.exception("highlights_data.json is unreadable; serving moments only")
        return {}


def clips_for(fixture_id: int) -> dict[str, Any]:
    """Any configured video for this fixture. Empty when none is licensed.

    `provenance` is built here rather than in `build` on purpose: the moments
    are cached but the clip config is re-read on every request, so adding
    footage takes effect immediately. If the sentence lived in the cached blob
    it would keep saying no video exists after one had been added.
    """
    entry = _clip_config().get(str(fixture_id)) or {}
    clips = entry.get("clips") or []
    return {
        "clips": clips,
        "poster": entry.get("poster"),
        "credit": entry.get("credit"),
        "has_video": bool(clips),
        # Stated in the payload so the client never has to imply footage exists.
        "provenance": (
            "Moments derived from the official timestamped event feed. "
            + ("Video supplied from the configured licensed source."
               if clips else
               "No licensed video is configured for this match.")
        ),
    }


def build(context: dict[str, Any], fixture_id: int) -> dict[str, Any]:
    return {
        "moments": build_moments(context),
        "shootout": build_shootout(context),
        **clips_for(fixture_id),
    }
