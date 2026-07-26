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

from services import momentum

logger = logging.getLogger(__name__)

_CLIP_DATA = Path(__file__).resolve().parent / "highlights_data.json"

# VAR outcomes that uphold the original decision. These are folded into the
# event they confirm rather than listed separately — the alternative is the
# same goal appearing twice, one line apart.
_VAR_UPHELD = ("confirmed", "upheld")

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
        if momentum.is_shootout_event(event):
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
            # A VAR check that upholds what already happened is not a separate
            # moment — "57' Goal confirmed" directly under "57' Goal" is the
            # same event twice. Fold it into the goal as a checked marker and
            # drop the row. Overturns are the opposite: they change the score,
            # so they earn a row of their own.
            if any(word in detail for word in _VAR_UPHELD):
                for earlier in reversed(moments):
                    same_player = earlier.get("player") and earlier["player"] == player
                    close_enough = abs(earlier["minute"] - ((event.get("time") or {}).get("elapsed") or 0)) <= 2
                    if same_player or (close_enough and earlier["team_id"] == team_id):
                        earlier["var_checked"] = True
                        break
                continue
            kind, title = "var_overturn", (event.get("detail") or "VAR decision")

        if not kind:
            continue

        player_id = (event.get("player") or {}).get("id")
        moments.append({
            "id": f"m{len(moments)}",
            "minute": (event.get("time") or {}).get("elapsed") or 0,
            "minute_label": _minute_label(event.get("time") or {}),
            "kind": kind,
            "title": title,
            "player": player,
            # A real photograph of the player who produced the moment. The
            # provider has no action photography — /fixtures and /photos on its
            # media CDN are both 404 — so this is the closest real image of the
            # moment that exists in any source available to us. A licensed
            # action shot, where one is held, overrides it via `image` below.
            "player_id": player_id,
            "photo": f"/api/player-photo/{player_id}" if player_id else None,
            "assist": assist if kind in ("goal", "penalty") else None,
            "team_id": team_id,
            "team": (event.get("team") or {}).get("name"),
            "is_home": is_home,
            # No score badge on a miss or an overturn: neither advanced the
            # scoreline, and a badge there reads as though one had.
            "score": (f"{running_home}–{running_away}"
                      if kind not in ("miss", "var_overturn") else None),
        })
    return moments


def build_shootout(context: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The penalty shootout as an ordered sequence with a running tally."""
    fixture = context.get("fixture") or {}
    home_id = ((fixture.get("teams") or {}).get("home") or {}).get("id")
    kicks_raw = [e for e in context.get("events") or [] if momentum.is_shootout_event(e)]
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
        player_id = (event.get("player") or {}).get("id")
        kicks.append({
            "order": len(kicks) + 1,
            "player": (event.get("player") or {}).get("name"),
            "player_id": player_id,
            "photo": f"/api/player-photo/{player_id}" if player_id else None,
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


def _provenance(clips: list, youtube: Optional[dict], images: Optional[dict]) -> str:
    """Say exactly where the media in this panel came from.

    Written per-source rather than as one fixed sentence, because the panel can
    hold licensed video, an official YouTube embed, licensed action stills, or
    none of those — and a viewer should never have to guess whether the picture
    beside a goal is a photograph of that goal.
    """
    parts = ["Moments from the official timestamped event feed."]
    if youtube:
        channel = youtube.get("channel")
        parts.append(f"Highlights embedded from {channel}'s official YouTube channel."
                     if channel else "Highlights embedded from YouTube.")
    if clips:
        parts.append("Video from the configured licensed source.")
    if images:
        parts.append("Action photography from the configured licensed source.")
    else:
        parts.append(
            "Moment images are player portraits from the data provider, which "
            "publishes no action photography — they are pictures of the player, "
            "not of the goal."
        )
    return " ".join(parts)


def clips_for(fixture_id: int, resolved: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Any configured video for this fixture. Empty when none is licensed.

    `provenance` is built here rather than in `build` on purpose: the moments
    are cached but the clip config is re-read on every request, so adding
    footage takes effect immediately. If the sentence lived in the cached blob
    it would keep saying no video exists after one had been added.
    """
    entry = _clip_config().get(str(fixture_id)) or {}
    clips = entry.get("clips") or []
    # Hand-verified mapping first; an auto-resolved video only fills a gap.
    youtube_id = entry.get("youtube") or (resolved or {}).get("id")
    if not entry.get("youtube") and resolved:
        entry = {**entry,
                 "youtube_channel": resolved.get("channel"),
                 "youtube_title": resolved.get("title")}
    youtube = None
    if youtube_id:
        youtube = {
            "id": youtube_id,
            # Standard host, not youtube-nocookie. The privacy-enhanced domain
            # is preferable in principle but was verified to fail here: both
            # the FIFA and FC Barcelona uploads returned "Error 153 — video
            # player configuration error" through nocookie while playing
            # normally through www.youtube.com. The facade below is what keeps
            # this private in practice — nothing loads from YouTube at all
            # until the viewer presses play.
            "embed": (f"https://www.youtube.com/embed/{youtube_id}"
                      "?rel=0&modestbranding=1&enablejsapi=1"),
            # Proxied like every other remote image, so a media blocker cannot
            # leave the panel with an empty poster.
            "thumb": f"/api/youtube-thumb/{youtube_id}",
            "channel": entry.get("youtube_channel"),
            "title": entry.get("youtube_title"),
        }
    return {
        "clips": clips,
        "youtube": youtube,
        "poster": entry.get("poster"),
        "credit": entry.get("credit"),
        "has_video": bool(clips or youtube),
        # Stated in the payload so the client never has to imply that footage
        # or action photography exists when it doesn't.
        "provenance": _provenance(clips, youtube, entry.get("images")),
    }


def auto_map(context: dict[str, Any], fixture_id: int) -> Optional[dict[str, Any]]:
    """Look up an official highlights video for this fixture, once, and keep it.

    The manual map wins wherever it has an entry — those ids were each verified
    by hand against the uploading channel. This fills the rest in, and only
    with candidates that clear the resolver's checks.

    The result is cached permanently either way, including the misses: a match
    with no official upload today will not have one tomorrow, and re-searching
    would burn the API's daily quota discovering that repeatedly.
    """
    from services import summary_cache, youtube

    if (_clip_config().get(str(fixture_id)) or {}).get("youtube"):
        return None                       # already mapped by hand
    if not youtube.api_key():
        return None                       # no key configured; stay inert

    cache_key = f"yt-{fixture_id}"
    cached = summary_cache.get(cache_key)
    if cached is not None:
        return cached or None             # {} is a remembered miss

    fixture = context.get("fixture") or {}
    teams = fixture.get("teams") or {}
    home, away = teams.get("home") or {}, teams.get("away") or {}
    found = youtube.resolve(
        home.get("name") or "", away.get("name") or "",
        (fixture.get("fixture") or {}).get("date") or "",
        (fixture.get("league") or {}).get("name") or "",
        home_id=home.get("id"), away_id=away.get("id"),
    )
    summary_cache.set(cache_key, found or {})
    return found


def build(context: dict[str, Any], fixture_id: int) -> dict[str, Any]:
    from services import youtube

    moments = build_moments(context)
    config = _clip_config().get(str(fixture_id)) or {}
    fixture = context.get("fixture") or {}
    sides = fixture.get("teams") or {}

    # A licensed action photograph, keyed by moment id or by minute, replaces
    # the player portrait for that moment. Nothing here ships with the app.
    images = config.get("images") or {}
    if images:
        for moment in moments:
            override = images.get(moment["id"]) or images.get(str(moment["minute"]))
            if override:
                moment["image"] = override
                moment["image_credit"] = config.get("image_credit")

    payload = {
        "moments": moments,
        "shootout": build_shootout(context),
        **clips_for(fixture_id, auto_map(context, fixture_id)),
    }
    # Always present. It backs two cases: no video resolved at all, and a
    # resolved video that turns out to be unplayable for this viewer — highlight
    # rights are territorial, so an NBC upload is US-only and a Sky one UK-only.
    # A link, not an embed: YouTube removed search embeds, so an iframe pointed
    # at a search renders Error 153, which is the dead player worth avoiding.
    if True:
        payload["search_url"] = youtube.search_url(
            (sides.get("home") or {}).get("name") or "",
            (sides.get("away") or {}).get("name") or "",
            (fixture.get("fixture") or {}).get("date") or "",
        )
    return payload
