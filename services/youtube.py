"""
youtube.py
----------
Resolves the official highlights video for a fixture, so highlights stop
depending on someone hand-mapping every match.

Why this is careful rather than a plain search-and-take-the-first-hit: the
searches run while building this were actively misleading. Results presented as
"official FIFA" turned out to be FOX Soccer, ITV and TSN re-uploads; one
suggested UEFA video had been deleted; and a search for a 2017 Champions League
tie returned a FIFA-21 video-game simulation of it. Binding any of those into
the player would be worse than showing no video at all, because it looks
authoritative.

So a candidate has to earn its place:

  * the channel must be on the allowlist below — a rights holder, not a fan
    account or an aggregator;
  * the title must mention both sides;
  * it must have been published near the match, which is what separates the
    real highlights from anniversary re-cuts and simulations.

Nothing is guessed. If no candidate clears those bars the resolver returns
None, the manual map in highlights_data.json still wins where it exists, and
the panel simply has no video.

Requires YOUTUBE_API_KEY (YouTube Data API v3). Without one this is inert and
the manual map is used unchanged — search cannot be done reliably without it,
and scraping results would be both fragile and against YouTube's terms.
"""

import logging
import os
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
REQUEST_TIMEOUT = 15

# Channels whose uploads we are willing to embed. Order of usefulness is not
# what you would guess, and it is worth writing down because it drives the
# whole design:
#
#   Premier League highlights are NOT on club channels. The league sells those
#   rights exclusively, so Manchester United cannot post match highlights to
#   its own YouTube channel — MUTV carries them behind a subscription instead.
#   The 3-4 minute packages that are on YouTube come from the rights-holding
#   broadcasters: Sky Sports and TNT in the UK, NBC in the US. Searching club
#   channels for a Premier League fixture will find nothing, every time.
#
#   Club and competition channels DO work outside the Premier League — FC
#   Barcelona posts its own Champions League highlights, FIFA posts World Cup
#   ones — so both groups are kept.
OFFICIAL_CHANNELS = {
    # Rights-holding broadcasters — the only route to Premier League highlights.
    "nbc sports", "sky sports premier league", "sky sports football",
    "tnt sports football", "tnt sports", "bbc sport", "bbc match of the day",
    "itv sport", "cbs sports golazo", "bein sports", "espn fc",
    "optus sport", "fox soccer", "tsn", "dazn",
    # Competition organisers.
    "fifa", "uefa", "premier league", "efl", "laliga", "serie a", "bundesliga",
    "ligue 1", "ligue1", "uefa champions league",
    # Clubs — effective for European and domestic-cup ties, not the league.
    "mutv", "fc barcelona", "real madrid", "manchester city",
    "manchester united", "liverpool fc", "arsenal", "chelsea football club",
    "tottenham hotspur", "paris saint-germain", "juventus", "ac milan",
    "inter", "fc bayern münchen", "borussia dortmund", "atlético madrid",
}

# Broadcasters follow rigid title conventions, which makes them far easier to
# match than free-form club uploads. NBC's is
# "{Home} v. {Away} | PREMIER LEAGUE HIGHLIGHTS | M/D/YYYY | NBC Sports",
# so a title carrying the competition and the date is strong evidence that this
# is the packaged highlights reel and not a clip, preview or reaction.
_HIGHLIGHT_MARKERS = re.compile(
    r"\b(highlights?|extended highlights|match highlights)\b", re.I)

# How far from kick-off a genuine highlights upload can sit. Wide enough for
# timezones and next-morning packages, tight enough to exclude the anniversary
# retrospectives that dominate results for famous matches.
_MAX_DAYS_AFTER = 4
_MAX_DAYS_BEFORE = 1

_NOISE = re.compile(
    r"\b(fifa \d\d|efootball|pes \d+|prediction|reaction|review|podcast|"
    r"simulation|recreated|fan cam|full match replay|watchalong)\b", re.I)


def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", ascii_only.lower())


def _key_word(team_name: str) -> str:
    """The most identifying word of a club name.

    "Manchester United" and "Manchester City" share their first word, so the
    last is the discriminating one; for single-word names it is the name.
    """
    words = [w for w in _norm(team_name).split()
             if w not in ("fc", "cf", "afc", "sc", "ac", "club", "de", "the")]
    return words[-1] if words else _norm(team_name)


def _identifiers(team_name: str, team_id: Optional[int]) -> set[str]:
    """Every form a broadcaster might use for this club.

    Matching only the full name misses almost everything: Sky writes "Man Utd",
    NBC writes "Manchester United", the club itself writes "United". The team
    registry already carries these aliases for the search bar, so they are
    reused here rather than maintained twice.
    """
    forms = {_key_word(team_name), _norm(team_name)}
    if team_id is not None:
        from services import teams
        entry = teams.get_team(team_id)
        if entry:
            forms.add(_norm(entry.get("name") or ""))
            forms.update(_norm(a) for a in entry.get("aliases") or [])
        forms.add(_norm(teams.display_name(team_id, team_name)))
    return {f.strip() for f in forms if f and f.strip()}


def _mentions(title_norm: str, forms: set[str]) -> bool:
    return any(form in title_norm for form in forms)


def _date_in_title(title: str, kickoff: datetime) -> bool:
    """Whether the title carries the fixture's date in a broadcaster's format.

    NBC writes 5/17/2026, others 17/05/2026 or 2026-05-17. Both orderings are
    checked because getting this wrong silently costs a match rather than
    raising anything.
    """
    d, m, y = kickoff.day, kickoff.month, kickoff.year
    candidates = {
        f"{m}/{d}/{y}", f"{m:02d}/{d:02d}/{y}",
        f"{d}/{m}/{y}", f"{d:02d}/{m:02d}/{y}",
        f"{y}-{m:02d}-{d:02d}",
    }
    return any(c in title for c in candidates)


def search_url(home: str, away: str, kickoff_iso: str) -> str:
    """A YouTube search that a person can follow when no video was resolved.

    A link, deliberately, not an embed: YouTube removed search embeds — an
    iframe pointed at `listType=search` renders "Error 153", which would put a
    dead player on the page instead of removing one.
    """
    date = (kickoff_iso or "")[:10]
    query = " ".join(x for x in (home, "vs", away, date, "highlights") if x)
    return "https://www.youtube.com/results?search_query=" + quote_plus(query)


def api_key() -> Optional[str]:
    return os.environ.get("YOUTUBE_API_KEY") or None


def _channel_is_official(title: str) -> bool:
    norm = _norm(title)
    return any(_norm(c) in norm for c in OFFICIAL_CHANNELS)


def _score(item: dict[str, Any], home_forms: set[str], away_forms: set[str],
           kickoff: datetime) -> Optional[tuple[float, dict[str, Any]]]:
    snippet = item.get("snippet") or {}
    title = snippet.get("title") or ""
    channel = snippet.get("channelTitle") or ""

    if not _channel_is_official(channel):
        return None
    if _NOISE.search(title):
        return None

    norm_title = _norm(title)
    if not (_mentions(norm_title, home_forms) and _mentions(norm_title, away_forms)):
        return None

    published = snippet.get("publishedAt")
    try:
        when = datetime.fromisoformat((published or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    delta = (when - kickoff).total_seconds() / 86400.0
    if not (-_MAX_DAYS_BEFORE <= delta <= _MAX_DAYS_AFTER):
        return None

    # Prefer same-day uploads and titles that say "highlights" outright.
    score = 10.0 - abs(delta)
    if _HIGHLIGHT_MARKERS.search(title):
        score += 4
    if "extended" in norm_title:
        score += 1
    # A broadcaster stamping the fixture date into the title is about as strong
    # a signal as this gets that it is the packaged reel for that exact match.
    if _date_in_title(title, kickoff):
        score += 3
    return score, {
        "id": (item.get("id") or {}).get("videoId"),
        "title": title,
        "channel": channel,
        "published": published,
    }


def resolve_all(home: str, away: str, kickoff_iso: str, competition: str = "",
                home_id: Optional[int] = None, away_id: Optional[int] = None,
                limit: int = 4) -> list[dict[str, Any]]:
    """Every candidate that clears the bar, best first.

    More than one is returned on purpose. Highlight rights are territorial —
    NBC's upload plays in the US and refuses everywhere else, Sky's the reverse
    — so the best-ranked candidate is frequently unplayable for a given viewer.
    Handing the client the whole shortlist lets it try the next one instead of
    giving up on the first refusal.
    """
    key = api_key()
    if not key:
        return []
    try:
        kickoff = datetime.fromisoformat((kickoff_iso or "").replace("Z", "+00:00"))
    except ValueError:
        logger.info("Unparseable kickoff %r; skipping video lookup", kickoff_iso)
        return []

    query = f"{home} vs {away} highlights {competition} {kickoff.year}".strip()
    try:
        resp = requests.get(SEARCH_URL, timeout=REQUEST_TIMEOUT, params={
            "key": key, "part": "snippet", "type": "video", "maxResults": 25,
            "videoEmbeddable": "true",      # unembeddable results are useless here
            "q": query,
            "publishedAfter": (kickoff - timedelta(days=_MAX_DAYS_BEFORE)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "publishedBefore": (kickoff + timedelta(days=_MAX_DAYS_AFTER)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("YouTube search failed for %s v %s: %s", home, away, exc)
        return []

    home_forms = _identifiers(home, home_id)
    away_forms = _identifiers(away, away_id)
    scored = []
    for item in resp.json().get("items", []):
        result = _score(item, home_forms, away_forms, kickoff)
        if result and result[1]["id"]:
            scored.append(result)

    if not scored:
        logger.info("No official highlights cleared the bar for %s v %s", home, away)
        return []
    scored.sort(key=lambda s: s[0], reverse=True)
    found = [entry for _, entry in scored[:limit]]
    logger.info("Resolved %d highlight candidate(s) for %s v %s: %s",
                len(found), home, away,
                ", ".join(f"{f['id']} ({f['channel']})" for f in found))
    return found


def resolve(home: str, away: str, kickoff_iso: str, competition: str = "",
            home_id: Optional[int] = None,
            away_id: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Best official highlights video for a fixture, or None."""
    found = resolve_all(home, away, kickoff_iso, competition, home_id, away_id)
    return found[0] if found else None
