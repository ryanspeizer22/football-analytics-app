"""
api_service.py
--------------
Modular client for a free-tier football data provider (API-Football via
api-sports.io). Every function returns plain dicts/lists so the rest of the
app never depends on the provider's response shape directly — swap providers
by editing only this file.

Setup:
    1. Sign up at https://www.api-football.com/ (free tier: 100 req/day).
    2. Export your key:  export FOOTBALL_API_KEY="your-key"
"""

import logging
import os
import threading
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

API_BASE_URL = "https://v3.football.api-sports.io"
API_KEY_ENV_VAR = "FOOTBALL_API_KEY"
REQUEST_TIMEOUT_SECONDS = 15

# Stop short of the hard daily cap so a burst of discovery calls can't strand
# the app with zero requests left for an in-flight analysis.
QUOTA_RESERVE = int(os.environ.get("FOOTBALL_API_QUOTA_RESERVE", "5"))


class FootballAPIError(Exception):
    """The provider returned an error or was unreachable."""


class SeasonUnavailableError(FootballAPIError):
    """The requested season exists but this subscription plan cannot read it.

    Distinct from a generic failure so callers can tell the user which
    seasons *are* reachable instead of showing a dead end.
    """


class QuotaExhaustedError(FootballAPIError):
    """The account's daily request allowance is spent (or nearly so)."""


# --- Quota state (updated from provider response headers) -------------------
_quota_lock = threading.Lock()
_quota: dict[str, Optional[int]] = {"limit": None, "remaining": None}


def quota_snapshot() -> dict[str, Optional[int]]:
    with _quota_lock:
        return dict(_quota)


def _record_quota(resp: requests.Response) -> None:
    """Track the provider's own accounting — authoritative over local counting."""
    with _quota_lock:
        for header, key in (
            ("x-ratelimit-requests-limit", "limit"),
            ("x-ratelimit-requests-remaining", "remaining"),
        ):
            raw = resp.headers.get(header)
            if raw is not None:
                try:
                    _quota[key] = int(raw)
                except ValueError:
                    pass


def _check_quota() -> None:
    with _quota_lock:
        remaining = _quota["remaining"]
    if remaining is not None and remaining <= QUOTA_RESERVE:
        raise QuotaExhaustedError(
            f"Daily football-data quota nearly exhausted ({remaining} requests left). "
            "Cached matches remain available; the quota resets at midnight UTC."
        )


def _headers() -> dict[str, str]:
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise FootballAPIError(
            f"Missing API key. Set the {API_KEY_ENV_VAR} environment variable."
        )
    return {"x-apisports-key": api_key}


def _raise_for_payload_errors(endpoint: str, errors: Any) -> None:
    """Translate the provider's in-body error object into typed exceptions.

    `errors` is `[]` on success but a dict like {"plan": "..."} on failure,
    delivered with HTTP 200 — so it must be inspected explicitly.
    """
    if not errors:
        return
    if isinstance(errors, dict):
        plan_msg = errors.get("plan")
        if plan_msg:
            if "season" in plan_msg.lower():
                raise SeasonUnavailableError(plan_msg)
            raise FootballAPIError(plan_msg)
        rate_msg = errors.get("rateLimit") or errors.get("requests")
        if rate_msg:
            raise QuotaExhaustedError(str(rate_msg))
    raise FootballAPIError(f"Provider error on {endpoint}: {errors}")


def _get(endpoint: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Perform a GET against the provider and unwrap the standard envelope.

    API-Football wraps every payload as:
        {"get": ..., "parameters": ..., "errors": [...], "results": N, "response": [...]}
    """
    _check_quota()
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(
            url, headers=_headers(), params=params or {}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FootballAPIError(f"Request to {endpoint} failed: {exc}") from exc

    _record_quota(resp)
    payload = resp.json()
    _raise_for_payload_errors(endpoint, payload.get("errors"))
    logger.debug(
        "%s %s -> %s results (quota left: %s)",
        endpoint, params, payload.get("results"), _quota["remaining"],
    )
    return payload


def get_fixture(fixture_id: int) -> dict[str, Any]:
    """Fetch a single fixture (match) with teams, goals, and status."""
    payload = _get("fixtures", {"id": fixture_id})
    response = payload.get("response", [])
    if not response:
        raise FootballAPIError(f"No fixture found with id {fixture_id}")
    return response[0]


def get_fixture_events(fixture_id: int) -> list[dict[str, Any]]:
    """Fetch match events (goals, cards, substitutions) for a fixture."""
    return _get("fixtures/events", {"fixture": fixture_id}).get("response", [])


def get_fixture_statistics(fixture_id: int) -> list[dict[str, Any]]:
    """Fetch team-level statistics (possession, shots, passes) for a fixture."""
    return _get("fixtures/statistics", {"fixture": fixture_id}).get("response", [])


def get_fixture_player_stats(fixture_id: int) -> list[dict[str, Any]]:
    """Fetch per-player statistics for both squads in a fixture."""
    return _get("fixtures/players", {"fixture": fixture_id}).get("response", [])


def get_player_season_stats(player_id: int, season: int) -> dict[str, Any]:
    """Fetch a player's aggregated statistics for a season."""
    payload = _get("players", {"id": player_id, "season": season})
    response = payload.get("response", [])
    if not response:
        raise FootballAPIError(f"No player found with id {player_id} for season {season}")
    return response[0]


def get_recent_fixtures(league_id: int, season: int, last: int = 10) -> list[dict[str, Any]]:
    """Fetch the most recent fixtures for a league/season (dashboard listing)."""
    return _get(
        "fixtures", {"league": league_id, "season": season, "last": last}
    ).get("response", [])


def get_head_to_head(team1_id: int, team2_id: int) -> list[dict[str, Any]]:
    """Fetch all head-to-head fixtures between two teams."""
    return _get("fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}"}).get(
        "response", []
    )


def get_team_season_fixtures(team_id: int, season: int) -> list[dict[str, Any]]:
    """Fetch all of a team's fixtures for one season."""
    return _get("fixtures", {"team": team_id, "season": season}).get("response", [])


# ---------------------------------------------------------------------------
# Dynamic discovery — used by the universal search engine
# ---------------------------------------------------------------------------

def search_teams_api(query: str) -> list[dict[str, Any]]:
    """Look up teams by name across the provider's whole database.

    Used only when the local registry has no match, so common searches stay
    free and instant.
    """
    if len(query.strip()) < 3:
        # The provider rejects very short search terms outright.
        return []
    return _get("teams", {"search": query.strip()}).get("response", [])


def get_league_fixtures(
    league_id: int, season: int, **filters: Any
) -> list[dict[str, Any]]:
    """Fetch fixtures for a competition/season.

    API-Football returns a whole league-season in one payload (no cursor), so
    there is no page loop to run; optional `filters` narrow it server-side
    (e.g. round="Regular Season - 38", team=42, from_/to for date ranges).
    """
    params: dict[str, Any] = {"league": league_id, "season": season}
    for key, value in filters.items():
        if value is None:
            continue
        # `from`/`to` are Python-reserved-ish; accept trailing-underscore forms.
        params[key.rstrip("_")] = value
    return _get("fixtures", params).get("response", [])


def get_team_league_fixtures(
    team_id: int, league_id: int, season: int
) -> list[dict[str, Any]]:
    """Fetch one team's fixtures within a specific competition and season."""
    return _get(
        "fixtures", {"team": team_id, "league": league_id, "season": season}
    ).get("response", [])


def build_match_context(fixture_id: int) -> dict[str, Any]:
    """Assemble everything the AI summarizer needs for one match in one call.

    Bundles fixture info, events, team stats, and player stats so the
    summarizer receives a single self-contained context dict.
    """
    return {
        "fixture": get_fixture(fixture_id),
        "events": get_fixture_events(fixture_id),
        "team_statistics": get_fixture_statistics(fixture_id),
        "player_statistics": get_fixture_player_stats(fixture_id),
    }
