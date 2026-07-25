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

import os
from typing import Any, Optional

import requests

API_BASE_URL = "https://v3.football.api-sports.io"
API_KEY_ENV_VAR = "FOOTBALL_API_KEY"
REQUEST_TIMEOUT_SECONDS = 15


class FootballAPIError(Exception):
    """Raised when the football data provider returns an error or is unreachable."""


def _headers() -> dict[str, str]:
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise FootballAPIError(
            f"Missing API key. Set the {API_KEY_ENV_VAR} environment variable."
        )
    return {"x-apisports-key": api_key}


def _get(endpoint: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Perform a GET against the provider and unwrap the standard envelope.

    API-Football wraps every payload as:
        {"get": ..., "parameters": ..., "errors": [...], "results": N, "response": [...]}
    """
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(
            url, headers=_headers(), params=params or {}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FootballAPIError(f"Request to {endpoint} failed: {exc}") from exc

    payload = resp.json()
    if payload.get("errors"):
        raise FootballAPIError(f"Provider error on {endpoint}: {payload['errors']}")
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
