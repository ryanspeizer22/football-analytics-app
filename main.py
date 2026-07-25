"""
main.py
-------
FastAPI entry point for the football analytics summary app.

Run locally:
    pip install -r requirements.txt
    cp .env.example .env               # then fill in your API keys
    uvicorn main:app --reload

Routes:
    GET  /                          Dashboard (Jinja2 template)
    GET  /api/match/{fixture_id}/summary   Three-layer AI match summary (JSON)
    GET  /api/player/{player_id}/stats     Player season stats + AI notes (JSON)
    GET  /api/premium/tactical/{fixture_id}  Premium tactical tier (placeholder)
"""

from dotenv import load_dotenv

# Load .env before importing services — the Anthropic client reads
# ANTHROPIC_API_KEY from the environment when the module is imported.
load_dotenv()

import logging
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from services import ai_summarizer, api_service, rate_limit, summary_cache, teams
from services.trending import TRENDING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("pitchsense")

app = FastAPI(
    title="Football Analytics Summary",
    description="AI-powered match summaries, team breakdowns, and player notes.",
    version="0.1.0",
)

templates = Jinja2Templates(directory="templates")


def _client_id(request: Request) -> str:
    """Identify the caller for rate limiting.

    Uses the direct peer address. X-Forwarded-For is deliberately ignored:
    without a trusted-proxy allowlist any client could spoof it and bypass
    the per-client limit. Behind a real proxy, configure uvicorn's
    --proxy-headers/--forwarded-allow-ips so request.client is correct.
    """
    return request.client.host if request.client else "unknown"


@contextmanager
def _paid_call(request: Request, label: str):
    """Reserve a slot for a billable model call, surfacing a breach as HTTP 429.

    Wrap only the generation itself — cache hits must never enter this block,
    or free reads would consume the budget.
    """
    try:
        with rate_limit.guard(_client_id(request), label):
            yield
    except rate_limit.RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.message,
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main dashboard shell; data loads client-side from the API."""
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# API — match summaries
# ---------------------------------------------------------------------------

# Fixture status codes that mean the match is over and its summary is final.
FINISHED_STATUSES = {"FT", "AET", "PEN"}

# API-Football free tier only serves seasons 2021-2023.
SUPPORTED_SEASONS = range(2021, 2024)


def _fixture_card(f: dict) -> dict:
    """Map a raw API-Football fixture to the compact card shape the UI renders."""
    home, away = f["teams"]["home"], f["teams"]["away"]
    home_color, away_color = teams.distinct_colors(home["id"], away["id"])
    return {
        "fixture_id": f["fixture"]["id"],
        "date": f["fixture"]["date"][:10],
        "competition": f["league"]["name"],
        "home": {
            "id": home["id"],
            "name": teams.display_name(home["id"], home["name"]),
            "full_name": home["name"],
            "score": f["goals"]["home"],
            "color": home_color,
        },
        "away": {
            "id": away["id"],
            "name": teams.display_name(away["id"], away["name"]),
            "full_name": away["name"],
            "score": f["goals"]["away"],
            "color": away_color,
        },
    }


def _summarizable(f: dict) -> bool:
    """Finished and inside the free-tier season window."""
    return (
        f["fixture"]["status"]["short"] in FINISHED_STATUSES
        and f["league"]["season"] in SUPPORTED_SEASONS
    )


# ---------------------------------------------------------------------------
# API — search & fixture discovery
# ---------------------------------------------------------------------------

@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=80)):
    """Autocomplete: match team names/aliases locally; understands 'A vs B'."""
    return teams.parse_query(q)


@app.get("/api/trending")
def trending():
    """Pre-seeded legendary fixtures for the homepage grid."""
    return {"fixtures": TRENDING}


@app.get("/api/fixtures")
def fixtures(team1: int = Query(...), team2: int = Query(None)):
    """List summarizable fixtures: head-to-head if two teams, else recent
    matches for one team. Responses are cached to protect the API quota."""
    cache_key = f"fixtures-{team1}-{team2}" if team2 else f"fixtures-{team1}"
    cached = summary_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        if team2:
            raw = api_service.get_head_to_head(team1, team2)
        else:
            # Most recent supported season's fixtures for the team.
            raw = api_service.get_team_season_fixtures(team1, SUPPORTED_SEASONS[-1])
    except api_service.FootballAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    cards = [_fixture_card(f) for f in raw if _summarizable(f)]
    cards.sort(key=lambda c: c["date"], reverse=True)
    result = {"fixtures": cards[:20]}
    summary_cache.set(cache_key, result)
    return result


@app.get("/api/match/{fixture_id}/summary")
def match_summary(request: Request, fixture_id: int, refresh: bool = False):
    """Return the three-layer AI summary for a fixture.

    Layer 1: tldr — quick TL;DR
    Layer 2: team_breakdowns — per-team narrative + key stats
    Layer 3: player_notes — player-by-player performance notes

    Finished matches are cached (memory + disk); pass ?refresh=true to
    regenerate. Only cache misses consume the rate-limit budget.
    """
    cache_key = f"match-{fixture_id}"
    if not refresh:
        cached = summary_cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        context = api_service.build_match_context(fixture_id)
    except api_service.FootballAPIError as exc:
        logger.warning("Fixture %s unavailable: %s", fixture_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    with _paid_call(request, f"match summary {fixture_id}"):
        logger.info("Generating summary for fixture %s", fixture_id)
        try:
            summary = ai_summarizer.generate_match_summary(context)
        except RuntimeError as exc:
            logger.error("Summary generation failed for %s: %s", fixture_id, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    result = summary.model_dump()

    # Only cache summaries of finished matches — an in-play summary goes
    # stale the moment the next goal is scored.
    status = context["fixture"].get("fixture", {}).get("status", {}).get("short")
    if status in FINISHED_STATUSES:
        summary_cache.set(cache_key, result)
    else:
        logger.info("Fixture %s not final (%s) — not cached", fixture_id, status)

    return result


# ---------------------------------------------------------------------------
# API — player stats
# ---------------------------------------------------------------------------

@app.get("/api/player/{player_id}/stats")
def player_stats(
    request: Request,
    player_id: int,
    season: int = Query(..., ge=2000, le=2100),
    refresh: bool = False,
):
    """Return raw season stats plus an AI scouting note for one player.

    A completed season's stats don't change, so the whole response is cached
    the same way match summaries are; only cache misses cost a model call.
    """
    cache_key = f"player-{player_id}-{season}"
    if not refresh:
        cached = summary_cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        stats = api_service.get_player_season_stats(player_id, season)
    except api_service.FootballAPIError as exc:
        logger.warning("Player %s (%s) unavailable: %s", player_id, season, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    notes = None
    note_error = None
    with _paid_call(request, f"player notes {player_id}"):
        logger.info("Generating scouting note for player %s (%s)", player_id, season)
        try:
            notes = ai_summarizer.generate_player_notes(stats)
        except Exception as exc:
            # The stats alone are still useful, so degrade rather than fail —
            # but never silently: log it and tell the client what happened.
            note_error = str(exc)
            logger.exception(
                "Scouting note failed for player %s (%s)", player_id, season
            )

    result = {
        "player_id": player_id,
        "season": season,
        "stats": stats,
        "ai_notes": notes,
        "ai_notes_error": note_error,
    }

    # Don't cache a response whose AI half failed — the next call should retry.
    if notes is not None:
        summary_cache.set(cache_key, result)

    return result


# ---------------------------------------------------------------------------
# API — premium tactical tier (placeholder)
# ---------------------------------------------------------------------------

@app.get("/api/premium/tactical/{fixture_id}")
def premium_tactical(fixture_id: int):
    """Placeholder for the premium tactical-analysis tier.

    Planned: formation/shape analysis, pressing patterns, buildup structure,
    turning points — gated behind a subscription check (add auth middleware
    here before launch).
    """
    return {
        "fixture_id": fixture_id,
        "tier": "premium",
        "status": "coming_soon",
        "message": "Deep tactical analysis is part of the upcoming premium tier.",
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness plus the current AI-spend budget, for monitoring."""
    return {"status": "ok", "budget": rate_limit.snapshot()}
