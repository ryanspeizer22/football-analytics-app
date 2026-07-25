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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from services import ai_summarizer, api_service, summary_cache

app = FastAPI(
    title="Football Analytics Summary",
    description="AI-powered match summaries, team breakdowns, and player notes.",
    version="0.1.0",
)

templates = Jinja2Templates(directory="templates")


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


@app.get("/api/match/{fixture_id}/summary")
def match_summary(fixture_id: int, refresh: bool = False):
    """Return the three-layer AI summary for a fixture.

    Layer 1: tldr — quick TL;DR
    Layer 2: team_breakdowns — per-team narrative + key stats
    Layer 3: player_notes — player-by-player performance notes

    Finished matches are cached (memory + disk); pass ?refresh=true to
    regenerate.
    """
    cache_key = f"match-{fixture_id}"
    if not refresh:
        cached = summary_cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        context = api_service.build_match_context(fixture_id)
    except api_service.FootballAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        summary = ai_summarizer.generate_match_summary(context)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    result = summary.model_dump()

    # Only cache summaries of finished matches — an in-play summary goes
    # stale the moment the next goal is scored.
    status = context["fixture"].get("fixture", {}).get("status", {}).get("short")
    if status in FINISHED_STATUSES:
        summary_cache.set(cache_key, result)

    return result


# ---------------------------------------------------------------------------
# API — player stats
# ---------------------------------------------------------------------------

@app.get("/api/player/{player_id}/stats")
def player_stats(player_id: int, season: int = Query(..., ge=2000, le=2100)):
    """Return raw season stats plus an AI scouting note for one player."""
    try:
        stats = api_service.get_player_season_stats(player_id, season)
    except api_service.FootballAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        notes = ai_summarizer.generate_player_notes(stats)
    except RuntimeError as exc:
        notes = None  # stats are still useful even if the AI note fails

    return {"player_id": player_id, "season": season, "stats": stats, "ai_notes": notes}


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
    return {"status": "ok"}
