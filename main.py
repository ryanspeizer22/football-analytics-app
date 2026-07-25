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
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services import (
    ai_summarizer,
    api_service,
    competitions,
    enrich,
    momentum,
    rate_limit,
    stats,
    summary_cache,
    teams,
)
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
app.mount("/static", StaticFiles(directory="static"), name="static")

# Player headshots are proxied rather than hotlinked: serving them same-origin
# keeps them out of reach of third-party-media blockers, and the on-disk cache
# means a given player is fetched from the provider CDN at most once.
PHOTO_CACHE_DIR = Path(".cache/photos")
PHOTO_CDN = "https://media.api-sports.io/football/players/{player_id}.png"


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


def _season_window() -> range:
    """Seasons this subscription can actually read (inclusive of both ends)."""
    lo, hi = competitions.plan_window()
    return range(lo, hi + 1)


def _api_error(exc: Exception) -> HTTPException:
    """Map a provider failure onto an HTTP response the UI can explain.

    A blocked season and a spent quota are ordinary, recoverable states — they
    get their own status codes so the frontend can say something useful rather
    than showing a generic failure.
    """
    if isinstance(exc, api_service.SeasonUnavailableError):
        lo, hi = competitions.plan_window()
        return HTTPException(
            status_code=409,
            detail=(
                f"That season isn't available on the current API-Football plan. "
                f"Seasons {lo}–{hi} are covered. ({exc})"
            ),
        )
    if isinstance(exc, api_service.QuotaExhaustedError):
        return HTTPException(status_code=429, detail=str(exc),
                             headers={"Retry-After": "3600"})
    return HTTPException(status_code=502, detail=str(exc))


def _fixture_card(f: dict) -> dict:
    """Map a raw API-Football fixture to the compact card shape the UI renders."""
    home, away = f["teams"]["home"], f["teams"]["away"]
    # Adopt unknown clubs so dynamically discovered teams get a real palette
    # instead of all rendering in the same fallback gray.
    teams.ensure_team(home["id"], home["name"], home.get("logo"))
    teams.ensure_team(away["id"], away["name"], away.get("logo"))
    home_color, away_color = teams.distinct_colors(home["id"], away["id"])
    return {
        "fixture_id": f["fixture"]["id"],
        "date": f["fixture"]["date"][:10],
        "competition": f["league"]["name"],
        "home": {
            "id": home["id"],
            "name": teams.display_name(home["id"], home["name"]),
            "full_name": home["name"],
            "short": (teams.get_team(home["id"]) or {}).get("short", ""),
            "score": f["goals"]["home"],
            "color": home_color,
        },
        "away": {
            "id": away["id"],
            "name": teams.display_name(away["id"], away["name"]),
            "full_name": away["name"],
            "short": (teams.get_team(away["id"]) or {}).get("short", ""),
            "score": f["goals"]["away"],
            "color": away_color,
        },
    }


def _use_photo_proxy(summary: dict) -> dict:
    """Point player photos at the same-origin proxy.

    Applied on the way out so summaries cached before the proxy existed pick
    it up too, rather than needing a paid regeneration.
    """
    for note in summary.get("player_notes") or []:
        pid = note.get("player_id")
        if isinstance(pid, int) and pid > 0:
            note["photo"] = f"/api/player-photo/{pid}"
    return summary


def _summarizable(f: dict) -> bool:
    """Finished, and in a season this subscription can actually fetch."""
    return (
        f["fixture"]["status"]["short"] in FINISHED_STATUSES
        and f["league"]["season"] in _season_window()
    )


# ---------------------------------------------------------------------------
# API — search & fixture discovery
# ---------------------------------------------------------------------------

@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=80)):
    """Universal autocomplete across teams, matchups, and competitions.

    Teams resolve against the curated registry first and fall back to a live
    provider lookup, so any club in a covered competition is reachable.
    """
    result = teams.parse_query(q)
    result["competitions"] = competitions.search(q)
    return result


@app.get("/api/competitions")
def list_competitions():
    """Every tracked competition, annotated with the seasons this plan can read."""
    lo, hi = competitions.plan_window()
    return {
        "competitions": competitions.public_list(),
        "plan_window": {"min_season": lo, "max_season": hi},
        "default_season": competitions.default_season(),
    }


@app.get("/api/competitions/{competition_id}/fixtures")
def competition_fixtures(
    competition_id: int,
    season: int = Query(None),
    team: int = Query(None),
):
    """Fixtures for a competition/season, resolved live and cached.

    This is the core of dynamic ingestion: any tracked competition can be
    browsed on demand without a hardcoded fixture list.
    """
    comp = competitions.get_competition(competition_id)
    if comp is None:
        raise HTTPException(status_code=404, detail="Unknown competition.")

    season = season or competitions.default_season(competition_id)
    reachable = competitions.accessible_seasons(competition_id)
    if reachable and season not in reachable:
        lo, hi = competitions.plan_window()
        raise HTTPException(
            status_code=409,
            detail=(
                f"{comp['name']} {season}/{str(season + 1)[-2:]} isn't on the current "
                f"API-Football plan. Available: {', '.join(map(str, reachable))} "
                f"(plan covers {lo}–{hi})."
            ),
        )

    cache_key = f"comp-{competition_id}-{season}" + (f"-t{team}" if team else "")
    cached = summary_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        if team:
            raw = api_service.get_team_league_fixtures(team, competition_id, season)
        else:
            raw = api_service.get_league_fixtures(competition_id, season)
    except api_service.FootballAPIError as exc:
        logger.warning("Competition %s/%s fetch failed: %s", competition_id, season, exc)
        raise _api_error(exc) from exc

    cards = [_fixture_card(f) for f in raw if _summarizable(f)]
    cards.sort(key=lambda c: c["date"], reverse=True)
    result = {
        "competition": {
            "id": comp["id"], "name": comp["name"], "emoji": comp["emoji"],
        },
        "season": season,
        "available_seasons": reachable,
        "count": len(cards),
        "fixtures": cards[:60],
    }
    summary_cache.set(cache_key, result)
    return result


@app.get("/api/player-photo/{player_id}")
def player_photo(player_id: int):
    """Serve a player headshot same-origin, caching it on first request.

    Hotlinking the provider CDN made rendering dependent on a third-party
    host that privacy extensions commonly block; proxying removes that
    failure mode and costs no football-API quota (the media CDN is separate).
    """
    if player_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid player id.")

    cached = PHOTO_CACHE_DIR / f"{player_id}.png"
    if cached.exists():
        return Response(
            content=cached.read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    try:
        resp = requests.get(PHOTO_CDN.format(player_id=player_id), timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.info("Headshot unavailable for player %s: %s", player_id, exc)
        raise HTTPException(status_code=404, detail="Headshot unavailable.") from exc

    if not resp.headers.get("content-type", "").startswith("image/"):
        raise HTTPException(status_code=404, detail="Headshot unavailable.")

    PHOTO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(resp.content)
    return Response(
        content=resp.content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/api/trending")
def trending():
    """Pre-seeded legendary fixtures for the homepage grid."""
    return {"fixtures": TRENDING}


@app.get("/api/fixtures")
def fixtures(
    team1: int = Query(...),
    team2: int = Query(None),
    season: int = Query(None),
):
    """List summarizable fixtures: head-to-head if two teams, else a team's
    season. Responses are cached to protect the provider quota."""
    season = season or competitions.default_season()
    cache_key = (
        f"fixtures-{team1}-{team2}" if team2 else f"fixtures-{team1}-s{season}"
    )
    cached = summary_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        if team2:
            raw = api_service.get_head_to_head(team1, team2)
        else:
            raw = api_service.get_team_season_fixtures(team1, season)
    except api_service.FootballAPIError as exc:
        logger.warning("Fixture lookup failed (team=%s): %s", team1, exc)
        raise _api_error(exc) from exc

    cards = [_fixture_card(f) for f in raw if _summarizable(f)]
    cards.sort(key=lambda c: c["date"], reverse=True)
    result = {"fixtures": cards[:40], "season": season}
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
            return _use_photo_proxy(cached)

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

    # Attach real headshots/ratings, and derive the momentum timeline from the
    # timestamped events. Both are enrichment: a failure here must not lose the
    # analysis we just paid to generate.
    try:
        result["player_notes"] = enrich.enrich_player_notes(
            result["player_notes"], context
        )
    except Exception:
        logger.exception("Player enrichment failed for fixture %s", fixture_id)

    try:
        fixture_block = context["fixture"]["teams"]
        result["momentum"] = momentum.build_timeline(
            context["events"],
            fixture_block["home"]["id"],
            fixture_block["away"]["id"],
        )
        result["momentum"]["caption"] = momentum.summarize_shift(result["momentum"])
    except Exception:
        logger.exception("Momentum build failed for fixture %s", fixture_id)

    # Stat deltas come straight from the provider's totals, never from prose.
    try:
        fixture_block = context["fixture"]["teams"]
        comparisons = stats.build_comparisons(
            context["team_statistics"],
            fixture_block["home"]["id"],
            fixture_block["away"]["id"],
        )
        result["stat_comparisons"] = comparisons
        result["headline_metrics"] = stats.headline_metrics(comparisons)
    except Exception:
        logger.exception("Stat comparison failed for fixture %s", fixture_id)

    # Only cache summaries of finished matches — an in-play summary goes
    # stale the moment the next goal is scored.
    status = context["fixture"].get("fixture", {}).get("status", {}).get("short")
    if status in FINISHED_STATUSES:
        summary_cache.set(cache_key, result)
    else:
        logger.info("Fixture %s not final (%s) — not cached", fixture_id, status)

    return _use_photo_proxy(result)


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
    """Liveness plus AI-spend budget and provider quota, for monitoring."""
    lo, hi = competitions.plan_window()
    return {
        "status": "ok",
        "budget": rate_limit.snapshot(),
        "football_api_quota": api_service.quota_snapshot(),
        "seasons": {"min": lo, "max": hi, "default": competitions.default_season()},
    }
