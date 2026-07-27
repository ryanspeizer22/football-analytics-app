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

import hashlib
import json
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services import (
    ai_summarizer,
    api_service,
    competitions,
    enrich,
    highlights,
    leaderboard,
    momentum,
    og_image,
    player_profile,
    prematch,
    rate_limit,
    slugs,
    squads,
    stats,
    cache_migrate,
    summary_cache,
    teams,
    textclean,
    youtube,
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

@app.on_event("startup")
def _preflight() -> None:
    """Shout about configurations that are quietly expensive or unsafe.

    Each of these is silent at runtime and only shows up on the bill or in an
    incident, so they are checked once, out loud, at boot.
    """
    is_production = BASE_URL and not any(
        h in BASE_URL for h in ("127.0.0.1", "localhost"))

    if os.environ.get("FORWARDED_ALLOW_IPS", "").strip() == "*":
        logger.error(
            "FORWARDED_ALLOW_IPS='*' — uvicorn will trust X-Forwarded-For from "
            "anyone, so request.client is caller-controlled and the per-client "
            "rate limit can be bypassed by rotating the header. Set this to the "
            "platform's proxy address.")

    if is_production:
        if rate_limit._bypass_enabled():
            logger.error(
                "Local rate-limit bypass is ON with a public base URL (%s). "
                "Unset RATE_LIMIT_BYPASS_LOCAL unless this is deliberate.", BASE_URL)
        for key in ("ANTHROPIC_API_KEY", "FOOTBALL_API_KEY"):
            if not os.environ.get(key):
                logger.error("%s is not set — the app will fail on first use.", key)
        if not os.environ.get("ADMIN_TOKEN"):
            logger.info("ADMIN_TOKEN unset: founder access by token is disabled.")
    else:
        logger.info("Local mode: rate-limit bypass %s, base URL %s",
                    "on" if rate_limit._bypass_enabled() else "off", BASE_URL)

    cache_root = PHOTO_CACHE_DIR.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    if is_production and not os.path.ismount(str(cache_root)):
        logger.warning(
            "%s is not a mount point. Without a persistent volume there, every "
            "deploy discards paid analyses and cached provider data.", cache_root)


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Provider media (headshots, club crests, competition logos) is proxied rather
# than hotlinked: serving it same-origin keeps it out of reach of third-party-
# media blockers, and the on-disk cache means a given asset is fetched from the
# provider CDN at most once. None of it costs football-API quota — the media
# CDN is a separate host from the data API.
PHOTO_CACHE_DIR = Path(".cache/photos")
PHOTO_CDN = "https://media.api-sports.io/football/players/{player_id}.png"

# The CDN addresses every asset by id under a per-kind path, so a crest or a
# competition logo needs no stored URL — only the id we already have.
_MEDIA_CDN = "https://media.api-sports.io/football/{kind}/{asset_id}.png"
_MEDIA_KINDS = {"players": "photos", "teams": "crests",
                "leagues": "competitions"}


def _client_id(request: Request) -> str:
    """Identify the caller for rate limiting.

    Uses the direct peer address. X-Forwarded-For is deliberately ignored:
    without a trusted-proxy allowlist any client could spoof it and bypass
    the per-client limit. Behind a real proxy, configure uvicorn's
    --proxy-headers/--forwarded-allow-ips so request.client is correct.
    """
    return request.client.host if request.client else "unknown"


# Kept beside the client-side constant of the same value; both describe the
# same allowance and should move together.
FREE_FRESH_ANALYSES = 5


def _admin_token(request: Request) -> Optional[str]:
    return request.headers.get(rate_limit.ADMIN_HEADER)


def _is_founder(request: Request) -> bool:
    return rate_limit.is_founder(_client_id(request), _admin_token(request))


@app.get("/api/session")
def session_state(request: Request):
    """Who the caller is, as far as quotas are concerned.

    The client asks once and shows unlimited access rather than a countdown
    when this says so. Deliberately server-answered: letting the page decide it
    was a founder would make the whole thing a client-side claim anyone could
    set for themselves.
    """
    token = _admin_token(request)
    # Token first: a loopback caller with a valid token was admitted by the
    # token, and reporting "localhost" there would misdescribe how access was
    # granted — which matters when checking that token auth actually works.
    by_token = rate_limit.admin_token_valid(token)
    founder = by_token or _is_founder(request)
    return {
        "founder": founder,
        "reason": "token" if by_token else ("localhost" if founder else None),
        "free_fresh_analyses": None if founder else FREE_FRESH_ANALYSES,
    }


@contextmanager
def _paid_call(request: Request, label: str):
    """Reserve a slot for a billable model call, surfacing a breach as HTTP 429.

    Wrap only the generation itself — cache hits must never enter this block,
    or free reads would consume the budget.
    """
    try:
        with rate_limit.guard(_client_id(request), label,
                              admin_token=_admin_token(request)):
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

# Absolute URLs are required for canonical links and OpenGraph images —
# crawlers and link unfurlers won't resolve relative paths. Set this to the
# real origin once a domain is pointed at the app.
BASE_URL = os.environ.get("PITCHSENSE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main dashboard shell; data loads client-side from the API."""
    return templates.TemplateResponse(request, "index.html", {"base_url": BASE_URL})


def _shell(request: Request, *, title: str, description: str,
           path: str, og_image: str, boot: Optional[dict] = None) -> HTMLResponse:
    """Render the SPA shell with route-specific metadata.

    Link unfurlers and crawlers never execute the page's JavaScript, so the
    title/description/og:image for a shared URL have to be in the HTML that
    comes back from the server — a client-side router alone would leave every
    shared link showing the generic homepage card.
    """
    return templates.TemplateResponse(request, "index.html", {
        "base_url": BASE_URL,
        "meta_title": title,
        "meta_description": description,
        "canonical_path": path,
        "og_image": og_image,
        # Handed to the client so the view opens directly, without a round
        # trip to work out what the URL meant. Rendered with `| safe`, because
        # Jinja's HTML autoescaping would turn the JSON quotes into &#34; and
        # break the whole script — so `<` is neutralised here instead, which
        # is what actually matters inside a <script> block.
        "boot_state": json.dumps(boot).replace("<", "\\u003c") if boot else None,
    })


@app.get("/match/{slug}", response_class=HTMLResponse)
def match_page(request: Request, slug: str):
    """Deep link to one match analysis."""
    fixture_id = slugs.trailing_id(slug)
    if fixture_id is None:
        raise HTTPException(status_code=404, detail="Unknown match link.")

    title, description = "Match analysis — PitchSense", "AI football match intelligence."
    meta = summary_cache.get(f"meta-{fixture_id}")
    if meta:
        title = f"{meta['home']} {meta['home_score']}–{meta['away_score']} {meta['away']} — PitchSense"
        description = meta.get("headline") or f"{meta['competition']}, {meta['date']}."

    return _shell(
        request, title=title, description=description,
        path=f"/match/{slug}", og_image=f"{BASE_URL}/og/match/{fixture_id}.png",
        boot={"view": "match", "fixture_id": fixture_id},
    )


@app.get("/compare/{slug}", response_class=HTMLResponse)
def compare_page(request: Request, slug: str):
    """Deep link to a head-to-head comparison."""
    pair = slugs.parse_compare(slug)
    if pair is None:
        raise HTTPException(status_code=404, detail="Unknown comparison link.")
    a_id, b_id = pair
    return _shell(
        request,
        title="Head-to-head player comparison — PitchSense",
        description="Compare two players' seasons side by side: ratings, goals, key passes and more.",
        path=f"/compare/{slug}",
        og_image=f"{BASE_URL}/og/compare/{a_id}-{b_id}.png",
        boot={"view": "compare", "players": [a_id, b_id]},
    )


@app.get("/players", response_class=HTMLResponse)
def players_page(request: Request):
    """Deep link to the player and team directory."""
    return _shell(
        request,
        title="Player & team search — PitchSense",
        description=("Search any club or player: season ratings, goals, assists, "
                     "current club and every analysed match they featured in."),
        path="/players",
        og_image=f"{BASE_URL}/og/leaderboard/39-2025.png",
        boot={"view": "players"},
    )


@app.get("/leaderboard/{slug}", response_class=HTMLResponse)
def leaderboard_page(request: Request, slug: str, metric: str = Query("rating")):
    parsed = slugs.parse_leaderboard(slug)
    if parsed is None:
        raise HTTPException(status_code=404, detail="Unknown leaderboard link.")
    league_id, season = parsed
    comp = competitions.get_competition(league_id)
    name = comp["name"] if comp else "League"
    return _shell(
        request,
        title=f"{name} {season}/{str(season + 1)[-2:]} player rankings — PitchSense",
        description=f"Every player in {name} {season}/{str(season + 1)[-2:]}, ranked by rating, goals, key passes and defensive output.",
        path=f"/leaderboard/{slug}",
        og_image=f"{BASE_URL}/og/leaderboard/{league_id}-{season}.png",
        boot={"view": "leaderboard", "league": league_id, "season": season, "metric": metric},
    )


# ---------------------------------------------------------------------------
# Dynamic OpenGraph images
# ---------------------------------------------------------------------------

_OG_HEADERS = {"Cache-Control": "public, max-age=86400"}


@app.get("/og/match/{fixture_id}.png")
def og_match(fixture_id: int):
    """Branded preview card for a shared match link."""
    cache_file = PHOTO_CACHE_DIR.parent / "og" / f"match-{fixture_id}.png"
    if cache_file.exists():
        return Response(cache_file.read_bytes(), media_type="image/png", headers=_OG_HEADERS)

    meta = summary_cache.get(f"meta-{fixture_id}")
    narrative = summary_cache.get(f"narr-{fixture_id}") or {}
    if not meta:
        png = og_image.generic_card("Match analysis", "PitchSense")
    else:
        png = og_image.match_card(
            meta["home"], meta["away"], meta.get("home_score"), meta.get("away_score"),
            meta.get("competition", ""), meta.get("date", ""),
            meta.get("home_color", "#37f0a2"), meta.get("away_color", "#5078ff"),
            narrative.get("headline", ""),
        )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(png)
    return Response(png, media_type="image/png", headers=_OG_HEADERS)


@app.get("/og/compare/{a_id}-{b_id}.png")
def og_compare(a_id: int, b_id: int):
    """Branded preview card for a shared comparison link."""
    cache_file = PHOTO_CACHE_DIR.parent / "og" / f"cmp-{a_id}-{b_id}.png"
    if cache_file.exists():
        return Response(cache_file.read_bytes(), media_type="image/png", headers=_OG_HEADERS)

    season = competitions.default_season()

    def profile(pid: int):
        return summary_cache.get(f"profile-{pid}-{season}")

    a, b = profile(a_id), profile(b_id)
    if a and b:
        png = og_image.compare_card(
            a["player"].get("name", ""), a.get("average_rating"), a["player"].get("team", ""),
            b["player"].get("name", ""), b.get("average_rating"), b["player"].get("team", ""),
            season,
        )
    else:
        # Don't block the unfurl on an uncached profile — a provider round trip
        # here would make the crawler wait, and crawlers time out fast.
        png = og_image.generic_card("Head to head", "Compare two players on PitchSense")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(png)
    return Response(png, media_type="image/png", headers=_OG_HEADERS)


@app.get("/og/leaderboard/{league_id}-{season}.png")
def og_leaderboard(league_id: int, season: int):
    comp = competitions.get_competition(league_id)
    name = comp["name"] if comp else "League"
    png = og_image.generic_card(
        f"{name} {season}/{str(season + 1)[-2:]}", "Season player rankings · PitchSense"
    )
    return Response(png, media_type="image/png", headers=_OG_HEADERS)


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    """Allow the app shell, keep crawlers out of the paid API surface."""
    return (
        "User-agent: *\n"
        "Allow: /$\n"
        "Allow: /static/\n"
        "Disallow: /api/\n"
        f"\nSitemap: {BASE_URL}/sitemap.xml\n"
    )


@app.get("/sitemap.xml")
def sitemap():
    """The stable landing pages.

    Match, comparison and leaderboard URLs are unbounded and generated on
    demand, so only the root is listed.
    """
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{BASE_URL}/</loc><changefreq>daily</changefreq>"
        "<priority>1.0</priority></url>\n"
        f"  <url><loc>{BASE_URL}/players</loc><changefreq>weekly</changefreq>"
        "<priority>0.7</priority></url>\n"
        "</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml")


@app.get("/manifest.webmanifest")
def manifest():
    """PWA manifest — lets iOS/Android users install the app to the home screen."""
    return Response(
        content=json.dumps({
            "name": "PitchSense — Match Intelligence",
            "short_name": "PitchSense",
            "description": "AI football match analysis, momentum charts and player ratings.",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#07090f",
            "theme_color": "#07090f",
            "orientation": "portrait-primary",
            "icons": [
                {"src": "/static/icon.svg", "sizes": "any",
                 "type": "image/svg+xml", "purpose": "any maskable"}
            ],
        }),
        media_type="application/manifest+json",
    )


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


def _generation_failed(exc: Exception, label: str, fixture_id: int) -> HTTPException:
    """Map any failed generation onto the 502 the client already handles.

    Catching only the summarizer's own RuntimeError here meant everything else
    the call can raise — the SDK's connection, timeout, rate-limit and upstream
    5xx errors, and schema validation failures — left the handler unconverted
    and reached the browser as a 500 on a half-painted page. Upstream trouble is
    a bad gateway, which is what 502 is for and what the UI already explains.

    Only RuntimeError's message is passed through: those are written for a
    reader. The rest describe our own infrastructure, and some of them name our
    configuration, so they are logged in full and reported generically.
    """
    logger.exception("%s generation failed for fixture %s", label, fixture_id)
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(
        status_code=502,
        detail="The analysis service is temporarily unavailable. "
               "Please try again in a moment.",
    )


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
        # Addresses the competition badge on the card; without it the card
        # falls back to a bare text label.
        "competition_id": f["league"].get("id"),
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


def _with_faces(payload: dict, fixture_id: int) -> dict:
    """Restore portraits on player notes written before ids were stored.

    The oldest notes carry only a name, so there is no id for the photo proxy
    to address and every card falls back to initials. The ids are still in the
    match context on disk, so the names are resolved against it and the repaired
    payload is written back — the lookup then happens once, not per request.
    """
    notes = payload.get("player_notes") or []
    if not notes or all(isinstance(n.get("player_id"), int) for n in notes):
        return payload
    try:
        context = api_service.build_match_context(fixture_id)
    except Exception:
        logger.info("Context unavailable for %s; player cards keep initials", fixture_id)
        return payload

    repaired = cache_migrate.backfill_player_photos(notes, context)
    if repaired:
        logger.info("Restored %d player portraits for fixture %s", repaired, fixture_id)
        summary_cache.set(f"players-{fixture_id}", payload)
    return payload


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


def _stamp_competition(payload: dict) -> dict:
    """Ensure every fixture card carries the competition id for its badge.

    These payloads are cached as fully-built cards, so entries written before
    the badge existed have no competition id and would render a bare text
    label forever. The id is on the payload already, so filling it in here
    repairs them on read rather than needing the caches thrown away.
    """
    comp_id = (payload.get("competition") or {}).get("id")
    if comp_id:
        for card in payload.get("fixtures") or []:
            card.setdefault("competition_id", comp_id)
    return payload


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
        return _stamp_competition(cached)

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


_MEDIA_HEADERS = {"Cache-Control": "public, max-age=604800"}


def _cache_media(cached: Path, payload: bytes) -> None:
    """Store one fetched image, tolerating a cache that will not take it.

    The cache saves a CDN round trip, not the response — so a cache that cannot
    be written is a slow request, not a failed one. It used to be a failed one:
    an unwritable root (a persistent volume that never mounted, one owned by
    another user, or one that has filled up) raised straight out of the request
    and turned every headshot, crest and competition logo on the page into a
    500 at once. The image is already in hand by this point; serve it.
    """
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)
    except OSError as exc:
        logger.warning("Could not cache %s (%s); serving without caching",
                       cached, exc)


def _serve_media(kind: str, asset_id: int, label: str) -> Response:
    """Serve one provider image same-origin, caching it on first request.

    Hotlinking the provider CDN made rendering dependent on a third-party host
    that privacy extensions commonly block; proxying removes that failure mode.
    """
    if asset_id <= 0:
        raise HTTPException(status_code=400, detail=f"Invalid {label} id.")

    cache_dir = PHOTO_CACHE_DIR.parent / _MEDIA_KINDS[kind]
    cached = cache_dir / f"{asset_id}.png"
    if cached.exists():
        # Sniffed, not assumed: venue art is JPEG served from a .png path, and
        # labelling it image/png left the browser to guess.
        kind_hint = "image/jpeg" if cached.read_bytes()[:2] == b"\xff\xd8" else "image/png"
        return Response(cached.read_bytes(), media_type=kind_hint,
                        headers=_MEDIA_HEADERS)

    try:
        resp = requests.get(_MEDIA_CDN.format(kind=kind, asset_id=asset_id), timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.info("%s image unavailable for %s: %s", label, asset_id, exc)
        raise HTTPException(status_code=404, detail=f"{label} image unavailable.") from exc

    if not resp.headers.get("content-type", "").startswith("image/"):
        raise HTTPException(status_code=404, detail=f"{label} image unavailable.")

    _cache_media(cached, resp.content)
    return Response(resp.content, headers=_MEDIA_HEADERS,
                    media_type=resp.headers.get("content-type", "image/png"))


@app.get("/api/player-photo/{player_id}")
def player_photo(player_id: int):
    """Player headshot. Path kept as-is — it is baked into cached summaries."""
    return _serve_media("players", player_id, "Headshot")


@app.get("/api/crest/{team_id}")
def team_crest(team_id: int):
    """Official club crest."""
    return _serve_media("teams", team_id, "Crest")


# The provider returns a generic grey shield, byte-identical every time, for
# competitions it holds no artwork for — the World Cup among them. It arrives
# as a normal 200 with a valid PNG, so only the content identifies it.
_PLACEHOLDER_LOGO_MD5 = "3617b8094f9ea8c81f6d0beff671978b"
_TROPHY_EMBLEM = Path("static/trophy-emblem.svg")


# YouTube publishes several thumbnail sizes and only guarantees the small one;
# maxres is absent for older or lower-resolution uploads, so fall back in order.
_YT_THUMBS = ("maxresdefault", "sddefault", "hqdefault", "mqdefault")
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


@app.get("/api/youtube-thumb/{video_id}")
def youtube_thumb(video_id: str):
    """Poster frame for an embedded highlight video, served same-origin.

    Proxied for the same reason the rest of the media is: a blocker that eats
    third-party images would otherwise leave the panel with an empty poster.
    """
    if not _YT_ID.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid video id.")

    cache_dir = PHOTO_CACHE_DIR.parent / "youtube"
    cached = cache_dir / f"{video_id}.jpg"
    if cached.exists():
        return Response(cached.read_bytes(), media_type="image/jpeg",
                        headers=_MEDIA_HEADERS)

    for name in _YT_THUMBS:
        try:
            resp = requests.get(f"https://img.youtube.com/vi/{video_id}/{name}.jpg",
                                timeout=10)
        except requests.RequestException:
            continue
        # YouTube answers a missing size with a 120x90 grey placeholder rather
        # than a 404, so size is the only reliable signal that it is real.
        if resp.ok and resp.headers.get("content-type", "").startswith("image/") \
                and len(resp.content) > 3000:
            _cache_media(cached, resp.content)
            return Response(resp.content, media_type="image/jpeg",
                            headers=_MEDIA_HEADERS)

    raise HTTPException(status_code=404, detail="Thumbnail unavailable.")


@app.get("/api/competition-logo/{league_id}")
def competition_logo(league_id: int):
    """Official competition logo, or a typographic emblem if there isn't one.

    Never falls back to a national flag: England, Spain, Italy, Germany and
    France each run several competitions here, so a flag identifies the country
    and not the competition — which is the regression this replaced.
    """
    try:
        response = _serve_media("leagues", league_id, "Competition")
        if hashlib.md5(response.body).hexdigest() != _PLACEHOLDER_LOGO_MD5:
            return response
        logger.info("Provider has no artwork for competition %s; using trophy", league_id)
    except (HTTPException, OSError):
        # OSError as well as the 404: reading the cached copy can fail on a
        # volume that has gone away mid-flight, and the emblem below is a
        # perfectly good answer in that case too.
        logger.info("Competition logo unavailable for %s; using trophy", league_id)

    return Response(_TROPHY_EMBLEM.read_bytes(),
                    media_type="image/svg+xml", headers=_MEDIA_HEADERS)


@app.get("/api/fixtures/upcoming")
def upcoming_fixtures(league: int = Query(39), count: int = Query(8, ge=1, le=30)):
    """Next scheduled fixtures — pre-season and early-season.

    These have not kicked off, so they carry no statistics and are explicitly
    marked unanalysable; the UI lists them rather than offering a summary.
    Cached briefly: the schedule changes far more slowly than it's viewed, but
    it is not immutable the way a finished match is.
    """
    comp = competitions.get_competition(league)
    if comp is None:
        raise HTTPException(status_code=404, detail="Unknown competition.")

    # The upcoming season is the newest the plan can reach, not the completed
    # default the rest of the app opens on.
    seasons = competitions.accessible_seasons(league)
    season = seasons[-1] if seasons else competitions.default_season(league)

    cache_key = f"upcoming-{league}-{season}-{count}"
    cached = summary_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = api_service.get_upcoming_fixtures(league, season, count)
    except api_service.FootballAPIError as exc:
        logger.warning("Upcoming fetch failed for %s/%s: %s", league, season, exc)
        raise _api_error(exc) from exc

    fixtures_out = []
    for f in raw:
        card = _fixture_card(f)
        card["kickoff"] = f["fixture"]["date"]
        card["status"] = f["fixture"]["status"]["short"]
        card["venue"] = (f["fixture"].get("venue") or {}).get("name")
        card["analysable"] = False  # no stats exist until it's played
        card["home"]["score"] = None
        card["away"]["score"] = None
        fixtures_out.append(card)

    result = {
        "competition": {"id": comp["id"], "name": comp["name"], "emoji": comp["emoji"]},
        "season": season,
        "count": len(fixtures_out),
        "fixtures": fixtures_out,
    }
    summary_cache.set(cache_key, result)
    return result


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


def _store_match_meta(context: dict, fixture_id: int) -> None:
    """Persist the few fields an OG card needs.

    Kept separate from the summary caches so rendering a preview image never
    depends on a generated analysis existing — a link can be shared the moment
    a match is opened.
    """
    try:
        fx = context["fixture"]
        home, away = fx["teams"]["home"], fx["teams"]["away"]
        summary_cache.set(f"meta-{fixture_id}", {
            "home": teams.display_name(home["id"], home["name"]),
            "away": teams.display_name(away["id"], away["name"]),
            # Ids, not URLs: they address the crest and drive the colour, so a
            # palette change reaches old entries without a cache flush.
            "home_id": home["id"],
            "away_id": away["id"],
            "home_short": (teams.get_team(home["id"]) or {}).get("short", ""),
            "away_short": (teams.get_team(away["id"]) or {}).get("short", ""),
            "home_score": (fx.get("goals") or {}).get("home"),
            "away_score": (fx.get("goals") or {}).get("away"),
            "competition": (fx.get("league") or {}).get("name", ""),
            "competition_id": (fx.get("league") or {}).get("id"),
            "venue": ((fx.get("fixture") or {}).get("venue") or {}).get("name"),
            "date": (fx.get("fixture") or {}).get("date", "")[:10],
            "slug": slugs.match_slug(
                teams.display_name(home["id"], home["name"]),
                teams.display_name(away["id"], away["name"]),
                fixture_id,
                (fx.get("fixture") or {}).get("date", "")[:10],
            ),
        })
    except Exception:
        logger.exception("Failed to store OG metadata for fixture %s", fixture_id)


def _refresh_derived(cached: dict, fixture_id: int) -> dict:
    """Recompute the derived half of a cached narrative on the way out.

    Momentum and the stat comparisons are pure functions of data we already
    hold — no model call, no provider call, since the match context is on disk.
    Serving them from the narrative cache froze them at whatever the engine did
    when the analysis was first generated: the Euro 2020 final kept reporting a
    130-minute match with ten phantom goals long after the shootout fix landed.
    The prose stays cached; only the computed parts are rebuilt.
    """
    try:
        context = api_service.build_match_context(fixture_id)
    except Exception:
        logger.info("Context unavailable for %s; serving cached derived data", fixture_id)
        return cached
    out = dict(cached)
    out["team_breakdowns"] = _home_first(out.get("team_breakdowns") or [], context)
    out.update(_derived(context, fixture_id))
    return out


def _home_first(breakdowns: list[dict], context: dict) -> list[dict]:
    """Order the team cards home-then-away, from the fixture payload.

    The model returns the two breakdowns in whatever order it wrote them. It
    has always happened to lead with the home side, but "happens to" is not an
    ordering: at a neutral venue there is no kick-off-side cue in the data to
    keep it consistent, so the pairing with the chart's home axis is enforced
    here rather than left to the prose.
    """
    teams_block = (context.get("fixture") or {}).get("teams") or {}
    home_name = ((teams_block.get("home") or {}).get("name") or "").strip().lower()
    # Split once, and test the words rather than the string: a name that is
    # blank after stripping is still truthy before it, so guarding on the
    # string alone let a whitespace-only name reach an empty [0] below.
    home_words = home_name.split()
    if not home_words or len(breakdowns) != 2:
        return breakdowns

    def is_home(entry: dict) -> bool:
        name = (entry.get("team_name") or "").strip().lower()
        words = name.split()
        # Names differ between feed and prose ("Paris Saint Germain" / "PSG"),
        # so compare on containment in either direction rather than equality.
        return bool(words) and (name in home_name or home_name in name
                                or words[0] == home_words[0])

    if is_home(breakdowns[1]) and not is_home(breakdowns[0]):
        logger.info("Team breakdowns arrived away-first; reordering to home-first")
        return [breakdowns[1], breakdowns[0]]
    return breakdowns


def _derived(context: dict, fixture_id: int) -> dict:
    """Momentum + stat deltas. Pure computation over cached data — no model,
    no provider call, so it's regenerated freely on every request."""
    out: dict = {}
    teams_block = context["fixture"]["teams"]
    home_id, away_id = teams_block["home"]["id"], teams_block["away"]["id"]
    try:
        out["momentum"] = momentum.build_timeline(
            context["events"], home_id, away_id, context.get("team_statistics")
        )
        out["momentum"]["caption"] = momentum.summarize_shift(out["momentum"])
    except Exception:
        logger.exception("Momentum build failed for fixture %s", fixture_id)
    # Free, and from the fixture itself rather than the model: the manager who
    # was actually in the dugout that day.
    out["managers"] = [
        {"team_id": (side.get("team") or {}).get("id"),
         "team": (side.get("team") or {}).get("name"),
         "manager": (side.get("coach") or {}).get("name")}
        for side in (context["fixture"].get("lineups") or [])
        if (side.get("coach") or {}).get("name")
    ]
    try:
        comparisons = stats.build_comparisons(
            context["team_statistics"], home_id, away_id
        )
        out["stat_comparisons"] = comparisons
        out["headline_metrics"] = stats.headline_metrics(comparisons)
    except Exception:
        logger.exception("Stat comparison failed for fixture %s", fixture_id)
    return out


def _with_club_colors(rows: list[dict]) -> list[dict]:
    """Attach each row's club colour on the way out.

    Applied at serve time rather than stored on the row, because these rows
    live in long-lived pool caches — baking the colour in would leave every
    cached pool serving whatever the palette looked like when it was built.
    """
    for row in rows:
        team_id = row.get("team_id")
        if team_id:
            row["team_color"] = teams.readable_color(*teams.team_colors(team_id))
    return rows


def _with_colors(meta: dict) -> dict:
    """Resolve match colours from team ids at read time.

    Colours used to be frozen into the cache when a match was first opened, so
    every entry written before a palette change kept serving the old pair —
    PSG stayed Arsenal red in every cached match. Deriving them here means the
    registry is the single source of truth and old entries correct themselves.
    Entries predating the stored ids keep whatever colours they were given.
    """
    home_id, away_id = meta.get("home_id"), meta.get("away_id")
    if home_id and away_id:
        meta = dict(meta)
        meta["home_color"], meta["away_color"] = teams.distinct_colors(home_id, away_id)
    return meta


@app.get("/api/match/{fixture_id}/meta")
def match_meta(fixture_id: int):
    """Scoreline and colours for a match, without generating anything.

    A shared link arrives with no card object, so the client needs this to
    paint the scoreboard before (or without) any analysis.
    """
    cached = summary_cache.get(f"meta-{fixture_id}")
    if cached is not None:
        return _with_colors(cached)
    try:
        context = api_service.build_match_context(fixture_id)
    except api_service.FootballAPIError as exc:
        raise _api_error(exc) from exc
    _store_match_meta(context, fixture_id)
    return _with_colors(summary_cache.get(f"meta-{fixture_id}") or {})


def _migrated(fixture_id: int, half: str) -> Optional[dict]:
    """Best cached copy of one half, brought forward to the current shape.

    Ordered newest-format first so a current entry always beats a legacy one
    for the same match. On a hit the normalised copy is written back under the
    current key, so the reshaping happens once rather than on every read.
    """
    candidates = [
        summary_cache.get(f"{'open' if half == 'opening' else 'anal'}-{fixture_id}"),
        summary_cache.get(f"narr-{fixture_id}"),   # pre-split single blob
        summary_cache.get(f"match-{fixture_id}"),  # original three-layer summary
    ]
    migrated = cache_migrate.first_usable(candidates, half)
    if migrated and candidates[0] != migrated:
        summary_cache.set(f"{'open' if half == 'opening' else 'anal'}-{fixture_id}", migrated)
        logger.info("Migrated cached %s for fixture %s to the current shape",
                    half, fixture_id)
    return migrated


def _derived_for(fixture_id: int) -> dict:
    """Momentum and stat comparisons, recomputed. No model or provider call."""
    try:
        return _derived(api_service.build_match_context(fixture_id), fixture_id)
    except Exception:
        logger.info("Context unavailable for %s; serving without derived data", fixture_id)
        return {}


def _narrative_prelude(fixture_id: int):
    """Context, stored metadata and the free derived data, for either pass."""
    try:
        context = api_service.build_match_context(fixture_id)
    except api_service.FootballAPIError as exc:
        logger.warning("Fixture %s unavailable: %s", fixture_id, exc)
        raise _api_error(exc) from exc
    _store_match_meta(context, fixture_id)
    # Derived first: the stat tiles it produces are handed to the model as the
    # figures already on screen, so the prose stops restating them.
    return context, _derived(context, fixture_id)


def _cacheable(context: dict) -> bool:
    return context["fixture"].get("fixture", {}).get("status", {}).get("short") in FINISHED_STATUSES


def _analysis_cached(fixture_id: int) -> tuple[bool, bool]:
    """(opening cached, analysis cached) without writing anything back."""
    def half(name: str) -> bool:
        key = "open" if name == "opening" else "anal"
        return cache_migrate.first_usable([
            summary_cache.get(f"{key}-{fixture_id}"),
            summary_cache.get(f"narr-{fixture_id}"),
            summary_cache.get(f"match-{fixture_id}"),
        ], name) is not None
    return half("opening"), half("analysis")


@app.get("/api/players/search")
def player_search(q: str = Query(..., min_length=2, max_length=60),
                  league: int = Query(None), season: int = Query(None)):
    """Find players by name, with the club they are at now.

    Searches the cached league pools first, which is free and instant and
    covers every player the app has already ranked. The provider's own search
    is the fallback for anyone outside those pools.
    """
    needle = q.strip().lower()
    season = season or competitions.default_season(league or 39)
    seen: dict[int, dict] = {}

    # Cached pools first — no quota, no latency.
    pool_keys = [f"pool-{league}-{season}"] if league else [
        f"pool-{c['id']}-{competitions.default_season(c['id'])}"
        for c in competitions.COMPETITIONS if c["kind"] == "domestic_league"
    ]
    for key in pool_keys:
        for row in summary_cache.get(key) or []:
            name = (row.get("name") or "").lower()
            if needle in name and row.get("id") not in seen:
                seen[row["id"]] = {
                    "id": row["id"],
                    "name": row.get("name"),
                    "photo": row.get("photo") or f"/api/player-photo/{row['id']}",
                    "season_team": row.get("team"),
                    "team_id": row.get("team_id"),
                    "position": row.get("position"),
                    "season": row.get("season"),
                    "rating": row.get("rating"),
                    "appearances": row.get("appearances"),
                    "goals": row.get("goals"),
                    "assists": row.get("assists"),
                    "source": "pool",
                }

    if not seen:
        try:
            payload = api_service._get("players/profiles", {"search": needle})
            for entry in (payload.get("response") or [])[:12]:
                info = entry.get("player") or {}
                pid = info.get("id")
                if not pid or not info.get("name"):
                    continue
                seen[pid] = {
                    "id": pid, "name": info.get("name"),
                    "photo": f"/api/player-photo/{pid}",
                    "season_team": None, "team_id": None,
                    "position": info.get("position"),
                    "nationality": info.get("nationality"),
                    "source": "provider",
                }
        except Exception as exc:
            logger.info("Provider player search failed for %r: %s", needle, exc)

    results = sorted(seen.values(),
                     key=lambda r: (-(r.get("rating") or 0), r["name"]))[:24]
    return {"query": q, "count": len(results), "players": results}


@app.get("/api/player/{player_id}/card")
def player_card(player_id: int, season: int = Query(None)):
    """Everything the board shows for one player.

    Keeps two club facts apart on purpose: `current_club` comes from the
    transfer ledger and is a statement about today, while `season_team` is who
    they played for in the season being shown. Conflating them is how a page
    ends up asserting a move that has not happened.
    """
    season = season or competitions.default_season()
    profile = summary_cache.get(f"profile-{player_id}-{season}")
    if profile is None:
        try:
            raw = api_service.get_player_season_stats(player_id, season)
            profile = player_profile.build_profile(raw, season)
            summary_cache.set(f"profile-{player_id}-{season}", profile)
        except Exception as exc:
            raise _api_error(exc) if isinstance(exc, api_service.FootballAPIError)                 else HTTPException(status_code=404, detail="Player not found.")

    appearances = _mvp_appearances(player_id)
    return {
        **profile,
        "current_club": squads.current_club(player_id),
        "transfers": squads.history(player_id),
        "mvp_appearances": appearances,
    }


def _mvp_appearances(player_id: int, limit: int = 12) -> list[dict]:
    """Matches this app has analysed where the player earned a card.

    Read straight out of the cached player notes, so it costs nothing and only
    ever reports matches that genuinely exist in the app.
    """
    found = []
    for path in sorted(Path(".cache/summaries").glob("players-*.json")):
        fixture_id = path.stem.replace("players-", "")
        payload = summary_cache.get(f"players-{fixture_id}")
        if not payload:
            continue
        for note in payload.get("player_notes") or []:
            if note.get("player_id") != player_id:
                continue
            meta = summary_cache.get(f"meta-{fixture_id}") or {}
            found.append({
                "fixture_id": int(fixture_id),
                "match": f"{meta.get('home','?')} {meta.get('home_score','')}–"
                         f"{meta.get('away_score','')} {meta.get('away','?')}".strip(),
                "date": meta.get("date"),
                "competition": meta.get("competition"),
                "slug": meta.get("slug"),
                "rating": note.get("rating"),
                "verdict": note.get("rating_comment"),
            })
            break
    found.sort(key=lambda f: f.get("date") or "", reverse=True)
    return found[:limit]


@app.get("/api/match-status")
def match_status_bulk(ids: str = Query(..., max_length=1200)):
    """Cache state for a page of fixtures at once.

    The grids show a badge per card, and one request per card would be absurd
    for a 60-match league page. Reads only in-memory and on-disk caches, so it
    stays in the low milliseconds regardless of how many ids are asked for.
    """
    wanted, seen = [], set()
    for raw in ids.split(",")[:120]:
        raw = raw.strip()
        if raw.isdigit() and raw not in seen:
            seen.add(raw)
            wanted.append(int(raw))

    out = {}
    for fixture_id in wanted:
        opening, analysis = _analysis_cached(fixture_id)
        out[str(fixture_id)] = {
            # Only the analysis decides whether opening this is charged —
            # player notes are free for everyone, so they do not gate anything.
            "cached": opening and analysis,
        }
    return {"statuses": out}


@app.get("/api/match/{fixture_id}/status")
def match_status(fixture_id: int):
    """Whether this match has already been analysed.

    Free, instant, and read-only — the client asks before opening a match so it
    can label it and decide whether a generation is about to be paid for. It
    deliberately does not write anything back, unlike the read paths, so a
    status probe never has side effects.
    """
    opening, analysis = _analysis_cached(fixture_id)
    return {
        # "Cached" to a visitor means the whole thing lands instantly. A half
        # cached match still bills a generation, so it does not qualify.
        "cached": opening and analysis,
        "opening_cached": opening,
        "analysis_cached": analysis,
        "players_cached": summary_cache.get(f"players-{fixture_id}") is not None,
    }


@app.get("/api/match/{fixture_id}/derived")
def match_derived(fixture_id: int):
    """Momentum, stat comparisons and the headline tiles — no model involved.

    Pure computation over data already on disk, so it answers in about a
    millisecond. Served separately so the chart and the stat tiles can paint
    immediately instead of riding along with a generation that takes ten
    seconds or more.
    """
    return _derived_for(fixture_id)


@app.get("/api/match/{fixture_id}/opening")
def match_opening(request: Request, fixture_id: int, refresh: bool = False):
    """Headline, summary and momentum bullets — the first thing on screen.

    Split from the analysis because generation latency tracks output volume:
    the full narrative emits ~3,700 tokens and takes about a minute, while this
    emits ~470 and lands in ten seconds. The client requests both at once, so
    the page fills in from the top instead of waiting on the whole thing.
    """
    cache_key = f"open-{fixture_id}"
    if not refresh:
        cached = _migrated(fixture_id, "opening")
        if cached:
            # Derived data is recomputed rather than served from the cache —
            # it is free, and freezing it is how the Euro final kept reporting
            # a 130-minute match after the shootout fix landed.
            return {**textclean.clean(cached), **_derived_for(fixture_id)}

    context, derived = _narrative_prelude(fixture_id)
    with _paid_call(request, f"opening {fixture_id}"):
        logger.info("Generating opening for fixture %s", fixture_id)
        try:
            opening = ai_summarizer.generate_opening(context, derived.get("headline_metrics"))
        except Exception as exc:
            raise _generation_failed(exc, "Opening", fixture_id) from exc

    result = textclean.clean(opening.model_dump())
    if _cacheable(context):
        summary_cache.set(cache_key, result)
    result.update(derived)
    return result


@app.get("/api/match/{fixture_id}/analysis")
def match_analysis(request: Request, fixture_id: int, refresh: bool = False):
    """The analytical reads and both team breakdowns — fills in behind."""
    cache_key = f"anal-{fixture_id}"
    if not refresh:
        cached = _migrated(fixture_id, "analysis")
        if cached:
            return textclean.clean(cached)

    context, derived = _narrative_prelude(fixture_id)
    with _paid_call(request, f"analysis {fixture_id}"):
        logger.info("Generating analysis for fixture %s", fixture_id)
        try:
            analysis = ai_summarizer.generate_analysis(context, derived.get("headline_metrics"))
        except Exception as exc:
            raise _generation_failed(exc, "Analysis", fixture_id) from exc

    result = textclean.clean(analysis.model_dump())
    result["team_breakdowns"] = _home_first(result.get("team_breakdowns") or [], context)
    if _cacheable(context):
        summary_cache.set(cache_key, result)
    return result


@app.get("/api/match/{fixture_id}/narrative")
def match_narrative(request: Request, fixture_id: int, refresh: bool = False):
    """Both halves merged. Kept for callers that want one payload; the UI uses
    the two split routes so it can paint as soon as the opening lands."""
    opening = match_opening(request, fixture_id, refresh)
    analysis = match_analysis(request, fixture_id, refresh)
    return {**analysis, **opening}


@app.get("/api/match/{fixture_id}/highlights")
def match_highlights(fixture_id: int):
    """Key moments for a match, plus any licensed video attached to them.

    Deliberately AI-free and unmetered: everything here is derived from the
    event feed we already hold, so opening the tab costs nothing and is
    instant. Cached because a finished match's moments never change.
    """
    cache_key = f"highlights-{fixture_id}"
    cached = summary_cache.get(cache_key)
    if cached is not None:
        # Clip configuration is re-read every time, so adding footage to
        # highlights_data.json takes effect without clearing the cache. The
        # auto-resolved video is looked up from its own cache, not re-searched.
        # The search fallback is stamped here too: entries cached before it
        # existed would otherwise never gain one, and it is what the client
        # falls back to when a territory-locked embed refuses to play.
        try:
            context = api_service.build_match_context(fixture_id)
            resolved = highlights.auto_map(context, fixture_id)
            fresh = highlights.clips_for(fixture_id, resolved)
            sides = (context.get("fixture") or {}).get("teams") or {}
            fresh["search_url"] = youtube.search_url(
                (sides.get("home") or {}).get("name") or "",
                (sides.get("away") or {}).get("name") or "",
                ((context.get("fixture") or {}).get("fixture") or {}).get("date") or "",
            )
        except Exception:
            fresh = highlights.clips_for(fixture_id, None)
        return {**cached, **fresh}

    try:
        context = api_service.build_match_context(fixture_id)
    except api_service.FootballAPIError as exc:
        raise _api_error(exc) from exc

    result = highlights.build(context, fixture_id)
    status = context["fixture"].get("fixture", {}).get("status", {}).get("short")
    if status in FINISHED_STATUSES:
        summary_cache.set(cache_key, result)
    return result


@app.get("/api/match/{fixture_id}/players")
def match_players(request: Request, fixture_id: int, refresh: bool = False):
    """Per-player notes with headshots — the long half of the generation."""
    cache_key = f"players-{fixture_id}"
    if not refresh:
        cached = summary_cache.get(cache_key)
        if cached is not None:
            return _use_photo_proxy(_with_faces(textclean.clean(cached), fixture_id))

    try:
        context = api_service.build_match_context(fixture_id)
    except api_service.FootballAPIError as exc:
        raise _api_error(exc) from exc

    # A failure here no longer 502s. The MVP tab is meant to be populated for
    # every match, so the measured fallback below stands in — real players,
    # real ratings, real portraits, assembled from the statistics with no model
    # involved. Better a card without commentary than an empty grid.
    #
    # Caught broadly on purpose. Narrowing this to the summarizer's own
    # RuntimeError left the fallback unreachable for most of what actually goes
    # wrong — the SDK's transport, rate-limit and authentication errors, and
    # schema validation failures, all of which surfaced as a 500 on the tab
    # this fallback exists to keep populated. Anything short of a cancelled
    # request is better answered with the statistics than with an error.
    try:
        with _paid_call(request, f"player notes {fixture_id}"):
            logger.info("Generating player notes for fixture %s", fixture_id)
            notes = ai_summarizer.generate_player_pass(context)
    except Exception as exc:
        logger.warning("Player pass unavailable for %s (%s: %s); using statistics",
                       fixture_id, type(exc).__name__, exc)
        fallback = enrich.enrich_player_notes(
            enrich.notes_from_statistics(context), context)
        # Deliberately not cached: this is a stand-in, and a later request
        # should get the real notes rather than inherit the substitute.
        return _use_photo_proxy({"player_notes": fallback, "source": "statistics"})

    result = textclean.clean({"player_notes": notes.model_dump()["player_notes"],
                              "source": "generated"})
    try:
        result["player_notes"] = enrich.enrich_player_notes(
            result["player_notes"], context
        )
    except Exception:
        logger.exception("Player enrichment failed for fixture %s", fixture_id)

    # The generation is paid for and in hand by this point, so nothing below is
    # allowed to lose it: storing the result is an optimisation for the next
    # reader, and the notes are returned whether or not it succeeds. Failing
    # here used to discard a pass that had already been billed, and because
    # nothing was cached, every retry was billed again.
    try:
        status = ((context.get("fixture") or {}).get("fixture", {})
                  .get("status", {}).get("short"))
        if status in FINISHED_STATUSES:
            summary_cache.set(cache_key, result)
    except Exception:
        logger.exception("Could not cache player notes for fixture %s", fixture_id)
    return _use_photo_proxy(result)


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
            return _use_photo_proxy(textclean.clean(cached))

    try:
        context = api_service.build_match_context(fixture_id)
    except api_service.FootballAPIError as exc:
        logger.warning("Fixture %s unavailable: %s", fixture_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    with _paid_call(request, f"match summary {fixture_id}"):
        logger.info("Generating summary for fixture %s", fixture_id)
        try:
            summary = ai_summarizer.generate_match_summary(context)
        except Exception as exc:
            raise _generation_failed(exc, "Summary", fixture_id) from exc

    result = textclean.clean(summary.model_dump())

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

@app.get("/api/leaderboard")
def get_leaderboard(
    league: int = Query(39),
    season: int = Query(None),
    metric: str = Query("rating"),
    limit: int = Query(25, ge=1, le=100),
):
    """Season-wide player rankings for one competition.

    The player pool is expensive to assemble (~34 provider calls) but never
    changes for a completed season, so it's cached and every re-sort is free.
    """
    season = season or competitions.default_season(league)
    comp = competitions.get_competition(league)
    if comp is None:
        raise HTTPException(status_code=404, detail="Unknown competition.")

    pool_key = f"pool-{league}-{season}"
    pool = summary_cache.get(pool_key)
    if pool is None:
        try:
            pool = leaderboard.fetch_league_players(league, season)
        except api_service.FootballAPIError as exc:
            logger.warning("Leaderboard pool failed for %s/%s: %s", league, season, exc)
            raise _api_error(exc) from exc
        if pool:
            summary_cache.set(pool_key, pool)

    return {
        "competition": {"id": comp["id"], "name": comp["name"], "emoji": comp["emoji"]},
        "season": season,
        "metric": metric,
        "metrics": leaderboard.metric_options(),
        "pool_size": len(pool),
        "leaders": _with_club_colors(leaderboard.rank(pool, metric, limit)),
    }


@app.get("/api/match/{fixture_id}/dossier")
def match_dossier(request: Request, fixture_id: int, refresh: bool = False):
    """Pre-match tactical dossier for an upcoming fixture.

    Manual trigger. Scheduling this 48 hours out needs a persistent host with a
    scheduler; the generation itself is identical either way, so wiring a cron
    to this endpoint at deploy time is all that's left.
    """
    cache_key = f"dossier-{fixture_id}"
    if not refresh:
        cached = summary_cache.get(cache_key)
        if cached is not None:
            return textclean.clean(cached)

    try:
        context = prematch.build_dossier_context(fixture_id)
    except api_service.FootballAPIError as exc:
        logger.warning("Dossier context failed for %s: %s", fixture_id, exc)
        raise _api_error(exc) from exc

    if not prematch.is_upcoming(context):
        raise HTTPException(
            status_code=409,
            detail=("That match has already been played — use the match analysis "
                    "instead of a pre-match dossier."),
        )

    with _paid_call(request, f"dossier {fixture_id}"):
        logger.info("Generating pre-match dossier for fixture %s", fixture_id)
        try:
            dossier = ai_summarizer.generate_prematch_dossier(context)
        except Exception as exc:
            raise _generation_failed(exc, "Dossier", fixture_id) from exc

    result = {
        "fixture": context["fixture"],
        "home_team": context["home_team"],
        "away_team": context["away_team"],
        "home_form": context["home_form"]["record"],
        "away_form": context["away_form"]["record"],
        "head_to_head": context["head_to_head"],
        # Stated so a thin dossier is visibly thin rather than quietly
        # confident — a newly promoted side has almost no usable record.
        "evidence": prematch.evidence_summary(context),
        "dossier": textclean.clean(dossier.model_dump()),
    }
    # Safe to cache: the inputs are historical and don't change before kickoff.
    summary_cache.set(cache_key, result)
    return result


@app.get("/api/player/{player_id}/season")
def player_season(player_id: int, season: int = Query(None)):
    """Aggregated season profile for the player-card drawer.

    Deliberately AI-free: this opens on a click, so it must be instant and
    free. Cached, since a completed season's totals never change.
    """
    season = season or competitions.default_season()
    cache_key = f"profile-{player_id}-{season}"
    cached = summary_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = api_service.get_player_season_stats(player_id, season)
    except api_service.FootballAPIError as exc:
        logger.info("Season profile unavailable for %s (%s): %s", player_id, season, exc)
        raise _api_error(exc) from exc

    profile = player_profile.build_profile(raw, season)
    summary_cache.set(cache_key, profile)
    return profile


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
