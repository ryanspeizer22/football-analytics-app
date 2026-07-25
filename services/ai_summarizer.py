"""
ai_summarizer.py
----------------
Generates structured, three-layer match summaries with Claude:

    Layer 1 — Quick TL;DR:       2-3 sentences, the "elevator" recap.
    Layer 2 — Team Breakdown:    tactical/statistical story per team.
    Layer 3 — Player-by-Player:  notes on every notable performer.

All three layers are produced in a single API call using structured outputs
(JSON schema), so the response is guaranteed to parse. Auth: set the
ANTHROPIC_API_KEY environment variable (or log in with `ant auth login`).
"""

import json
from typing import Any, Optional

import anthropic
from pydantic import BaseModel

MODEL = "claude-opus-5"

# Beta flag + fallback so a rare safety-classifier decline is transparently
# re-served by another model instead of failing the request.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = (
    "You are a football (soccer) analytics writer for a modern sports dashboard. "
    "You receive raw match data (fixture info, events, team statistics, player "
    "statistics) as JSON and produce accurate, engaging summaries. Ground every "
    "claim in the provided data — never invent scores, minutes, or statistics. "
    "Write for knowledgeable fans: concrete, vivid, and free of filler."
)


# ---------------------------------------------------------------------------
# Response models (validated structure for the three summary layers)
# ---------------------------------------------------------------------------

class PlayerNote(BaseModel):
    player_id: int        # copied from the supplied stats, used to attach a headshot
    player_name: str
    team: str
    rating_comment: str   # one-line verdict, e.g. "Ran the midfield; 91% pass accuracy"
    detail: str           # 2-3 sentence note on the performance


class TeamBreakdown(BaseModel):
    team_name: str
    narrative: str        # how the match unfolded for this team
    key_stats: list[str]  # standout numbers, e.g. "62% possession", "18 shots"


class MatchSummary(BaseModel):
    tldr: str                          # Layer 1 — quick TL;DR
    team_breakdowns: list[TeamBreakdown]  # Layer 2 — one entry per team
    player_notes: list[PlayerNote]        # Layer 3 — notable performers


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client_instance: Optional[anthropic.Anthropic] = None


def _client() -> anthropic.Anthropic:
    """Create the Anthropic client on first use, not at import time.

    Deferring construction means the .env file has been loaded by the time
    credentials are resolved, regardless of module import order.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client_instance


def generate_match_summary(match_context: dict[str, Any]) -> MatchSummary:
    """Produce all three summary layers for one match in a single call.

    Args:
        match_context: the dict returned by api_service.build_match_context().

    Returns:
        A validated MatchSummary instance.
    """
    response = _client().messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # The system prompt never changes — cache it across requests.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this match at all three layers "
                    "(TL;DR, team breakdowns, player-by-player notes).\n\n"
                    "For every player note, copy that player's numeric `id` from "
                    "the player_statistics data verbatim into player_id — do not "
                    "invent or guess an id.\n\n"
                    f"Match data:\n{json.dumps(match_context, ensure_ascii=False)}"
                ),
            }
        ],
        output_format=MatchSummary,
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to summarize this match data.")

    summary = response.parsed_output
    if summary is None:
        raise RuntimeError("Failed to parse structured summary from model response.")
    return summary


def generate_player_notes(player_season_stats: dict[str, Any]) -> str:
    """Generate a free-text scouting-style write-up of a player's season.

    Used by the /player/{id}/stats route; returns Markdown text.
    """
    response = _client().beta.messages.create(
        model=MODEL,
        max_tokens=4096,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a concise scouting-style summary (Markdown, ~200 words) "
                    "of this player's season based on the data below. Cover form, "
                    "strengths, and one area to improve.\n\n"
                    f"{json.dumps(player_season_stats, ensure_ascii=False)}"
                ),
            }
        ],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to summarize this player data.")

    return next(b.text for b in response.content if b.type == "text")


def generate_tactical_analysis(match_context: dict[str, Any]) -> str:
    """PREMIUM TIER (placeholder): deep tactical analysis of a match.

    Intended scope: formations and in-game shape changes, pressing patterns,
    buildup structure, and turning-point analysis. Wire this to the premium
    route once the tier is live; use streaming for the longer output.
    """
    raise NotImplementedError("Premium tactical analysis tier is not yet available.")
