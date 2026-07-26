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
    verdict: str          # 3-6 word punchy summary, e.g. "Ruthless on the break"
    narrative: str        # how the match unfolded for this team
    bullets: list[str]    # 3-4 scannable takeaways, each under ~14 words
    key_stats: list[str]  # standout numbers, e.g. "62% possession", "18 shots"


class MatchSummary(BaseModel):
    headline: str                      # 6-10 word hook, the match in one line
    tldr: str                          # Layer 1 — quick TL;DR
    momentum_takeaways: list[str]      # 3-4 bullets on how the match swung
    team_breakdowns: list[TeamBreakdown]  # Layer 2 — one entry per team
    player_notes: list[PlayerNote]        # Layer 3 — notable performers


class MatchNarrative(BaseModel):
    """The Shockwave + Tactical halves — everything except player notes.

    Split out because it is a fraction of the output volume and can therefore
    be shown while the far longer per-player pass is still running.
    """
    headline: str
    tldr: str
    momentum_takeaways: list[str]
    team_breakdowns: list[TeamBreakdown]


class PlayerNotes(BaseModel):
    player_notes: list[PlayerNote]


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
                    "This feeds a scannable dashboard, so write for glanceability:\n"
                    "- headline: 6-10 words, the match in one punchy line.\n"
                    "- momentum_takeaways: 3-4 bullets tracing how control shifted, "
                    "each naming the minute or passage that caused the swing.\n"
                    "- Per team, `verdict` is 3-6 words; `bullets` are 3-4 concrete "
                    "takeaways under ~14 words each, leading with the fact rather "
                    "than a preamble. Keep `narrative` as the fuller prose account.\n"
                    "Bullets must not restate each other or the headline.\n\n"
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


class PreMatchDossier(BaseModel):
    headline: str                  # 6-10 words framing the fixture
    overview: str                  # 2-3 sentences on what's at stake
    tactical_matchups: list[str]   # where the game is likely decided
    key_duels: list[str]           # individual battles worth watching
    historical_trends: list[str]   # what the H2H and form record actually shows
    watch_points: list[str]        # concrete things to look for on the day
    caveats: list[str]             # what the data can't tell us


def generate_prematch_dossier(context: dict[str, Any]) -> PreMatchDossier:
    """Tactical preview of an upcoming fixture, grounded in the record.

    The prompt is explicit that this is analysis of history, not prediction:
    the match hasn't happened, so anything stated as fact must be traceable to
    a past result, and everything else has to be marked as inference. Without
    that instruction a model will happily invent lineups, injuries and
    scorelines that read exactly like reporting.
    """
    response = _client().messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": (
                "Write a pre-match tactical dossier for the upcoming fixture below.\n\n"
                "This match has NOT been played. You are given only historical "
                "evidence: recent form and head-to-head results.\n"
                "- Every factual claim must trace to something in that data. "
                "Cite the match or run it comes from.\n"
                "- Do NOT invent lineups, injuries, transfers, formations, "
                "quotes, or a predicted scoreline. You have none of that.\n"
                "- Where you reason beyond the data, mark it as inference "
                "('on this record you would expect…').\n"
                "- `caveats` must name what this data genuinely cannot tell us "
                "— squad changes since these matches, absentees, and the fact "
                "that pre-season form is a weak signal.\n"
                "- tactical_matchups, key_duels, historical_trends and "
                "watch_points: 3-4 entries each, concrete and under ~20 words.\n\n"
                f"Fixture and historical record:\n{json.dumps(context, ensure_ascii=False)}"
            ),
        }],
        output_format=PreMatchDossier,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to preview this fixture.")
    if response.parsed_output is None:
        raise RuntimeError("Failed to parse dossier from model response.")
    return response.parsed_output


def _strip_players(match_context: dict[str, Any]) -> dict[str, Any]:
    """Drop the per-player rows — ~14k of the ~39k input tokens.

    The narrative pass reasons about the match, not individuals, so sending
    every player's stat line only slows it down.
    """
    return {k: v for k, v in match_context.items() if k != "player_statistics"}


def generate_narrative(match_context: dict[str, Any]) -> MatchNarrative:
    """Generate the headline, TL;DR, momentum bullets and team breakdowns."""
    response = _client().messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": (
                "Write the match narrative for a scannable dashboard:\n"
                "- headline: 6-10 words, the match in one punchy line.\n"
                "- momentum_takeaways: 3-4 bullets tracing how control shifted, "
                "each naming the minute or passage that caused the swing.\n"
                "- Per team, `verdict` is 3-6 words; `bullets` are 3-4 concrete "
                "takeaways under ~14 words each; `narrative` is the fuller prose.\n"
                "Bullets must not restate each other or the headline.\n\n"
                f"Match data:\n{json.dumps(_strip_players(match_context), ensure_ascii=False)}"
            ),
        }],
        output_format=MatchNarrative,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to summarize this match data.")
    if response.parsed_output is None:
        raise RuntimeError("Failed to parse narrative from model response.")
    return response.parsed_output


def generate_player_pass(match_context: dict[str, Any]) -> PlayerNotes:
    """Generate the per-player notes — the long half of the output."""
    response = _client().messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": (
                "Write one note per player who appeared, plus a short line for "
                "unused substitutes. Copy each player's numeric `id` from the "
                "player_statistics data verbatim into player_id — never invent "
                "one. `rating_comment` is a short verdict; `detail` is 2-3 "
                "sentences grounded in that player's numbers.\n\n"
                f"Match data:\n{json.dumps(match_context, ensure_ascii=False)}"
            ),
        }],
        output_format=PlayerNotes,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to summarize this match data.")
    if response.parsed_output is None:
        raise RuntimeError("Failed to parse player notes from model response.")
    return response.parsed_output


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
