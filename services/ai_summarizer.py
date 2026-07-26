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
import logging
from typing import Any, Optional

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
    manager_takeaway: str  # what this result tells that manager, as a read


class MatchSummary(BaseModel):
    headline: str                      # 6-10 word hook, the match in one line
    tldr: str                          # Layer 1 — quick TL;DR
    momentum_takeaways: list[str]      # 3-4 bullets on how the match swung
    team_breakdowns: list[TeamBreakdown]  # Layer 2 — one entry per team
    player_notes: list[PlayerNote]        # Layer 3 — notable performers


class MatchOpening(BaseModel):
    """The hook: what a reader needs within a couple of breaths.

    Split from the rest because latency here is bounded by how much text the
    model has to produce, not by how hard it thinks. The full narrative emits
    ~3,700 tokens and takes about a minute no matter how the prompt is shaped;
    this emits ~470 and lands in ten seconds. Running the two in parallel means
    the page has something real on it almost immediately.
    """
    headline: str
    tldr: str
    momentum_takeaways: list[str]


class MatchAnalysis(BaseModel):
    """The deeper read, filled in behind the opening."""
    decisive_factor: str
    misleading_stat: str
    tactical_shifts: list[str]
    team_breakdowns: list[TeamBreakdown]


class MatchNarrative(BaseModel):
    """The Shockwave + Tactical halves — everything except player notes.

    Split out because it is a fraction of the output volume and can therefore
    be shown while the far longer per-player pass is still running.

    The analytical fields exist because a recap that restates the scoreboard
    adds nothing the stat tiles above it haven't already shown. These four ask
    the questions a reader actually has once they know the score.
    """
    headline: str
    tldr: str
    decisive_factor: str          # what actually settled the match
    misleading_stat: str          # a headline number that misreads the game
    tactical_shifts: list[str]    # what changed after substitutions / at the break
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
                "- `computed_trends` holds every aggregate already tallied for "
                "you: the head-to-head record, the venue record, goal totals, "
                "and which meetings went over three goals. Use those numbers "
                "verbatim. Do NOT count across the fixture rows yourself, and "
                "never assert a tally ('only one meeting…', 'four of the last "
                "five…') that isn't in computed_trends — quote individual "
                "results from the rows, but take all counting from there.\n"
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
    """Build the leanest context the narrative pass actually reasons over.

    This used to drop only the top-level `player_statistics` key, which looked
    right and wasn't: the provider also nests the whole per-player block inside
    `fixture.players`, and that alone was 30k of a 53k payload. `events` and
    `statistics` were each being sent twice as well, once at the top level and
    again inside `fixture`.

    So the pass was reading ~25k tokens to write ~500. Latency here is bounded
    by input, not output — an opening pass producing 486 tokens still took 13s
    against the full context. Sending only what the pass uses cuts the payload
    by about 80%.

    Formations are kept because the tactical fields reason about shape; the
    lineups themselves are not, since individual players are the other pass's
    job.
    """
    fixture = match_context.get("fixture") or {}
    lineups = fixture.get("lineups") or []

    return {
        "fixture": {
            key: fixture.get(key)
            for key in ("fixture", "league", "teams", "goals", "score")
            if fixture.get(key) is not None
        },
        "formations": [
            {"team": (side.get("team") or {}).get("name"),
             "formation": side.get("formation")}
            for side in lineups if side.get("formation")
        ],
        # Deliberately from the top level, and only once.
        "events": match_context.get("events") or fixture.get("events") or [],
        "team_statistics": (match_context.get("team_statistics")
                            or fixture.get("statistics") or []),
    }


def _displayed_metrics(headline_metrics: Optional[list[dict[str, Any]]]) -> str:
    """Render the stat tiles the page already shows, for the prompt to avoid.

    Without this the model has no way to know those numbers are on screen, so
    it spends its bullets and its prose repeating them — the same figure in the
    tile, the bullet and the paragraph.
    """
    if not headline_metrics:
        return ""
    lines = [
        f"- {m.get('label')}: {m.get('home')} vs {m.get('away')}"
        for m in headline_metrics
    ]
    return (
        "\nThese figures are ALREADY displayed on the page as large stat tiles "
        "directly above your text:\n" + "\n".join(lines) +
        "\nDo not restate them as the point of a bullet or a sentence. You may "
        "refer to one only when you are adding something it does not say — what "
        "it caused, what it conceals, or how it changed within the match.\n"
    )


def _call(output_format: Any, instruction: str, match_context: dict[str, Any],
          headline_metrics: Optional[list[dict[str, Any]]], max_tokens: int) -> Any:
    """One structured pass over the lean match context."""
    response = _client().messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": (instruction + _displayed_metrics(headline_metrics) +
                        f"\nMatch data:\n{json.dumps(_strip_players(match_context), ensure_ascii=False)}"),
        }],
        output_format=output_format,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to summarize this match data.")
    if response.parsed_output is None:
        raise RuntimeError("Failed to parse the model response.")
    return response.parsed_output


_OPENING_PROMPT = (
    "Write the opening of a match report for a scannable dashboard.\n\n"
    "The reader can already see the scoreline and the headline statistics. "
    "Explain the match, don't report it.\n"
    "- headline: 6-10 words, the match in one punchy line.\n"
    "- tldr: 2-3 sentences on what happened and why. No stat dump.\n"
    "- momentum_takeaways: 3-4 bullets tracing how control moved, each naming "
    "the minute or passage that caused the swing.\n\n"
    "Write only these three fields, and keep them tight."
)

_ANALYSIS_PROMPT = (
    "Write the analytical half of a match report. The headline, summary and "
    "momentum bullets are being written separately — do not repeat them.\n\n"
    "- decisive_factor: the one thing that actually settled it — a tactical "
    "mismatch, a passage of play, a substitution, an individual. Name it and "
    "say why it was decisive.\n"
    "- misleading_stat: a headline number that misreads this game, and what "
    "the data shows was really going on. If every number genuinely reflects "
    "the match, say so plainly instead of manufacturing a contradiction.\n"
    "- tactical_shifts: 2-4 bullets on what changed after the substitutions or "
    "at the break, tied to the minute. Fewer bullets if the data shows no shift.\n"
    "- team_breakdowns: one per team. `verdict` is 3-6 words; `bullets` are 3-4 "
    "concrete takeaways under ~14 words; `narrative` is the fuller prose; "
    "`manager_takeaway` is one sentence on what this result tells that manager "
    "about their side.\n\n"
    "Never repeat a number in more than one place."
)


def generate_opening(match_context: dict[str, Any],
                     headline_metrics: Optional[list[dict[str, Any]]] = None) -> MatchOpening:
    """The fast pass — headline, summary and momentum bullets."""
    opening = _call(MatchOpening, _OPENING_PROMPT, match_context, headline_metrics, 2000)
    if not (opening.headline and opening.tldr):
        raise RuntimeError("The model returned an empty opening for this match.")
    return opening


def generate_analysis(match_context: dict[str, Any],
                      headline_metrics: Optional[list[dict[str, Any]]] = None) -> MatchAnalysis:
    """The slower pass — the analytical reads and both team breakdowns."""
    for attempt in (1, 2):
        analysis = _call(MatchAnalysis, _ANALYSIS_PROMPT, match_context, headline_metrics, 8000)
        if analysis.team_breakdowns:
            return analysis
        logger.warning("Analysis attempt %d came back without team breakdowns; %s",
                       attempt, "retrying" if attempt == 1 else "giving up")
    raise RuntimeError("The model returned an incomplete analysis for this match.")


def _is_complete(narrative: MatchNarrative) -> bool:
    """Whether a parsed narrative actually carries an analysis.

    A structured response can satisfy the schema while being hollow — every
    required field present, but the lists empty and the analytical strings
    blank. That parses cleanly, so nothing downstream objects, and it gets
    cached as though it were a real analysis. Seen in practice: one generation
    returned only a headline and TL;DR and was cached in that state.
    """
    return bool(
        narrative.headline
        and narrative.tldr
        and narrative.team_breakdowns
        and narrative.momentum_takeaways
    )


def generate_narrative(match_context: dict[str, Any],
                       headline_metrics: Optional[list[dict[str, Any]]] = None) -> MatchNarrative:
    """Generate the headline, TL;DR, analysis and team breakdowns.

    Retries once on a hollow response rather than returning it: the result is
    cached by the caller, so a stub would persist until someone forced a
    refresh.
    """
    for attempt in (1, 2):
        narrative = _narrative_attempt(match_context, headline_metrics)
        if _is_complete(narrative):
            return narrative
        logger.warning(
            "Narrative attempt %d returned an incomplete analysis "
            "(breakdowns=%d, takeaways=%d); %s",
            attempt, len(narrative.team_breakdowns), len(narrative.momentum_takeaways),
            "retrying" if attempt == 1 else "giving up",
        )
    raise RuntimeError("The model returned an incomplete analysis for this match.")


def _narrative_attempt(match_context: dict[str, Any],
                       headline_metrics: Optional[list[dict[str, Any]]]) -> MatchNarrative:
    response = _client().messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": (
                "Write the match analysis for a scannable dashboard.\n\n"
                "The reader can already see the scoreline, the scorers and the "
                "headline statistics. Your job is to explain the match, not to "
                "report it. Every field should tell them something they could "
                "not get by looking at the numbers.\n\n"
                "- headline: 6-10 words, the match in one punchy line.\n"
                "- tldr: 2-3 sentences on what happened and why. No stat dump.\n"
                "- decisive_factor: the one thing that actually settled it — a "
                "tactical mismatch, a passage of play, a substitution, an "
                "individual. Name it and say why it was decisive.\n"
                "- misleading_stat: a headline number that misreads this game, "
                "and what the data shows was really going on. If every number "
                "genuinely reflects the match, say so plainly instead of "
                "manufacturing a contradiction.\n"
                "- tactical_shifts: 2-4 bullets on what changed after the "
                "substitutions or at the break, tied to the minute. If the "
                "data does not show a shift, return fewer bullets.\n"
                "- momentum_takeaways: 3-4 bullets tracing how control moved, "
                "each naming the minute or passage that caused the swing.\n"
                "- Per team: `verdict` is 3-6 words; `bullets` are 3-4 concrete "
                "takeaways under ~14 words; `narrative` is the fuller prose; "
                "`manager_takeaway` is one sentence on what this result tells "
                "that manager about their side.\n\n"
                "Never repeat a number in more than one place. A figure used in "
                "a bullet must not reappear in that team's narrative, and "
                "nothing should echo the headline."
                f"{_displayed_metrics(headline_metrics)}\n"
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
