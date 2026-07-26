"""
prematch.py
-----------
Assembles the evidence for a pre-match tactical dossier.

Everything here is retrospective: head-to-head history, each side's recent
results, and last season's squad output. That is deliberate. A post-match
report describes what happened and can be checked against the data; a
pre-match report is inherently about a match that does not exist yet, so the
only defensible version is one that says what the record shows and labels the
rest as inference. The prompt in `ai_summarizer` is written to that rule, and
the UI marks the output as a preview rather than a prediction.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

RECENT_FORM_COUNT = 6
H2H_COUNT = 8


def _result_for(fixture: dict[str, Any], team_id: int) -> str:
    goals = fixture.get("goals") or {}
    home_id = ((fixture.get("teams") or {}).get("home") or {}).get("id")
    home, away = goals.get("home"), goals.get("away")
    if home is None or away is None:
        return "?"
    ours, theirs = (home, away) if home_id == team_id else (away, home)
    return "W" if ours > theirs else ("D" if ours == theirs else "L")


def _summarise_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    teams = fixture.get("teams") or {}
    goals = fixture.get("goals") or {}
    return {
        "date": (fixture.get("fixture") or {}).get("date", "")[:10],
        "competition": (fixture.get("league") or {}).get("name"),
        "home": (teams.get("home") or {}).get("name"),
        "away": (teams.get("away") or {}).get("name"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
    }


def build_dossier_context(fixture_id: int) -> dict[str, Any]:
    """Gather the historical record behind an upcoming fixture."""
    from services import api_service

    fixture = api_service.get_fixture(fixture_id)
    teams = fixture.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    league = fixture.get("league") or {}
    season = league.get("season")

    def recent(team_id: int) -> list[dict[str, Any]]:
        """Most recent completed fixtures, newest first."""
        try:
            played = [
                f for f in api_service.get_team_season_fixtures(team_id, season)
                if (f.get("fixture") or {}).get("status", {}).get("short") in ("FT", "AET", "PEN")
            ]
        except api_service.FootballAPIError:
            logger.info("No form data for team %s season %s", team_id, season)
            return []
        played.sort(key=lambda f: (f.get("fixture") or {}).get("date", ""), reverse=True)
        return played[:RECENT_FORM_COUNT]

    home_recent = recent(home.get("id"))
    away_recent = recent(away.get("id"))

    try:
        h2h_raw = api_service.get_head_to_head(home.get("id"), away.get("id"))
    except api_service.FootballAPIError:
        h2h_raw = []
    h2h = [
        _summarise_fixture(f) for f in sorted(
            [f for f in h2h_raw
             if (f.get("fixture") or {}).get("status", {}).get("short") in ("FT", "AET", "PEN")],
            key=lambda f: (f.get("fixture") or {}).get("date", ""), reverse=True,
        )[:H2H_COUNT]
    ]

    return {
        "fixture": {
            "id": fixture_id,
            "date": (fixture.get("fixture") or {}).get("date"),
            "status": (fixture.get("fixture") or {}).get("status", {}).get("short"),
            "venue": ((fixture.get("fixture") or {}).get("venue") or {}).get("name"),
            "competition": league.get("name"),
            "season": season,
            "round": league.get("round"),
        },
        "home_team": {"id": home.get("id"), "name": home.get("name")},
        "away_team": {"id": away.get("id"), "name": away.get("name")},
        "home_form": {
            "record": "".join(_result_for(f, home.get("id")) for f in home_recent),
            "fixtures": [_summarise_fixture(f) for f in home_recent],
        },
        "away_form": {
            "record": "".join(_result_for(f, away.get("id")) for f in away_recent),
            "fixtures": [_summarise_fixture(f) for f in away_recent],
        },
        "head_to_head": h2h,
    }


def is_upcoming(context: dict[str, Any]) -> bool:
    return (context.get("fixture") or {}).get("status") in ("NS", "TBD", "PST")


def evidence_summary(context: dict[str, Any]) -> dict[str, Optional[Any]]:
    """What the dossier is actually built on, so thin evidence is visible."""
    return {
        "h2h_matches": len(context.get("head_to_head") or []),
        "home_form_matches": len((context.get("home_form") or {}).get("fixtures") or []),
        "away_form_matches": len((context.get("away_form") or {}).get("fixtures") or []),
    }
