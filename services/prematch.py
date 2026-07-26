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


def _trends(h2h: list[dict[str, Any]], home_name: str, away_name: str,
            home_fixtures: list[dict[str, Any]], away_fixtures: list[dict[str, Any]],
            home_id: int, away_id: int) -> dict[str, Any]:
    """Pre-compute the aggregates a writer would otherwise tally by hand.

    A model reads a result off a single row reliably and counts across rows
    unreliably. In testing, one dossier correctly cited a 2-2 draw in one
    bullet and then claimed no meeting in the same list had exceeded three
    goals in another — the facts were right, the arithmetic over them wasn't.
    Deriving the totals here and handing them over as findings takes the
    counting out of the model's job, which is the half it is worst at.
    """
    def side(match: dict[str, Any], name: str) -> tuple[Optional[int], Optional[int]]:
        """(goals for, goals against) for `name` in this match."""
        if match["home"] == name:
            return match["home_goals"], match["away_goals"]
        return match["away_goals"], match["home_goals"]

    played = [m for m in h2h if m["home_goals"] is not None and m["away_goals"] is not None]
    wins = draws = losses = 0
    goals_for = goals_against = 0
    at_home = {"played": 0, "wins": 0, "draws": 0, "losses": 0}
    for m in played:
        gf, ga = side(m, home_name)
        goals_for += gf
        goals_against += ga
        outcome = "wins" if gf > ga else ("draws" if gf == ga else "losses")
        wins += outcome == "wins"
        draws += outcome == "draws"
        losses += outcome == "losses"
        if m["home"] == home_name:
            at_home["played"] += 1
            at_home[outcome] += 1

    def form_totals(fixtures: list[dict[str, Any]], name: str) -> dict[str, Any]:
        scored = conceded = clean_sheets = blanks = 0
        counted = 0
        for f in fixtures:
            gf, ga = side(f, name)
            if gf is None or ga is None:
                continue
            counted += 1
            scored += gf
            conceded += ga
            clean_sheets += ga == 0
            blanks += gf == 0
        return {"matches": counted, "goals_scored": scored, "goals_conceded": conceded,
                "clean_sheets": clean_sheets, "failed_to_score": blanks}

    return {
        "note": ("Aggregates computed from the data below. Use these figures "
                 "verbatim for any counting claim."),
        "h2h_record": {
            "described_from": home_name,
            "played": len(played),
            "wins": wins, "draws": draws, "losses": losses,
            "unbeaten_run_covers_whole_record": losses == 0 and len(played) > 0,
            "goals_for": goals_for, "goals_against": goals_against,
        },
        # Deliberately not "at this venue": clubs move grounds (Everton left
        # Goodison for Hill Dickinson), so past home wins may have been at a
        # different stadium. The key says what is actually true of the data.
        "h2h_with_home_side_at_home": at_home,
        "h2h_matches_over_three_goals": [
            f"{m['date']}: {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']}"
            for m in played if m["home_goals"] + m["away_goals"] > 3
        ],
        "h2h_tight_games": {
            "decided_by_one_goal_or_drawn": sum(
                1 for m in played if abs(m["home_goals"] - m["away_goals"]) <= 1),
            "of_total": len(played),
        },
        "h2h_competitions": sorted({m["competition"] for m in played if m["competition"]}),
        "h2h_by_competition": {
            comp: sum(1 for m in played if m["competition"] == comp)
            for comp in sorted({m["competition"] for m in played if m["competition"]})
        },
        "h2h_date_range": (f"{played[-1]['date']} to {played[0]['date']}" if played else None),
        "recent_form_totals": {
            home_name: form_totals(home_fixtures, home_name),
            away_name: form_totals(away_fixtures, away_name),
        },
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

    home_fx = [_summarise_fixture(f) for f in home_recent]
    away_fx = [_summarise_fixture(f) for f in away_recent]

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
            "fixtures": home_fx,
        },
        "away_form": {
            "record": "".join(_result_for(f, away.get("id")) for f in away_recent),
            "fixtures": away_fx,
        },
        "head_to_head": h2h,
        "computed_trends": _trends(
            h2h, home.get("name"), away.get("name"),
            home_fx, away_fx, home.get("id"), away.get("id"),
        ),
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
