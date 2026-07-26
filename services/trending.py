"""
trending.py
-----------
The heritage grid: matches that earned their place in the game's history,
not simply recent lopsided scorelines.

Every fixture id here was resolved against API-Football and each was checked
to build a full analysis context, so a card never opens onto a dead end. Two
things are worth knowing before adding to this list:

* The provider reaches further back than our own plan window suggests — 2014,
  2016 and 2020 seasons all return data — but not indefinitely. The 2004-05
  Champions League returns nothing, which is why Istanbul is absent and La
  Remontada carries the comeback slot instead.
* Coverage thins with age. The 2014 World Cup has events but no team or player
  statistics, so those matches render the narrative and the Match State Index
  without stat bars or player cards. `thin_data` marks them so the UI can say
  so rather than looking broken.
"""

from typing import Any

from services.teams import display_name, distinct_colors, get_team


def _card(
    fixture_id: int,
    competition: str,
    competition_id: int,
    date: str,
    home_id: int,
    home_name: str,
    home_score: int,
    away_id: int,
    away_name: str,
    away_score: int,
    tagline: str,
    badge: str,
    thin_data: bool = False,
) -> dict[str, Any]:
    home_color, away_color = distinct_colors(home_id, away_id)
    return {
        "fixture_id": fixture_id,
        "competition": competition,
        # The label above is editorial ("Champions League SF"); this is the
        # real league id, which is what addresses the competition logo.
        "competition_id": competition_id,
        "date": date,
        "tagline": tagline,
        "badge": badge,
        # True where the provider has events but no team/player statistics, so
        # the match view can set expectations instead of showing empty panels.
        "thin_data": thin_data,
        "home": {
            "id": home_id,
            "name": display_name(home_id, home_name),
            "full_name": home_name,
            # Feeds the chart's axis labels. Without it they fall back to the
            # words HOME and AWAY, which at a neutral venue asserts something
            # the fixture doesn't mean — the designation there is a coin toss.
            "short": (get_team(home_id) or {}).get("short", ""),
            "score": home_score,
            "color": home_color,
        },
        "away": {
            "id": away_id,
            "name": display_name(away_id, away_name),
            "full_name": away_name,
            "short": (get_team(away_id) or {}).get("short", ""),
            "score": away_score,
            "color": away_color,
        },
    }


TRENDING: list[dict[str, Any]] = [
    _card(979139, "World Cup Final", 1, "2022-12-18",
          26, "Argentina", 3, 2, "France", 3,
          "Messi and Mbappé trade blows across 120 minutes; settled from twelve yards",
          "🏆 IMMORTAL"),
    _card(152855, "Champions League", 2, "2017-03-08",
          529, "Barcelona", 6, 85, "Paris Saint Germain", 1,
          "Four down from the first leg — and three goals in the last seven minutes",
          "🌋 LA REMONTADA"),
    _card(20496, "La Liga", 140, "2017-04-23",
          541, "Real Madrid", 2, 529, "Barcelona", 3,
          "Messi's 500th goal, struck in the 92nd minute, silences the Bernabéu",
          "👑 EL CLÁSICO"),
    _card(723370, "Euro Final", 4, "2021-07-11",
          768, "Italy", 1, 10, "England", 1,
          "Shaw scores inside two minutes; Italy take Wembley on penalties",
          "🇮🇹 EURO FINAL"),
    _card(208317, "World Cup Semi-final", 1, "2014-07-08",
          6, "Brazil", 1, 25, "Germany", 7,
          "Five German goals inside half an hour — the host nation's darkest night",
          "💔 MINEIRAÇO", thin_data=True),
    _card(857631, "Champions League", 2, "2022-05-04",
          541, "Real Madrid", 3, 50, "Manchester City", 1,
          "Rodrygo twice in 90 seconds at the death, then Benzema in extra time",
          "⚡ BERNABÉU MIRACLE"),
    _card(868201, "Premier League", 39, "2023-03-05",
          40, "Liverpool", 7, 33, "Man United", 0,
          "United's heaviest league defeat in more than ninety years",
          "💀 ANFIELD ROUT"),
    _card(868033, "Premier League", 39, "2022-10-02",
          50, "Man City", 6, 33, "Man United", 3,
          "Haaland and Foden both take hat-tricks off their neighbours",
          "🔥 DERBY DAY"),
]
