"""
trending.py
-----------
Pre-seeded legendary fixtures for the homepage grid. Fixture IDs were
resolved against API-Football (free-tier seasons 2021-2023) and verified,
so every card dives straight into a summarizable match with zero friction.
"""

from typing import Any

from services.teams import display_name, distinct_colors


def _card(
    fixture_id: int,
    competition: str,
    date: str,
    home_id: int,
    home_name: str,
    home_score: int,
    away_id: int,
    away_name: str,
    away_score: int,
    tagline: str,
    badge: str,
) -> dict[str, Any]:
    home_color, away_color = distinct_colors(home_id, away_id)
    return {
        "fixture_id": fixture_id,
        "competition": competition,
        "date": date,
        "tagline": tagline,
        "badge": badge,
        "home": {
            "id": home_id,
            "name": display_name(home_id, home_name),
            "full_name": home_name,
            "score": home_score,
            "color": home_color,
        },
        "away": {
            "id": away_id,
            "name": display_name(away_id, away_name),
            "full_name": away_name,
            "score": away_score,
            "color": away_color,
        },
    }


TRENDING: list[dict[str, Any]] = [
    _card(979139, "World Cup Final", "2022-12-18",
          26, "Argentina", 3, 2, "France", 3,
          "The greatest final ever played — Messi vs Mbappé, settled on penalties", "🏆 LEGENDARY"),
    _card(868201, "Premier League", "2023-03-05",
          40, "Liverpool", 7, 33, "Man United", 0,
          "Anfield demolition — United's record league defeat", "💀 MASSACRE"),
    _card(868033, "Premier League", "2022-10-02",
          50, "Man City", 6, 33, "Man United", 3,
          "Haaland and Foden hat-tricks in the derby", "⚡ DERBY DAY"),
    _card(1022982, "Champions League SF", "2023-05-17",
          50, "Man City", 4, 541, "Real Madrid", 0,
          "The Etihad statement that launched the treble", "👑 STATEMENT"),
    _card(857630, "Champions League SF", "2022-04-26",
          50, "Man City", 4, 541, "Real Madrid", 3,
          "Seven goals of pure chaos in the first leg", "🔥 CLASSIC"),
    _card(710643, "Premier League", "2021-10-24",
          33, "Man United", 0, 40, "Liverpool", 5,
          "Salah's hat-trick silences Old Trafford", "💀 MASSACRE"),
    _card(1035545, "Premier League", "2024-05-19",
          55, "Brentford", 2, 34, "Newcastle", 4,
          "Six goals and a VAR storm at the Gtech", "⚡ GOAL FEST"),
    _card(1035544, "Premier League", "2024-05-19",
          42, "Arsenal", 2, 45, "Everton", 1,
          "Title-race final day — so close, yet so far", "🎬 FINALE"),
]
