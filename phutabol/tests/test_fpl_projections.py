"""Projection model: shrinkage, luck/age adjustments, availability."""

from datetime import date

from phutabol.fpl.projections import build_projections


def element(pid, **overrides):
    base = {
        "id": pid,
        "web_name": f"P{pid}",
        "team": 1,
        "element_type": 3,
        "now_cost": 60,
        "birth_date": "2000-01-01",
        "status": "a",
        "chance_of_playing_next_round": None,
        "can_select": True,
        "removed": False,
        "news": "",
        "selected_by_percent": "1.0",
        "minutes": 3000,
        "points_per_game": "5.0",
        "total_points": 190,
        "goals_scored": 10,
        "assists": 8,
        "expected_goal_involvements": "18.0",
    }
    base.update(overrides)
    return base


def project(elements, fixtures=None):
    bootstrap = {
        "elements": elements,
        "teams": [{"id": 1, "short_name": "AAA"},
                  {"id": 2, "short_name": "BBB"}],
    }
    fixtures = fixtures if fixtures is not None else [
        {"event": 1, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 3, "team_a_difficulty": 3},
    ]
    return {
        p.id: p
        for p in build_projections(
            bootstrap, fixtures, next_event=1, horizon=6,
            as_of=date(2026, 8, 1),
        )
    }


def test_minutes_shrinkage():
    """Same PPG: the low-minutes player is pulled toward baseline."""
    players = project([
        element(1, minutes=3000),
        element(2, minutes=200),
    ])
    assert players[1].projected_ppg > players[2].projected_ppg
    # 5.0 ppg is above the positional baseline, so shrinkage lowers it.
    assert players[2].projected_ppg < 5.0


def test_luck_regression_is_symmetric():
    """Overperformers get trimmed, underperformers get boosted."""
    players = project([
        element(1, goals_scored=15, assists=10,
                expected_goal_involvements="12.0"),  # 25 GI vs 12 xGI
        element(2, goals_scored=6, assists=4,
                expected_goal_involvements="18.0"),  # 10 GI vs 18 xGI
        element(3),  # 18 GI vs 18 xGI: neutral
    ])
    assert players[1].projected_ppg < players[3].projected_ppg
    assert players[2].projected_ppg > players[3].projected_ppg


def test_age_decline():
    """Identical stats: the 33-year-old projects below the 25-year-old,
    but a young player is never discounted."""
    players = project([
        element(1, birth_date="2001-08-01"),
        element(2, birth_date="1993-08-01"),
    ])
    assert players[2].projected_ppg < players[1].projected_ppg


def test_availability_flags():
    players = project([
        element(1, status="a"),
        element(2, status="d", chance_of_playing_next_round=25),
        element(3, status="i", chance_of_playing_next_round=0),
        element(4, status="u"),
    ])
    assert players[2].projected_ppg < players[1].projected_ppg
    assert players[3].projected_ppg == 0.0
    assert players[4].projected_ppg == 0.0


def test_unselectable_players_excluded():
    players = project([
        element(1),
        element(2, can_select=False),
        element(3, removed=True),
        element(4, element_type=5),  # non-playing element (manager)
    ])
    assert set(players) == {1}


def test_fixture_difficulty():
    """Easier fixtures raise the projection."""
    easy = [{"event": e, "team_h": 1, "team_a": 2,
             "team_h_difficulty": 2, "team_a_difficulty": 5}
            for e in range(1, 7)]
    hard = [{"event": e, "team_h": 1, "team_a": 2,
             "team_h_difficulty": 5, "team_a_difficulty": 2}
            for e in range(1, 7)]
    easy_proj = project([element(1)], fixtures=easy)[1].projected_ppg
    hard_proj = project([element(1)], fixtures=hard)[1].projected_ppg
    assert easy_proj > hard_proj


def test_base_ppg_composition():
    """projected = base * availability * fixture factor (neutral=1)."""
    players = project([element(1, chance_of_playing_next_round=None)])
    p = players[1]
    assert p.availability == 1.0
    assert abs(p.projected_ppg - p.base_ppg) < 1e-6
