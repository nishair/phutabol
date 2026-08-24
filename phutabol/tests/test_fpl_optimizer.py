"""Squad optimizer: legality constraints and objective sanity."""

from collections import Counter

import pytest

from phutabol.fpl.optimizer import SQUAD_QUOTAS, XI_MAX, XI_MIN, optimize_squad
from .conftest import make_player


def test_squad_is_legal(player_pool):
    squad = optimize_squad(player_pool, budget=100.0)

    assert len(squad.players) == 15
    by_position = Counter(p.position for p in squad.players)
    assert dict(by_position) == SQUAD_QUOTAS

    assert len(squad.starters) == 11
    xi_positions = Counter(p.position for p in squad.starters)
    for position in SQUAD_QUOTAS:
        assert XI_MIN[position] <= xi_positions[position] <= XI_MAX[position]

    assert squad.total_cost <= 100.0
    assert max(Counter(p.team_id for p in squad.players).values()) <= 3


def test_budget_binds():
    """With a tight budget the optimizer must pick cheaper players."""
    pool = []
    pid = 0
    for position, count in {1: 4, 2: 8, 3: 8, 4: 6}.items():
        for i in range(count):
            # Half expensive-and-good, half cheap-and-worse; teams
            # spread wide so the club limit never binds here.
            expensive = i % 2 == 0
            pid += 1
            pool.append(make_player(
                pid=position * 100 + i, position=position,
                cost=10.0 if expensive else 4.0,
                projected=6.0 if expensive else 2.0,
                team_id=(pid % 9) + 1,
            ))
    squad = optimize_squad(pool, budget=80.0)
    assert squad.total_cost <= 80.0
    # An unconstrained squad would cost far more than 80.
    assert any(p.cost == 4.0 for p in squad.players)


def test_club_limit_binds():
    """Four best-value players from one club: at most 3 selected."""
    pool = []
    for position, count in {1: 4, 2: 8, 3: 8, 4: 6}.items():
        for i in range(count):
            pool.append(make_player(
                pid=position * 100 + i, position=position,
                cost=5.0, projected=2.0, team_id=(i % 6) + 2,
            ))
    # Stack team 1 with four wildly superior midfielders/forwards.
    for i, position in enumerate((3, 3, 4, 4)):
        pool.append(make_player(
            pid=900 + i, position=position,
            cost=5.0, projected=50.0, team_id=1,
        ))
    squad = optimize_squad(pool, budget=100.0)
    team_one = [p for p in squad.players if p.team_id == 1]
    assert len(team_one) == 3


def test_captain_is_top_projected_starter(player_pool):
    squad = optimize_squad(player_pool, budget=100.0)
    top = max(squad.starters, key=lambda p: p.projected_ppg)
    assert squad.captain.projected_ppg == top.projected_ppg
    assert squad.vice_captain is not squad.captain


def test_better_player_preferred():
    """Identical price, higher projection: must be selected."""
    pool = []
    for position, count in {1: 3, 2: 7, 3: 7, 4: 5}.items():
        for i in range(count):
            pool.append(make_player(
                pid=position * 100 + i, position=position,
                cost=5.0, projected=3.0, team_id=(i % 8) + 1,
            ))
    star = make_player(pid=999, position=3, cost=5.0,
                       projected=9.0, team_id=8)
    pool.append(star)
    squad = optimize_squad(pool, budget=100.0)
    assert star in squad.players
    assert star in squad.starters


def test_empty_pool_raises():
    with pytest.raises(ValueError):
        optimize_squad([])
