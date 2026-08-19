"""Shared fixtures: synthetic player pools and season data."""

import pytest

from phutabol.fpl.projections import ProjectedPlayer


def make_player(
    pid: int,
    position: int,
    cost: float = 5.0,
    projected: float = 3.0,
    team_id: int = None,
    name: str = None,
) -> ProjectedPlayer:
    return ProjectedPlayer(
        id=pid,
        name=name or f"P{pid}",
        team_id=team_id if team_id is not None else (pid % 10) + 1,
        team=f"T{team_id if team_id is not None else (pid % 10) + 1}",
        position=position,
        cost=cost,
        projected_ppg=projected,
        last_season_points=0,
        last_season_ppg=0.0,
        minutes=0,
        selected_by_percent=0.0,
        status="a",
        news="",
        base_ppg=projected,
        availability=1.0,
    )


@pytest.fixture
def player_pool():
    """A pool big enough for a legal 15-player squad with headroom.

    Player ids encode position (1xx GKP, 2xx DEF, 3xx MID, 4xx FWD);
    projections descend with id so lower ids are strictly better.
    """
    pool = []
    counts = {1: 5, 2: 10, 3: 10, 4: 7}
    for position, count in counts.items():
        for i in range(count):
            pool.append(make_player(
                pid=position * 100 + i,
                position=position,
                cost=6.0 - 0.2 * i,
                projected=5.0 - 0.3 * i,
                team_id=(i % 8) + 1,
            ))
    return pool
