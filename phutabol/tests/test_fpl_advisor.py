"""Advisor logic that needs no network: the free-transfer walk and the
wildcard rebuild, on a hand-wired Advisor."""

from collections import Counter

from phutabol.fpl.advisor import Advisor, TeamState, estimate_free_transfers
from phutabol.fpl.season import ManagerConfig


def transfers(*events):
    return [{"event": e} for e in events]


def test_no_transfers_caps_at_five():
    assert estimate_free_transfers([], [], current_event=10) == 5


def test_start_of_season():
    # Advice for GW2: exactly the one accrued free transfer.
    assert estimate_free_transfers([], [], current_event=1) == 1


def test_transfers_consume():
    # 1 FT accrued by GW2, used; 1 more by GW3.
    assert estimate_free_transfers(
        transfers(2), [], current_event=2
    ) == 1


def test_hits_floor_at_zero():
    # GW2: 1 FT, 3 transfers (2 hits) -> 0, then +1 for GW3.
    assert estimate_free_transfers(
        transfers(2, 2, 2), [], current_event=2
    ) == 1


def test_wildcard_week_is_free():
    # 5 transfers on a GW8 wildcard cost nothing.
    chips = [{"name": "wildcard", "event": 8}]
    assert estimate_free_transfers(
        transfers(8, 8, 8, 8, 8), chips, current_event=8
    ) == 5


def test_free_hit_week_is_free():
    chips = [{"name": "freehit", "event": 4}]
    with_chip = estimate_free_transfers(
        transfers(4, 4), chips, current_event=4
    )
    without = estimate_free_transfers(
        transfers(4, 4), [], current_event=4
    )
    assert with_chip > without


def test_banked_then_spent():
    # 4 FTs accrued by the GW5 deadline, spend 2 -> 2 left, +1 for GW6.
    assert estimate_free_transfers(
        transfers(5, 5), [], current_event=5
    ) == 3


def make_advisor(pool, event=10, **config):
    """Hand-wire an Advisor (no network, no snapshot cache)."""
    advisor = Advisor.__new__(Advisor)
    advisor.config = ManagerConfig(**config)
    advisor.projections = pool
    advisor.players = {p.id: p for p in pool}
    advisor.next_event = {
        "id": event, "deadline_time": "2026-11-01T11:00:00Z"
    }
    advisor.bootstrap = {"events": [{"id": i} for i in range(1, 39)]}
    teams = {p.team_id for p in pool}
    advisor.fixture_map = {
        e: {t: [3] for t in teams} for e in range(1, 45)
    }
    return advisor


# The legal-but-worst 15 from the conftest pool (projections descend
# with id, so these leave plenty of headroom for a rebuild).
WORST_15 = [103, 104, 205, 206, 207, 208, 209,
            305, 306, 307, 308, 309, 404, 405, 406]


def worst_squad_state(pool):
    costs = {p.id: int(p.cost * 10) for p in pool}
    return TeamState(
        name="t",
        squad_ids=list(WORST_15),
        purchase_price={pid: costs[pid] for pid in WORST_15},
        bank=200,  # £20m spare: leaves the rebuild room to upgrade
        free_transfers=1,
        chips_used=Counter(),
        current_event=9,
    )


def test_wildcard_advice_includes_rebuilt_squad(player_pool):
    advisor = make_advisor(player_pool, wildcard_gain=0.5)
    advice = advisor.advise(worst_squad_state(player_pool))

    assert advice.chip == "wildcard"
    assert advice.transfers == []  # no swap advice on a wildcard
    shown = {p.id for p in advice.starters + advice.bench}
    assert len(advice.starters + advice.bench) == 15
    assert advice.chip_in and advice.chip_out
    names = {advisor.players[pid].name for pid in shown}
    assert set(advice.chip_in) <= names
    assert not set(advice.chip_out) & names


def test_bench_boost_advice_keeps_current_squad(player_pool):
    advisor = make_advisor(
        player_pool, wildcard_gain=1e9, free_hit_gain=1e9,
        bench_boost_min=0.0, ft_gain=1e9,
    )
    advice = advisor.advise(worst_squad_state(player_pool))

    assert advice.chip == "bboost"
    assert advice.chip_out == [] and advice.chip_in == []
    assert {p.id for p in advice.starters + advice.bench} == set(WORST_15)
