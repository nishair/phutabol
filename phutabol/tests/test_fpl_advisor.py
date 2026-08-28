"""Advisor logic that needs no network: the free-transfer walk and the
wildcard rebuild, on a hand-wired Advisor."""

from collections import Counter

import pytest

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


# --- confidence gate ---------------------------------------------------
#
# The conftest pool leaves ProjectedPlayer.reliability at its 1.0
# default, so these tests dial it down explicitly to model an early
# season, where projected_ppg is mostly the positional prior.

def with_reliability(pool, value):
    from copy import copy
    out = []
    for player in pool:
        clone = copy(player)
        clone.reliability = value
        out.append(clone)
    return out


def test_squad_confidence_averages_reliability(player_pool):
    advisor = make_advisor(with_reliability(player_pool, 0.4))
    assert advisor.squad_confidence(WORST_15) == pytest.approx(0.4)


def test_squad_confidence_ignores_unknown_ids(player_pool):
    advisor = make_advisor(with_reliability(player_pool, 0.5))
    assert advisor.squad_confidence(
        WORST_15 + [99999]
    ) == pytest.approx(0.5)


def test_low_confidence_suppresses_wildcard(player_pool):
    """The same squad that fires a wildcard at full reliability must
    not fire one when the projections are still mostly prior."""
    state = worst_squad_state(player_pool)

    confident = make_advisor(
        with_reliability(player_pool, 1.0), wildcard_gain=0.5
    )
    assert confident.advise(state).chip == "wildcard"

    early = make_advisor(
        with_reliability(player_pool, 0.07), wildcard_gain=0.5
    )
    advice = early.advise(state)
    assert advice.chip is None
    assert "confidence" in advice.chip_note
    assert advice.confidence == pytest.approx(0.07)


def test_confidence_gate_threshold_is_configurable(player_pool):
    state = worst_squad_state(player_pool)
    pool = with_reliability(player_pool, 0.05)
    # 0.05 clears a 0.01 floor but not the 0.10 default.
    assert make_advisor(
        pool, wildcard_gain=0.5, chip_confidence_min=0.01
    ).advise(state).chip == "wildcard"
    assert make_advisor(
        pool, wildcard_gain=0.5
    ).advise(state).chip is None


def test_gate_does_not_block_mandatory_picks(player_pool):
    """XI, captain and bench are compulsory every week, so the gate
    must leave them alone -- there is no option to abstain."""
    advisor = make_advisor(with_reliability(player_pool, 0.01))
    advice = advisor.advise(worst_squad_state(player_pool))
    assert len(advice.starters) == 11
    assert len(advice.bench) == 4
    assert advice.captain is not None
    assert advice.vice is not None


def test_low_confidence_raises_the_transfer_bar(player_pool):
    """Transfers are not blocked outright, only made more expensive:
    the same swap that passes at full reliability is declined when the
    projections behind it are noise."""
    state = worst_squad_state(player_pool)
    kwargs = dict(wildcard_gain=1e9, free_hit_gain=1e9, ft_gain=4.0)

    confident = make_advisor(
        with_reliability(player_pool, 1.0), **kwargs
    ).advise(state)
    early = make_advisor(
        with_reliability(player_pool, 0.07), **kwargs
    ).advise(state)
    assert len(early.transfers) <= len(confident.transfers)


def test_transfer_penalty_can_be_disabled(player_pool):
    """confidence_penalty=0 restores the pre-gate transfer behaviour."""
    state = worst_squad_state(player_pool)
    kwargs = dict(wildcard_gain=1e9, free_hit_gain=1e9, ft_gain=4.0)
    baseline = make_advisor(
        with_reliability(player_pool, 1.0), **kwargs
    ).advise(state)
    unpenalised = make_advisor(
        with_reliability(player_pool, 0.07),
        confidence_penalty=0.0, **kwargs
    ).advise(state)
    assert len(unpenalised.transfers) == len(baseline.transfers)


def test_no_chip_note_when_confident(player_pool):
    advisor = make_advisor(
        with_reliability(player_pool, 1.0), wildcard_gain=1e9,
        free_hit_gain=1e9,
    )
    assert advisor.advise(worst_squad_state(player_pool)).chip_note == ""
