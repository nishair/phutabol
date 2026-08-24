"""Advisor arithmetic that needs no network: the free-transfer walk."""

from phutabol.fpl.advisor import estimate_free_transfers


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
