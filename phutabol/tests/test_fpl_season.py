"""Season mechanics: XI picking, auto-subs, prices, transfers."""

from collections import Counter

from phutabol.fpl.season import (
    ManagerConfig, SeasonData, SeasonManager, apply_auto_subs,
    difficulty_factor_sum, pick_weekly_xi,
)
from .conftest import make_player


def build_squad_15():
    """2 GKP, 5 DEF, 5 MID, 3 FWD with descending projections."""
    squad = []
    for position, count in {1: 2, 2: 5, 3: 5, 4: 3}.items():
        for i in range(count):
            squad.append(make_player(
                pid=position * 100 + i, position=position,
                projected=5.0 - i, team_id=(position + i) % 8 + 1,
            ))
    return squad


class TestPickWeeklyXI:
    def test_legal_formation(self):
        squad = build_squad_15()
        weekly = {p.id: p.projected_ppg for p in squad}
        xi, bench = pick_weekly_xi(squad, weekly)
        counts = Counter(p.position for p in xi)
        assert len(xi) == 11 and len(bench) == 4
        assert counts[1] == 1
        assert 3 <= counts[2] <= 5
        assert 2 <= counts[3] <= 5
        assert 1 <= counts[4] <= 3

    def test_bench_keeper_first(self):
        squad = build_squad_15()
        weekly = {p.id: p.projected_ppg for p in squad}
        _, bench = pick_weekly_xi(squad, weekly)
        assert bench[0].position == 1

    def test_loaded_position_gets_max_slots(self):
        squad = build_squad_15()
        weekly = {p.id: p.projected_ppg for p in squad}
        for p in squad:
            if p.position == 3:
                weekly[p.id] = 50.0  # all five midfielders massive
        xi, _ = pick_weekly_xi(squad, weekly)
        assert Counter(p.position for p in xi)[3] == 5


class TestAutoSubs:
    def make_week(self, squad, blanked_ids):
        return {
            p.id: {
                "points": 0 if p.id in blanked_ids else 2,
                "minutes": 0 if p.id in blanked_ids else 90,
            }
            for p in squad
        }

    def test_outfield_swap(self):
        squad = build_squad_15()
        weekly = {p.id: p.projected_ppg for p in squad}
        xi, bench = pick_weekly_xi(squad, weekly)
        blank = next(p for p in xi if p.position == 3)
        week = self.make_week(squad, {blank.id})
        lineup = apply_auto_subs(xi, bench, week)
        assert blank not in lineup
        assert len(lineup) == 11

    def test_keeper_only_replaced_by_keeper(self):
        squad = build_squad_15()
        weekly = {p.id: p.projected_ppg for p in squad}
        xi, bench = pick_weekly_xi(squad, weekly)
        keeper = next(p for p in xi if p.position == 1)
        bench_keeper = next(p for p in bench if p.position == 1)
        # Keeper blanks AND bench keeper blanks: no outfielder may
        # replace the keeper, so the blank stays in the lineup.
        week = self.make_week(squad, {keeper.id, bench_keeper.id})
        lineup = apply_auto_subs(xi, bench, week)
        assert keeper in lineup

    def test_formation_minimum_respected(self):
        squad = build_squad_15()
        weekly = {p.id: p.projected_ppg for p in squad}
        for p in squad:
            if p.position == 2:
                weekly[p.id] = 0.1  # force a 3-defender XI
        xi, bench = pick_weekly_xi(squad, weekly)
        assert Counter(p.position for p in xi)[2] == 3
        blank_def = next(p for p in xi if p.position == 2)
        # Only non-DEF bench players available: swap would drop DEF
        # below 3, so it must not happen.
        week = self.make_week(squad, {blank_def.id} | {
            p.id for p in bench if p.position == 2
        })
        lineup = apply_auto_subs(xi, bench, week)
        assert Counter(p.position for p in lineup)[2] == 3
        assert blank_def in lineup


def test_difficulty_factor_sum():
    assert difficulty_factor_sum([]) == 0.0  # blank gameweek
    assert difficulty_factor_sum([3]) == 1.0  # neutral fixture
    assert difficulty_factor_sum([2]) > 1.0 > difficulty_factor_sum([4])
    # Double gameweek roughly doubles.
    assert difficulty_factor_sum([3, 3]) == 2.0


class TestManagerEconomics:
    def make_manager(self):
        pool = []
        for position, count in {1: 4, 2: 8, 3: 8, 4: 6}.items():
            for i in range(count):
                pool.append(make_player(
                    pid=position * 100 + i, position=position,
                    cost=5.0, projected=4.0 - 0.2 * i,
                    team_id=(i % 8) + 1,
                ))
        prices = {p.id: int(p.cost * 10) for p in pool}
        data = SeasonData(
            scores={gw: {} for gw in (1, 2, 3)},
            prices={gw: dict(prices) for gw in (1, 2, 3)},
            fixtures={
                gw: {t: [3] for t in range(1, 9)} for gw in (1, 2, 3)
            },
            n_events=3,
        )
        return SeasonManager(pool, data, budget=100.0), pool

    def test_sell_on_rule(self):
        manager, _ = self.make_manager()
        pid = manager.squad_ids[0]
        manager.purchase_price[pid] = 50

        manager.data.prices[2][pid] = 56  # rose 0.6: keep half, floored
        assert manager._sale_price(2, pid) == 53
        manager.data.prices[2][pid] = 55  # 0.5 rise: floor(5/2)=2
        assert manager._sale_price(2, pid) == 52
        manager.data.prices[2][pid] = 46  # drops are fully borne
        assert manager._sale_price(2, pid) == 46

    def test_transfer_updates_bank_and_squad(self):
        manager, pool = self.make_manager()
        out_id = manager.squad_ids[0]
        in_id = next(
            p.id for p in pool
            if p.position == manager.players[out_id].position
            and p.id not in manager.squad_ids
        )
        bank_before = manager.bank
        sale = manager._sale_price(1, out_id)
        cost = manager._price(1, in_id)
        manager._apply_transfer(1, out_id, in_id)

        assert manager.bank == bank_before + sale - cost
        assert out_id not in manager.squad_ids
        assert in_id in manager.squad_ids
        assert len(manager.squad_ids) == 15
        assert manager.purchase_price[in_id] == cost

    def test_free_transfer_accrual_and_cap(self):
        manager, _ = self.make_manager()
        assert manager.free_transfers == 0  # none before GW1
        manager.config = ManagerConfig(
            ft_gain=1e9, wildcard_gain=1e9, free_hit_gain=1e9,
            bench_boost_min=1e9, triple_captain_min=1e9,
        )  # thresholds so high nothing ever fires
        results = manager.run()
        assert len(results) == 3
        assert all(w.points == 0 for w in results)  # empty score weeks
        # +1 per week, capped at 5.
        assert manager.free_transfers == 3
