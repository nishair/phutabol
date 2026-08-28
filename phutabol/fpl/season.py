"""
In-season FPL manager simulation.

Plays a full season the way a human manager does, week by week, using
only information available before each deadline:

- weekly projections that blend the pre-season projection with accruing
  current-season points-per-appearance (evidence-weighted by minutes),
  a recent-minutes availability proxy, and that week's actual fixture
  count/difficulty (double gameweeks project double, blanks zero)
- best legal XI, captain, and vice re-picked every week
- 1 free transfer per week, bankable to 5; extra transfers cost 4 points
- chips: two wildcards (one per half), free hit, bench boost, triple
  captain — triggered by projection thresholds (typically fired by
  blank/double gameweeks), with use-it-or-lose-it at the final gameweek
- FPL sell-on rule: half of any price rise is kept on sale

The simulation is data-source agnostic: it consumes a SeasonData bundle
of per-gameweek scores, prices, and fixtures.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from copy import copy
from typing import Dict, List, Optional, Tuple

from .projections import ProjectedPlayer
from .optimizer import Squad, optimize_squad, XI_MIN

# Evidence weight: current-season minutes at which pre-season priors get
# half weight.
BLEND_MINUTES = 900

FIXTURE_SWING = 0.05
HORIZON = 6

VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
    (4, 5, 1), (5, 2, 3), (5, 3, 2), (5, 4, 1),
]


def difficulty_factor_sum(difficulties: List[int]) -> float:
    """Summed fixture factor for one team-gameweek (0 for a blank,
    >1 per easy fixture, doubled naturally on double gameweeks)."""
    return sum(1.0 + (3 - d) * FIXTURE_SWING for d in difficulties)


@dataclass
class SeasonData:
    """Everything knowable about a season, keyed by gameweek."""

    scores: Dict[int, Dict[int, Dict]]  # gw -> pid -> {points, minutes}
    prices: Dict[int, Dict[int, int]]   # gw -> pid -> price (tenths)
    fixtures: Dict[int, Dict[int, List[int]]]  # gw -> team -> difficulties
    n_events: int = 38


@dataclass
class ManagerConfig:
    """Decision thresholds (projected points) and learning rates."""

    ft_gain: float = 4.0        # horizon gain to spend a free transfer
    hit_gain: float = 10.0      # horizon gain to take a -4 hit
    max_hits_per_week: int = 1
    wildcard_gain: float = 14.0  # horizon XI gain to fire a wildcard
    free_hit_gain: float = 12.0  # single-week XI gain to fire free hit
    # A cheap bench projects ~9-10 in a normal week, so demand clearly
    # more (in practice: a double gameweek) before boosting.
    # Confidence gate. Projections shrink toward a positional prior
    # until a player has minutes, so early in a season every gain is
    # mostly noise -- and the thresholds above were tuned on mature
    # projections. Chips are irreversible, so refuse them outright
    # below chip_confidence_min; transfers are cheap and sometimes
    # forced (injuries), so inflate their threshold instead of
    # blocking. Mandatory picks (XI, captain, bench order) are never
    # gated: there is no option to abstain.
    # 0.10 ~= one full gameweek of minutes. Swept over 8 seasons: 0.05
    # and 0.10 gain ~+17/+20 pts/season with small downside, while 0.15
    # and above lose points badly (-347 in 2024-25) by blocking early
    # wildcards that paid off. The gate is meant to veto the opening
    # gameweeks only, not to enforce patience.
    chip_confidence_min: float = 0.10
    confidence_penalty: float = 1.0  # transfer threshold inflation
    bench_boost_min: float = 14.0  # projected bench points to boost
    triple_captain_min: float = 11.0  # projected captain points to triple
    # Learning rates: current-season minutes at which the pre-season
    # prior drops to half weight, and the smoothing prior (in minutes)
    # on the recent-minutes availability estimate.
    blend_minutes: float = BLEND_MINUTES
    avail_prior: float = 90.0
    avail_window: int = 6
    # Bench discount used when optimizing squads (GW1 and wildcards).
    bench_weight: float = 0.15
    # Crowd wisdom: scale on the net-transfer share of ownership at the
    # deadline (the crowd reacts to news the minutes-proxy sees late).
    # 0 disables.
    crowd_weight: float = 0.0
    # Captaincy ceiling bias: the armband doubles points, so prefer
    # high-ceiling (pricier) starters — score = proj + bias * cost.
    # +0.3 was worth ~+10 pts/season across 8 backtest seasons; the
    # crowd_weight and form_halflife knobs tested flat-to-negative and
    # default off.
    captain_bias: float = 0.3
    # Half-life (in gameweeks) for recency weighting of current-season
    # points-per-appearance. None = plain season average.
    form_halflife: Optional[float] = None


@dataclass
class WeekResult:
    gw: int
    points: int
    hits: int = 0
    chip: Optional[str] = None
    transfers: List[Tuple[str, str]] = field(default_factory=list)


def pick_weekly_xi(
    squad: List[ProjectedPlayer], weekly: Dict[int, float]
) -> Tuple[List[ProjectedPlayer], List[ProjectedPlayer]]:
    """Best legal XI (and ordered bench) for one week's projections."""
    by_position = defaultdict(list)
    for player in squad:
        by_position[player.position].append(player)
    for players in by_position.values():
        players.sort(key=lambda p: -weekly.get(p.id, 0.0))

    best_xi, best_score = None, float("-inf")
    for defenders, midfielders, forwards in VALID_FORMATIONS:
        counts = {2: defenders, 3: midfielders, 4: forwards}
        if any(len(by_position[pos]) < n for pos, n in counts.items()):
            continue
        xi = by_position[1][:1] + [
            p for pos, n in counts.items() for p in by_position[pos][:n]
        ]
        score = sum(weekly.get(p.id, 0.0) for p in xi)
        if score > best_score:
            best_xi, best_score = xi, score

    bench = [p for p in squad if p not in best_xi]
    bench.sort(key=lambda p: (p.position != 1, -weekly.get(p.id, 0.0)))
    return best_xi, bench


def apply_auto_subs(
    starters: List[ProjectedPlayer],
    bench: List[ProjectedPlayer],
    week: Dict[int, Dict],
) -> List[ProjectedPlayer]:
    """Replace non-playing starters from the bench, FPL-style."""
    def played(p: ProjectedPlayer) -> bool:
        return week.get(p.id, {}).get("minutes", 0) > 0

    lineup = list(starters)
    available_bench = [p for p in bench if played(p)]

    for i, starter in enumerate(list(lineup)):
        if played(starter):
            continue
        for sub in list(available_bench):
            if (starter.position == 1) != (sub.position == 1):
                continue
            counts = defaultdict(int)
            for p in lineup:
                counts[p.position] += 1
            counts[starter.position] -= 1
            counts[sub.position] += 1
            if all(counts[pos] >= XI_MIN[pos] for pos in XI_MIN):
                lineup[i] = sub
                available_bench.remove(sub)
                break
    return lineup


class SeasonManager:
    """Simulates managing a squad through a season."""

    def __init__(
        self,
        projections: List[ProjectedPlayer],
        data: SeasonData,
        budget: float = 100.0,
        config: Optional[ManagerConfig] = None,
    ):
        self.players = {p.id: p for p in projections}
        self.preseason_ppg = {p.id: p.projected_ppg for p in projections}
        self.data = data
        self.config = config or ManagerConfig()

        initial = optimize_squad(
            projections, budget=budget,
            bench_weight=self.config.bench_weight,
        )
        self.squad_ids = [p.id for p in initial.players]
        self.purchase_price = {
            p.id: data.prices[1].get(p.id, int(p.cost * 10))
            for p in initial.players
        }
        self.bank = int(budget * 10) - sum(self.purchase_price.values())
        self.initial_squad = initial

        # Accrues to 1 after GW1 (transfers first possible at GW2).
        self.free_transfers = 0
        self.chips = {"WC1", "WC2", "FH", "BB", "TC"}
        self.points_scored = defaultdict(int)   # pid -> pts
        self.minutes_played = defaultdict(int)  # pid -> mins
        self.appearances = defaultdict(int)     # pid -> games with minutes
        self.minute_history = defaultdict(list)  # pid -> [(fixtures, mins)]
        self.appearance_history = defaultdict(list)  # pid -> [(gw, pts)]

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def _fixture_sum(self, gw: int, team_id: int) -> float:
        """Summed difficulty factor over a team's fixtures in one GW."""
        return difficulty_factor_sum(
            self.data.fixtures.get(gw, {}).get(team_id, [])
        )

    def _availability(self, pid: int) -> float:
        """Share of recent possible minutes, lightly smoothed."""
        history = self.minute_history[pid][-self.config.avail_window:]
        fixtures = sum(f for f, _ in history)
        minutes = sum(m for _, m in history)
        if fixtures == 0:
            return 1.0
        prior = self.config.avail_prior
        return min(1.0, (minutes + prior) / (90 * fixtures + prior))

    def _current_ppg(self, pid: int, gw: int) -> float:
        """Current-season points per appearance, optionally recency-
        weighted with an exponential half-life."""
        halflife = self.config.form_halflife
        if halflife is None:
            return self.points_scored[pid] / max(1, self.appearances[pid])
        history = self.appearance_history[pid]
        if not history:
            return 0.0
        weighted = total = 0.0
        for played_gw, points in history:
            weight = 0.5 ** ((gw - played_gw) / halflife)
            weighted += weight * points
            total += weight
        return weighted / total

    def _blended_ppg(self, pid: int, gw: int) -> float:
        minutes = self.minutes_played[pid]
        blend = self.config.blend_minutes
        weight = blend / (blend + minutes)
        current = self._current_ppg(pid, gw)
        return weight * self.preseason_ppg[pid] + (1 - weight) * current

    def squad_confidence(self) -> float:
        """Mean evidence weight behind the squad's projections, [0, 1].

        The complement of the prior weight in `_blended_ppg`: near 0
        early in a season, when projections are mostly pre-season prior
        and any computed gain is indistinguishable from noise.
        """
        blend = self.config.blend_minutes
        if not self.squad_ids:
            return 0.0
        return sum(
            self.minutes_played[pid] / (blend + self.minutes_played[pid])
            for pid in self.squad_ids
        ) / len(self.squad_ids)

    def _crowd_factor(self, gw: int, pid: int) -> float:
        """Deadline transfer momentum as an availability/news proxy."""
        weight = self.config.crowd_weight
        if not weight:
            return 1.0
        row = self.data.scores.get(gw, {}).get(pid)
        if not row or not row.get("selected"):
            return 1.0
        ratio = row.get("transfers_balance", 0) / row["selected"]
        return 1.0 + max(-0.5, min(0.1, weight * ratio))

    def weekly_projection(self, gw: int) -> Dict[int, float]:
        return {
            pid: self._blended_ppg(pid, gw) * self._availability(pid)
            * self._crowd_factor(gw, pid)
            * self._fixture_sum(gw, p.team_id)
            for pid, p in self.players.items()
        }

    def horizon_projection(self, gw: int) -> Dict[int, float]:
        end = min(gw + HORIZON, self.data.n_events + 1)
        return {
            pid: self._blended_ppg(pid, gw) * self._availability(pid)
            * self._crowd_factor(gw, pid)
            * sum(self._fixture_sum(e, p.team_id) for e in range(gw, end))
            for pid, p in self.players.items()
        }

    # ------------------------------------------------------------------
    # Squad value and transfers
    # ------------------------------------------------------------------

    def _price(self, gw: int, pid: int) -> int:
        return self.data.prices[gw].get(pid, self.purchase_price.get(pid, 45))

    def _sale_price(self, gw: int, pid: int) -> int:
        """FPL sell-on rule: keep half of any rise; drops are fully
        borne (a fallen player sells at his current price)."""
        bought = self.purchase_price[pid]
        current = self._price(gw, pid)
        if current <= bought:
            return current
        return bought + (current - bought) // 2

    def _squad_sale_value(self, gw: int) -> int:
        return sum(self._sale_price(gw, pid) for pid in self.squad_ids)

    def _club_counts(self, squad_ids: List[int]) -> Dict[int, int]:
        counts = defaultdict(int)
        for pid in squad_ids:
            counts[self.players[pid].team_id] += 1
        return counts

    def _best_swap(
        self, gw: int, horizon: Dict[int, float]
    ) -> Optional[Tuple[float, int, int]]:
        """Highest-gain legal single transfer, or None."""
        club_counts = self._club_counts(self.squad_ids)
        best = None
        for out_id in self.squad_ids:
            out = self.players[out_id]
            funds = self.bank + self._sale_price(gw, out_id)
            for candidate in self.players.values():
                if (candidate.position != out.position
                        or candidate.id in self.squad_ids):
                    continue
                if self._price(gw, candidate.id) > funds:
                    continue
                if (club_counts[candidate.team_id]
                        - (candidate.team_id == out.team_id)) >= 3:
                    continue
                gain = horizon[candidate.id] - horizon[out_id]
                if best is None or gain > best[0]:
                    best = (gain, out_id, candidate.id)
        return best

    def _apply_transfer(self, gw: int, out_id: int, in_id: int) -> None:
        self.bank += self._sale_price(gw, out_id)
        self.bank -= self._price(gw, in_id)
        self.squad_ids.remove(out_id)
        self.squad_ids.append(in_id)
        del self.purchase_price[out_id]
        self.purchase_price[in_id] = self._price(gw, in_id)

    def _make_transfers(self, gw: int, horizon, result: WeekResult) -> None:
        config = self.config
        inflation = 1 + config.confidence_penalty * (
            1 - self.squad_confidence()
        )
        ft_gain = config.ft_gain * inflation
        hit_gain = config.hit_gain * inflation
        while True:
            swap = self._best_swap(gw, horizon)
            if swap is None:
                break
            gain, out_id, in_id = swap
            if self.free_transfers > 0 and gain > ft_gain:
                self.free_transfers -= 1
            elif (gain > hit_gain
                    and result.hits < config.max_hits_per_week):
                result.hits += 1
            else:
                break
            result.transfers.append(
                (self.players[out_id].name, self.players[in_id].name)
            )
            self._apply_transfer(gw, out_id, in_id)

    def _rebuild_squad(
        self, gw: int, projection: Dict[int, float], budget_tenths: int
    ) -> List[int]:
        """Optimize a fresh squad at current prices (wildcard/free hit)."""
        candidates = []
        for player in self.players.values():
            clone = copy(player)
            clone.projected_ppg = projection[player.id]
            clone.cost = self._price(gw, player.id) / 10.0
            candidates.append(clone)
        squad = optimize_squad(
            candidates, budget=budget_tenths / 10.0,
            bench_weight=self.config.bench_weight,
        )
        return [p.id for p in squad.players]

    # ------------------------------------------------------------------
    # Season loop
    # ------------------------------------------------------------------

    def _projected_xi_points(self, projection, squad_ids) -> float:
        squad = [self.players[pid] for pid in squad_ids]
        xi, _ = pick_weekly_xi(squad, projection)
        return sum(projection.get(p.id, 0.0) for p in xi)

    def _score_week(
        self, gw: int, squad_ids: List[int], weekly, result: WeekResult
    ) -> None:
        week = self.data.scores.get(gw, {})
        squad = [self.players[pid] for pid in squad_ids]
        starters, bench = pick_weekly_xi(squad, weekly)
        lineup = apply_auto_subs(starters, bench, week)

        points = sum(week.get(p.id, {}).get("points", 0) for p in lineup)

        bias = self.config.captain_bias
        ranked = sorted(
            starters,
            key=lambda p: -(weekly.get(p.id, 0.0) + bias * p.cost),
        )
        multiplier = 3 if result.chip == "TC" else 2
        for leader in ranked[:2]:  # captain, then vice
            if week.get(leader.id, {}).get("minutes", 0) > 0:
                points += week[leader.id]["points"] * (multiplier - 1)
                break

        if result.chip == "BB":
            points += sum(week.get(p.id, {}).get("points", 0) for p in bench)

        result.points = points - 4 * result.hits

    def _consider_chips(self, gw: int, weekly, horizon) -> Optional[str]:
        config = self.config
        half = self.data.n_events // 2
        last = gw == self.data.n_events
        confident = self.squad_confidence() >= config.chip_confidence_min

        # Wildcard: fire when a rebuilt squad clearly out-projects ours.
        wildcard = "WC1" if gw <= half else "WC2"
        if wildcard in self.chips and confident and gw > 1:
            budget = self._squad_sale_value(gw) + self.bank
            rebuilt = self._rebuild_squad(gw, horizon, budget)
            gain = (self._projected_xi_points(horizon, rebuilt)
                    - self._projected_xi_points(horizon, self.squad_ids))
            if gain > config.wildcard_gain:
                self.chips.discard(wildcard)
                self.squad_ids = rebuilt
                self.purchase_price = {
                    pid: self._price(gw, pid) for pid in rebuilt
                }
                self.bank = budget - sum(self.purchase_price.values())
                return wildcard

        # Free hit: one-week rebuild, typically on a blank gameweek.
        if "FH" in self.chips and confident and gw > 1:
            budget = self._squad_sale_value(gw) + self.bank
            rebuilt = self._rebuild_squad(gw, weekly, budget)
            gain = (self._projected_xi_points(weekly, rebuilt)
                    - self._projected_xi_points(weekly, self.squad_ids))
            if gain > config.free_hit_gain:
                self.chips.discard("FH")
                self._free_hit_squad = rebuilt
                return "FH"

        # Bench boost / triple captain on strong (usually double) weeks.
        squad = [self.players[pid] for pid in self.squad_ids]
        starters, bench = pick_weekly_xi(squad, weekly)
        bench_projection = sum(weekly.get(p.id, 0.0) for p in bench)
        captain_projection = max(weekly.get(p.id, 0.0) for p in starters)

        if "BB" in self.chips and (
            (bench_projection > config.bench_boost_min and confident)
            or last
        ):
            self.chips.discard("BB")
            return "BB"
        if "TC" in self.chips and (
            (captain_projection > config.triple_captain_min and confident)
            or last
        ):
            self.chips.discard("TC")
            return "TC"
        return None

    def _record_week(self, gw: int) -> None:
        week = self.data.scores.get(gw, {})
        for pid, player in self.players.items():
            fixtures = len(self.data.fixtures.get(gw, {}).get(player.team_id, []))
            stats = week.get(pid, {"points": 0, "minutes": 0})
            self.points_scored[pid] += stats["points"]
            self.minutes_played[pid] += stats["minutes"]
            if stats["minutes"] > 0:
                self.appearances[pid] += 1
                self.appearance_history[pid].append((gw, stats["points"]))
            if fixtures:
                self.minute_history[pid].append((fixtures, stats["minutes"]))

    def run(self) -> List[WeekResult]:
        results = []
        for gw in range(1, self.data.n_events + 1):
            weekly = self.weekly_projection(gw)
            horizon = self.horizon_projection(gw)
            result = WeekResult(gw=gw, points=0)

            result.chip = self._consider_chips(gw, weekly, horizon)
            if result.chip is None and gw > 1:
                self._make_transfers(gw, horizon, result)

            if result.chip == "FH":
                self._score_week(gw, self._free_hit_squad, weekly, result)
            else:
                self._score_week(gw, self.squad_ids, weekly, result)

            self._record_week(gw)
            if result.chip not in ("WC1", "WC2", "FH"):
                self.free_transfers = min(5, self.free_transfers + 1)
            results.append(result)
        return results
