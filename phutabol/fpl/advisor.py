"""
Weekly deadline advisor for a real FPL team.

Reads a manager's actual squad, bank, chips, and transfer ledger from
the public FPL entry endpoints, projects the coming gameweeks with the
tuned model, and recommends the deadline plan: transfers, starting XI,
captain, and chip. All state comes from public endpoints — no login —
so it works for any team ID once GW1's deadline has passed.

Projections blend a pre-season snapshot (cached to disk the first time
the advisor runs before GW1) with accruing current-season data, exactly
like the backtested SeasonManager, but with one live-only upgrade: the
API's real injury/suspension flags and chance-of-playing replace the
backtest's minutes-based availability proxy.
"""

import json
from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .client import FPLClient
from .optimizer import optimize_squad
from .projections import ProjectedPlayer, build_projections
from .season import (
    HORIZON, ManagerConfig, difficulty_factor_sum, pick_weekly_xi,
)

CACHE_DIR = Path(".fpl_cache")

# Chip names as the API reports them in entry history.
CHIP_NAMES = {"wildcard", "freehit", "bboost", "3xc"}


@dataclass
class TeamState:
    """A manager's squad as of the last passed deadline."""

    name: str
    squad_ids: List[int]
    purchase_price: Dict[int, int]  # tenths
    bank: int  # tenths
    free_transfers: int
    chips_used: Counter  # chip name -> times played (wildcard can be 2)
    current_event: int


@dataclass
class Advice:
    """The recommended plan for the next deadline."""

    event: int
    deadline: str
    transfers: List[Tuple[str, str, float]] = field(default_factory=list)
    hit_cost: int = 0
    starters: List[ProjectedPlayer] = field(default_factory=list)
    bench: List[ProjectedPlayer] = field(default_factory=list)
    captain: Optional[ProjectedPlayer] = None
    vice: Optional[ProjectedPlayer] = None
    chip: Optional[str] = None
    chip_reason: str = ""
    notes: List[str] = field(default_factory=list)


def estimate_free_transfers(
    transfers: List[Dict], chips: List[Dict], current_event: int
) -> int:
    """Walk the transfer ledger to estimate banked free transfers."""
    used_by_event = Counter(t["event"] for t in transfers)
    no_cost_events = {
        c["event"] for c in chips if c["name"] in ("wildcard", "freehit")
    }
    free = 0
    for event in range(2, current_event + 2):
        free = min(5, free + 1)
        if event <= current_event and event not in no_cost_events:
            free = max(0, free - used_by_event.get(event, 0))
    return max(1, free)


def fetch_team_state(
    client: FPLClient, team_id: int
) -> Optional[TeamState]:
    """Reconstruct a team's current state from public endpoints."""
    entry = client.get_entry(team_id)
    if entry is None:
        raise ValueError(f"No FPL team with ID {team_id}")

    current_event = entry.get("current_event")
    if not current_event:
        return None  # season hasn't started for this team yet

    picks = client.get_entry_picks(team_id, current_event)
    if picks is None:
        return None

    history = client.get_entry_history(team_id) or {}
    chips = history.get("chips", [])
    transfers = client.get_entry_transfers(team_id)

    # Purchase price: the cost on the transfer that brought the player
    # in, else their season-start price (exact for original picks).
    bootstrap = client.get_bootstrap()
    start_price = {
        e["id"]: e["now_cost"] - e["cost_change_start"]
        for e in bootstrap["elements"]
    }
    bought_at = {}
    for transfer in sorted(transfers, key=lambda t: t["time"]):
        bought_at[transfer["element_in"]] = transfer["element_in_cost"]

    squad_ids = [p["element"] for p in picks["picks"]]
    return TeamState(
        name=f"{entry['player_first_name']} {entry['player_last_name']}",
        squad_ids=squad_ids,
        purchase_price={
            pid: bought_at.get(pid, start_price.get(pid, 45))
            for pid in squad_ids
        },
        bank=picks["entry_history"]["bank"],
        free_transfers=estimate_free_transfers(
            transfers, chips, current_event
        ),
        chips_used=Counter(c["name"] for c in chips),
        current_event=current_event,
    )


class Advisor:
    """Builds a deadline recommendation for one team."""

    def __init__(
        self,
        client: FPLClient,
        config: Optional[ManagerConfig] = None,
    ):
        self.client = client
        self.config = config or ManagerConfig()
        self.bootstrap = client.get_bootstrap()
        self.fixtures = client.get_fixtures()
        self.next_event = client.get_next_event()

        self.projections = build_projections(
            self.bootstrap, self.fixtures, self.next_event["id"]
        )
        self.players = {p.id: p for p in self.projections}
        self._blend_preseason_snapshot()

        self.fixture_map: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for fixture in self.fixtures:
            if fixture.get("event") is None:
                continue
            self.fixture_map[fixture["event"]][fixture["team_h"]].append(
                fixture["team_h_difficulty"]
            )
            self.fixture_map[fixture["event"]][fixture["team_a"]].append(
                fixture["team_a_difficulty"]
            )

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def _snapshot_path(self) -> Path:
        # Season is identifiable from the number of finished events vs
        # the first event's year in the deadline timestamp.
        year = self.next_event["deadline_time"][:4]
        return CACHE_DIR / f"preseason_{year}.json"

    def _blend_preseason_snapshot(self) -> None:
        """Blend cached pre-season base PPG with live current data.

        Before GW1 the live bootstrap still shows last season, so the
        first run just saves a snapshot. In-season runs blend it back
        in, weighted down as current-season minutes accrue.
        """
        path = self._snapshot_path()
        if self.next_event["id"] == 1:
            CACHE_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps({
                str(p.id): p.base_ppg for p in self.projections
            }))
            return
        if not path.exists():
            return  # mid-season start: live-only projections
        snapshot = {
            int(k): v for k, v in json.loads(path.read_text()).items()
        }
        blend = self.config.blend_minutes
        for player in self.projections:
            prior = snapshot.get(player.id)
            if prior is None:
                continue
            weight = blend / (blend + player.minutes)
            player.base_ppg = round(
                weight * prior + (1 - weight) * player.base_ppg, 3
            )

    def _fixture_sum(self, event: int, team_id: int) -> float:
        return difficulty_factor_sum(
            self.fixture_map.get(event, {}).get(team_id, [])
        )

    def weekly_projection(self, event: int) -> Dict[int, float]:
        return {
            p.id: p.base_ppg * p.availability
            * self._fixture_sum(event, p.team_id)
            for p in self.projections
        }

    def horizon_projection(self, event: int) -> Dict[int, float]:
        end = event + HORIZON
        return {
            p.id: p.base_ppg * p.availability * sum(
                self._fixture_sum(e, p.team_id) for e in range(event, end)
            )
            for p in self.projections
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _sale_price(self, state: TeamState, pid: int) -> int:
        """FPL sell-on rule: keep half of any rise; drops are fully
        borne (a fallen player sells at his current price)."""
        bought = state.purchase_price.get(pid, 45)
        current = int(self.players[pid].cost * 10)
        if current <= bought:
            return current
        return bought + (current - bought) // 2

    def _best_swaps(
        self, state: TeamState, horizon: Dict[int, float], limit: int
    ) -> List[Tuple[float, int, int]]:
        """Top `limit` sequential swaps by horizon gain."""
        squad = list(state.squad_ids)
        bank = state.bank
        swaps = []
        for _ in range(limit):
            club_counts = Counter(
                self.players[pid].team_id for pid in squad
            )
            best = None
            for out_id in squad:
                out = self.players[out_id]
                funds = bank + self._sale_price(state, out_id)
                for candidate in self.projections:
                    if (candidate.position != out.position
                            or candidate.id in squad):
                        continue
                    if int(candidate.cost * 10) > funds:
                        continue
                    if (club_counts[candidate.team_id]
                            - (candidate.team_id == out.team_id)) >= 3:
                        continue
                    gain = horizon[candidate.id] - horizon[out_id]
                    if best is None or gain > best[0]:
                        best = (gain, out_id, candidate.id)
            if best is None or best[0] <= 0:
                break
            gain, out_id, in_id = best
            bank += self._sale_price(state, out_id)
            bank -= int(self.players[in_id].cost * 10)
            squad.remove(out_id)
            squad.append(in_id)
            swaps.append(best)
        return swaps

    def _chip_advice(
        self, state: TeamState, squad_ids: List[int], weekly, horizon
    ) -> Tuple[Optional[str], str]:
        config = self.config
        n_events = len(self.bootstrap["events"])
        half = n_events // 2
        event = self.next_event["id"]

        def xi_points(projection, ids):
            xi, _ = pick_weekly_xi(
                [self.players[pid] for pid in ids], projection
            )
            return sum(projection.get(p.id, 0.0) for p in xi)

        budget = (
            sum(self._sale_price(state, pid) for pid in state.squad_ids)
            + state.bank
        ) / 10.0

        def rebuild(projection):
            candidates = []
            for player in self.projections:
                clone = copy(player)
                clone.projected_ppg = projection[player.id]
                candidates.append(clone)
            squad = optimize_squad(
                candidates, budget=budget,
                bench_weight=config.bench_weight,
            )
            return [p.id for p in squad.players]

        wildcards_played = state.chips_used["wildcard"]
        wildcard_ok = wildcards_played == 0 or (
            event > half and wildcards_played < 2
        )
        if wildcard_ok and event > 1:
            gain = (xi_points(horizon, rebuild(horizon))
                    - xi_points(horizon, squad_ids))
            if gain > config.wildcard_gain:
                return "wildcard", (
                    f"a rebuilt squad projects +{gain:.0f} pts over the "
                    f"next {HORIZON} GWs"
                )

        if "freehit" not in state.chips_used and event > 1:
            gain = (xi_points(weekly, rebuild(weekly))
                    - xi_points(weekly, squad_ids))
            if gain > config.free_hit_gain:
                return "freehit", (
                    f"a one-week squad projects +{gain:.0f} pts this GW"
                )

        squad = [self.players[pid] for pid in squad_ids]
        starters, bench = pick_weekly_xi(squad, weekly)
        bench_pts = sum(weekly.get(p.id, 0.0) for p in bench)
        captain_pts = max(weekly.get(p.id, 0.0) for p in starters)

        if "bboost" not in state.chips_used and (
            bench_pts > config.bench_boost_min or event == n_events
        ):
            return "bboost", f"bench projects {bench_pts:.1f} pts"
        if "3xc" not in state.chips_used and (
            captain_pts > config.triple_captain_min or event == n_events
        ):
            return "3xc", f"captain projects {captain_pts:.1f} pts"
        return None, ""

    def advise(self, state: TeamState) -> Advice:
        event = self.next_event["id"]
        weekly = self.weekly_projection(event)
        horizon = self.horizon_projection(event)
        advice = Advice(
            event=event, deadline=self.next_event["deadline_time"]
        )

        chip, reason = self._chip_advice(
            state, state.squad_ids, weekly, horizon
        )
        advice.chip, advice.chip_reason = chip, reason

        squad_ids = list(state.squad_ids)
        if chip in (None, "bboost", "3xc"):
            swaps = self._best_swaps(
                state, horizon, limit=max(state.free_transfers, 2)
            )
            config = self.config
            for i, (gain, out_id, in_id) in enumerate(swaps):
                if i < state.free_transfers:
                    if gain <= config.ft_gain:
                        break
                elif gain <= config.hit_gain or advice.hit_cost >= 4 * (
                    config.max_hits_per_week
                ):
                    break
                else:
                    advice.hit_cost += 4
                advice.transfers.append((
                    self.players[out_id].name,
                    self.players[in_id].name,
                    round(gain, 1),
                ))
                squad_ids.remove(out_id)
                squad_ids.append(in_id)

        squad = [self.players[pid] for pid in squad_ids]
        advice.starters, advice.bench = pick_weekly_xi(squad, weekly)
        bias = self.config.captain_bias
        ranked = sorted(
            advice.starters,
            key=lambda p: -(weekly.get(p.id, 0.0) + bias * p.cost),
        )
        advice.captain, advice.vice = ranked[0], ranked[1]

        for player in squad:
            if player.news:
                advice.notes.append(f"{player.name}: {player.news}")
        return advice
