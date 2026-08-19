"""
Season-start player projections for FPL.

Early in a season the FPL API still exposes each player's previous-season
stats (points, minutes, xG/xA). The projection here is deliberately simple
and transparent:

    projected_ppg = shrunk_ppg * luck_adjustment * availability * fixture_factor

- shrunk_ppg: last season's points-per-game, shrunk toward a positional
  baseline in proportion to minutes played (low-minute players carry
  little evidence).
- luck_adjustment: attackers whose actual goal involvements ran ahead of
  their xGI are regressed down, and vice versa.
- age_adjustment: premium-age decline discount for players past their
  positional peak (backtests showed prior-season PPG badly overrates
  ageing premiums).
- availability: injury/suspension status and chance-of-playing.
- fixture_factor: average FPL fixture difficulty over the next few
  gameweeks, centred on a neutral difficulty of 3.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Any, Optional

# Positional baselines (points per game) that low-evidence players are
# shrunk toward — roughly what a fringe starter returns.
POSITION_BASELINE_PPG = {1: 2.5, 2: 2.4, 3: 2.6, 4: 2.6}

# Minutes at which last season's evidence gets half weight (~10 full games).
SHRINKAGE_MINUTES = 900

# Per-point-of-difficulty swing in the fixture factor (difficulty 2 -> +5%).
FIXTURE_SWING = 0.05

POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class ProjectedPlayer:
    """A player with a projected points-per-game for the coming gameweeks."""

    id: int
    name: str
    team_id: int
    team: str
    position: int  # 1=GKP, 2=DEF, 3=MID, 4=FWD
    cost: float  # in millions
    projected_ppg: float
    last_season_points: int
    last_season_ppg: float
    minutes: int
    selected_by_percent: float
    status: str
    news: str

    @property
    def position_name(self) -> str:
        return POSITION_NAMES[self.position]


def _availability(element: Dict[str, Any]) -> float:
    """Probability-of-playing multiplier from FPL status flags."""
    status = element["status"]
    if status == "u":  # unavailable / left the league
        return 0.0
    chance = element.get("chance_of_playing_next_round")
    if status in ("i", "s", "d") and chance is not None:
        return chance / 100.0
    if status in ("i", "s"):
        return 0.0
    return 1.0


def _luck_adjustment(element: Dict[str, Any], position: int) -> float:
    """Regress attackers whose returns outran (or lagged) their xGI."""
    if position == 1:  # keepers: xGI is irrelevant
        return 1.0
    goal_involvements = element["goals_scored"] + element["assists"]
    expected = float(element["expected_goal_involvements"])
    if goal_involvements < 3 or expected <= 0:
        return 1.0
    ratio = expected / goal_involvements
    # Blend half of the xGI signal in. The band is deliberately wide:
    # a 0.85 floor left 2024/25 overperformers (Salah, Wood, Wissa)
    # barely trimmed, which the 2025/26 backtest punished.
    adjustment = 0.5 + 0.5 * ratio
    return max(0.72, min(1.15, adjustment))


# Age (at season start) past which projections are discounted, per
# position, and the per-year discount beyond it. Keepers age slowest.
AGE_PEAK_END = {1: 34, 2: 30, 3: 29, 4: 29}
AGE_DECLINE_PER_YEAR = 0.04
AGE_FLOOR = 0.78


def _age_adjustment(
    element: Dict[str, Any], position: int, as_of: date
) -> float:
    """Discount players past their positional peak age."""
    birth = element.get("birth_date")
    if not birth:
        return 1.0
    try:
        born = date.fromisoformat(str(birth)[:10])
    except ValueError:
        return 1.0
    age = (as_of - born).days / 365.25
    years_past_peak = age - AGE_PEAK_END[position]
    if years_past_peak <= 0:
        return 1.0
    return max(AGE_FLOOR, 1.0 - AGE_DECLINE_PER_YEAR * years_past_peak)


def _fixture_factors(
    fixtures: List[Dict[str, Any]], next_event: int, horizon: int
) -> Dict[int, float]:
    """Average fixture-difficulty multiplier per team over the horizon."""
    difficulties: Dict[int, List[int]] = {}
    for fixture in fixtures:
        event = fixture.get("event")
        if event is None or not (next_event <= event < next_event + horizon):
            continue
        difficulties.setdefault(fixture["team_h"], []).append(
            fixture["team_h_difficulty"]
        )
        difficulties.setdefault(fixture["team_a"], []).append(
            fixture["team_a_difficulty"]
        )

    factors = {}
    for team_id, team_difficulties in difficulties.items():
        average = sum(team_difficulties) / len(team_difficulties)
        factors[team_id] = 1.0 + (3.0 - average) * FIXTURE_SWING
    return factors


def build_projections(
    bootstrap: Dict[str, Any],
    fixtures: List[Dict[str, Any]],
    next_event: int,
    horizon: int = 6,
    as_of: Optional[date] = None,
) -> List[ProjectedPlayer]:
    """Project points-per-game for every selectable player.

    `as_of` anchors the age adjustment (defaults to today); backtests
    should pass the historical season-start date.
    """
    as_of = as_of or date.today()
    team_names = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    fixture_factors = _fixture_factors(fixtures, next_event, horizon)

    players = []
    for element in bootstrap["elements"]:
        if not element.get("can_select", True) or element.get("removed"):
            continue

        position = element["element_type"]
        if position not in POSITION_BASELINE_PPG:
            continue  # ignore non-standard element types (e.g. managers)

        availability = _availability(element)
        minutes = element["minutes"]
        last_ppg = float(element["points_per_game"] or 0.0)

        reliability = minutes / (minutes + SHRINKAGE_MINUTES)
        baseline = POSITION_BASELINE_PPG[position]
        shrunk_ppg = reliability * last_ppg + (1 - reliability) * baseline

        projected = (
            shrunk_ppg
            * _luck_adjustment(element, position)
            * _age_adjustment(element, position, as_of)
            * availability
            * fixture_factors.get(element["team"], 1.0)
        )

        players.append(
            ProjectedPlayer(
                id=element["id"],
                name=element["web_name"],
                team_id=element["team"],
                team=team_names[element["team"]],
                position=position,
                cost=element["now_cost"] / 10.0,
                projected_ppg=round(projected, 3),
                last_season_points=element["total_points"],
                last_season_ppg=last_ppg,
                minutes=minutes,
                selected_by_percent=float(element["selected_by_percent"]),
                status=element["status"],
                news=element.get("news", ""),
            )
        )

    return players
