"""
FPL squad optimizer.

Solves squad selection as a mixed-integer linear program (scipy.optimize.milp):

- 15-player squad: 2 GKP, 5 DEF, 5 MID, 3 FWD
- budget cap (default 100.0m)
- at most 3 players per club
- a legal starting XI is chosen jointly with the squad
  (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 total)

The objective maximises projected points of the XI plus a discounted
contribution from the bench, so the solver buys premium starters and
cheap-but-playing bench fodder rather than spreading the budget evenly.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

from .projections import ProjectedPlayer

SQUAD_QUOTAS = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3
BENCH_WEIGHT = 0.15


@dataclass
class Squad:
    """An optimised 15-player squad split into XI and bench."""

    starters: List[ProjectedPlayer]
    bench: List[ProjectedPlayer]
    captain: ProjectedPlayer
    vice_captain: ProjectedPlayer

    @property
    def players(self) -> List[ProjectedPlayer]:
        return self.starters + self.bench

    @property
    def total_cost(self) -> float:
        return round(sum(p.cost for p in self.players), 1)

    @property
    def projected_xi_ppg(self) -> float:
        """Projected XI points per gameweek, captain counted double."""
        return round(
            sum(p.projected_ppg for p in self.starters)
            + self.captain.projected_ppg,
            2,
        )


def optimize_squad(
    players: List[ProjectedPlayer], budget: float = 100.0
) -> Squad:
    """Pick the highest-projected legal squad within the budget."""
    n = len(players)
    if n == 0:
        raise ValueError("No players to optimize over")

    projections = np.array([p.projected_ppg for p in players])
    costs = np.array([p.cost for p in players])

    # Variables: x (in squad) followed by y (in starting XI).
    # Objective (maximise): sum(y * proj) + BENCH_WEIGHT * sum((x - y) * proj)
    objective = -np.concatenate(
        [BENCH_WEIGHT * projections, (1 - BENCH_WEIGHT) * projections]
    )

    constraints = []

    def selector(indices, on_x=False, on_y=False):
        row = np.zeros(2 * n)
        for i in indices:
            if on_x:
                row[i] = 1
            if on_y:
                row[n + i] = 1
        return row

    everyone = range(n)

    # Budget.
    constraints.append(
        LinearConstraint(np.concatenate([costs, np.zeros(n)]), 0, budget)
    )

    # Squad quotas and XI formation limits per position.
    for position, quota in SQUAD_QUOTAS.items():
        indices = [i for i, p in enumerate(players) if p.position == position]
        constraints.append(
            LinearConstraint(selector(indices, on_x=True), quota, quota)
        )
        constraints.append(
            LinearConstraint(
                selector(indices, on_y=True), XI_MIN[position], XI_MAX[position]
            )
        )

    # XI size.
    constraints.append(LinearConstraint(selector(everyone, on_y=True), 11, 11))

    # Club limit.
    for team_id in {p.team_id for p in players}:
        indices = [i for i, p in enumerate(players) if p.team_id == team_id]
        constraints.append(
            LinearConstraint(selector(indices, on_x=True), 0, MAX_PER_CLUB)
        )

    # Linking: y_i <= x_i.
    linking = lil_matrix((n, 2 * n))
    for i in range(n):
        linking[i, i] = 1
        linking[i, n + i] = -1
    constraints.append(LinearConstraint(linking.tocsr(), 0, np.inf))

    result = milp(
        c=objective,
        constraints=constraints,
        integrality=np.ones(2 * n),
        bounds=Bounds(0, 1),
    )
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    in_squad = result.x[:n] > 0.5
    in_xi = result.x[n:] > 0.5

    starters = [p for i, p in enumerate(players) if in_xi[i]]
    bench = [p for i, p in enumerate(players) if in_squad[i] and not in_xi[i]]

    starters.sort(key=lambda p: (p.position, -p.projected_ppg))
    # Bench order: keeper first, then outfielders by projection.
    bench.sort(key=lambda p: (p.position != 1, -p.projected_ppg))

    by_projection = sorted(starters, key=lambda p: -p.projected_ppg)
    return Squad(
        starters=starters,
        bench=bench,
        captain=by_projection[0],
        vice_captain=by_projection[1],
    )
