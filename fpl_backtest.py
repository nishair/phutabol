#!/usr/bin/env python3
"""
Backtest the FPL squad optimizer against a completed season.

Rebuilds the information state at the start of a season (previous-season
stats, GW1 prices, fixture difficulty), runs the exact production
pipeline (build_projections -> optimize_squad), then replays every
gameweek with actual scores, FPL auto-substitution rules, and
captain/vice doubling.

Assumptions (stated, since they differ from a real manager's season):
- no transfers, no chips: the GW1 squad is held all season
- season-long captain = model captain (vice steps in on blanks)
- everyone assumed fit at the season-start pick (no injury news)

Data: vaastav/Fantasy-Premier-League GitHub archive (per-GW scores,
season-end stats, fixtures). Players are joined across seasons on their
permanent FPL `code`.

Usage:
    python fpl_backtest.py [--season 2025-26] [--budget 100.0]
"""

import argparse
import io
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import requests

from phutabol.fpl import build_projections, optimize_squad
from phutabol.fpl.projections import ProjectedPlayer

ARCHIVE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League"
    "/master/data"
)

# Formation minimums that auto-substitution must preserve.
XI_MINIMUMS = {1: 1, 2: 3, 3: 2, 4: 1}


def fetch_csv(path: str) -> pd.DataFrame:
    response = requests.get(
        f"{ARCHIVE}/{path}",
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (phutabol)"},
    )
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def previous_season(season: str) -> str:
    start = int(season.split("-")[0])
    return f"{start - 1}-{str(start)[-2:]}"


def build_season_start_bootstrap(
    players: pd.DataFrame, prior: pd.DataFrame, teams: pd.DataFrame
) -> Dict:
    """Reconstruct a bootstrap-static payload as of GW1 of the season.

    Current-season stat fields are filled from the *prior* season (joined
    on permanent player code), prices are rolled back to their season-start
    values, and everyone is marked available — injury news from the
    season's opening week is not in the archive.
    """
    prior_by_code = prior.set_index("code")
    elements = []
    for row in players.itertuples():
        stats = (
            prior_by_code.loc[row.code]
            if row.code in prior_by_code.index
            else None
        )
        elements.append({
            "id": row.id,
            "web_name": row.web_name,
            "team": row.team,
            "element_type": row.element_type,
            "now_cost": row.now_cost - row.cost_change_start,
            "status": "a",
            "chance_of_playing_next_round": None,
            "can_select": True,
            "removed": False,
            "news": "",
            "selected_by_percent": row.selected_by_percent,
            "minutes": int(stats.minutes) if stats is not None else 0,
            "points_per_game": (
                float(stats.points_per_game) if stats is not None else 0.0
            ),
            "total_points": int(stats.total_points) if stats is not None else 0,
            "goals_scored": int(stats.goals_scored) if stats is not None else 0,
            "assists": int(stats.assists) if stats is not None else 0,
            "expected_goal_involvements": (
                float(stats.expected_goal_involvements)
                if stats is not None else 0.0
            ),
        })

    return {
        "elements": elements,
        "teams": teams.to_dict("records"),
    }


def gameweek_scores(gws: pd.DataFrame) -> Dict[int, Dict[int, Dict]]:
    """{gw: {element_id: {points, minutes}}}, double gameweeks summed."""
    scores: Dict[int, Dict[int, Dict]] = defaultdict(dict)
    grouped = gws.groupby(["GW", "element"]).agg(
        points=("total_points", "sum"), minutes=("minutes", "sum")
    )
    for (gw, element), row in grouped.iterrows():
        scores[gw][element] = {
            "points": int(row.points), "minutes": int(row.minutes)
        }
    return scores


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
            # Keepers swap only with keepers; outfielders with outfielders.
            if (starter.position == 1) != (sub.position == 1):
                continue
            counts = defaultdict(int)
            for p in lineup:
                counts[p.position] += 1
            counts[starter.position] -= 1
            counts[sub.position] += 1
            if all(counts[pos] >= XI_MINIMUMS[pos] for pos in XI_MINIMUMS):
                lineup[i] = sub
                available_bench.remove(sub)
                break
    return lineup


def simulate_season(squad, scores: Dict[int, Dict[int, Dict]]) -> pd.DataFrame:
    """Replay the season week by week with a static squad."""
    rows = []
    for gw in sorted(scores):
        week = scores[gw]
        lineup = apply_auto_subs(squad.starters, squad.bench, week)
        base = sum(week.get(p.id, {}).get("points", 0) for p in lineup)

        armband = 0
        for leader in (squad.captain, squad.vice_captain):
            if week.get(leader.id, {}).get("minutes", 0) > 0:
                armband = week[leader.id]["points"]
                break
        rows.append({"GW": gw, "points": base + armband})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the FPL optimizer")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--budget", type=float, default=100.0)
    args = parser.parse_args()

    prior = previous_season(args.season)
    print(f"Backtesting {args.season} (projections from {prior} stats)…")

    players = fetch_csv(f"{args.season}/players_raw.csv")
    prior_players = fetch_csv(f"{prior}/players_raw.csv")
    teams = fetch_csv(f"{args.season}/teams.csv")
    gws = fetch_csv(f"{args.season}/gws/merged_gw.csv")
    fixtures = fetch_csv(f"{args.season}/fixtures.csv")

    bootstrap = build_season_start_bootstrap(players, prior_players, teams)
    projections = build_projections(
        bootstrap, fixtures.to_dict("records"), next_event=1
    )
    squad = optimize_squad(projections, budget=args.budget)

    season_totals = players.set_index("id")["total_points"]
    print(f"\nGW1 squad (£{squad.total_cost}m):")
    for player in squad.players:
        role = ""
        if player is squad.captain:
            role = " (C)"
        elif player is squad.vice_captain:
            role = " (V)"
        bench_tag = "  [bench]" if player in squad.bench else ""
        print(f"  {player.position_name}  {player.name+role:<22}"
              f"{player.team:<5}£{player.cost:<6.1f}"
              f"actual {args.season} pts: "
              f"{season_totals.get(player.id, 0)}{bench_tag}")

    scores = gameweek_scores(gws)
    results = simulate_season(squad, scores)

    total = results["points"].sum()
    print(f"\n{'='*56}")
    print(f"Season total (no transfers, no chips): {total} pts")
    print(f"Average per GW: {results['points'].mean():.1f}   "
          f"Best GW: {results['points'].max()} (GW"
          f"{results.loc[results['points'].idxmax(), 'GW']})   "
          f"Worst GW: {results['points'].min()} (GW"
          f"{results.loc[results['points'].idxmin(), 'GW']})")

    # Hindsight ceiling: same optimizer fed actual season totals.
    for projection in projections:
        projection.projected_ppg = season_totals.get(projection.id, 0) / 38.0
    ceiling = optimize_squad(projections, budget=args.budget)
    ceiling_results = simulate_season(ceiling, scores)
    print(f"Perfect-hindsight static squad, same rules: "
          f"{ceiling_results['points'].sum()} pts")


if __name__ == "__main__":
    main()
