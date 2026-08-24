#!/usr/bin/env python3
"""
Backtest the FPL squad optimizer against a completed season.

Rebuilds the information state at the start of a season (previous-season
stats, GW1 prices, fixture difficulty), runs the exact production
pipeline (build_projections -> optimize_squad), then replays the season
two ways:

1. static  — the GW1 squad held all season (no transfers, no chips)
2. managed — SeasonManager plays every week: re-picked XI and captain,
   1 bankable free transfer, -4 hits, wildcards, free hit, bench boost,
   triple captain

Data: vaastav/Fantasy-Premier-League GitHub archive (per-GW scores,
season-end stats, fixtures). Players are joined across seasons on their
permanent FPL `code`.

Usage:
    python fpl_backtest.py [--season 2025-26] [--budget 100.0]
"""

import argparse
import io
from collections import defaultdict
from datetime import date
from typing import Dict, Optional

import pandas as pd
import requests

from phutabol.fpl import build_projections, optimize_squad
from phutabol.fpl.season import (
    SeasonData, SeasonManager, apply_auto_subs, pick_weekly_xi,
)

ARCHIVE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League"
    "/master/data"
)


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
    players: pd.DataFrame,
    prior: pd.DataFrame,
    teams: pd.DataFrame,
    birth_dates: Optional[Dict[int, str]] = None,
) -> Dict:
    """Reconstruct a bootstrap-static payload as of GW1 of the season.

    Current-season stat fields are filled from the *prior* season (joined
    on permanent player code), prices are rolled back to their season-start
    values, and everyone is marked available — injury news from the
    season's opening week is not in the archive.

    Old archive seasons lack xGI (pre-2022/23) and birth dates
    (pre-2024/25); missing xGI disables the luck regression for those
    players, and `birth_dates` (permanent code -> ISO date) lets callers
    backfill ages from newer data, which is hindsight-free.
    """
    prior_by_code = prior.set_index("code")
    birth_dates = birth_dates or {}
    elements = []
    for row in players.itertuples():
        stats = (
            prior_by_code.loc[row.code]
            if row.code in prior_by_code.index
            else None
        )

        def stat(name, cast, default):
            value = getattr(stats, name, None) if stats is not None else None
            return cast(value) if pd.notna(value) else default

        elements.append({
            "id": row.id,
            "web_name": row.web_name,
            "team": row.team,
            "element_type": row.element_type,
            "now_cost": row.now_cost - row.cost_change_start,
            "birth_date": (
                getattr(row, "birth_date", None)
                or birth_dates.get(row.code)
            ),
            "status": "a",
            "chance_of_playing_next_round": None,
            "can_select": True,
            "removed": False,
            "news": "",
            "selected_by_percent": row.selected_by_percent,
            "minutes": stat("minutes", int, 0),
            "points_per_game": stat("points_per_game", float, 0.0),
            "total_points": stat("total_points", int, 0),
            "goals_scored": stat("goals_scored", int, 0),
            "assists": stat("assists", int, 0),
            "expected_goal_involvements": stat(
                "expected_goal_involvements", float, 0.0
            ),
        })

    return {
        "elements": elements,
        "teams": teams.to_dict("records"),
    }


def build_season_data(
    gws: pd.DataFrame, fixtures: pd.DataFrame, start_prices: Dict[int, int]
) -> SeasonData:
    """Bundle per-GW scores, prices (carried forward), and fixtures."""
    scores: Dict[int, Dict[int, Dict]] = defaultdict(dict)
    aggregates = {
        "points": ("total_points", "sum"),
        "minutes": ("minutes", "sum"),
        "value": ("value", "first"),
    }
    # Crowd signal, where the archive has it. Ownership and transfer
    # counts are frozen at each deadline, so they are knowable inputs
    # for that gameweek's decisions, not hindsight.
    for crowd in ("selected", "transfers_balance"):
        if crowd in gws.columns:
            aggregates[crowd] = (crowd, "first")
    grouped = gws.groupby(["GW", "element"]).agg(**aggregates)
    for (gw, element), row in grouped.iterrows():
        scores[gw][element] = {
            "points": int(row.points),
            "minutes": int(row.minutes),
            "selected": int(getattr(row, "selected", 0) or 0),
            "transfers_balance": int(
                getattr(row, "transfers_balance", 0) or 0
            ),
        }

    n_events = int(gws["GW"].max())
    prices: Dict[int, Dict[int, int]] = {}
    last = dict(start_prices)
    for gw in range(1, n_events + 1):
        if gw in scores:
            for (gw_key, element), row in grouped.loc[[gw]].iterrows():
                last[element] = int(row.value)
        prices[gw] = dict(last)

    fixture_map: Dict[int, Dict[int, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in fixtures.itertuples():
        if pd.isna(row.event):
            continue
        event = int(row.event)
        fixture_map[event][row.team_h].append(int(row.team_h_difficulty))
        fixture_map[event][row.team_a].append(int(row.team_a_difficulty))

    return SeasonData(
        scores=dict(scores),
        prices=prices,
        fixtures={k: dict(v) for k, v in fixture_map.items()},
        n_events=n_events,
    )


def simulate_static(squad, data: SeasonData) -> int:
    """Hold the GW1 squad all season; fixed captain/vice; auto-subs."""
    total = 0
    for gw in range(1, data.n_events + 1):
        week = data.scores.get(gw, {})
        lineup = apply_auto_subs(squad.starters, squad.bench, week)
        total += sum(week.get(p.id, {}).get("points", 0) for p in lineup)
        for leader in (squad.captain, squad.vice_captain):
            if week.get(leader.id, {}).get("minutes", 0) > 0:
                total += week[leader.id]["points"]
                break
    return total


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
    season_start = date(int(args.season.split("-")[0]), 8, 1)
    projections = build_projections(
        bootstrap, fixtures.to_dict("records"), next_event=1,
        as_of=season_start,
    )

    start_prices = {e["id"]: e["now_cost"] for e in bootstrap["elements"]}
    data = build_season_data(gws, fixtures, start_prices)
    season_totals = players.set_index("id")["total_points"]

    # --- Static replay -------------------------------------------------
    squad = optimize_squad(projections, budget=args.budget)
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

    static_total = simulate_static(squad, data)

    # --- Managed replay ------------------------------------------------
    manager = SeasonManager(projections, data, budget=args.budget)
    weeks = manager.run()
    managed_total = sum(w.points for w in weeks)
    n_transfers = sum(len(w.transfers) for w in weeks)
    n_hits = sum(w.hits for w in weeks)

    print(f"\n{'='*60}")
    print(f"Static squad (no transfers, no chips):  {static_total} pts")
    print(f"Managed season:                         {managed_total} pts")
    print(f"{'='*60}")
    print(f"Transfers: {n_transfers} ({n_hits} hits, -{4*n_hits} pts)   "
          f"Avg/GW: {managed_total / len(weeks):.1f}")
    for week in weeks:
        if week.chip:
            print(f"  {week.chip:<4} played GW{week.gw}: {week.points} pts")
    best = max(weeks, key=lambda w: w.points)
    worst = min(weeks, key=lambda w: w.points)
    print(f"  Best GW{best.gw}: {best.points}   Worst GW{worst.gw}: "
          f"{worst.points}")

    # --- Hindsight ceiling (static, same rules) ------------------------
    for projection in projections:
        projection.projected_ppg = season_totals.get(projection.id, 0) / 38.0
    ceiling = optimize_squad(projections, budget=args.budget)
    print(f"\nPerfect-hindsight static squad, same static rules: "
          f"{simulate_static(ceiling, data)} pts")


if __name__ == "__main__":
    main()
