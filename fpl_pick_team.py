#!/usr/bin/env python3
"""
Pick an optimal FPL squad from live Fantasy Premier League data.

Usage:
    python fpl_pick_team.py [--budget 100.0] [--horizon 6]

Fetches current player prices, availability, and fixture difficulty from
the official FPL API, projects points per game for the coming gameweeks,
and solves squad selection as an integer program.
"""

import argparse

from phutabol.fpl import FPLClient, build_projections, optimize_squad


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize an FPL squad")
    parser.add_argument(
        "--budget", type=float, default=100.0, help="budget in millions"
    )
    parser.add_argument(
        "--horizon", type=int, default=6,
        help="gameweeks of fixtures to average over",
    )
    args = parser.parse_args()

    client = FPLClient()
    bootstrap = client.get_bootstrap()
    fixtures = client.get_fixtures()
    next_event = client.get_next_event()

    print(f"⚽ Optimizing for {next_event['name']} "
          f"(deadline {next_event['deadline_time']})")

    players = build_projections(
        bootstrap, fixtures, next_event["id"], horizon=args.horizon
    )
    squad = optimize_squad(players, budget=args.budget)

    print(f"\n{'='*74}\nSTARTING XI\n{'='*74}")
    row = ("{name:<22}{team:<6}{pos:<5}{cost:>6}{proj:>8}{last:>12}{sel:>9}")
    header = row.format(
        name="Player", team="Team", pos="Pos", cost="£m",
        proj="Proj/GW", last="Pts 25/26", sel="Sel%",
    )
    print(header + "\n" + "-" * 74)
    for player in squad.starters:
        armband = ""
        if player is squad.captain:
            armband = " (C)"
        elif player is squad.vice_captain:
            armband = " (V)"
        print(row.format(
            name=player.name + armband, team=player.team,
            pos=player.position_name, cost=f"{player.cost:.1f}",
            proj=f"{player.projected_ppg:.2f}",
            last=player.last_season_points,
            sel=f"{player.selected_by_percent:.1f}",
        ))

    print(f"\nBENCH (in order)\n" + "-" * 74)
    for player in squad.bench:
        print(row.format(
            name=player.name, team=player.team, pos=player.position_name,
            cost=f"{player.cost:.1f}", proj=f"{player.projected_ppg:.2f}",
            last=player.last_season_points,
            sel=f"{player.selected_by_percent:.1f}",
        ))

    print("-" * 74)
    print(f"Total cost: £{squad.total_cost}m of £{args.budget}m   |   "
          f"Projected XI pts/GW (captain doubled): {squad.projected_xi_ppg}")


if __name__ == "__main__":
    main()
