#!/usr/bin/env python3
"""
Weekly deadline advisor for your real FPL team.

Usage:
    python fpl_manage.py TEAM_ID [--free-transfers N]

Your team ID is in the URL on the FPL site: Points tab ->
fantasy.premierleague.com/entry/<TEAM_ID>/event/1

Reads your actual squad, bank, chips, and transfer ledger from the
public FPL API (no login), and recommends this deadline's transfers,
starting XI, captain, and chip using the model tuned on eight seasons
of backtests. Before GW1 (when no picks exist yet) it falls back to
recommending an initial squad, and caches the pre-season projections
that in-season runs blend from.

The model handles the stats; you hold the veto on team news.
"""

import argparse
import sys

from phutabol.fpl import FPLClient, optimize_squad
from phutabol.fpl.advisor import Advisor, fetch_team_state


def print_squad_fallback(advisor: Advisor, budget: float) -> None:
    print("No picks found for this team yet (pre-GW1?).\n"
          "Recommended initial squad — enter it on the FPL site:\n")
    squad = optimize_squad(advisor.projections, budget=budget)
    for player in squad.players:
        role = ""
        if player is squad.captain:
            role = " (C)"
        elif player is squad.vice_captain:
            role = " (V)"
        bench = "  [bench]" if player in squad.bench else ""
        print(f"  {player.position_name}  {player.name+role:<22}"
              f"{player.team:<5}£{player.cost:.1f}{bench}")
    print(f"\nTotal: £{squad.total_cost}m")


def main() -> None:
    parser = argparse.ArgumentParser(description="FPL deadline advisor")
    parser.add_argument("team_id", type=int, help="your FPL team ID")
    parser.add_argument(
        "--free-transfers", type=int, default=None,
        help="override the estimated banked free transfers",
    )
    parser.add_argument("--budget", type=float, default=100.0,
                        help="budget for the pre-GW1 fallback squad")
    args = parser.parse_args()

    client = FPLClient()
    advisor = Advisor(client)
    state = fetch_team_state(client, args.team_id)

    if state is None:
        print_squad_fallback(advisor, args.budget)
        return

    if args.free_transfers is not None:
        state.free_transfers = args.free_transfers

    advice = advisor.advise(state)
    weekly = advisor.weekly_projection(advice.event)

    print(f"{state.name} — GW{advice.event} plan "
          f"(deadline {advice.deadline})")
    print(f"Bank £{state.bank / 10:.1f}m · "
          f"~{state.free_transfers} free transfer(s) "
          f"(estimated; override with --free-transfers)\n")

    if advice.chip:
        print(f"CHIP: play your {advice.chip.upper()} — "
              f"{advice.chip_reason}")
        if advice.chip_out or advice.chip_in:
            print(f"  OUT: {', '.join(advice.chip_out)}")
            print(f"  IN:  {', '.join(advice.chip_in)}")
            print("  (XI and bench below are the rebuilt squad)")
        print()

    if advice.chip_note:
        print(f"CHIP: {advice.chip_note}\n")

    if advice.transfers:
        print("TRANSFERS:")
        for out_name, in_name, gain in advice.transfers:
            print(f"  OUT {out_name:<20} IN {in_name:<20} "
                  f"(+{gain} pts over horizon)")
        if advice.hit_cost:
            print(f"  (includes -{advice.hit_cost} hit — "
                  f"projected gain covers it)")
        print()
    elif advice.chip not in ("wildcard", "freehit"):
        print("TRANSFERS: none — bank the free transfer\n")

    print("STARTING XI:")
    for player in advice.starters:
        role = ""
        if player is advice.captain:
            role = " (C)"
        elif player is advice.vice:
            role = " (V)"
        print(f"  {player.position_name}  {player.name+role:<22}"
              f"{player.team:<5}proj {weekly.get(player.id, 0):.2f}")
    print("BENCH (in order):")
    for player in advice.bench:
        print(f"  {player.position_name}  {player.name:<22}"
              f"{player.team:<5}proj {weekly.get(player.id, 0):.2f}")

    if advice.notes:
        print("\n⚠ NEWS ON YOUR PLAYERS (your veto beats the model):")
        for note in advice.notes:
            print(f"  {note}")


if __name__ == "__main__":
    sys.exit(main())
