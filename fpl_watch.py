#!/usr/bin/env python3
"""
Change watcher and deadline alerter for a real FPL team.

Designed to be run repeatedly by a scheduler (launchd/cron, every ~30
minutes). Each run is one cheap pass:

1. If the next deadline is within --hours-before (default 24) and no
   plan has been produced for that gameweek yet, run the full advisor
   (fpl_manage.py), save the plan to disk, and send a notification.
2. Diff your squad's news, availability, chance-of-playing, and prices
   against the previous run; notify on any change. If news breaks after
   the plan was produced, the plan is regenerated and re-notified.

State lives in ~/.phutabol/ so runs are independent and idempotent —
you get at most one plan alert per gameweek, and one alert per actual
change. Notifications use macOS `osascript`; --no-notify prints to
stdout instead (useful for logs and testing).

Usage:
    python fpl_watch.py TEAM_ID [--hours-before 24] [--no-notify]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from phutabol.fpl import FPLClient
from phutabol.fpl.advisor import fetch_team_state

REPO_DIR = Path(__file__).resolve().parent
STATE_DIR = Path.home() / ".phutabol"

WATCHED_FIELDS = {
    "news": "news",
    "status": "status",
    "chance": "chance_of_playing_next_round",
    "cost": "now_cost",
}


def notify(title: str, message: str, enabled: bool) -> None:
    print(f"[{title}] {message}")
    if not enabled:
        return
    script = 'display notification "{}" with title "{}"'.format(
        message.replace("\\", "\\\\").replace('"', '\\"'),
        title.replace("\\", "\\\\").replace('"', '\\"'),
    )
    subprocess.run(["osascript", "-e", script], check=False)


def load_state(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"planned_event": None, "players": {}}


def snapshot_squad(bootstrap: Dict, squad_ids: List[int]) -> Dict[str, Dict]:
    elements = {e["id"]: e for e in bootstrap["elements"]}
    return {
        str(pid): {
            key: elements[pid].get(field)
            for key, field in WATCHED_FIELDS.items()
        }
        for pid in squad_ids
        if pid in elements
    }


def describe_changes(
    bootstrap: Dict, old: Dict[str, Dict], new: Dict[str, Dict]
) -> List[str]:
    names = {str(e["id"]): e["web_name"] for e in bootstrap["elements"]}
    changes = []
    for pid, now in new.items():
        before = old.get(pid)
        if before is None or before == now:
            continue
        name = names.get(pid, f"player {pid}")
        parts = []
        if before["news"] != now["news"]:
            parts.append(f"news: {now['news'] or 'cleared'}")
        if (before["status"], before["chance"]) != (
            now["status"], now["chance"]
        ):
            chance = now["chance"]
            parts.append(
                "available" if now["status"] == "a" and chance in (None, 100)
                else f"status {now['status']}, {chance}% to play"
            )
        if before["cost"] != now["cost"]:
            parts.append(
                f"price £{before['cost'] / 10:.1f}m → £{now['cost'] / 10:.1f}m"
            )
        if parts:
            changes.append(f"{name}: {'; '.join(parts)}")
    return changes


def generate_plan(
    team_id: int, event: int, free_transfers: Optional[int]
) -> Path:
    command = [sys.executable, str(REPO_DIR / "fpl_manage.py"), str(team_id)]
    if free_transfers is not None:
        command += ["--free-transfers", str(free_transfers)]
    result = subprocess.run(
        command, cwd=REPO_DIR, capture_output=True, text=True, timeout=600
    )
    plan_path = STATE_DIR / "plans" / f"gw{event}.txt"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    output = result.stdout
    if result.returncode != 0:
        output += f"\n[fpl_manage exited {result.returncode}]\n{result.stderr}"
    plan_path.write_text(output)
    return plan_path


def main() -> int:
    parser = argparse.ArgumentParser(description="FPL squad watcher")
    parser.add_argument("team_id", type=int, help="your FPL team ID")
    parser.add_argument(
        "--hours-before", type=float, default=24.0,
        help="produce the deadline plan this many hours ahead",
    )
    parser.add_argument(
        "--free-transfers", type=int, default=None,
        help="passed through to fpl_manage.py",
    )
    parser.add_argument(
        "--no-notify", action="store_true",
        help="print alerts instead of sending macOS notifications",
    )
    args = parser.parse_args()

    client = FPLClient()
    bootstrap = client.get_bootstrap()
    event = client.get_next_event()
    deadline = datetime.fromisoformat(
        event["deadline_time"].replace("Z", "+00:00")
    )
    hours_left = (
        deadline - datetime.now(timezone.utc)
    ).total_seconds() / 3600

    team_state = fetch_team_state(client, args.team_id)
    if team_state is None:
        print("No picks yet (pre-GW1) — run fpl_manage.py for an "
              "initial squad; watching starts after GW1.")
        return 0

    state_path = STATE_DIR / f"watch_{args.team_id}.json"
    state = load_state(state_path)
    squad_now = snapshot_squad(bootstrap, team_state.squad_ids)
    changes = describe_changes(bootstrap, state["players"], squad_now)

    plan_current = state["planned_event"] == event["id"]
    if hours_left <= args.hours_before and (
        not plan_current or (changes and hours_left > 0)
    ):
        plan_path = generate_plan(
            args.team_id, event["id"], args.free_transfers
        )
        verb = "refreshed (squad news)" if plan_current else "ready"
        notify(
            f"FPL — GW{event['id']} plan {verb}",
            f"Deadline in {hours_left:.0f}h. Plan: {plan_path}",
            enabled=not args.no_notify,
        )
        state["planned_event"] = event["id"]
    elif changes:
        summary = changes[0]
        if len(changes) > 1:
            summary += f" (+{len(changes) - 1} more)"
        notify("FPL — squad news", summary, enabled=not args.no_notify)

    for change in changes:
        print(f"  {change}")

    state["players"] = squad_now
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
