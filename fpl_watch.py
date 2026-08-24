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
change.

Notifications go to every channel configured in ~/.phutabol/notify.json
(Telegram gets the full plan text; ntfy.sh gets a push), plus a local
macOS notification when a GUI session exists. With no config the local
notification is all you get. --no-notify prints to stdout instead
(useful for logs and testing).

One-time Telegram setup: create a bot with @BotFather, send it any
message from your account, then:
    python fpl_watch.py --setup-telegram BOT_TOKEN

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

import requests

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


def load_notify_config() -> Dict:
    path = STATE_DIR / "notify.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def send_telegram(config: Dict, text: str) -> None:
    response = requests.post(
        "https://api.telegram.org/bot{}/sendMessage".format(
            config["telegram_token"]
        ),
        json={"chat_id": config["telegram_chat_id"], "text": text[:4000]},
        timeout=20,
    )
    if not response.ok:
        print(f"telegram send failed: {response.text[:200]}")


def send_ntfy(config: Dict, title: str, message: str) -> None:
    response = requests.post(
        f"https://ntfy.sh/{config['ntfy_topic']}",
        data=message.encode(),
        headers={"Title": title.encode("ascii", "ignore")},
        timeout=20,
    )
    if not response.ok:
        print(f"ntfy send failed: {response.text[:200]}")


def notify(
    title: str, message: str, enabled: bool, body: Optional[str] = None,
    local: bool = True,
) -> None:
    """Print always; when enabled, fan out to configured channels.

    `body` (the full plan text) rides along on Telegram, which fits it;
    push/local notifications carry only the short message.
    """
    print(f"[{title}] {message}")
    if not enabled:
        return
    config = load_notify_config()
    if config.get("telegram_token") and config.get("telegram_chat_id"):
        text = f"{title}\n{message}"
        if body:
            text += f"\n\n{body}"
        send_telegram(config, text)
    if config.get("ntfy_topic"):
        send_ntfy(config, title, message)
    if not local:
        return
    # Local notification too, when a GUI session exists (a headless
    # daemon run just fails this quietly).
    script = 'display notification "{}" with title "{}"'.format(
        message.replace("\\", "\\\\").replace('"', '\\"'),
        title.replace("\\", "\\\\").replace('"', '\\"'),
    )
    subprocess.run(
        ["osascript", "-e", script], check=False, capture_output=True
    )


def setup_telegram(token: str) -> int:
    """Discover the chat ID for `token` and save the notify config."""
    response = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates", timeout=20
    )
    payload = response.json()
    if not payload.get("ok"):
        print(f"Token rejected by Telegram: {payload}")
        return 1
    chats = {}
    for update in payload["result"]:
        chat = update.get("message", {}).get("chat")
        if chat:
            chats[chat["id"]] = chat.get(
                "username", chat.get("first_name", "?")
            )
    if not chats:
        print("No messages found — open Telegram, send your bot any "
              "message, then rerun this command.")
        return 1
    chat_id, username = list(chats.items())[-1]
    config = load_notify_config()
    config.update(telegram_token=token, telegram_chat_id=chat_id)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "notify.json"
    path.write_text(json.dumps(config, indent=1))
    path.chmod(0o600)
    send_telegram(config, "FPL watcher connected — alerts will "
                          "arrive here.")
    print(f"Saved {path} for chat {chat_id} (@{username}) and sent a "
          f"test message.")
    return 0


def heartbeat_due(state: Dict, every_hours: float) -> bool:
    """True on the first run, then once per `every_hours`."""
    last = state.get("last_heartbeat")
    if not last:
        return True
    elapsed = (
        datetime.now(timezone.utc) - datetime.fromisoformat(last)
    ).total_seconds() / 3600
    return elapsed >= every_hours


def build_heartbeat(
    state: Dict,
    bootstrap: Dict,
    squad_now: Dict[str, Dict],
    event: Dict,
    hours_left: float,
    hours_before: float,
) -> str:
    """A proof-of-life digest: what the scheduler did since last time."""
    deadline = datetime.fromisoformat(
        event["deadline_time"].replace("Z", "+00:00")
    ).astimezone()
    names = {str(e["id"]): e["web_name"] for e in bootstrap["elements"]}

    lines = [
        "Watcher alive - {} pass(es) since last check-in.".format(
            state.get("runs", 0)
        ),
        "GW{} deadline: {} (in {:.0f}h)".format(
            event["id"], deadline.strftime("%a %d %b %H:%M %Z"), hours_left
        ),
    ]
    if state.get("planned_event") == event["id"]:
        lines.append(f"Plan: GW{event['id']} sent - see plans/gw{event['id']}.txt")
    else:
        lines.append(
            "Plan: generates {:.0f}h before deadline (in {:.0f}h)".format(
                hours_before, max(0.0, hours_left - hours_before)
            )
        )

    flagged = []
    for pid, info in squad_now.items():
        chance = info.get("chance")
        if info.get("status") != "a" or chance not in (None, 100):
            flagged.append(
                "{} ({}%)".format(names.get(pid, pid), chance
                                  if chance is not None else "?")
            )
    lines.append(
        "Squad: {} tracked, {}".format(
            len(squad_now),
            "all available" if not flagged
            else f"{len(flagged)} flagged - " + ", ".join(flagged[:5]),
        )
    )

    recent = state.get("recent_changes", [])
    if recent:
        lines.append(f"Changes since last check-in ({len(recent)}):")
        lines.extend(f"  {c}" for c in recent[-10:])
    else:
        lines.append("Changes since last check-in: none")
    return "\n".join(lines)


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
    parser.add_argument(
        "team_id", type=int, nargs="?", help="your FPL team ID"
    )
    parser.add_argument(
        "--setup-telegram", metavar="BOT_TOKEN",
        help="save Telegram credentials to ~/.phutabol/notify.json "
             "(message your bot first) and exit",
    )
    parser.add_argument(
        "--hours-before", type=float, default=24.0,
        help="produce the deadline plan this many hours ahead",
    )
    parser.add_argument(
        "--free-transfers", type=int, default=None,
        help="passed through to fpl_manage.py",
    )
    parser.add_argument(
        "--heartbeat-hours", type=float, default=24.0,
        help="send a proof-of-life digest this often (0 disables)",
    )
    parser.add_argument(
        "--no-notify", action="store_true",
        help="print alerts instead of sending macOS notifications",
    )
    args = parser.parse_args()

    if args.setup_telegram:
        return setup_telegram(args.setup_telegram)
    if args.team_id is None:
        parser.error("team_id is required")

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

    state.setdefault("runs", 0)
    state.setdefault("recent_changes", [])
    state.setdefault("last_heartbeat", None)
    state["runs"] += 1
    if changes:
        stamp = datetime.now(timezone.utc).strftime("%a %H:%M UTC")
        state["recent_changes"].extend(f"{stamp} - {c}" for c in changes)
        state["recent_changes"] = state["recent_changes"][-25:]

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
            body=plan_path.read_text(),
        )
        state["planned_event"] = event["id"]
    elif changes:
        summary = changes[0]
        if len(changes) > 1:
            summary += f" (+{len(changes) - 1} more)"
        notify("FPL — squad news", summary, enabled=not args.no_notify)

    for change in changes:
        print(f"  {change}")

    if args.heartbeat_hours > 0 and heartbeat_due(
        state, args.heartbeat_hours
    ):
        notify(
            "FPL watcher - check-in",
            build_heartbeat(
                state, bootstrap, squad_now, event, hours_left,
                args.hours_before,
            ),
            enabled=not args.no_notify,
            local=False,
        )
        state["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        state["runs"] = 0
        state["recent_changes"] = []

    state["players"] = squad_now
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
