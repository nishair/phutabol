"""
Client for the official Fantasy Premier League API.

The FPL API is free and requires no authentication. Endpoints used:
- bootstrap-static: all players, teams, positions, and gameweeks
- fixtures: full season fixture list with difficulty ratings
"""

import requests
from typing import Dict, List, Any, Optional


class FPLClient:
    """Fetches data from the official Fantasy Premier League API."""

    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (phutabol)"})
        self._bootstrap: Dict[str, Any] = {}
        self._fixtures: List[Dict[str, Any]] = []

    def get_bootstrap(self) -> Dict[str, Any]:
        """Fetch (and cache) the bootstrap-static payload."""
        if not self._bootstrap:
            response = self.session.get(
                f"{self.BASE_URL}/bootstrap-static/", timeout=self.timeout
            )
            response.raise_for_status()
            self._bootstrap = response.json()
        return self._bootstrap

    def get_fixtures(self) -> List[Dict[str, Any]]:
        """Fetch (and cache) the full fixture list."""
        if not self._fixtures:
            response = self.session.get(
                f"{self.BASE_URL}/fixtures/", timeout=self.timeout
            )
            response.raise_for_status()
            self._fixtures = response.json()
        return self._fixtures

    def _get_json(self, path: str) -> Optional[Any]:
        response = self.session.get(
            f"{self.BASE_URL}/{path}", timeout=self.timeout
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def get_entry(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Public metadata for a manager's team; None if it doesn't exist."""
        return self._get_json(f"entry/{team_id}/")

    def get_entry_history(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Season history and chips played."""
        return self._get_json(f"entry/{team_id}/history/")

    def get_entry_picks(
        self, team_id: int, event: int
    ) -> Optional[Dict[str, Any]]:
        """Picks for a gameweek (public once its deadline has passed)."""
        return self._get_json(f"entry/{team_id}/event/{event}/picks/")

    def get_entry_transfers(self, team_id: int) -> List[Dict[str, Any]]:
        """Full transfer ledger for a team."""
        return self._get_json(f"entry/{team_id}/transfers/") or []

    def get_next_event(self) -> Dict[str, Any]:
        """Return the next gameweek whose deadline hasn't passed.

        FPL flags: `is_next` marks the upcoming deadline; `is_current`
        marks the latest gameweek whose deadline already passed, so it
        is only a fallback (season over / pre-season edge cases).
        """
        events = self.get_bootstrap()["events"]
        for event in events:
            if event.get("is_next"):
                return event
        for event in events:
            if event.get("is_current"):
                return event
        return events[0]
