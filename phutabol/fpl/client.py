"""
Client for the official Fantasy Premier League API.

The FPL API is free and requires no authentication. Endpoints used:
- bootstrap-static: all players, teams, positions, and gameweeks
- fixtures: full season fixture list with difficulty ratings
"""

import requests
from typing import Dict, List, Any


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

    def get_next_event(self) -> Dict[str, Any]:
        """Return the next (or current) gameweek."""
        events = self.get_bootstrap()["events"]
        for event in events:
            if event.get("is_current"):
                return event
        for event in events:
            if event.get("is_next"):
                return event
        return events[0]
