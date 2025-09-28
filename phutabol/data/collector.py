import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json

from ..models.team import Team, TeamForm
from ..models.match import Match, MatchStats, MatchContext, MatchStatus


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    async def get_teams(self, league: str) -> List[Team]:
        """Fetch teams for a specific league."""
        pass

    @abstractmethod
    async def get_matches(self, league: str, start_date: datetime, end_date: datetime) -> List[Match]:
        """Fetch matches for a specific league and date range."""
        pass

    @abstractmethod
    async def get_match_stats(self, match_id: str) -> Optional[MatchStats]:
        """Fetch detailed statistics for a specific match."""
        pass

    @abstractmethod
    async def get_team_form(self, team_id: str, num_matches: int = 5) -> TeamForm:
        """Fetch recent form for a team."""
        pass


class MockDataSource(DataSource):
    """Mock data source for testing and development."""

    def __init__(self):
        self.teams_data = self._generate_mock_teams()
        self.matches_data = self._generate_mock_matches()

    def _generate_mock_teams(self) -> List[Team]:
        """Generate mock team data for testing."""
        teams = [
            Team(
                id="team_1", name="Manchester United", league="Premier League", country="England",
                elo_rating=1650, matches_played=20, wins=12, draws=5, losses=3,
                goals_for=35, goals_against=18, expected_goals_for=32.5, expected_goals_against=20.1
            ),
            Team(
                id="team_2", name="Liverpool", league="Premier League", country="England",
                elo_rating=1720, matches_played=20, wins=14, draws=4, losses=2,
                goals_for=42, goals_against=15, expected_goals_for=38.7, expected_goals_against=16.3
            ),
            Team(
                id="team_3", name="Arsenal", league="Premier League", country="England",
                elo_rating=1680, matches_played=20, wins=13, draws=3, losses=4,
                goals_for=38, goals_against=22, expected_goals_for=36.2, expected_goals_against=19.8
            ),
            Team(
                id="team_4", name="Chelsea", league="Premier League", country="England",
                elo_rating=1620, matches_played=20, wins=11, draws=6, losses=3,
                goals_for=31, goals_against=19, expected_goals_for=29.8, expected_goals_against=21.5
            ),
        ]
        return teams

    def _generate_mock_matches(self) -> List[Match]:
        """Generate mock match data for testing."""
        matches = []
        base_date = datetime.now() - timedelta(days=30)

        for i in range(10):
            match = Match(
                id=f"match_{i+1}",
                home_team_id="team_1" if i % 2 == 0 else "team_2",
                away_team_id="team_3" if i % 2 == 0 else "team_4",
                league="Premier League",
                season="2023-24",
                scheduled_datetime=base_date + timedelta(days=i*3),
                status=MatchStatus.FINISHED,
                home_score=2 if i % 3 == 0 else 1,
                away_score=1 if i % 3 == 0 else 0,
            )
            matches.append(match)

        return matches

    async def get_teams(self, league: str) -> List[Team]:
        await asyncio.sleep(0.1)  # Simulate API delay
        return [team for team in self.teams_data if team.league == league]

    async def get_matches(self, league: str, start_date: datetime, end_date: datetime) -> List[Match]:
        await asyncio.sleep(0.1)
        return [
            match for match in self.matches_data
            if match.league == league and
            start_date <= match.scheduled_datetime <= end_date
        ]

    async def get_match_stats(self, match_id: str) -> Optional[MatchStats]:
        await asyncio.sleep(0.1)
        return MatchStats(
            possession_home=55.0, possession_away=45.0,
            shots_home=12, shots_away=8,
            shots_on_target_home=5, shots_on_target_away=3,
            xg_home=1.8, xg_away=1.2,
            fouls_home=10, fouls_away=14,
            corners_home=6, corners_away=4
        )

    async def get_team_form(self, team_id: str, num_matches: int = 5) -> TeamForm:
        await asyncio.sleep(0.1)
        recent_matches = [
            {"result": "W", "goals_for": 2, "goals_against": 1},
            {"result": "W", "goals_for": 3, "goals_against": 0},
            {"result": "D", "goals_for": 1, "goals_against": 1},
            {"result": "W", "goals_for": 2, "goals_against": 0},
            {"result": "L", "goals_for": 0, "goals_against": 2},
        ][:num_matches]

        form_string = "".join([match["result"] for match in recent_matches])

        return TeamForm(
            team_id=team_id,
            recent_matches=recent_matches,
            form_string=form_string
        )


class FootballAPISource(DataSource):
    """Data source that fetches from football APIs (placeholder for real implementation)."""

    def __init__(self, api_key: str, base_url: str = "https://api.football-data.org/v4"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-Auth-Token": self.api_key}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_teams(self, league: str) -> List[Team]:
        """Fetch teams from API (placeholder implementation)."""
        # This would make actual API calls to fetch team data
        # For now, return empty list as placeholder
        return []

    async def get_matches(self, league: str, start_date: datetime, end_date: datetime) -> List[Match]:
        """Fetch matches from API (placeholder implementation)."""
        return []

    async def get_match_stats(self, match_id: str) -> Optional[MatchStats]:
        """Fetch match statistics from API (placeholder implementation)."""
        return None

    async def get_team_form(self, team_id: str, num_matches: int = 5) -> TeamForm:
        """Fetch team form from API (placeholder implementation)."""
        return TeamForm(team_id=team_id)


class DataCollector:
    """Main data collection orchestrator."""

    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        self.cache = {}

    async def collect_league_data(self, league: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect comprehensive data for a league."""

        # Fetch teams and matches concurrently
        teams_task = self.data_source.get_teams(league)
        matches_task = self.data_source.get_matches(league, start_date, end_date)

        teams, matches = await asyncio.gather(teams_task, matches_task)

        # Update team statistics based on matches
        self._update_team_stats(teams, matches)

        # Collect recent form for each team
        form_tasks = [
            self.data_source.get_team_form(team.id)
            for team in teams
        ]
        team_forms = await asyncio.gather(*form_tasks)

        # Create team form mapping
        team_form_map = {form.team_id: form for form in team_forms}

        # Collect match statistics
        stat_tasks = [
            self.data_source.get_match_stats(match.id)
            for match in matches if match.is_finished
        ]
        match_stats = await asyncio.gather(*stat_tasks)

        # Update matches with statistics
        for match, stats in zip(matches, match_stats):
            if stats:
                match.stats = stats

        return {
            "teams": teams,
            "matches": matches,
            "team_forms": team_form_map,
            "collection_timestamp": datetime.now()
        }

    def _update_team_stats(self, teams: List[Team], matches: List[Match]):
        """Update team statistics based on match results."""
        team_map = {team.id: team for team in teams}

        for match in matches:
            if not match.is_finished:
                continue

            home_team = team_map.get(match.home_team_id)
            away_team = team_map.get(match.away_team_id)

            if home_team and away_team:
                # Update match counts
                home_team.matches_played += 1
                away_team.matches_played += 1

                # Update goals
                home_team.goals_for += match.home_score
                home_team.goals_against += match.away_score
                away_team.goals_for += match.away_score
                away_team.goals_against += match.home_score

                # Update results
                if match.home_score > match.away_score:
                    home_team.wins += 1
                    away_team.losses += 1
                elif match.home_score < match.away_score:
                    home_team.losses += 1
                    away_team.wins += 1
                else:
                    home_team.draws += 1
                    away_team.draws += 1

                # Update home/away specific stats
                home_team.home_wins += 1 if match.home_score > match.away_score else 0
                home_team.home_draws += 1 if match.home_score == match.away_score else 0
                home_team.home_losses += 1 if match.home_score < match.away_score else 0
                home_team.home_goals_for += match.home_score
                home_team.home_goals_against += match.away_score

                away_team.away_wins += 1 if match.away_score > match.home_score else 0
                away_team.away_draws += 1 if match.away_score == match.home_score else 0
                away_team.away_losses += 1 if match.away_score < match.home_score else 0
                away_team.away_goals_for += match.away_score
                away_team.away_goals_against += match.home_score

    async def get_head_to_head(self, team1_id: str, team2_id: str, num_matches: int = 10) -> List[Match]:
        """Get head-to-head match history between two teams."""
        # This would typically query historical data
        # For now, return empty list as placeholder
        return []

    def export_to_csv(self, data: Dict[str, Any], output_dir: str = "data/exports"):
        """Export collected data to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export teams
        teams_df = pd.DataFrame([
            {
                "id": team.id,
                "name": team.name,
                "league": team.league,
                "elo_rating": team.elo_rating,
                "matches_played": team.matches_played,
                "wins": team.wins,
                "draws": team.draws,
                "losses": team.losses,
                "goals_for": team.goals_for,
                "goals_against": team.goals_against,
                "goal_difference": team.goal_difference,
                "win_rate": team.win_rate,
                "goals_per_game": team.goals_per_game,
            }
            for team in data["teams"]
        ])
        teams_df.to_csv(f"{output_dir}/teams.csv", index=False)

        # Export matches
        matches_df = pd.DataFrame([
            {
                "id": match.id,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "league": match.league,
                "scheduled_datetime": match.scheduled_datetime,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "total_goals": match.total_goals,
                "result": match.result,
            }
            for match in data["matches"]
        ])
        matches_df.to_csv(f"{output_dir}/matches.csv", index=False)