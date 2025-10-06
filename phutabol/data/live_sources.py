import asyncio
import aiohttp
import os
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json

from ..models.team import Team, TeamForm
from ..models.match import Match, MatchStats, MatchContext, MatchStatus, VenueType
from .collector import DataSource


class FootballDataOrgSource(DataSource):
    """
    Real-time data source using Football-Data.org API

    Get your free API key at: https://www.football-data.org/client/register
    Provides current season data for major European leagues.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FOOTBALL_DATA_API_KEY')
        self.base_url = "https://api.football-data.org/v4"
        self.session = None

        # League ID mappings
        self.league_ids = {
            "Premier League": 2021,
            "La Liga": 2014,
            "Bundesliga": 2002,
            "Serie A": 2019,
            "Ligue 1": 2015,
            "Champions League": 2001,
            "Europa League": 2018
        }

    async def __aenter__(self):
        if not self.api_key:
            raise ValueError("Football Data API key is required. Get one at https://www.football-data.org/client/register")

        self.session = aiohttp.ClientSession(
            headers={
                "X-Auth-Token": self.api_key,
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_teams(self, league: str) -> List[Team]:
        """Fetch current teams for a specific league."""
        league_id = self.league_ids.get(league)
        if not league_id:
            raise ValueError(f"League '{league}' not supported. Available: {list(self.league_ids.keys())}")

        async with self.session.get(f"{self.base_url}/competitions/{league_id}/teams") as response:
            if response.status == 200:
                data = await response.json()
                teams = []

                for team_data in data.get("teams", []):
                    team = Team(
                        id=str(team_data["id"]),
                        name=team_data["name"],
                        league=league,
                        country=team_data.get("area", {}).get("name", "Unknown"),
                        founded_year=team_data.get("founded"),
                        home_stadium=team_data.get("venue"),
                        elo_rating=1500.0  # Will be calculated from recent performance
                    )
                    teams.append(team)

                # Get current season standings to update team stats
                await self._update_team_standings(teams, league_id)

                return teams
            else:
                raise Exception(f"API request failed: {response.status}")

    async def _update_team_standings(self, teams: List[Team], league_id: int):
        """Update team statistics from current standings."""
        async with self.session.get(f"{self.base_url}/competitions/{league_id}/standings") as response:
            if response.status == 200:
                data = await response.json()
                standings = data.get("standings", [])

                # Get the overall table (usually the first one)
                if standings:
                    table = standings[0].get("table", [])
                    team_map = {team.id: team for team in teams}

                    for position in table:
                        team_id = str(position["team"]["id"])
                        if team_id in team_map:
                            team = team_map[team_id]

                            # Update team statistics
                            team.matches_played = position["playedGames"]
                            team.wins = position["won"]
                            team.draws = position["draw"]
                            team.losses = position["lost"]
                            team.goals_for = position["goalsFor"]
                            team.goals_against = position["goalsAgainst"]
                            team.points = position["points"]

                            # Calculate Elo rating based on performance
                            win_rate = team.wins / max(team.matches_played, 1)
                            goal_diff_ratio = team.goal_difference / max(team.matches_played, 1)
                            team.elo_rating = 1500 + (win_rate * 200) + (goal_diff_ratio * 50)

    async def get_matches(self, league: str, start_date: datetime, end_date: datetime) -> List[Match]:
        """Fetch matches for a specific league and date range."""
        league_id = self.league_ids.get(league)
        if not league_id:
            raise ValueError(f"League '{league}' not supported")

        # Format dates for API
        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")

        url = f"{self.base_url}/competitions/{league_id}/matches"
        params = {"dateFrom": date_from, "dateTo": date_to}

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                matches = []

                for match_data in data.get("matches", []):
                    # Parse match status
                    status_map = {
                        "SCHEDULED": MatchStatus.SCHEDULED,
                        "LIVE": MatchStatus.IN_PROGRESS,
                        "IN_PLAY": MatchStatus.IN_PROGRESS,
                        "PAUSED": MatchStatus.IN_PROGRESS,
                        "FINISHED": MatchStatus.FINISHED,
                        "POSTPONED": MatchStatus.POSTPONED,
                        "CANCELLED": MatchStatus.CANCELLED
                    }

                    status = status_map.get(match_data["status"], MatchStatus.SCHEDULED)

                    # Parse datetime
                    utc_date = datetime.fromisoformat(match_data["utcDate"].replace("Z", "+00:00"))

                    # Create match context
                    context = MatchContext(
                        is_home_advantage=True,
                        attendance=match_data.get("attendance"),
                        referee=match_data.get("referees", [{}])[0].get("name") if match_data.get("referees") else None
                    )

                    match = Match(
                        id=str(match_data["id"]),
                        home_team_id=str(match_data["homeTeam"]["id"]),
                        away_team_id=str(match_data["awayTeam"]["id"]),
                        league=league,
                        season=match_data.get("season", {}).get("startDate", "")[:4],
                        matchday=match_data.get("matchday"),
                        scheduled_datetime=utc_date,
                        status=status,
                        venue=match_data.get("venue"),
                        home_score=match_data["score"]["fullTime"]["home"],
                        away_score=match_data["score"]["fullTime"]["away"],
                        home_score_ht=match_data["score"]["halfTime"]["home"],
                        away_score_ht=match_data["score"]["halfTime"]["away"],
                        context=context
                    )
                    matches.append(match)

                return matches
            else:
                raise Exception(f"API request failed: {response.status}")

    async def get_match_stats(self, match_id: str) -> Optional[MatchStats]:
        """Fetch detailed statistics for a specific match."""
        async with self.session.get(f"{self.base_url}/matches/{match_id}") as response:
            if response.status == 200:
                data = await response.json()
                match_data = data.get("match", data)

                # Football-Data.org doesn't provide detailed match stats in free tier
                # This would require additional data sources or premium access
                return MatchStats(
                    # Basic stats only available in free tier
                    shots_home=None,
                    shots_away=None,
                    shots_on_target_home=None,
                    shots_on_target_away=None,
                )
            return None

    async def get_team_form(self, team_id: str, num_matches: int = 5) -> TeamForm:
        """Fetch recent form for a team."""
        # Get recent matches for the team
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Look back 60 days

        async with self.session.get(f"{self.base_url}/teams/{team_id}/matches",
                                   params={
                                       "dateFrom": start_date.strftime("%Y-%m-%d"),
                                       "dateTo": end_date.strftime("%Y-%m-%d"),
                                       "status": "FINISHED"
                                   }) as response:
            if response.status == 200:
                data = await response.json()
                matches = data.get("matches", [])

                recent_matches = []
                form_string = ""

                # Sort by date and get most recent
                matches.sort(key=lambda x: x["utcDate"], reverse=True)

                for match in matches[:num_matches]:
                    home_id = str(match["homeTeam"]["id"])
                    away_id = str(match["awayTeam"]["id"])
                    home_score = match["score"]["fullTime"]["home"]
                    away_score = match["score"]["fullTime"]["away"]

                    if team_id == home_id:
                        # Team was playing at home
                        goals_for = home_score
                        goals_against = away_score
                    else:
                        # Team was playing away
                        goals_for = away_score
                        goals_against = home_score

                    # Determine result
                    if goals_for > goals_against:
                        result = "W"
                    elif goals_for < goals_against:
                        result = "L"
                    else:
                        result = "D"

                    recent_matches.append({
                        "result": result,
                        "goals_for": goals_for,
                        "goals_against": goals_against,
                        "date": match["utcDate"]
                    })
                    form_string += result

                return TeamForm(
                    team_id=team_id,
                    recent_matches=recent_matches,
                    form_string=form_string
                )

            return TeamForm(team_id=team_id)


class RapidAPIFootballSource(DataSource):
    """
    Alternative data source using RapidAPI Football API

    Get API key at: https://rapidapi.com/api-sports/api/api-football
    Provides extensive stats including player injuries, weather, etc.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('RAPIDAPI_KEY')
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.session = None

        # League ID mappings for RapidAPI
        self.league_ids = {
            "Premier League": 39,
            "La Liga": 140,
            "Bundesliga": 78,
            "Serie A": 135,
            "Ligue 1": 61
        }

    async def __aenter__(self):
        if not self.api_key:
            raise ValueError("RapidAPI key is required")

        self.session = aiohttp.ClientSession(
            headers={
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_teams(self, league: str) -> List[Team]:
        """Implementation for RapidAPI source."""
        # Implementation would be similar but using RapidAPI endpoints
        # This provides more detailed data including player information
        pass

    async def get_matches(self, league: str, start_date: datetime, end_date: datetime) -> List[Match]:
        """Implementation for RapidAPI source."""
        pass

    async def get_match_stats(self, match_id: str) -> Optional[MatchStats]:
        """RapidAPI provides much more detailed match statistics."""
        pass

    async def get_team_form(self, team_id: str, num_matches: int = 5) -> TeamForm:
        """Implementation for RapidAPI source."""
        pass


class LiveDataCollector:
    """Enhanced data collector that combines multiple real-time sources."""

    def __init__(self, primary_source: DataSource, backup_sources: List[DataSource] = None):
        self.primary_source = primary_source
        self.backup_sources = backup_sources or []
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes

    async def get_current_season_data(self, league: str) -> Dict[str, Any]:
        """Get comprehensive current season data."""

        cache_key = f"{league}_current_season"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            async with self.primary_source as source:
                # Get current teams
                teams = await source.get_teams(league)

                # Get recent matches (last 30 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                matches = await source.get_matches(league, start_date, end_date)

                # Get upcoming fixtures (next 14 days)
                fixture_end = end_date + timedelta(days=14)
                fixtures = await source.get_matches(league, end_date, fixture_end)

                # Get team forms
                team_forms = {}
                for team in teams:
                    try:
                        form = await source.get_team_form(team.id)
                        team_forms[team.id] = form
                    except Exception as e:
                        print(f"Failed to get form for team {team.name}: {e}")
                        team_forms[team.id] = TeamForm(team_id=team.id)

                data = {
                    "teams": teams,
                    "recent_matches": matches,
                    "upcoming_fixtures": fixtures,
                    "team_forms": team_forms,
                    "last_updated": datetime.now(),
                    "source": "live_data"
                }

                # Cache the data
                self.cache[cache_key] = data

                return data

        except Exception as e:
            print(f"Primary source failed: {e}")
            # Try backup sources
            for backup_source in self.backup_sources:
                try:
                    async with backup_source as source:
                        return await self._get_basic_data(source, league)
                except Exception as backup_error:
                    print(f"Backup source failed: {backup_error}")
                    continue

            # If all sources fail, raise the original error
            raise e

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False

        cached_data = self.cache[cache_key]
        if "last_updated" not in cached_data:
            return False

        age = datetime.now() - cached_data["last_updated"]
        return age < self.cache_duration

    async def _get_basic_data(self, source: DataSource, league: str) -> Dict[str, Any]:
        """Get basic data from a source (fallback method)."""
        teams = await source.get_teams(league)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        matches = await source.get_matches(league, start_date, end_date)

        return {
            "teams": teams,
            "recent_matches": matches,
            "upcoming_fixtures": [],
            "team_forms": {team.id: TeamForm(team_id=team.id) for team in teams},
            "last_updated": datetime.now(),
            "source": "backup_data"
        }