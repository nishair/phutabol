from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class Team:
    """Represents a soccer team with basic information and performance metrics."""

    id: str
    name: str
    league: str
    country: str
    founded_year: Optional[int] = None
    home_stadium: Optional[str] = None

    # Elo rating system
    elo_rating: float = 1500.0

    # Current season stats
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0

    # Home/Away performance
    home_wins: int = 0
    home_draws: int = 0
    home_losses: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0

    away_wins: int = 0
    away_draws: int = 0
    away_losses: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0

    # Advanced metrics
    expected_goals_for: float = 0.0
    expected_goals_against: float = 0.0
    shots_per_game: float = 0.0
    shots_on_target_per_game: float = 0.0
    possession_percentage: float = 0.0

    def __post_init__(self):
        self.points = self.wins * 3 + self.draws
        self.goal_difference = self.goals_for - self.goals_against

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches_played if self.matches_played > 0 else 0.0

    @property
    def home_win_rate(self) -> float:
        home_matches = self.home_wins + self.home_draws + self.home_losses
        return self.home_wins / home_matches if home_matches > 0 else 0.0

    @property
    def away_win_rate(self) -> float:
        away_matches = self.away_wins + self.away_draws + self.away_losses
        return self.away_wins / away_matches if away_matches > 0 else 0.0

    @property
    def goals_per_game(self) -> float:
        return self.goals_for / self.matches_played if self.matches_played > 0 else 0.0

    @property
    def goals_conceded_per_game(self) -> float:
        return self.goals_against / self.matches_played if self.matches_played > 0 else 0.0

    def update_elo(self, opponent_elo: float, actual_score: float, k_factor: float = 32.0):
        """Update Elo rating based on match result."""
        expected_score = 1 / (1 + 10 ** ((opponent_elo - self.elo_rating) / 400))
        self.elo_rating += k_factor * (actual_score - expected_score)


@dataclass
class TeamForm:
    """Represents recent form data for a team."""

    team_id: str
    recent_matches: List[Dict] = None  # Last N matches
    form_string: str = ""  # e.g., "WWLDW"

    def __post_init__(self):
        if self.recent_matches is None:
            self.recent_matches = []

    @property
    def recent_wins(self) -> int:
        return sum(1 for match in self.recent_matches if match.get('result') == 'W')

    @property
    def recent_losses(self) -> int:
        return sum(1 for match in self.recent_matches if match.get('result') == 'L')

    @property
    def recent_draws(self) -> int:
        return sum(1 for match in self.recent_matches if match.get('result') == 'D')

    @property
    def recent_goals_for(self) -> int:
        return sum(match.get('goals_for', 0) for match in self.recent_matches)

    @property
    def recent_goals_against(self) -> int:
        return sum(match.get('goals_against', 0) for match in self.recent_matches)

    @property
    def recent_form_score(self) -> float:
        """Calculate form score based on recent results (3 for win, 1 for draw, 0 for loss)."""
        total_points = self.recent_wins * 3 + self.recent_draws
        max_possible = len(self.recent_matches) * 3
        return total_points / max_possible if max_possible > 0 else 0.0