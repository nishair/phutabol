from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MatchStatus(Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class VenueType(Enum):
    HOME = "home"
    AWAY = "away"
    NEUTRAL = "neutral"


@dataclass
class MatchContext:
    """Context information that can affect match outcome."""

    is_home_advantage: bool = True
    weather_conditions: Optional[str] = None
    temperature: Optional[float] = None
    pitch_condition: Optional[str] = None
    attendance: Optional[int] = None
    referee: Optional[str] = None

    # Team-specific context
    home_team_rest_days: Optional[int] = None
    away_team_rest_days: Optional[int] = None
    home_team_travel_distance: Optional[float] = None
    away_team_travel_distance: Optional[float] = None

    # Match importance
    is_cup_match: bool = False
    is_playoff: bool = False
    is_rivalry: bool = False
    tournament_stage: Optional[str] = None


@dataclass
class MatchStats:
    """Detailed match statistics."""

    # Basic stats
    possession_home: Optional[float] = None
    possession_away: Optional[float] = None

    # Shooting stats
    shots_home: Optional[int] = None
    shots_away: Optional[int] = None
    shots_on_target_home: Optional[int] = None
    shots_on_target_away: Optional[int] = None

    # Expected goals
    xg_home: Optional[float] = None
    xg_away: Optional[float] = None

    # Defensive stats
    fouls_home: Optional[int] = None
    fouls_away: Optional[int] = None
    yellow_cards_home: Optional[int] = None
    yellow_cards_away: Optional[int] = None
    red_cards_home: Optional[int] = None
    red_cards_away: Optional[int] = None

    # Passing stats
    passes_home: Optional[int] = None
    passes_away: Optional[int] = None
    pass_accuracy_home: Optional[float] = None
    pass_accuracy_away: Optional[float] = None

    # Other stats
    corners_home: Optional[int] = None
    corners_away: Optional[int] = None
    offsides_home: Optional[int] = None
    offsides_away: Optional[int] = None


@dataclass
class Match:
    """Represents a soccer match with all relevant information."""

    id: str
    home_team_id: str
    away_team_id: str
    league: str
    season: str
    matchday: Optional[int] = None

    # Timing
    scheduled_datetime: Optional[datetime] = None
    kickoff_datetime: Optional[datetime] = None
    status: MatchStatus = MatchStatus.SCHEDULED

    # Venue
    venue: Optional[str] = None
    venue_type: VenueType = VenueType.HOME

    # Scores
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_score_ht: Optional[int] = None  # Half-time score
    away_score_ht: Optional[int] = None

    # Context and stats
    context: MatchContext = field(default_factory=MatchContext)
    stats: Optional[MatchStats] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_finished(self) -> bool:
        return self.status == MatchStatus.FINISHED

    @property
    def total_goals(self) -> Optional[int]:
        if self.home_score is not None and self.away_score is not None:
            return self.home_score + self.away_score
        return None

    @property
    def goal_difference(self) -> Optional[int]:
        if self.home_score is not None and self.away_score is not None:
            return self.home_score - self.away_score
        return None

    @property
    def result(self) -> Optional[str]:
        """Return match result from home team perspective: W, D, L."""
        if not self.is_finished:
            return None

        if self.home_score > self.away_score:
            return "W"
        elif self.home_score < self.away_score:
            return "L"
        else:
            return "D"

    @property
    def winner_team_id(self) -> Optional[str]:
        """Return the ID of the winning team, None if draw."""
        if not self.is_finished:
            return None

        if self.home_score > self.away_score:
            return self.home_team_id
        elif self.home_score < self.away_score:
            return self.away_team_id
        else:
            return None

    def get_team_score(self, team_id: str) -> Optional[int]:
        """Get score for a specific team."""
        if team_id == self.home_team_id:
            return self.home_score
        elif team_id == self.away_team_id:
            return self.away_score
        else:
            raise ValueError(f"Team {team_id} is not playing in this match")

    def get_opponent_id(self, team_id: str) -> str:
        """Get opponent team ID for a given team."""
        if team_id == self.home_team_id:
            return self.away_team_id
        elif team_id == self.away_team_id:
            return self.home_team_id
        else:
            raise ValueError(f"Team {team_id} is not playing in this match")

    def is_home_team(self, team_id: str) -> bool:
        """Check if team is playing at home."""
        return team_id == self.home_team_id