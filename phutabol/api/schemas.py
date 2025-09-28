from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.team import Team
from ..models.match import Match, MatchContext
from ..models.prediction import MatchPrediction, ScorePrediction


class TeamResponse(BaseModel):
    """Response model for team data."""
    id: str
    name: str
    league: str
    country: str
    elo_rating: float
    matches_played: int
    wins: int
    draws: int
    losses: int
    points: int
    goals_for: int
    goals_against: int
    goal_difference: int
    win_rate: float
    goals_per_game: float
    goals_conceded_per_game: float

    @classmethod
    def from_team(cls, team: Team) -> "TeamResponse":
        return cls(
            id=team.id,
            name=team.name,
            league=team.league,
            country=team.country,
            elo_rating=team.elo_rating,
            matches_played=team.matches_played,
            wins=team.wins,
            draws=team.draws,
            losses=team.losses,
            points=team.points,
            goals_for=team.goals_for,
            goals_against=team.goals_against,
            goal_difference=team.goal_difference,
            win_rate=team.win_rate,
            goals_per_game=team.goals_per_game,
            goals_conceded_per_game=team.goals_conceded_per_game
        )


class MatchResponse(BaseModel):
    """Response model for match data."""
    id: str
    home_team_id: str
    away_team_id: str
    league: str
    season: str
    scheduled_datetime: Optional[datetime]
    status: str
    home_score: Optional[int]
    away_score: Optional[int]
    total_goals: Optional[int]
    result: Optional[str]

    @classmethod
    def from_match(cls, match: Match) -> "MatchResponse":
        return cls(
            id=match.id,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            league=match.league,
            season=match.season,
            scheduled_datetime=match.scheduled_datetime,
            status=match.status.value,
            home_score=match.home_score,
            away_score=match.away_score,
            total_goals=match.total_goals,
            result=match.result
        )


class ScorePredictionResponse(BaseModel):
    """Response model for score predictions."""
    home_score: int
    away_score: int
    probability: float
    total_goals: int
    result: str

    @classmethod
    def from_score_prediction(cls, score_pred: ScorePrediction) -> "ScorePredictionResponse":
        return cls(
            home_score=score_pred.home_score,
            away_score=score_pred.away_score,
            probability=score_pred.probability,
            total_goals=score_pred.total_goals,
            result=score_pred.result
        )


class PredictionResponse(BaseModel):
    """Main prediction response."""
    # Outcome probabilities
    home_win_probability: float = Field(..., description="Probability of home team winning")
    draw_probability: float = Field(..., description="Probability of draw")
    away_win_probability: float = Field(..., description="Probability of away team winning")

    # Expected goals
    expected_goals_home: float = Field(..., description="Expected goals for home team")
    expected_goals_away: float = Field(..., description="Expected goals for away team")
    expected_total_goals: float = Field(..., description="Expected total goals in match")

    # Most likely outcome
    most_likely_result: str = Field(..., description="Most likely result (W/D/L)")
    most_likely_score: ScorePredictionResponse = Field(..., description="Most likely exact score")

    # Top score predictions
    top_score_predictions: List[ScorePredictionResponse] = Field(
        ..., description="Top 5-10 most likely scores"
    )

    # Over/Under probabilities
    over_under_probabilities: Dict[str, float] = Field(
        ..., description="Probabilities for different goal totals"
    )

    # Other markets
    both_teams_score_probability: float = Field(
        ..., description="Probability both teams will score"
    )

    # Team information
    home_team: TeamResponse
    away_team: TeamResponse

    # Analysis
    team_comparison: Dict[str, float] = Field(
        ..., description="Comparative analysis between teams"
    )

    # Model metadata
    model_used: str = Field(..., description="Prediction model used")
    confidence_score: float = Field(..., description="Model confidence (0-1)")
    factors_considered: List[str] = Field(..., description="Factors included in prediction")

    # Timestamps
    prediction_time: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_prediction_and_teams(cls, prediction: MatchPrediction,
                                home_team: Team, away_team: Team,
                                team_comparison: Dict[str, float],
                                model_name: str) -> "PredictionResponse":
        return cls(
            home_win_probability=prediction.home_win_probability,
            draw_probability=prediction.draw_probability,
            away_win_probability=prediction.away_win_probability,
            expected_goals_home=prediction.expected_goals_home,
            expected_goals_away=prediction.expected_goals_away,
            expected_total_goals=prediction.expected_total_goals,
            most_likely_result=prediction.most_likely_result,
            most_likely_score=ScorePredictionResponse.from_score_prediction(
                prediction.most_likely_score
            ),
            top_score_predictions=[
                ScorePredictionResponse.from_score_prediction(sp)
                for sp in prediction.top_score_predictions[:5]
            ],
            over_under_probabilities={
                str(k): v for k, v in prediction.over_under_probabilities.items()
            },
            both_teams_score_probability=prediction.both_teams_score_probability,
            home_team=TeamResponse.from_team(home_team),
            away_team=TeamResponse.from_team(away_team),
            team_comparison=team_comparison,
            model_used=model_name,
            confidence_score=prediction.model_confidence,
            factors_considered=prediction.factors_considered
        )

    # Update the existing from_prediction method
    prediction: MatchPrediction
    home_team: TeamResponse
    away_team: TeamResponse
    team_comparison: Dict[str, float]
    model_used: str
    confidence_score: float


class PredictionRequest(BaseModel):
    """Request model for match predictions."""
    home_team_id: str = Field(..., description="ID of the home team")
    away_team_id: str = Field(..., description="ID of the away team")
    league: str = Field(..., description="League name")
    model: Optional[str] = Field("ensemble", description="Prediction model to use")
    match_context: Optional[MatchContext] = Field(None, description="Additional match context")


class LeagueStandingsResponse(BaseModel):
    """Response model for league standings."""
    league: str
    standings: List[Dict[str, Any]]
    last_updated: datetime


class ModelPerformanceResponse(BaseModel):
    """Response model for model performance metrics."""
    model_name: str
    total_predictions: int
    result_accuracy: float
    score_accuracy: float
    log_loss: float
    brier_score: float
    goals_mae: float
    high_confidence_accuracy: float


class BulkPredictionRequest(BaseModel):
    """Request for predicting multiple matches."""
    matches: List[Dict[str, str]] = Field(
        ...,
        description="List of matches with home_team_id, away_team_id, league"
    )
    model: Optional[str] = Field("ensemble", description="Prediction model to use")


class BulkPredictionResponse(BaseModel):
    """Response for bulk predictions."""
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_seconds: float
    model_used: str