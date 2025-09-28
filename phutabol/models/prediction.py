from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class PredictionModel(Enum):
    POISSON = "poisson"
    DIXON_COLES = "dixon_coles"
    BIVARIATE_POISSON = "bivariate_poisson"
    ZERO_INFLATED_POISSON = "zero_inflated_poisson"
    ELO_BASED = "elo_based"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class ScorePrediction:
    """Prediction for specific match score."""

    home_score: int
    away_score: int
    probability: float

    @property
    def total_goals(self) -> int:
        return self.home_score + self.away_score

    @property
    def result(self) -> str:
        """Result from home team perspective."""
        if self.home_score > self.away_score:
            return "W"
        elif self.home_score < self.away_score:
            return "L"
        else:
            return "D"


@dataclass
class MatchPrediction:
    """Complete prediction for a match."""

    match_id: str
    home_team_id: str
    away_team_id: str

    # Overall outcome probabilities
    home_win_probability: float
    draw_probability: float
    away_win_probability: float

    # Expected goals
    expected_goals_home: float
    expected_goals_away: float

    # Most likely scores
    most_likely_score: ScorePrediction
    top_score_predictions: List[ScorePrediction] = field(default_factory=list)

    # Goal probability distributions
    over_under_probabilities: Dict[float, float] = field(default_factory=dict)  # e.g., {2.5: 0.65}
    both_teams_score_probability: float = 0.0

    # Model information
    model_used: PredictionModel = PredictionModel.POISSON
    model_confidence: float = 0.0
    model_version: str = "1.0"

    # Context factors considered
    factors_considered: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    prediction_horizon_hours: Optional[float] = None

    def __post_init__(self):
        # Ensure probabilities sum to 1
        total_prob = self.home_win_probability + self.draw_probability + self.away_win_probability
        if abs(total_prob - 1.0) > 0.01:
            raise ValueError(f"Outcome probabilities must sum to 1.0, got {total_prob}")

    @property
    def expected_total_goals(self) -> float:
        return self.expected_goals_home + self.expected_goals_away

    @property
    def most_likely_result(self) -> str:
        """Most likely result based on probabilities."""
        probs = {
            "W": self.home_win_probability,
            "D": self.draw_probability,
            "L": self.away_win_probability
        }
        return max(probs, key=probs.get)

    def get_over_under_probability(self, line: float) -> Optional[float]:
        """Get probability of total goals being over a specific line."""
        return self.over_under_probabilities.get(line)

    def add_score_prediction(self, home_score: int, away_score: int, probability: float):
        """Add a specific score prediction."""
        score_pred = ScorePrediction(home_score, away_score, probability)
        self.top_score_predictions.append(score_pred)
        # Keep only top 10 most likely scores
        self.top_score_predictions.sort(key=lambda x: x.probability, reverse=True)
        self.top_score_predictions = self.top_score_predictions[:10]


@dataclass
class ModelPerformance:
    """Track performance metrics for prediction models."""

    model_name: str
    total_predictions: int = 0
    correct_results: int = 0
    correct_scores: int = 0

    # Accuracy by prediction confidence
    high_confidence_correct: int = 0
    high_confidence_total: int = 0
    medium_confidence_correct: int = 0
    medium_confidence_total: int = 0
    low_confidence_correct: int = 0
    low_confidence_total: int = 0

    # Calibration metrics
    log_loss: float = 0.0
    brier_score: float = 0.0

    # Goal prediction accuracy
    goals_mae: float = 0.0  # Mean Absolute Error for goals
    goals_rmse: float = 0.0  # Root Mean Square Error for goals

    @property
    def result_accuracy(self) -> float:
        return self.correct_results / self.total_predictions if self.total_predictions > 0 else 0.0

    @property
    def score_accuracy(self) -> float:
        return self.correct_scores / self.total_predictions if self.total_predictions > 0 else 0.0

    @property
    def high_confidence_accuracy(self) -> float:
        return (self.high_confidence_correct / self.high_confidence_total
                if self.high_confidence_total > 0 else 0.0)