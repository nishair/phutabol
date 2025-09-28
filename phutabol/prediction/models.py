import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import math

from ..models.team import Team
from ..models.match import Match
from ..models.prediction import MatchPrediction, ScorePrediction, PredictionModel
from ..analysis.performance import PerformanceMetrics


class PredictionModelBase(ABC):
    """Abstract base class for prediction models."""

    @abstractmethod
    def predict_match(self, home_team: Team, away_team: Team,
                     home_metrics: PerformanceMetrics, away_metrics: PerformanceMetrics,
                     **kwargs) -> MatchPrediction:
        """Predict the outcome of a match."""
        pass

    @abstractmethod
    def get_model_name(self) -> PredictionModel:
        """Return the model identifier."""
        pass


class PoissonModel(PredictionModelBase):
    """Basic Poisson distribution model for goal prediction."""

    def __init__(self, home_advantage: float = 1.3):
        self.home_advantage = home_advantage

    def predict_match(self, home_team: Team, away_team: Team,
                     home_metrics: PerformanceMetrics, away_metrics: PerformanceMetrics,
                     **kwargs) -> MatchPrediction:
        """Predict match using Poisson distribution."""

        # Calculate expected goals
        home_attack = home_metrics.attack_strength
        home_defense = home_metrics.defense_strength
        away_attack = away_metrics.attack_strength
        away_defense = away_metrics.defense_strength

        # League average goals per game (approximation)
        league_avg_goals = 2.5

        # Calculate expected goals for each team
        home_expected = (home_attack / away_defense) * self.home_advantage * league_avg_goals
        away_expected = (away_attack / home_defense) * league_avg_goals

        # Calculate outcome probabilities
        max_goals = 10
        score_probabilities = {}
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (stats.poisson.pmf(home_goals, home_expected) *
                       stats.poisson.pmf(away_goals, away_expected))

                score_probabilities[(home_goals, away_goals)] = prob

                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals < away_goals:
                    away_win_prob += prob
                else:
                    draw_prob += prob

        # Find most likely score
        most_likely_score_tuple = max(score_probabilities.items(), key=lambda x: x[1])
        most_likely_score = ScorePrediction(
            home_score=most_likely_score_tuple[0][0],
            away_score=most_likely_score_tuple[0][1],
            probability=most_likely_score_tuple[1]
        )

        # Calculate over/under probabilities
        over_under_probs = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            over_prob = sum(
                prob for (h, a), prob in score_probabilities.items()
                if h + a > line
            )
            over_under_probs[line] = over_prob

        # Both teams to score probability
        btts_prob = sum(
            prob for (h, a), prob in score_probabilities.items()
            if h > 0 and a > 0
        )

        # Top score predictions
        top_scores = sorted(score_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
        top_score_predictions = [
            ScorePrediction(home_score=score[0], away_score=score[1], probability=prob)
            for score, prob in top_scores
        ]

        return MatchPrediction(
            match_id="",
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_win_probability=home_win_prob,
            draw_probability=draw_prob,
            away_win_probability=away_win_prob,
            expected_goals_home=home_expected,
            expected_goals_away=away_expected,
            most_likely_score=most_likely_score,
            top_score_predictions=top_score_predictions,
            over_under_probabilities=over_under_probs,
            both_teams_score_probability=btts_prob,
            model_used=PredictionModel.POISSON,
            factors_considered=["attack_strength", "defense_strength", "home_advantage"]
        )

    def get_model_name(self) -> PredictionModel:
        return PredictionModel.POISSON


class DixonColesModel(PredictionModelBase):
    """Dixon-Coles model that adjusts for low-scoring draws."""

    def __init__(self, home_advantage: float = 1.3, rho: float = -0.03):
        self.home_advantage = home_advantage
        self.rho = rho  # Correlation parameter for low scores

    def _dixon_coles_adjustment(self, home_goals: int, away_goals: int) -> float:
        """Apply Dixon-Coles adjustment for low-scoring matches."""
        if home_goals == 0 and away_goals == 0:
            return 1 - self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - self.rho
        else:
            return 1.0

    def predict_match(self, home_team: Team, away_team: Team,
                     home_metrics: PerformanceMetrics, away_metrics: PerformanceMetrics,
                     **kwargs) -> MatchPrediction:
        """Predict match using Dixon-Coles model."""

        # Start with basic Poisson calculation
        home_attack = home_metrics.attack_strength
        home_defense = home_metrics.defense_strength
        away_attack = away_metrics.attack_strength
        away_defense = away_metrics.defense_strength

        league_avg_goals = 2.5

        home_expected = (home_attack / away_defense) * self.home_advantage * league_avg_goals
        away_expected = (away_attack / home_defense) * league_avg_goals

        # Calculate adjusted probabilities
        max_goals = 10
        score_probabilities = {}
        total_probability = 0.0

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Basic Poisson probability
                poisson_prob = (stats.poisson.pmf(home_goals, home_expected) *
                              stats.poisson.pmf(away_goals, away_expected))

                # Apply Dixon-Coles adjustment
                adjustment = self._dixon_coles_adjustment(home_goals, away_goals)
                adjusted_prob = poisson_prob * adjustment

                score_probabilities[(home_goals, away_goals)] = adjusted_prob
                total_probability += adjusted_prob

        # Normalize probabilities
        for key in score_probabilities:
            score_probabilities[key] /= total_probability

        # Calculate outcome probabilities
        home_win_prob = sum(
            prob for (h, a), prob in score_probabilities.items() if h > a
        )
        draw_prob = sum(
            prob for (h, a), prob in score_probabilities.items() if h == a
        )
        away_win_prob = sum(
            prob for (h, a), prob in score_probabilities.items() if h < a
        )

        # Find most likely score
        most_likely_score_tuple = max(score_probabilities.items(), key=lambda x: x[1])
        most_likely_score = ScorePrediction(
            home_score=most_likely_score_tuple[0][0],
            away_score=most_likely_score_tuple[0][1],
            probability=most_likely_score_tuple[1]
        )

        # Calculate over/under and other probabilities
        over_under_probs = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            over_prob = sum(
                prob for (h, a), prob in score_probabilities.items()
                if h + a > line
            )
            over_under_probs[line] = over_prob

        btts_prob = sum(
            prob for (h, a), prob in score_probabilities.items()
            if h > 0 and a > 0
        )

        top_scores = sorted(score_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
        top_score_predictions = [
            ScorePrediction(home_score=score[0], away_score=score[1], probability=prob)
            for score, prob in top_scores
        ]

        return MatchPrediction(
            match_id="",
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_win_probability=home_win_prob,
            draw_probability=draw_prob,
            away_win_probability=away_win_prob,
            expected_goals_home=home_expected,
            expected_goals_away=away_expected,
            most_likely_score=most_likely_score,
            top_score_predictions=top_score_predictions,
            over_under_probabilities=over_under_probs,
            both_teams_score_probability=btts_prob,
            model_used=PredictionModel.DIXON_COLES,
            factors_considered=["attack_strength", "defense_strength", "home_advantage", "low_score_adjustment"]
        )

    def get_model_name(self) -> PredictionModel:
        return PredictionModel.DIXON_COLES


class EloBasedModel(PredictionModelBase):
    """Prediction model based on Elo ratings."""

    def __init__(self, home_advantage_elo: float = 100.0):
        self.home_advantage_elo = home_advantage_elo

    def predict_match(self, home_team: Team, away_team: Team,
                     home_metrics: PerformanceMetrics, away_metrics: PerformanceMetrics,
                     **kwargs) -> MatchPrediction:
        """Predict match using Elo ratings."""

        # Adjust home team Elo for home advantage
        adjusted_home_elo = home_team.elo_rating + self.home_advantage_elo
        away_elo = away_team.elo_rating

        # Calculate win probabilities using Elo formula
        elo_diff = adjusted_home_elo - away_elo
        home_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        away_win_prob = 1 / (1 + 10 ** (elo_diff / 400))

        # Estimate draw probability (simplified approach)
        max_win_prob = max(home_win_prob, away_win_prob)
        draw_prob = max(0.1, 0.5 - abs(elo_diff) / 800)  # More draws in close matches

        # Normalize probabilities
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total

        # Estimate expected goals based on Elo difference
        # Higher-rated teams tend to score more and concede less
        base_goals = 1.4  # Base expected goals per team

        elo_factor = elo_diff / 400
        home_expected = base_goals * (1 + elo_factor * 0.3)
        away_expected = base_goals * (1 - elo_factor * 0.3)

        # Ensure realistic ranges
        home_expected = max(0.5, min(4.0, home_expected))
        away_expected = max(0.5, min(4.0, away_expected))

        # Generate score predictions using Poisson
        max_goals = 8
        score_probabilities = {}

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (stats.poisson.pmf(home_goals, home_expected) *
                       stats.poisson.pmf(away_goals, away_expected))
                score_probabilities[(home_goals, away_goals)] = prob

        # Find most likely score
        most_likely_score_tuple = max(score_probabilities.items(), key=lambda x: x[1])
        most_likely_score = ScorePrediction(
            home_score=most_likely_score_tuple[0][0],
            away_score=most_likely_score_tuple[0][1],
            probability=most_likely_score_tuple[1]
        )

        # Calculate over/under probabilities
        over_under_probs = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            over_prob = sum(
                prob for (h, a), prob in score_probabilities.items()
                if h + a > line
            )
            over_under_probs[line] = over_prob

        btts_prob = sum(
            prob for (h, a), prob in score_probabilities.items()
            if h > 0 and a > 0
        )

        top_scores = sorted(score_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
        top_score_predictions = [
            ScorePrediction(home_score=score[0], away_score=score[1], probability=prob)
            for score, prob in top_scores
        ]

        return MatchPrediction(
            match_id="",
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_win_probability=home_win_prob,
            draw_probability=draw_prob,
            away_win_probability=away_win_prob,
            expected_goals_home=home_expected,
            expected_goals_away=away_expected,
            most_likely_score=most_likely_score,
            top_score_predictions=top_score_predictions,
            over_under_probabilities=over_under_probs,
            both_teams_score_probability=btts_prob,
            model_used=PredictionModel.ELO_BASED,
            factors_considered=["elo_rating", "home_advantage"]
        )

    def get_model_name(self) -> PredictionModel:
        return PredictionModel.ELO_BASED


class BivariatePoissonModel(PredictionModelBase):
    """Bivariate Poisson model that accounts for correlation between team goals."""

    def __init__(self, home_advantage: float = 1.3):
        self.home_advantage = home_advantage

    def predict_match(self, home_team: Team, away_team: Team,
                     home_metrics: PerformanceMetrics, away_metrics: PerformanceMetrics,
                     **kwargs) -> MatchPrediction:
        """Predict match using Bivariate Poisson model."""

        # Calculate expected goals (similar to basic Poisson)
        home_attack = home_metrics.attack_strength
        home_defense = home_metrics.defense_strength
        away_attack = away_metrics.attack_strength
        away_defense = away_metrics.defense_strength

        league_avg_goals = 2.5

        lambda_home = (home_attack / away_defense) * self.home_advantage * league_avg_goals
        lambda_away = (away_attack / home_defense) * league_avg_goals

        # Estimate correlation parameter
        # In high-tempo games, both teams tend to score more
        # In defensive games, both teams tend to score less
        avg_expected = (lambda_home + lambda_away) / 2
        lambda_correlation = max(0.0, min(0.5, (avg_expected - 2.0) * 0.1))

        # Calculate bivariate Poisson probabilities
        max_goals = 10
        score_probabilities = {}

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Bivariate Poisson probability calculation
                prob = self._bivariate_poisson_pmf(
                    home_goals, away_goals, lambda_home, lambda_away, lambda_correlation
                )
                score_probabilities[(home_goals, away_goals)] = prob

        # Normalize probabilities
        total_prob = sum(score_probabilities.values())
        for key in score_probabilities:
            score_probabilities[key] /= total_prob

        # Calculate outcome probabilities
        home_win_prob = sum(
            prob for (h, a), prob in score_probabilities.items() if h > a
        )
        draw_prob = sum(
            prob for (h, a), prob in score_probabilities.items() if h == a
        )
        away_win_prob = sum(
            prob for (h, a), prob in score_probabilities.items() if h < a
        )

        # Rest of the implementation similar to Poisson model
        most_likely_score_tuple = max(score_probabilities.items(), key=lambda x: x[1])
        most_likely_score = ScorePrediction(
            home_score=most_likely_score_tuple[0][0],
            away_score=most_likely_score_tuple[0][1],
            probability=most_likely_score_tuple[1]
        )

        over_under_probs = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            over_prob = sum(
                prob for (h, a), prob in score_probabilities.items()
                if h + a > line
            )
            over_under_probs[line] = over_prob

        btts_prob = sum(
            prob for (h, a), prob in score_probabilities.items()
            if h > 0 and a > 0
        )

        top_scores = sorted(score_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
        top_score_predictions = [
            ScorePrediction(home_score=score[0], away_score=score[1], probability=prob)
            for score, prob in top_scores
        ]

        return MatchPrediction(
            match_id="",
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_win_probability=home_win_prob,
            draw_probability=draw_prob,
            away_win_probability=away_win_prob,
            expected_goals_home=lambda_home,
            expected_goals_away=lambda_away,
            most_likely_score=most_likely_score,
            top_score_predictions=top_score_predictions,
            over_under_probabilities=over_under_probs,
            both_teams_score_probability=btts_prob,
            model_used=PredictionModel.BIVARIATE_POISSON,
            factors_considered=["attack_strength", "defense_strength", "home_advantage", "goal_correlation"]
        )

    def _bivariate_poisson_pmf(self, x: int, y: int, lambda1: float, lambda2: float, lambda12: float) -> float:
        """Calculate bivariate Poisson probability mass function."""
        if lambda12 < 0:
            lambda12 = 0

        # Simplified bivariate Poisson calculation
        min_xy = min(x, y)
        prob = 0.0

        for k in range(min_xy + 1):
            term1 = math.exp(-(lambda1 + lambda2 + lambda12))
            term2 = (lambda1 ** (x - k)) / math.factorial(x - k)
            term3 = (lambda2 ** (y - k)) / math.factorial(y - k)
            term4 = (lambda12 ** k) / math.factorial(k)

            prob += term1 * term2 * term3 * term4

        return prob

    def get_model_name(self) -> PredictionModel:
        return PredictionModel.BIVARIATE_POISSON


class ModelEnsemble:
    """Ensemble of multiple prediction models."""

    def __init__(self, models: List[PredictionModelBase], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        if abs(sum(self.weights) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

    def predict_match(self, home_team: Team, away_team: Team,
                     home_metrics: PerformanceMetrics, away_metrics: PerformanceMetrics,
                     **kwargs) -> MatchPrediction:
        """Predict match using ensemble of models."""

        predictions = []
        for model in self.models:
            pred = model.predict_match(home_team, away_team, home_metrics, away_metrics, **kwargs)
            predictions.append(pred)

        # Weighted average of probabilities
        home_win_prob = sum(p.home_win_probability * w for p, w in zip(predictions, self.weights))
        draw_prob = sum(p.draw_probability * w for p, w in zip(predictions, self.weights))
        away_win_prob = sum(p.away_win_probability * w for p, w in zip(predictions, self.weights))

        # Weighted average of expected goals
        home_expected = sum(p.expected_goals_home * w for p, w in zip(predictions, self.weights))
        away_expected = sum(p.expected_goals_away * w for p, w in zip(predictions, self.weights))

        # Use most confident model's score predictions
        most_confident_pred = max(predictions, key=lambda p: p.model_confidence)

        # Combine factors considered
        all_factors = set()
        for pred in predictions:
            all_factors.update(pred.factors_considered)

        return MatchPrediction(
            match_id="",
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_win_probability=home_win_prob,
            draw_probability=draw_prob,
            away_win_probability=away_win_prob,
            expected_goals_home=home_expected,
            expected_goals_away=away_expected,
            most_likely_score=most_confident_pred.most_likely_score,
            top_score_predictions=most_confident_pred.top_score_predictions,
            over_under_probabilities=most_confident_pred.over_under_probabilities,
            both_teams_score_probability=most_confident_pred.both_teams_score_probability,
            model_used=PredictionModel.MACHINE_LEARNING,  # Using as ensemble identifier
            factors_considered=list(all_factors)
        )