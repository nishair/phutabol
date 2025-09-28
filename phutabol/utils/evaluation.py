import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

from ..models.team import Team
from ..models.match import Match
from ..models.prediction import MatchPrediction, ModelPerformance
from ..prediction.models import PredictionModelBase
from ..analysis.performance import PerformanceAnalyzer
from ..data.collector import DataCollector


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    model_name: str
    start_date: datetime
    end_date: datetime
    total_matches: int

    # Accuracy metrics
    result_accuracy: float
    score_accuracy: float

    # Probability calibration
    log_loss: float
    brier_score: float

    # Goal prediction accuracy
    goals_mae: float
    goals_rmse: float

    # By confidence level
    high_confidence_accuracy: float
    medium_confidence_accuracy: float
    low_confidence_accuracy: float

    # Market-specific accuracy
    over_under_accuracy: Dict[float, float]
    btts_accuracy: float

    # Profitability (if betting odds provided)
    roi_percentage: Optional[float] = None
    total_profit: Optional[float] = None


class ModelEvaluator:
    """Evaluates prediction model performance using historical data."""

    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
        self.performance_analyzer = PerformanceAnalyzer()

    async def backtest_model(self, model: PredictionModelBase, league: str,
                           start_date: datetime, end_date: datetime,
                           min_confidence: float = 0.0) -> BacktestResult:
        """Run a comprehensive backtest of a model."""

        # Collect historical data
        data = await self.data_collector.collect_league_data(league, start_date, end_date)
        teams = data["teams"]
        matches = data["matches"]
        team_forms = data["team_forms"]

        # Filter to completed matches only
        completed_matches = [m for m in matches if m.is_finished]

        if not completed_matches:
            raise ValueError("No completed matches found in the specified period")

        predictions = []
        actuals = []

        # Generate predictions for each match
        for match in completed_matches:
            try:
                # Get teams
                home_team = next(t for t in teams if t.id == match.home_team_id)
                away_team = next(t for t in teams if t.id == match.away_team_id)

                # Calculate performance metrics (using data up to match date)
                historical_matches = [
                    m for m in matches
                    if m.scheduled_datetime < match.scheduled_datetime and m.is_finished
                ]

                home_metrics = self.performance_analyzer.calculate_team_metrics(
                    home_team, historical_matches, team_forms.get(home_team.id), teams
                )
                away_metrics = self.performance_analyzer.calculate_team_metrics(
                    away_team, historical_matches, team_forms.get(away_team.id), teams
                )

                # Generate prediction
                prediction = model.predict_match(home_team, away_team, home_metrics, away_metrics)

                # Skip low confidence predictions if threshold is set
                if prediction.model_confidence < min_confidence:
                    continue

                predictions.append(prediction)
                actuals.append(match)

            except (StopIteration, Exception) as e:
                # Skip matches where we can't find teams or generate predictions
                continue

        if not predictions:
            raise ValueError("No valid predictions generated")

        # Calculate metrics
        result = self._calculate_metrics(predictions, actuals, model.get_model_name().value)
        result.start_date = start_date
        result.end_date = end_date

        return result

    def _calculate_metrics(self, predictions: List[MatchPrediction],
                         actuals: List[Match], model_name: str) -> BacktestResult:
        """Calculate comprehensive evaluation metrics."""

        total_matches = len(predictions)
        correct_results = 0
        correct_scores = 0

        # For probability calibration
        result_probs = []
        result_outcomes = []

        # For goal prediction accuracy
        home_goal_errors = []
        away_goal_errors = []

        # Confidence buckets
        high_conf_correct = 0
        high_conf_total = 0
        medium_conf_correct = 0
        medium_conf_total = 0
        low_conf_correct = 0
        low_conf_total = 0

        # Over/under accuracy
        over_under_results = {2.5: {"correct": 0, "total": 0}}
        btts_correct = 0

        for pred, actual in zip(predictions, actuals):
            # Result accuracy
            predicted_result = pred.most_likely_result
            actual_result = actual.result

            if predicted_result == actual_result:
                correct_results += 1

            # Score accuracy
            if (pred.most_likely_score.home_score == actual.home_score and
                pred.most_likely_score.away_score == actual.away_score):
                correct_scores += 1

            # Probability calibration data
            if actual_result == "W":
                result_probs.append(pred.home_win_probability)
                result_outcomes.append(1)
            elif actual_result == "D":
                result_probs.append(pred.draw_probability)
                result_outcomes.append(1)
            else:  # "L"
                result_probs.append(pred.away_win_probability)
                result_outcomes.append(1)

            # Goal prediction errors
            home_goal_errors.append(abs(pred.expected_goals_home - actual.home_score))
            away_goal_errors.append(abs(pred.expected_goals_away - actual.away_score))

            # Confidence-based accuracy
            is_correct = predicted_result == actual_result
            confidence = pred.model_confidence

            if confidence >= 0.7:
                high_conf_total += 1
                if is_correct:
                    high_conf_correct += 1
            elif confidence >= 0.5:
                medium_conf_total += 1
                if is_correct:
                    medium_conf_correct += 1
            else:
                low_conf_total += 1
                if is_correct:
                    low_conf_correct += 1

            # Over/under 2.5 goals accuracy
            actual_total = actual.total_goals
            predicted_over_prob = pred.over_under_probabilities.get(2.5, 0.5)
            predicted_over = predicted_over_prob > 0.5
            actual_over = actual_total > 2.5

            over_under_results[2.5]["total"] += 1
            if predicted_over == actual_over:
                over_under_results[2.5]["correct"] += 1

            # Both teams to score accuracy
            predicted_btts = pred.both_teams_score_probability > 0.5
            actual_btts = actual.home_score > 0 and actual.away_score > 0

            if predicted_btts == actual_btts:
                btts_correct += 1

        # Calculate final metrics
        result_accuracy = correct_results / total_matches
        score_accuracy = correct_scores / total_matches

        # Probability calibration metrics
        log_loss = self._calculate_log_loss(result_probs, result_outcomes)
        brier_score = self._calculate_brier_score(result_probs, result_outcomes)

        # Goal prediction accuracy
        all_goal_errors = home_goal_errors + away_goal_errors
        goals_mae = np.mean(all_goal_errors)
        goals_rmse = np.sqrt(np.mean(np.square(all_goal_errors)))

        # Confidence-based accuracy
        high_conf_acc = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
        medium_conf_acc = medium_conf_correct / medium_conf_total if medium_conf_total > 0 else 0
        low_conf_acc = low_conf_correct / low_conf_total if low_conf_total > 0 else 0

        # Market accuracy
        over_under_acc = {
            line: data["correct"] / data["total"] if data["total"] > 0 else 0
            for line, data in over_under_results.items()
        }
        btts_accuracy = btts_correct / total_matches

        return BacktestResult(
            model_name=model_name,
            start_date=datetime.now(),  # Will be overridden
            end_date=datetime.now(),    # Will be overridden
            total_matches=total_matches,
            result_accuracy=result_accuracy,
            score_accuracy=score_accuracy,
            log_loss=log_loss,
            brier_score=brier_score,
            goals_mae=goals_mae,
            goals_rmse=goals_rmse,
            high_confidence_accuracy=high_conf_acc,
            medium_confidence_accuracy=medium_conf_acc,
            low_confidence_accuracy=low_conf_acc,
            over_under_accuracy=over_under_acc,
            btts_accuracy=btts_accuracy
        )

    def _calculate_log_loss(self, probabilities: List[float], outcomes: List[int]) -> float:
        """Calculate logarithmic loss."""
        if not probabilities:
            return float('inf')

        # Clip probabilities to avoid log(0)
        clipped_probs = np.clip(probabilities, 1e-15, 1 - 1e-15)

        log_loss = -np.mean([
            outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)
            for prob, outcome in zip(clipped_probs, outcomes)
        ])

        return log_loss

    def _calculate_brier_score(self, probabilities: List[float], outcomes: List[int]) -> float:
        """Calculate Brier score."""
        if not probabilities:
            return 1.0

        brier = np.mean([(prob - outcome) ** 2 for prob, outcome in zip(probabilities, outcomes)])
        return brier

    async def compare_models(self, models: Dict[str, PredictionModelBase],
                           league: str, start_date: datetime, end_date: datetime) -> Dict[str, BacktestResult]:
        """Compare multiple models on the same dataset."""

        results = {}

        # Run backtests for all models
        tasks = [
            self.backtest_model(model, league, start_date, end_date)
            for model in models.values()
        ]

        backtest_results = await asyncio.gather(*tasks)

        # Map results to model names
        for model_name, result in zip(models.keys(), backtest_results):
            results[model_name] = result

        return results

    def generate_performance_report(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Generate a comprehensive performance comparison report."""

        report_data = []

        for model_name, result in results.items():
            report_data.append({
                "Model": model_name,
                "Total Matches": result.total_matches,
                "Result Accuracy": f"{result.result_accuracy:.3f}",
                "Score Accuracy": f"{result.score_accuracy:.3f}",
                "Log Loss": f"{result.log_loss:.3f}",
                "Brier Score": f"{result.brier_score:.3f}",
                "Goals MAE": f"{result.goals_mae:.3f}",
                "Goals RMSE": f"{result.goals_rmse:.3f}",
                "High Conf Accuracy": f"{result.high_confidence_accuracy:.3f}",
                "O/U 2.5 Accuracy": f"{result.over_under_accuracy.get(2.5, 0):.3f}",
                "BTTS Accuracy": f"{result.btts_accuracy:.3f}",
            })

        df = pd.DataFrame(report_data)

        # Sort by result accuracy (descending)
        df = df.sort_values("Result Accuracy", ascending=False)

        return df

    def calculate_feature_importance(self, model: PredictionModelBase,
                                   teams: List[Team], matches: List[Match]) -> Dict[str, float]:
        """Calculate feature importance for prediction models (simplified version)."""

        # This is a simplified feature importance calculation
        # In practice, you'd use more sophisticated methods

        features = {
            "elo_rating": 0.0,
            "recent_form": 0.0,
            "home_advantage": 0.0,
            "attack_strength": 0.0,
            "defense_strength": 0.0,
        }

        # For now, return equal importance for all features
        # This would be replaced with actual feature importance calculation
        num_features = len(features)
        equal_importance = 1.0 / num_features

        return {feature: equal_importance for feature in features}

    async def cross_validate_model(self, model: PredictionModelBase, league: str,
                                 start_date: datetime, end_date: datetime,
                                 num_folds: int = 5) -> List[BacktestResult]:
        """Perform cross-validation on a model."""

        total_days = (end_date - start_date).days
        fold_days = total_days // num_folds

        results = []

        for i in range(num_folds):
            fold_start = start_date + timedelta(days=i * fold_days)
            fold_end = min(start_date + timedelta(days=(i + 1) * fold_days), end_date)

            try:
                result = await self.backtest_model(model, league, fold_start, fold_end)
                results.append(result)
            except Exception as e:
                print(f"Fold {i+1} failed: {e}")
                continue

        return results

    def calculate_cv_statistics(self, cv_results: List[BacktestResult]) -> Dict[str, Dict[str, float]]:
        """Calculate cross-validation statistics."""

        if not cv_results:
            return {}

        metrics = {
            "result_accuracy": [r.result_accuracy for r in cv_results],
            "score_accuracy": [r.score_accuracy for r in cv_results],
            "log_loss": [r.log_loss for r in cv_results],
            "brier_score": [r.brier_score for r in cv_results],
            "goals_mae": [r.goals_mae for r in cv_results],
        }

        statistics = {}
        for metric_name, values in metrics.items():
            statistics[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }

        return statistics