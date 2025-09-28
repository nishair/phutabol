import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..models.team import Team
from ..models.match import Match, MatchContext
from ..analysis.performance import PerformanceMetrics


class InjuryImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InjuryReport:
    """Represents injury information for a player."""
    player_name: str
    position: str
    injury_type: str
    expected_return: Optional[datetime]
    importance_rating: float  # 0-1 scale, 1 being most important
    impact_level: InjuryImpact


@dataclass
class TeamNews:
    """Current team news affecting match preparation."""
    team_id: str
    injuries: List[InjuryReport]
    suspensions: List[str]  # Player names
    recent_transfers: List[Dict]
    manager_change: Optional[datetime]
    morale_level: float  # 0-1 scale


@dataclass
class ContextAdjustment:
    """Adjustments to team performance based on context."""
    team_id: str
    attack_multiplier: float = 1.0
    defense_multiplier: float = 1.0
    motivation_boost: float = 0.0
    fatigue_penalty: float = 0.0
    overall_adjustment: float = 0.0


class ContextAnalyzer:
    """Analyzes match context and its impact on team performance."""

    def __init__(self):
        self.injury_position_weights = {
            "goalkeeper": 0.8,
            "defender": 0.6,
            "midfielder": 0.7,
            "forward": 0.75,
            "striker": 0.8
        }

    def analyze_match_context(self, match: Match, home_team: Team, away_team: Team,
                            home_news: Optional[TeamNews] = None,
                            away_news: Optional[TeamNews] = None) -> Tuple[ContextAdjustment, ContextAdjustment]:
        """Analyze all contextual factors affecting the match."""

        home_adjustment = ContextAdjustment(team_id=home_team.id)
        away_adjustment = ContextAdjustment(team_id=away_team.id)

        # Home advantage analysis
        home_adv, away_disadv = self._analyze_home_advantage(match, home_team, away_team)
        home_adjustment.motivation_boost += home_adv
        away_adjustment.motivation_boost += away_disadv

        # Travel and rest analysis
        home_fatigue, away_fatigue = self._analyze_travel_fatigue(match)
        home_adjustment.fatigue_penalty += home_fatigue
        away_adjustment.fatigue_penalty += away_fatigue

        # Weather impact
        weather_home, weather_away = self._analyze_weather_impact(match, home_team, away_team)
        home_adjustment.overall_adjustment += weather_home
        away_adjustment.overall_adjustment += weather_away

        # Team news impact
        if home_news:
            injury_impact = self._analyze_injury_impact(home_news)
            suspension_impact = self._analyze_suspension_impact(home_news)
            morale_impact = self._analyze_morale_impact(home_news)

            home_adjustment.attack_multiplier *= (1 - injury_impact["attack"])
            home_adjustment.defense_multiplier *= (1 - injury_impact["defense"])
            home_adjustment.overall_adjustment += suspension_impact + morale_impact

        if away_news:
            injury_impact = self._analyze_injury_impact(away_news)
            suspension_impact = self._analyze_suspension_impact(away_news)
            morale_impact = self._analyze_morale_impact(away_news)

            away_adjustment.attack_multiplier *= (1 - injury_impact["attack"])
            away_adjustment.defense_multiplier *= (1 - injury_impact["defense"])
            away_adjustment.overall_adjustment += suspension_impact + morale_impact

        # Match importance
        importance_home, importance_away = self._analyze_match_importance(match, home_team, away_team)
        home_adjustment.motivation_boost += importance_home
        away_adjustment.motivation_boost += importance_away

        # Calculate final adjustments
        home_adjustment.overall_adjustment = (
            home_adjustment.motivation_boost +
            home_adjustment.fatigue_penalty +
            home_adjustment.overall_adjustment
        )

        away_adjustment.overall_adjustment = (
            away_adjustment.motivation_boost +
            away_adjustment.fatigue_penalty +
            away_adjustment.overall_adjustment
        )

        return home_adjustment, away_adjustment

    def _analyze_home_advantage(self, match: Match, home_team: Team, away_team: Team) -> Tuple[float, float]:
        """Analyze home advantage factors."""
        if not match.context.is_home_advantage:
            return 0.0, 0.0

        # Base home advantage
        home_boost = 0.15

        # Stadium-specific factors
        if match.context.attendance:
            # Higher attendance increases home advantage
            capacity = 50000  # Assumed average capacity
            attendance_ratio = min(1.0, match.context.attendance / capacity)
            home_boost *= (0.8 + 0.4 * attendance_ratio)

        # Recent home form
        if home_team.home_win_rate > 0.7:
            home_boost *= 1.2
        elif home_team.home_win_rate < 0.3:
            home_boost *= 0.8

        # Away team's away record
        away_penalty = 0.1
        if away_team.away_win_rate < 0.3:
            away_penalty *= 1.3
        elif away_team.away_win_rate > 0.6:
            away_penalty *= 0.7

        return home_boost, -away_penalty

    def _analyze_travel_fatigue(self, match: Match) -> Tuple[float, float]:
        """Analyze travel distance and rest days impact."""
        home_fatigue = 0.0
        away_fatigue = 0.0

        # Rest days impact
        if match.context.home_team_rest_days is not None:
            if match.context.home_team_rest_days < 3:
                home_fatigue -= 0.05 * (3 - match.context.home_team_rest_days)
            elif match.context.home_team_rest_days > 7:
                home_fatigue -= 0.02  # Too much rest can be detrimental

        if match.context.away_team_rest_days is not None:
            if match.context.away_team_rest_days < 3:
                away_fatigue -= 0.05 * (3 - match.context.away_team_rest_days)
            elif match.context.away_team_rest_days > 7:
                away_fatigue -= 0.02

        # Travel distance impact (mainly affects away team)
        if match.context.away_team_travel_distance is not None:
            if match.context.away_team_travel_distance > 500:  # Long distance travel
                travel_penalty = min(0.1, match.context.away_team_travel_distance / 5000)
                away_fatigue -= travel_penalty

        return home_fatigue, away_fatigue

    def _analyze_weather_impact(self, match: Match, home_team: Team, away_team: Team) -> Tuple[float, float]:
        """Analyze weather conditions impact."""
        if not match.context.weather_conditions:
            return 0.0, 0.0

        weather = match.context.weather_conditions.lower()
        temperature = match.context.temperature

        home_impact = 0.0
        away_impact = 0.0

        # Extreme weather affects both teams but home team adapts better
        if "rain" in weather or "snow" in weather:
            # Wet conditions generally reduce scoring
            home_impact -= 0.03
            away_impact -= 0.05  # Away team affected more

        if temperature is not None:
            # Extreme temperatures
            if temperature < 0 or temperature > 35:
                home_impact -= 0.02
                away_impact -= 0.04  # Home team better adapted to local climate

        if "wind" in weather:
            # Strong wind affects passing and shooting
            home_impact -= 0.02
            away_impact -= 0.02

        return home_impact, away_impact

    def _analyze_injury_impact(self, team_news: TeamNews) -> Dict[str, float]:
        """Analyze impact of injuries on team performance."""
        attack_impact = 0.0
        defense_impact = 0.0

        for injury in team_news.injuries:
            position_weight = self.injury_position_weights.get(
                injury.position.lower(), 0.5
            )

            impact_multiplier = {
                InjuryImpact.LOW: 0.1,
                InjuryImpact.MEDIUM: 0.3,
                InjuryImpact.HIGH: 0.6,
                InjuryImpact.CRITICAL: 1.0
            }.get(injury.impact_level, 0.3)

            total_impact = injury.importance_rating * position_weight * impact_multiplier

            # Distribute impact based on position
            if injury.position.lower() in ["forward", "striker"]:
                attack_impact += total_impact * 0.8
                defense_impact += total_impact * 0.2
            elif injury.position.lower() == "midfielder":
                attack_impact += total_impact * 0.5
                defense_impact += total_impact * 0.5
            elif injury.position.lower() == "defender":
                attack_impact += total_impact * 0.2
                defense_impact += total_impact * 0.8
            elif injury.position.lower() == "goalkeeper":
                defense_impact += total_impact

        return {
            "attack": min(0.5, attack_impact),  # Cap at 50% impact
            "defense": min(0.5, defense_impact)
        }

    def _analyze_suspension_impact(self, team_news: TeamNews) -> float:
        """Analyze impact of suspensions."""
        # Simple model: each suspension impacts team by 5%
        return -0.05 * len(team_news.suspensions)

    def _analyze_morale_impact(self, team_news: TeamNews) -> float:
        """Analyze team morale impact."""
        morale_impact = (team_news.morale_level - 0.5) * 0.2  # Scale to ±10%

        # Recent manager change can be disruptive
        if team_news.manager_change:
            days_since_change = (datetime.now() - team_news.manager_change).days
            if days_since_change < 30:
                morale_impact -= 0.05 * (30 - days_since_change) / 30

        return morale_impact

    def _analyze_match_importance(self, match: Match, home_team: Team, away_team: Team) -> Tuple[float, float]:
        """Analyze match importance and its motivational impact."""
        importance_boost = 0.0

        # Cup matches and playoffs
        if match.context.is_cup_match:
            importance_boost += 0.08
        if match.context.is_playoff:
            importance_boost += 0.12

        # Rivalry matches
        if match.context.is_rivalry:
            importance_boost += 0.05

        # Tournament stage importance
        if match.context.tournament_stage:
            stage_boosts = {
                "final": 0.15,
                "semi-final": 0.12,
                "quarter-final": 0.08,
                "round-of-16": 0.05
            }
            importance_boost += stage_boosts.get(
                match.context.tournament_stage.lower(), 0.0
            )

        # Both teams get the same importance boost
        return importance_boost, importance_boost

    def adjust_performance_metrics(self, metrics: PerformanceMetrics,
                                 adjustment: ContextAdjustment) -> PerformanceMetrics:
        """Apply context adjustments to performance metrics."""

        # Create a new metrics object with adjustments
        adjusted_metrics = PerformanceMetrics(
            team_id=metrics.team_id,
            attack_strength=metrics.attack_strength * adjustment.attack_multiplier,
            defense_strength=metrics.defense_strength * adjustment.defense_multiplier,
            overall_strength=metrics.overall_strength * (1 + adjustment.overall_adjustment),
            recent_form_score=metrics.recent_form_score,
            momentum=metrics.momentum + adjustment.motivation_boost,
            goal_scoring_consistency=metrics.goal_scoring_consistency,
            defensive_consistency=metrics.defensive_consistency,
            performance_vs_expectation=metrics.performance_vs_expectation,
            home_advantage_factor=metrics.home_advantage_factor,
            away_performance_factor=metrics.away_performance_factor,
            strength_of_schedule=metrics.strength_of_schedule,
            adjusted_goal_difference=metrics.adjusted_goal_difference
        )

        return adjusted_metrics

    def get_context_summary(self, match: Match, home_adjustment: ContextAdjustment,
                          away_adjustment: ContextAdjustment) -> Dict[str, str]:
        """Generate a human-readable summary of context factors."""
        summary = {}

        # Weather
        if match.context.weather_conditions:
            summary["weather"] = f"Weather: {match.context.weather_conditions}"
            if match.context.temperature:
                summary["weather"] += f", {match.context.temperature}°C"

        # Attendance
        if match.context.attendance:
            summary["attendance"] = f"Expected attendance: {match.context.attendance:,}"

        # Rest and travel
        if match.context.home_team_rest_days:
            summary["home_rest"] = f"Home team rest: {match.context.home_team_rest_days} days"
        if match.context.away_team_rest_days:
            summary["away_rest"] = f"Away team rest: {match.context.away_team_rest_days} days"

        # Match importance
        importance_factors = []
        if match.context.is_cup_match:
            importance_factors.append("Cup match")
        if match.context.is_playoff:
            importance_factors.append("Playoff")
        if match.context.is_rivalry:
            importance_factors.append("Derby/Rivalry")
        if match.context.tournament_stage:
            importance_factors.append(f"{match.context.tournament_stage}")

        if importance_factors:
            summary["importance"] = f"Special factors: {', '.join(importance_factors)}"

        # Adjustments impact
        if abs(home_adjustment.overall_adjustment) > 0.05:
            direction = "boosted" if home_adjustment.overall_adjustment > 0 else "reduced"
            summary["home_impact"] = f"Home team performance {direction}"

        if abs(away_adjustment.overall_adjustment) > 0.05:
            direction = "boosted" if away_adjustment.overall_adjustment > 0 else "reduced"
            summary["away_impact"] = f"Away team performance {direction}"

        return summary