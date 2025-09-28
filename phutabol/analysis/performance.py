import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..models.team import Team, TeamForm
from ..models.match import Match


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a team."""

    team_id: str

    # Basic metrics
    attack_strength: float
    defense_strength: float
    overall_strength: float

    # Form metrics
    recent_form_score: float
    momentum: float

    # Advanced metrics
    goal_scoring_consistency: float
    defensive_consistency: float
    performance_vs_expectation: float

    # Context-specific metrics
    home_advantage_factor: float
    away_performance_factor: float

    # Opponent-adjusted metrics
    strength_of_schedule: float
    adjusted_goal_difference: float


class PerformanceAnalyzer:
    """Analyzes team performance using various metrics and statistical methods."""

    def __init__(self):
        self.league_averages = {}

    def calculate_team_metrics(self, team: Team, matches: List[Match],
                             team_form: TeamForm, league_teams: List[Team]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for a team."""

        # Calculate league averages for context
        league_avg_goals = np.mean([t.goals_per_game for t in league_teams])
        league_avg_conceded = np.mean([t.goals_conceded_per_game for t in league_teams])

        # Basic strength metrics
        attack_strength = self._calculate_attack_strength(team, league_avg_goals)
        defense_strength = self._calculate_defense_strength(team, league_avg_conceded)
        overall_strength = self._calculate_overall_strength(attack_strength, defense_strength)

        # Form and momentum
        recent_form_score = team_form.recent_form_score
        momentum = self._calculate_momentum(team_form)

        # Consistency metrics
        team_matches = [m for m in matches if m.home_team_id == team.id or m.away_team_id == team.id]
        goal_scoring_consistency = self._calculate_scoring_consistency(team, team_matches)
        defensive_consistency = self._calculate_defensive_consistency(team, team_matches)

        # Performance vs expectation
        performance_vs_expectation = self._calculate_performance_vs_expectation(team)

        # Home/away factors
        home_advantage_factor = self._calculate_home_advantage(team)
        away_performance_factor = self._calculate_away_performance(team)

        # Opponent-adjusted metrics
        strength_of_schedule = self._calculate_strength_of_schedule(team, team_matches, league_teams)
        adjusted_goal_difference = self._calculate_adjusted_goal_difference(
            team, team_matches, league_teams
        )

        return PerformanceMetrics(
            team_id=team.id,
            attack_strength=attack_strength,
            defense_strength=defense_strength,
            overall_strength=overall_strength,
            recent_form_score=recent_form_score,
            momentum=momentum,
            goal_scoring_consistency=goal_scoring_consistency,
            defensive_consistency=defensive_consistency,
            performance_vs_expectation=performance_vs_expectation,
            home_advantage_factor=home_advantage_factor,
            away_performance_factor=away_performance_factor,
            strength_of_schedule=strength_of_schedule,
            adjusted_goal_difference=adjusted_goal_difference
        )

    def _calculate_attack_strength(self, team: Team, league_average: float) -> float:
        """Calculate attack strength relative to league average."""
        if team.matches_played == 0:
            return 1.0

        team_avg = team.goals_per_game
        return team_avg / league_average if league_average > 0 else 1.0

    def _calculate_defense_strength(self, team: Team, league_average: float) -> float:
        """Calculate defense strength relative to league average (lower is better)."""
        if team.matches_played == 0:
            return 1.0

        team_avg = team.goals_conceded_per_game
        # Invert the ratio since lower goals conceded is better
        return league_average / team_avg if team_avg > 0 else 2.0

    def _calculate_overall_strength(self, attack_strength: float, defense_strength: float) -> float:
        """Calculate overall team strength combining attack and defense."""
        # Weighted combination of attack and defense
        return (attack_strength * 0.6) + (defense_strength * 0.4)

    def _calculate_momentum(self, team_form: TeamForm) -> float:
        """Calculate team momentum based on recent results."""
        if not team_form.recent_matches:
            return 0.0

        # Weight recent matches more heavily
        weights = [0.4, 0.3, 0.2, 0.1]  # Most recent gets highest weight
        momentum = 0.0

        for i, match in enumerate(team_form.recent_matches[:4]):
            weight = weights[i] if i < len(weights) else 0.05

            if match["result"] == "W":
                momentum += 3 * weight
            elif match["result"] == "D":
                momentum += 1 * weight
            # Loss adds 0

        return momentum / 3.0  # Normalize to 0-1 scale

    def _calculate_scoring_consistency(self, team: Team, matches: List[Match]) -> float:
        """Calculate how consistent the team's goal scoring is."""
        if not matches:
            return 0.0

        goals_scored = []
        for match in matches:
            if match.is_finished:
                goals = match.get_team_score(team.id)
                if goals is not None:
                    goals_scored.append(goals)

        if len(goals_scored) < 2:
            return 0.0

        # Use coefficient of variation (lower = more consistent)
        mean_goals = np.mean(goals_scored)
        std_goals = np.std(goals_scored)

        if mean_goals == 0:
            return 0.0

        cv = std_goals / mean_goals
        # Convert to consistency score (higher = more consistent)
        return max(0.0, 1.0 - (cv / 2.0))

    def _calculate_defensive_consistency(self, team: Team, matches: List[Match]) -> float:
        """Calculate how consistent the team's defensive performance is."""
        if not matches:
            return 0.0

        goals_conceded = []
        for match in matches:
            if match.is_finished:
                opponent_id = match.get_opponent_id(team.id)
                opponent_goals = match.get_team_score(opponent_id)
                if opponent_goals is not None:
                    goals_conceded.append(opponent_goals)

        if len(goals_conceded) < 2:
            return 0.0

        # Use coefficient of variation (lower = more consistent)
        mean_conceded = np.mean(goals_conceded)
        std_conceded = np.std(goals_conceded)

        if mean_conceded == 0:
            return 1.0  # Perfect defensive consistency

        cv = std_conceded / mean_conceded
        # Convert to consistency score (higher = more consistent)
        return max(0.0, 1.0 - (cv / 2.0))

    def _calculate_performance_vs_expectation(self, team: Team) -> float:
        """Calculate how team is performing vs expected goals."""
        if team.expected_goals_for == 0 and team.expected_goals_against == 0:
            return 0.0

        # Compare actual vs expected goal difference
        actual_gd = team.goal_difference
        expected_gd = team.expected_goals_for - team.expected_goals_against

        if expected_gd == 0:
            return 0.0 if actual_gd == 0 else (1.0 if actual_gd > 0 else -1.0)

        performance_ratio = actual_gd / expected_gd

        # Normalize to reasonable range
        return max(-2.0, min(2.0, performance_ratio - 1.0))

    def _calculate_home_advantage(self, team: Team) -> float:
        """Calculate team's home advantage factor."""
        total_home_matches = team.home_wins + team.home_draws + team.home_losses
        total_matches = team.matches_played

        if total_home_matches == 0 or total_matches == 0:
            return 1.0

        home_points_per_game = (team.home_wins * 3 + team.home_draws) / total_home_matches
        overall_points_per_game = (team.wins * 3 + team.draws) / total_matches

        if overall_points_per_game == 0:
            return 1.0

        return home_points_per_game / overall_points_per_game

    def _calculate_away_performance(self, team: Team) -> float:
        """Calculate team's away performance factor."""
        total_away_matches = team.away_wins + team.away_draws + team.away_losses
        total_matches = team.matches_played

        if total_away_matches == 0 or total_matches == 0:
            return 1.0

        away_points_per_game = (team.away_wins * 3 + team.away_draws) / total_away_matches
        overall_points_per_game = (team.wins * 3 + team.draws) / total_matches

        if overall_points_per_game == 0:
            return 1.0

        return away_points_per_game / overall_points_per_game

    def _calculate_strength_of_schedule(self, team: Team, matches: List[Match],
                                      league_teams: List[Team]) -> float:
        """Calculate the strength of opponents faced."""
        if not matches:
            return 1.0

        team_map = {t.id: t for t in league_teams}
        opponent_elos = []

        for match in matches:
            if match.is_finished:
                opponent_id = match.get_opponent_id(team.id)
                opponent = team_map.get(opponent_id)
                if opponent:
                    opponent_elos.append(opponent.elo_rating)

        if not opponent_elos:
            return 1.0

        avg_opponent_elo = np.mean(opponent_elos)
        league_avg_elo = np.mean([t.elo_rating for t in league_teams])

        return avg_opponent_elo / league_avg_elo if league_avg_elo > 0 else 1.0

    def _calculate_adjusted_goal_difference(self, team: Team, matches: List[Match],
                                          league_teams: List[Team]) -> float:
        """Calculate goal difference adjusted for opponent strength."""
        if not matches:
            return 0.0

        team_map = {t.id: t for t in league_teams}
        adjusted_gd = 0.0
        match_count = 0

        for match in matches:
            if match.is_finished:
                opponent_id = match.get_opponent_id(team.id)
                opponent = team_map.get(opponent_id)

                if opponent:
                    team_goals = match.get_team_score(team.id)
                    opponent_goals = match.get_team_score(opponent_id)

                    if team_goals is not None and opponent_goals is not None:
                        match_gd = team_goals - opponent_goals

                        # Adjust based on opponent strength
                        opponent_strength = opponent.elo_rating / 1500.0  # Normalize around 1500
                        adjusted_match_gd = match_gd / opponent_strength

                        adjusted_gd += adjusted_match_gd
                        match_count += 1

        return adjusted_gd / match_count if match_count > 0 else 0.0

    def compare_teams(self, team1_metrics: PerformanceMetrics,
                     team2_metrics: PerformanceMetrics) -> Dict[str, float]:
        """Compare two teams across all performance metrics."""

        comparisons = {
            "attack_advantage": team1_metrics.attack_strength / team2_metrics.attack_strength,
            "defense_advantage": team1_metrics.defense_strength / team2_metrics.defense_strength,
            "overall_advantage": team1_metrics.overall_strength / team2_metrics.overall_strength,
            "form_advantage": team1_metrics.recent_form_score - team2_metrics.recent_form_score,
            "momentum_advantage": team1_metrics.momentum - team2_metrics.momentum,
            "consistency_advantage": (
                (team1_metrics.goal_scoring_consistency + team1_metrics.defensive_consistency) -
                (team2_metrics.goal_scoring_consistency + team2_metrics.defensive_consistency)
            ) / 2.0
        }

        return comparisons

    def get_league_rankings(self, teams: List[Team]) -> pd.DataFrame:
        """Generate league rankings based on various metrics."""

        rankings_data = []
        for team in teams:
            rankings_data.append({
                "team_name": team.name,
                "elo_rating": team.elo_rating,
                "points": team.points,
                "goal_difference": team.goal_difference,
                "goals_per_game": team.goals_per_game,
                "goals_conceded_per_game": team.goals_conceded_per_game,
                "win_rate": team.win_rate,
                "home_win_rate": team.home_win_rate,
                "away_win_rate": team.away_win_rate,
            })

        df = pd.DataFrame(rankings_data)
        return df.sort_values("elo_rating", ascending=False)