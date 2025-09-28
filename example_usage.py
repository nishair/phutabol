#!/usr/bin/env python3
"""
Example usage of the Phutabol Soccer Prediction System

This script demonstrates how to use the various components of the system
to make predictions and analyze team performance.
"""

import asyncio
from datetime import datetime, timedelta

from phutabol.data.collector import DataCollector, MockDataSource
from phutabol.analysis.performance import PerformanceAnalyzer
from phutabol.analysis.context import ContextAnalyzer
from phutabol.prediction.models import PoissonModel, DixonColesModel, EloBasedModel, ModelEnsemble
from phutabol.utils.evaluation import ModelEvaluator


async def main():
    """Main example function demonstrating the system capabilities."""

    print("🔮 Phutabol Soccer Prediction System - Example Usage")
    print("=" * 60)

    # Initialize components
    data_source = MockDataSource()
    data_collector = DataCollector(data_source)
    performance_analyzer = PerformanceAnalyzer()
    context_analyzer = ContextAnalyzer()

    # Example 1: Collect and analyze league data
    print("\n📊 1. Collecting League Data")
    print("-" * 30)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    league_data = await data_collector.collect_league_data(
        "Premier League", start_date, end_date
    )

    teams = league_data["teams"]
    matches = league_data["matches"]
    team_forms = league_data["team_forms"]

    print(f"✅ Collected data for {len(teams)} teams and {len(matches)} matches")

    # Example 2: Analyze team performance
    print("\n⚡ 2. Team Performance Analysis")
    print("-" * 35)

    for team in teams[:2]:  # Analyze first 2 teams
        metrics = performance_analyzer.calculate_team_metrics(
            team, matches, team_forms[team.id], teams
        )

        print(f"\n🏆 {team.name}")
        print(f"   Attack Strength: {metrics.attack_strength:.2f}")
        print(f"   Defense Strength: {metrics.defense_strength:.2f}")
        print(f"   Overall Strength: {metrics.overall_strength:.2f}")
        print(f"   Recent Form: {metrics.recent_form_score:.2f}")
        print(f"   Momentum: {metrics.momentum:.2f}")

    # Example 3: Generate league standings
    print("\n🏆 3. League Standings")
    print("-" * 25)

    standings = performance_analyzer.get_league_rankings(teams)
    print(standings[["team_name", "elo_rating", "points", "goal_difference", "win_rate"]].head())

    # Example 4: Make match predictions
    print("\n🔮 4. Match Predictions")
    print("-" * 25)

    # Get two teams for prediction
    home_team = teams[0]
    away_team = teams[1]

    # Calculate metrics for both teams
    home_metrics = performance_analyzer.calculate_team_metrics(
        home_team, matches, team_forms[home_team.id], teams
    )
    away_metrics = performance_analyzer.calculate_team_metrics(
        away_team, matches, team_forms[away_team.id], teams
    )

    # Test different models
    models = {
        "Poisson": PoissonModel(),
        "Dixon-Coles": DixonColesModel(),
        "Elo-based": EloBasedModel(),
        "Ensemble": ModelEnsemble([PoissonModel(), DixonColesModel(), EloBasedModel()])
    }

    print(f"\n🏠 {home_team.name} vs {away_team.name} 🚗")
    print("=" * 50)

    for model_name, model in models.items():
        prediction = model.predict_match(home_team, away_team, home_metrics, away_metrics)

        print(f"\n📊 {model_name} Model:")
        print(f"   Home Win: {prediction.home_win_probability:.1%}")
        print(f"   Draw:     {prediction.draw_probability:.1%}")
        print(f"   Away Win: {prediction.away_win_probability:.1%}")
        print(f"   Expected Goals: {prediction.expected_goals_home:.1f} - {prediction.expected_goals_away:.1f}")
        print(f"   Most Likely Score: {prediction.most_likely_score.home_score}-{prediction.most_likely_score.away_score}")
        print(f"   Over 2.5 Goals: {prediction.over_under_probabilities.get(2.5, 0):.1%}")
        print(f"   Both Teams Score: {prediction.both_teams_score_probability:.1%}")

    # Example 5: Model evaluation and backtesting
    print("\n📈 5. Model Evaluation")
    print("-" * 25)

    evaluator = ModelEvaluator(data_collector)

    try:
        # Backtest a single model
        backtest_result = await evaluator.backtest_model(
            PoissonModel(), "Premier League", start_date, end_date
        )

        print(f"\n🧪 Poisson Model Backtest Results:")
        print(f"   Total Matches: {backtest_result.total_matches}")
        print(f"   Result Accuracy: {backtest_result.result_accuracy:.1%}")
        print(f"   Score Accuracy: {backtest_result.score_accuracy:.1%}")
        print(f"   Goals MAE: {backtest_result.goals_mae:.2f}")
        print(f"   Log Loss: {backtest_result.log_loss:.3f}")
        print(f"   High Confidence Accuracy: {backtest_result.high_confidence_accuracy:.1%}")

        # Compare multiple models
        print(f"\n🔄 Comparing Multiple Models...")
        model_comparison = await evaluator.compare_models(
            {
                "poisson": PoissonModel(),
                "dixon_coles": DixonColesModel(),
                "elo": EloBasedModel()
            },
            "Premier League", start_date, end_date
        )

        # Generate performance report
        performance_report = evaluator.generate_performance_report(model_comparison)
        print("\n📋 Model Comparison Report:")
        print(performance_report.to_string(index=False))

    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")

    # Example 6: Team comparison
    print("\n⚔️  6. Team Comparison")
    print("-" * 25)

    comparison = performance_analyzer.compare_teams(home_metrics, away_metrics)

    print(f"\n{home_team.name} vs {away_team.name} Analysis:")
    for factor, value in comparison.items():
        advantage = "Home" if value > 1 or value > 0 else "Away"
        strength = abs(value - 1) if "advantage" in factor else abs(value)

        if strength > 0.1:
            print(f"   {factor.replace('_', ' ').title()}: {advantage} team advantage ({strength:.2f})")

    print(f"\n✅ Example completed successfully!")
    print("\n💡 To run the API server, use: python -m phutabol.api.main")
    print("💡 Visit http://localhost:8000/docs for interactive API documentation")


def run_api_example():
    """Example of how to start the API server."""
    print("\n🚀 Starting API Server...")
    print("Visit http://localhost:8000/docs for interactive documentation")

    import uvicorn
    from phutabol.api.main import app

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run prediction examples")
    print("2. Start API server")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        asyncio.run(main())
    elif choice == "2":
        run_api_example()
    else:
        print("Invalid choice. Running prediction examples...")
        asyncio.run(main())