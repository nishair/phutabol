#!/usr/bin/env python3
"""
Live Data Example for Phutabol Soccer Prediction System

This script demonstrates how to use the system with real-time data from
Football-Data.org and other live sources.
"""

import asyncio
import os
from datetime import datetime, timedelta

from phutabol.config import get_config, DataSourceType
from phutabol.data.live_sources import FootballDataOrgSource, LiveDataCollector
from phutabol.data.collector import MockDataSource
from phutabol.analysis.performance import PerformanceAnalyzer
from phutabol.prediction.models import PoissonModel, EloBasedModel, ModelEnsemble


async def demonstrate_live_data():
    """Demonstrate live data capabilities."""

    print("🔮 Phutabol - Live Data Example")
    print("=" * 50)

    # Get configuration
    config = get_config()
    print(f"📡 Data Source: {config.data_source_type.value}")
    print(f"🔑 Live Data Available: {config.has_live_data_access()}")

    if not config.has_live_data_access():
        print("\n⚠️  No live data access configured!")
        print(config.get_setup_instructions())
        print("\nFalling back to enhanced mock data demonstration...")
        await demonstrate_enhanced_mock_data()
        return

    # Initialize live data source
    print(f"\n🌐 Connecting to Football-Data.org API...")

    try:
        async with FootballDataOrgSource(config.api_config.football_data_api_key) as live_source:
            # Create live data collector
            live_collector = LiveDataCollector(live_source, [MockDataSource()])

            # Demonstrate live data for Premier League
            await demonstrate_league_data(live_collector, "Premier League")

            # Demonstrate predictions with live data
            await demonstrate_live_predictions(live_collector, "Premier League")

    except Exception as e:
        print(f"❌ Live data failed: {e}")
        print("📊 Falling back to mock data...")
        await demonstrate_enhanced_mock_data()


async def demonstrate_league_data(live_collector: LiveDataCollector, league: str):
    """Demonstrate live league data collection."""

    print(f"\n📊 Live Data for {league}")
    print("-" * 30)

    try:
        # Get current season data
        current_data = await live_collector.get_current_season_data(league)

        teams = current_data["teams"]
        recent_matches = current_data["recent_matches"]
        upcoming_fixtures = current_data.get("upcoming_fixtures", [])
        team_forms = current_data["team_forms"]

        print(f"✅ Loaded {len(teams)} teams")
        print(f"✅ Found {len(recent_matches)} recent matches")
        print(f"✅ Found {len(upcoming_fixtures)} upcoming fixtures")
        print(f"📅 Last updated: {current_data['last_updated']}")
        print(f"🔄 Data source: {current_data['source']}")

        # Show top teams by current standings
        print(f"\n🏆 Top 5 Teams (by Elo Rating):")
        sorted_teams = sorted(teams, key=lambda t: t.elo_rating, reverse=True)[:5]

        for i, team in enumerate(sorted_teams, 1):
            wins = team.wins
            played = team.matches_played
            points = team.points
            gd = team.goal_difference

            print(f"   {i}. {team.name}")
            print(f"      Elo: {team.elo_rating:.0f} | Record: {wins}W-{team.draws}D-{team.losses}L")
            print(f"      Points: {points} | GD: {gd:+d} | Played: {played}")

        # Show recent form for top teams
        print(f"\n📈 Recent Form (Last 5 Games):")
        for team in sorted_teams[:3]:
            form = team_forms.get(team.id)
            if form and form.form_string:
                print(f"   {team.name}: {form.form_string} (Form Score: {form.recent_form_score:.2f})")

        # Show upcoming fixtures
        if upcoming_fixtures:
            print(f"\n📅 Next 3 Upcoming Fixtures:")
            upcoming_sorted = sorted(upcoming_fixtures, key=lambda f: f.scheduled_datetime or datetime.now())

            for fixture in upcoming_sorted[:3]:
                home_team = next((t for t in teams if t.id == fixture.home_team_id), None)
                away_team = next((t for t in teams if t.id == fixture.away_team_id), None)

                if home_team and away_team:
                    date_str = fixture.scheduled_datetime.strftime("%Y-%m-%d %H:%M") if fixture.scheduled_datetime else "TBD"
                    print(f"   {date_str}: {home_team.name} vs {away_team.name}")

    except Exception as e:
        print(f"❌ Failed to get live data: {e}")


async def demonstrate_live_predictions(live_collector: LiveDataCollector, league: str):
    """Demonstrate predictions using live data."""

    print(f"\n🔮 Live Predictions for {league}")
    print("-" * 35)

    try:
        # Get current data
        current_data = await live_collector.get_current_season_data(league)
        teams = current_data["teams"]
        matches = current_data["recent_matches"]
        team_forms = current_data["team_forms"]

        if len(teams) < 2:
            print("❌ Not enough teams for prediction")
            return

        # Initialize analyzers and models
        performance_analyzer = PerformanceAnalyzer()

        # Get top 2 teams for prediction
        sorted_teams = sorted(teams, key=lambda t: t.elo_rating, reverse=True)
        home_team = sorted_teams[0]
        away_team = sorted_teams[1]

        print(f"🏠 Home: {home_team.name} (Elo: {home_team.elo_rating:.0f})")
        print(f"🚗 Away: {away_team.name} (Elo: {away_team.elo_rating:.0f})")

        # Calculate performance metrics using live data
        home_form = team_forms.get(home_team.id)
        away_form = team_forms.get(away_team.id)

        home_metrics = performance_analyzer.calculate_team_metrics(
            home_team, matches, home_form, teams
        )
        away_metrics = performance_analyzer.calculate_team_metrics(
            away_team, matches, away_form, teams
        )

        print(f"\n📊 Performance Analysis:")
        print(f"   {home_team.name}: Attack {home_metrics.attack_strength:.2f}, Defense {home_metrics.defense_strength:.2f}")
        print(f"   {away_team.name}: Attack {away_metrics.attack_strength:.2f}, Defense {away_metrics.defense_strength:.2f}")

        # Make predictions with different models
        models = {
            "Elo-based": EloBasedModel(),
            "Poisson": PoissonModel(),
            "Ensemble": ModelEnsemble([EloBasedModel(), PoissonModel()])
        }

        print(f"\n🎯 Match Predictions:")
        print("=" * 50)

        for model_name, model in models.items():
            prediction = model.predict_match(home_team, away_team, home_metrics, away_metrics)

            print(f"\n📊 {model_name} Model:")
            print(f"   🏠 {home_team.name} Win: {prediction.home_win_probability:.1%}")
            print(f"   🤝 Draw: {prediction.draw_probability:.1%}")
            print(f"   🚗 {away_team.name} Win: {prediction.away_win_probability:.1%}")
            print(f"   ⚽ Expected Goals: {prediction.expected_goals_home:.1f} - {prediction.expected_goals_away:.1f}")
            print(f"   🎲 Most Likely: {prediction.most_likely_score.home_score}-{prediction.most_likely_score.away_score}")
            print(f"   📈 Over 2.5 Goals: {prediction.over_under_probabilities.get(2.5, 0):.1%}")

        # Show current league context
        avg_goals = sum(m.total_goals or 0 for m in matches if m.total_goals) / max(len([m for m in matches if m.total_goals]), 1)
        print(f"\n📊 League Context:")
        print(f"   Average goals per game: {avg_goals:.1f}")
        print(f"   Total teams analyzed: {len(teams)}")
        print(f"   Recent matches analyzed: {len(matches)}")
        print(f"   Data freshness: {current_data['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")


async def demonstrate_enhanced_mock_data():
    """Enhanced demonstration using mock data with live-like features."""

    print(f"\n🧪 Enhanced Mock Data Demonstration")
    print("-" * 40)

    from phutabol.data.collector import DataCollector, MockDataSource

    # Use mock data source
    mock_source = MockDataSource()
    collector = DataCollector(mock_source)
    performance_analyzer = PerformanceAnalyzer()

    try:
        # Collect mock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        league_data = await collector.collect_league_data("Premier League", start_date, end_date)

        teams = league_data["teams"]
        matches = league_data["matches"]
        team_forms = league_data["team_forms"]

        print(f"📊 Mock Data Loaded:")
        print(f"   Teams: {len(teams)}")
        print(f"   Matches: {len(matches)}")
        print(f"   Collection time: {league_data['collection_timestamp']}")

        # Enhanced mock predictions
        home_team = teams[0]
        away_team = teams[1]

        print(f"\n🔮 Mock Prediction: {home_team.name} vs {away_team.name}")

        home_metrics = performance_analyzer.calculate_team_metrics(
            home_team, matches, team_forms[home_team.id], teams
        )
        away_metrics = performance_analyzer.calculate_team_metrics(
            away_team, matches, team_forms[away_team.id], teams
        )

        model = ModelEnsemble([EloBasedModel(), PoissonModel()])
        prediction = model.predict_match(home_team, away_team, home_metrics, away_metrics)

        print(f"   Home Win: {prediction.home_win_probability:.1%}")
        print(f"   Draw: {prediction.draw_probability:.1%}")
        print(f"   Away Win: {prediction.away_win_probability:.1%}")

        print(f"\n💡 To use live data:")
        print(f"   1. Get a free API key from https://www.football-data.org/client/register")
        print(f"   2. Set: export FOOTBALL_DATA_API_KEY='your_key'")
        print(f"   3. Set: export PHUTABOL_DATA_SOURCE='football_data_org'")
        print(f"   4. Restart the application")

    except Exception as e:
        print(f"❌ Mock demonstration failed: {e}")


def main():
    """Main function."""
    print("Choose a demonstration:")
    print("1. Live data example (requires API key)")
    print("2. Enhanced mock data example")
    print("3. Configuration status")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        asyncio.run(demonstrate_live_data())
    elif choice == "2":
        asyncio.run(demonstrate_enhanced_mock_data())
    elif choice == "3":
        config = get_config()
        print("\n📋 Configuration Status:")
        print(config.get_setup_instructions())
    else:
        print("Invalid choice. Running live data example...")
        asyncio.run(demonstrate_live_data())


if __name__ == "__main__":
    main()