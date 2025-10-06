from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio

from ..data.collector import DataCollector, MockDataSource
from ..data.live_sources import FootballDataOrgSource, LiveDataCollector
from ..config import get_config, DataSourceType
from ..analysis.performance import PerformanceAnalyzer
from ..analysis.context import ContextAnalyzer
from ..prediction.models import (
    PoissonModel, DixonColesModel, EloBasedModel,
    BivariatePoissonModel, ModelEnsemble
)
from ..models.team import Team
from ..models.match import Match
from ..models.prediction import MatchPrediction
from .schemas import (
    PredictionRequest, PredictionResponse, TeamResponse,
    MatchResponse, LeagueStandingsResponse
)


app = FastAPI(
    title="Phutabol - Soccer Prediction API",
    description="Advanced soccer match prediction system using multiple statistical models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration and instances
app_config = get_config()

# Initialize data source based on configuration
def create_data_source():
    """Create appropriate data source based on configuration."""
    if app_config.data_source_type == DataSourceType.FOOTBALL_DATA_ORG:
        if app_config.api_config.football_data_api_key:
            return FootballDataOrgSource(app_config.api_config.football_data_api_key)
        else:
            print("⚠️  Football-Data.org API key not found, falling back to mock data")
            return MockDataSource()
    elif app_config.data_source_type == DataSourceType.MIXED:
        # Use live data collector with fallbacks
        primary_source = FootballDataOrgSource(app_config.api_config.football_data_api_key)
        return LiveDataCollector(primary_source, [MockDataSource()])
    else:
        return MockDataSource()

# Global instances
data_source = create_data_source()
if isinstance(data_source, LiveDataCollector):
    data_collector = data_source
else:
    data_collector = DataCollector(data_source)
performance_analyzer = PerformanceAnalyzer()
context_analyzer = ContextAnalyzer()

# Available prediction models
models = {
    "poisson": PoissonModel(),
    "dixon_coles": DixonColesModel(),
    "elo": EloBasedModel(),
    "bivariate_poisson": BivariatePoissonModel(),
    "ensemble": ModelEnsemble([
        PoissonModel(),
        DixonColesModel(),
        EloBasedModel()
    ])
}


@app.get("/")
async def root():
    """API health check and basic info."""
    return {
        "message": "Phutabol Soccer Prediction API",
        "version": "1.0.0",
        "available_models": list(models.keys()),
        "data_source": app_config.data_source_type.value,
        "live_data_available": app_config.has_live_data_access(),
        "supported_leagues": list(app_config.supported_leagues.keys()),
        "status": "healthy"
    }


@app.get("/config")
async def get_configuration():
    """Get current configuration and setup instructions."""
    return {
        "data_source_type": app_config.data_source_type.value,
        "live_data_available": app_config.has_live_data_access(),
        "supported_leagues": app_config.supported_leagues,
        "default_model": app_config.default_prediction_model,
        "features": {
            "context_analysis": app_config.enable_context_analysis,
            "form_analysis": app_config.enable_form_analysis
        },
        "setup_instructions": app_config.get_setup_instructions()
    }


@app.get("/teams/{league}", response_model=List[TeamResponse])
async def get_teams(league: str):
    """Get all teams in a specific league."""
    try:
        # Check if using live data collector
        if isinstance(data_collector, LiveDataCollector):
            current_data = await data_collector.get_current_season_data(league)
            teams = current_data["teams"]
        else:
            # Use traditional data source
            if hasattr(data_source, '__aenter__'):
                async with data_source as source:
                    teams = await source.get_teams(league)
            else:
                teams = await data_source.get_teams(league)

        return [TeamResponse.from_team(team) for team in teams]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams/{league}/standings", response_model=LeagueStandingsResponse)
async def get_league_standings(league: str):
    """Get league standings and rankings."""
    try:
        teams = await data_source.get_teams(league)
        standings_df = performance_analyzer.get_league_rankings(teams)

        standings = []
        for _, row in standings_df.iterrows():
            standings.append({
                "position": len(standings) + 1,
                "team_name": row["team_name"],
                "elo_rating": row["elo_rating"],
                "points": row["points"],
                "goal_difference": row["goal_difference"],
                "goals_per_game": round(row["goals_per_game"], 2),
                "win_rate": round(row["win_rate"], 3)
            })

        return LeagueStandingsResponse(
            league=league,
            standings=standings,
            last_updated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{league}")
async def get_matches(
    league: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(50, description="Maximum number of matches")
):
    """Get matches for a league within a date range."""
    try:
        # Default date range: last 30 days to next 30 days
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.now() - timedelta(days=30)

        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.now() + timedelta(days=30)

        matches = await data_source.get_matches(league, start_dt, end_dt)
        matches = matches[:limit]  # Apply limit

        return [MatchResponse.from_match(match) for match in matches]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fixtures/{league}")
async def get_upcoming_fixtures(
    league: str,
    days_ahead: int = Query(7, description="Number of days to look ahead")
):
    """Get upcoming fixtures for a league."""
    try:
        if isinstance(data_collector, LiveDataCollector):
            current_data = await data_collector.get_current_season_data(league)
            fixtures = current_data.get("upcoming_fixtures", [])

            # Filter by days ahead
            end_date = datetime.now() + timedelta(days=days_ahead)
            filtered_fixtures = [
                f for f in fixtures
                if f.scheduled_datetime and f.scheduled_datetime <= end_date
            ]

            return [MatchResponse.from_match(match) for match in filtered_fixtures]
        else:
            # For mock data, generate some upcoming fixtures
            end_date = datetime.now() + timedelta(days=days_ahead)
            start_date = datetime.now()

            if hasattr(data_source, '__aenter__'):
                async with data_source as source:
                    matches = await source.get_matches(league, start_date, end_date)
            else:
                matches = await data_source.get_matches(league, start_date, end_date)

            return [MatchResponse.from_match(match) for match in matches]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live/{league}")
async def get_live_data_summary(league: str):
    """Get a summary of live data for a league."""
    try:
        if isinstance(data_collector, LiveDataCollector):
            current_data = await data_collector.get_current_season_data(league)

            return {
                "league": league,
                "total_teams": len(current_data["teams"]),
                "recent_matches": len(current_data["recent_matches"]),
                "upcoming_fixtures": len(current_data.get("upcoming_fixtures", [])),
                "last_updated": current_data["last_updated"],
                "data_source": current_data["source"],
                "cache_status": "cached" if data_collector._is_cache_valid(f"{league}_current_season") else "fresh"
            }
        else:
            return {
                "league": league,
                "message": "Using mock data - set up live data sources for real-time information",
                "data_source": "mock"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    """Predict the outcome of a match."""
    try:
        # Get teams data with live data support
        if isinstance(data_collector, LiveDataCollector):
            current_data = await data_collector.get_current_season_data(request.league)
            teams = current_data["teams"]
            matches = current_data["recent_matches"]
            team_forms = current_data["team_forms"]
        else:
            # Use traditional data source
            if hasattr(data_source, '__aenter__'):
                async with data_source as source:
                    teams = await source.get_teams(request.league)
            else:
                teams = await data_source.get_teams(request.league)

            # Get recent matches for context
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            if hasattr(data_source, '__aenter__'):
                async with data_source as source:
                    matches = await source.get_matches(request.league, start_date, end_date)
            else:
                matches = await data_source.get_matches(request.league, start_date, end_date)

            # Get team forms
            team_forms = {}
            for team in teams:
                if hasattr(data_source, '__aenter__'):
                    async with data_source as source:
                        team_forms[team.id] = await source.get_team_form(team.id)
                else:
                    team_forms[team.id] = await data_source.get_team_form(team.id)

        # Find the teams
        home_team = None
        away_team = None
        for team in teams:
            if team.id == request.home_team_id:
                home_team = team
            elif team.id == request.away_team_id:
                away_team = team

        if not home_team or not away_team:
            raise HTTPException(
                status_code=404,
                detail="One or both teams not found in the specified league"
            )

        # Get team forms from the data we already collected
        home_form = team_forms.get(home_team.id)
        away_form = team_forms.get(away_team.id)

        # Calculate performance metrics
        home_metrics = performance_analyzer.calculate_team_metrics(
            home_team, matches, home_form, teams
        )
        away_metrics = performance_analyzer.calculate_team_metrics(
            away_team, matches, away_form, teams
        )

        # Apply context adjustments if match context is provided
        if request.match_context:
            match = Match(
                id="prediction_match",
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                league=request.league,
                season="current",
                context=request.match_context
            )

            home_adjustment, away_adjustment = context_analyzer.analyze_match_context(
                match, home_team, away_team
            )

            home_metrics = context_analyzer.adjust_performance_metrics(
                home_metrics, home_adjustment
            )
            away_metrics = context_analyzer.adjust_performance_metrics(
                away_metrics, away_adjustment
            )

        # Get prediction model
        model_name = request.model or "ensemble"
        if model_name not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_name}. Available: {list(models.keys())}"
            )

        model = models[model_name]

        # Make prediction
        prediction = model.predict_match(home_team, away_team, home_metrics, away_metrics)

        # Add team comparison
        team_comparison = performance_analyzer.compare_teams(home_metrics, away_metrics)

        return PredictionResponse.from_prediction_and_teams(
            prediction=prediction,
            home_team=home_team,
            away_team=away_team,
            team_comparison=team_comparison,
            model_name=model_name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{home_team_id}/vs/{away_team_id}")
async def quick_predict(
    home_team_id: str,
    away_team_id: str,
    league: str = Query(..., description="League name"),
    model: str = Query("ensemble", description="Prediction model to use")
):
    """Quick prediction endpoint for two teams."""
    request = PredictionRequest(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        league=league,
        model=model
    )
    return await predict_match(request)


@app.get("/models")
async def get_available_models():
    """Get information about available prediction models."""
    model_info = {
        "poisson": {
            "name": "Basic Poisson",
            "description": "Simple Poisson distribution model",
            "strengths": ["Fast", "Simple", "Good baseline"],
            "best_for": "Quick predictions with basic data"
        },
        "dixon_coles": {
            "name": "Dixon-Coles",
            "description": "Enhanced Poisson with low-score adjustments",
            "strengths": ["Better low-score modeling", "Accounts for draws"],
            "best_for": "More accurate than basic Poisson"
        },
        "elo": {
            "name": "Elo-based",
            "description": "Predictions based on Elo ratings",
            "strengths": ["Accounts for team strength evolution", "Good for long-term trends"],
            "best_for": "When team strength is the main factor"
        },
        "bivariate_poisson": {
            "name": "Bivariate Poisson",
            "description": "Accounts for correlation between team goals",
            "strengths": ["Models goal correlation", "More sophisticated"],
            "best_for": "High-quality detailed predictions"
        },
        "ensemble": {
            "name": "Ensemble Model",
            "description": "Combines multiple models for better accuracy",
            "strengths": ["Best overall accuracy", "Robust predictions"],
            "best_for": "Most important predictions"
        }
    }

    return {
        "available_models": model_info,
        "recommended": "ensemble",
        "fastest": "poisson"
    }


@app.get("/health")
async def health_check():
    """API health check."""
    try:
        # Test data source connectivity
        test_teams = await data_source.get_teams("Premier League")

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "data_source": "connected",
            "models_loaded": len(models),
            "test_teams_count": len(test_teams)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)