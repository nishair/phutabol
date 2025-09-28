# Phutabol ⚽

A comprehensive soccer match prediction system that uses advanced statistical models and machine learning techniques to predict match outcomes.

## Features

🔮 **Multiple Prediction Models**
- Basic Poisson Distribution Model
- Dixon-Coles Model (enhanced for low-scoring games)
- Elo Rating-based Model
- Bivariate Poisson Model (accounts for goal correlation)
- Model Ensemble for improved accuracy

⚡ **Advanced Analytics**
- Team performance analysis with 15+ metrics
- Match context analysis (home advantage, weather, injuries)
- Expected Goals (xG) calculations
- Form and momentum analysis
- Strength of schedule adjustments

📊 **Comprehensive Data Models**
- Team statistics and historical performance
- Match data with detailed context
- Player injuries and suspensions tracking
- Weather and venue conditions

🚀 **REST API**
- FastAPI-powered prediction endpoints
- Interactive documentation at `/docs`
- Bulk prediction capabilities
- League standings and team comparisons

📈 **Model Evaluation**
- Backtesting with historical data
- Cross-validation support
- Performance metrics (accuracy, log-loss, Brier score)
- Model comparison and benchmarking

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd phutabol

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from phutabol.data.collector import DataCollector, MockDataSource
from phutabol.prediction.models import PoissonModel

# Initialize components
data_source = MockDataSource()
collector = DataCollector(data_source)

# Get teams and make prediction
teams = await data_source.get_teams("Premier League")
home_team, away_team = teams[0], teams[1]

model = PoissonModel()
prediction = model.predict_match(home_team, away_team, home_metrics, away_metrics)

print(f"Home Win: {prediction.home_win_probability:.1%}")
print(f"Draw: {prediction.draw_probability:.1%}")
print(f"Away Win: {prediction.away_win_probability:.1%}")
```

### Run the API Server

```bash
# Start the API server
python -m phutabol.api.main

# Or run the example script
python example_usage.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Run Examples

```bash
python example_usage.py
```

## API Endpoints

### Core Prediction Endpoints

- `POST /predict` - Predict match outcome
- `GET /predict/{home_team_id}/vs/{away_team_id}` - Quick prediction

### Data Endpoints

- `GET /teams/{league}` - Get teams in a league
- `GET /teams/{league}/standings` - League standings
- `GET /matches/{league}` - Get matches

### Model Information

- `GET /models` - Available prediction models
- `GET /health` - API health check

## Prediction Models

### 1. Poisson Model
Basic model assuming goals follow Poisson distribution.
- **Best for**: Quick predictions, baseline comparisons
- **Strengths**: Fast, simple, interpretable

### 2. Dixon-Coles Model
Enhanced Poisson model with adjustments for low-scoring games.
- **Best for**: More accurate than basic Poisson
- **Strengths**: Better modeling of 0-0, 1-0, 1-1 scores

### 3. Elo-based Model
Uses team strength ratings that evolve over time.
- **Best for**: Long-term team strength assessment
- **Strengths**: Accounts for team strength evolution

### 4. Bivariate Poisson Model
Accounts for correlation between home and away team goals.
- **Best for**: High-quality detailed predictions
- **Strengths**: Models goal correlation in tempo

### 5. Ensemble Model
Combines multiple models for improved accuracy.
- **Best for**: Most important predictions
- **Strengths**: Best overall accuracy, robust

## Performance Metrics

The system tracks multiple performance metrics:

- **Result Accuracy**: Percentage of correct match outcomes (W/D/L)
- **Score Accuracy**: Percentage of exact score predictions
- **Log Loss**: Probabilistic accuracy measure
- **Brier Score**: Probability calibration metric
- **Goals MAE/RMSE**: Goal prediction accuracy
- **Market-specific**: Over/Under, Both Teams to Score accuracy

## Data Sources

Currently supports:
- **Mock Data Source**: For development and testing
- **Football API Source**: Template for real API integration (requires API key)

Easily extensible to support:
- Football-Data.org API
- RapidAPI Sports APIs
- Custom data feeds

## Architecture

```
phutabol/
├── models/          # Data models (Team, Match, Prediction)
├── data/           # Data collection and sources
├── analysis/       # Performance and context analysis
├── prediction/     # Prediction models
├── api/           # REST API endpoints
├── utils/         # Evaluation and utilities
└── tests/         # Test suite
```

## Key Factors Considered

### Team Strength & Form
- Elo Ratings
- Recent Performance (last 3-8 games)
- League Position
- Win/Loss records

### Offensive and Defensive Capabilities
- Expected Goals (xG)
- Goals Scored & Conceded
- Shots on Target
- Defensive Actions

### Match Context
- Home Advantage
- Injuries and Suspensions
- Tactics and Formations
- Match Importance

### External Factors
- Weather and Pitch Conditions
- Travel Distance & Rest Days
- Head-to-Head Records

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Future Enhancements

- 🔗 Real-time data integration
- 🤖 Machine learning models (Random Forest, Neural Networks)
- 📱 Web dashboard
- 💰 Betting market analysis
- 🎯 Player-level analysis
- 📊 Advanced visualizations

---

Made with ⚽ for soccer analytics enthusiasts