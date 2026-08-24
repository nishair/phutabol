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

#### Option 1: Direct Python Module (Recommended)
```bash
cd phutabol
python -m phutabol.api.main
```

#### Option 2: Using the Example Script
```bash
python example_usage.py
# Then choose option 2 when prompted
```

#### Option 3: Using Uvicorn Directly
```bash
uvicorn phutabol.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### After Starting the API:

- **API Base URL**: `http://localhost:8000`
- **Interactive Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc`

#### Quick API Test:
```bash
# Health check
curl http://localhost:8000/health

# Get teams
curl http://localhost:8000/teams/Premier%20League

# Quick prediction
curl "http://localhost:8000/predict/team_1/vs/team_2?league=Premier%20League"

# Available models
curl http://localhost:8000/models
```

#### What You'll See After Starting:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### Next Steps - What You Can Do:

**1. Open Interactive Documentation (Recommended First Step)**
```
http://localhost:8000/docs
```
This gives you a **Swagger UI** where you can:
- See all available endpoints
- Test API calls directly in the browser
- View request/response schemas
- Try predictions without writing code

**2. Test Basic Endpoints**
```
# Health Check
http://localhost:8000/health

# Get Available Models
http://localhost:8000/models

# Get Teams
http://localhost:8000/teams/Premier%20League
```

**3. Make Your First Prediction**
```
http://localhost:8000/predict/team_1/vs/team_2?league=Premier%20League&model=ensemble
```

**4. Use Interactive Docs for POST Requests**
1. Go to `http://localhost:8000/docs`
2. Click on **"POST /predict"**
3. Click **"Try it out"**
4. Use this sample request:
```json
{
  "home_team_id": "team_1",
  "away_team_id": "team_2",
  "league": "Premier League",
  "model": "ensemble"
}
```
5. Click **"Execute"**

**5. Stop the Server**
Press **Ctrl+C** in the terminal when done.

> 💡 **Note**: The API comes with mock data, so you can immediately test all features!

#### Troubleshooting:
If you get import errors, ensure you're in the project root directory:
```bash
cd phutabol
pip install -r requirements.txt
python -m phutabol.api.main
```

### Run Examples

```bash
# Basic examples with mock data
python example_usage.py

# Live data examples (requires API key)
python live_example.py
```

## 🚀 **NEW: Live Data Integration**

### Setup Real-Time Data Sources

#### Option 1: Football-Data.org (Free - Recommended)
1. **Get API Key**: Visit [Football-Data.org](https://www.football-data.org/client/register)
2. **Set Environment Variables**:
   ```bash
   export FOOTBALL_DATA_API_KEY="your_api_key_here"
   export PHUTABOL_DATA_SOURCE="football_data_org"
   ```
3. **Restart API**: `python -m phutabol.api.main`

#### Option 2: RapidAPI Football (Premium Features)
1. **Get API Key**: Visit [RapidAPI Football](https://rapidapi.com/api-sports/api/api-football)
2. **Set Environment Variables**:
   ```bash
   export RAPIDAPI_KEY="your_api_key_here"
   export PHUTABOL_DATA_SOURCE="rapidapi"
   ```

#### Live Data Features:
- ✅ **Current 2024-25 season statistics**
- ✅ **Real team standings and Elo ratings**
- ✅ **Actual match results and fixtures**
- ✅ **Team form based on recent games**
- ✅ **Live injury and suspension data** (RapidAPI)
- ✅ **Weather and venue information**
- ✅ **15-minute intelligent caching**

#### Check Configuration:
```bash
# View current setup
curl http://localhost:8000/config

# Test live data access
curl http://localhost:8000/live/Premier%20League
```

## API Endpoints

### Core Prediction Endpoints

- `POST /predict` - Predict match outcome
- `GET /predict/{home_team_id}/vs/{away_team_id}` - Quick prediction

### Data Endpoints

- `GET /teams/{league}` - Get teams in a league
- `GET /teams/{league}/standings` - League standings
- `GET /matches/{league}` - Get matches
- `GET /fixtures/{league}` - Get upcoming fixtures
- `GET /live/{league}` - Live data summary

### Configuration & Status

- `GET /config` - Current configuration and setup instructions
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
## Fantasy Premier League Toolkit

The `phutabol.fpl` package picks and manages an FPL team using the
official (keyless) FPL API, with an MILP squad optimizer and projections
backtested over eight seasons:

```bash
python fpl_pick_team.py                  # optimal GW1 squad from live data
python fpl_manage.py <TEAM_ID>           # weekly deadline advisor for your team
python fpl_backtest.py [--season 2025-26]  # replay a season (static + managed)
python fpl_tune.py                       # multi-season config sweeps
```

`fpl_manage.py` reads your real squad from the public API (your team ID
is in the site URL), then recommends transfers, XI, captain, and chip
for the next deadline. When it recommends a wildcard or free hit it
prints the rebuilt squad it has in mind (OUT/IN plus the new XI). The
model handles the stats — you hold the veto on late team news.

### Automated alerts

`fpl_watch.py` runs the advisor for you on a schedule and alerts on
squad news:

```bash
./install_fpl_watch.sh <TEAM_ID>            # launchd agent, every 30 min
sudo ./install_fpl_watch.sh --daemon <TEAM_ID>  # from boot, no login (e.g. a Mac mini)
```

Each pass produces the full deadline plan 24 hours ahead (once per
gameweek, refreshed if news breaks after), and notifies on any change
to your players' injury flags, chance of playing, news, or price.
Alerts go to macOS notifications and, if configured via
`python fpl_watch.py --setup-telegram <BOT_TOKEN>` (or an `ntfy_topic`
in `~/.phutabol/notify.json`), to your phone — Telegram receives the
complete plan text. State and plans live in `~/.phutabol/`.
