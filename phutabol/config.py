import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class DataSourceType(Enum):
    MOCK = "mock"
    FOOTBALL_DATA_ORG = "football_data_org"
    RAPIDAPI = "rapidapi"
    MIXED = "mixed"


@dataclass
class APIConfig:
    """Configuration for external APIs."""

    # Football-Data.org API (Free tier: 10 requests/minute)
    football_data_api_key: Optional[str] = None
    football_data_base_url: str = "https://api.football-data.org/v4"

    # RapidAPI Football (Premium features)
    rapidapi_key: Optional[str] = None
    rapidapi_base_url: str = "https://api-football-v1.p.rapidapi.com/v3"

    # API rate limiting
    requests_per_minute: int = 10
    cache_duration_minutes: int = 15

    def __post_init__(self):
        # Try to load from environment variables
        if not self.football_data_api_key:
            self.football_data_api_key = os.getenv('FOOTBALL_DATA_API_KEY')

        if not self.rapidapi_key:
            self.rapidapi_key = os.getenv('RAPIDAPI_KEY')


@dataclass
class AppConfig:
    """Main application configuration."""

    # Data source configuration
    data_source_type: DataSourceType = DataSourceType.MOCK
    api_config: APIConfig = None

    # Model configuration
    default_prediction_model: str = "ensemble"
    enable_context_analysis: bool = True
    enable_form_analysis: bool = True

    # Performance settings
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 30

    # Supported leagues
    supported_leagues: Dict[str, Any] = None

    def __post_init__(self):
        if self.api_config is None:
            self.api_config = APIConfig()

        if self.supported_leagues is None:
            self.supported_leagues = {
                "Premier League": {
                    "country": "England",
                    "football_data_id": 2021,
                    "rapidapi_id": 39,
                    "current_season": "2024-25"
                },
                "La Liga": {
                    "country": "Spain",
                    "football_data_id": 2014,
                    "rapidapi_id": 140,
                    "current_season": "2024-25"
                },
                "Bundesliga": {
                    "country": "Germany",
                    "football_data_id": 2002,
                    "rapidapi_id": 78,
                    "current_season": "2024-25"
                },
                "Serie A": {
                    "country": "Italy",
                    "football_data_id": 2019,
                    "rapidapi_id": 135,
                    "current_season": "2024-25"
                },
                "Ligue 1": {
                    "country": "France",
                    "football_data_id": 2015,
                    "rapidapi_id": 61,
                    "current_season": "2024-25"
                },
                "Champions League": {
                    "country": "Europe",
                    "football_data_id": 2001,
                    "rapidapi_id": 2,
                    "current_season": "2024-25"
                }
            }

    @classmethod
    def load_from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""

        # Determine data source type
        data_source = os.getenv('PHUTABOL_DATA_SOURCE', 'mock').lower()
        data_source_type = DataSourceType.MOCK

        if data_source == 'football_data_org':
            data_source_type = DataSourceType.FOOTBALL_DATA_ORG
        elif data_source == 'rapidapi':
            data_source_type = DataSourceType.RAPIDAPI
        elif data_source == 'mixed':
            data_source_type = DataSourceType.MIXED

        config = cls(
            data_source_type=data_source_type,
            default_prediction_model=os.getenv('PHUTABOL_DEFAULT_MODEL', 'ensemble'),
            enable_context_analysis=os.getenv('PHUTABOL_ENABLE_CONTEXT', 'true').lower() == 'true',
            enable_form_analysis=os.getenv('PHUTABOL_ENABLE_FORM', 'true').lower() == 'true',
            max_concurrent_requests=int(os.getenv('PHUTABOL_MAX_CONCURRENT', '5')),
            request_timeout_seconds=int(os.getenv('PHUTABOL_TIMEOUT', '30'))
        )

        return config

    def has_live_data_access(self) -> bool:
        """Check if we have access to live data sources."""
        if self.data_source_type == DataSourceType.MOCK:
            return False

        if self.data_source_type == DataSourceType.FOOTBALL_DATA_ORG:
            return bool(self.api_config.football_data_api_key)

        if self.data_source_type == DataSourceType.RAPIDAPI:
            return bool(self.api_config.rapidapi_key)

        if self.data_source_type == DataSourceType.MIXED:
            return bool(self.api_config.football_data_api_key or self.api_config.rapidapi_key)

        return False

    def get_setup_instructions(self) -> str:
        """Get instructions for setting up live data access."""

        instructions = """
🔧 SETUP INSTRUCTIONS FOR LIVE DATA

To use real-time soccer data, you need to configure API keys:

1. Football-Data.org (Free tier - 10 requests/minute):
   • Visit: https://www.football-data.org/client/register
   • Sign up for a free account
   • Get your API key
   • Set environment variable: export FOOTBALL_DATA_API_KEY="your_key_here"

2. RapidAPI Football (Premium features):
   • Visit: https://rapidapi.com/api-sports/api/api-football
   • Subscribe to a plan
   • Get your API key
   • Set environment variable: export RAPIDAPI_KEY="your_key_here"

3. Configure data source:
   • For Football-Data.org: export PHUTABOL_DATA_SOURCE="football_data_org"
   • For RapidAPI: export PHUTABOL_DATA_SOURCE="rapidapi"
   • For both: export PHUTABOL_DATA_SOURCE="mixed"

4. Optional configuration:
   • export PHUTABOL_DEFAULT_MODEL="ensemble"
   • export PHUTABOL_ENABLE_CONTEXT="true"
   • export PHUTABOL_MAX_CONCURRENT="5"

5. Restart the API server after setting environment variables.

📊 CURRENT STATUS:
"""

        if self.has_live_data_access():
            instructions += "✅ Live data access configured!\n"
            if self.api_config.football_data_api_key:
                instructions += "✅ Football-Data.org API available\n"
            if self.api_config.rapidapi_key:
                instructions += "✅ RapidAPI Football available\n"
        else:
            instructions += "❌ No live data access configured - using mock data\n"

        instructions += f"🎯 Current data source: {self.data_source_type.value}\n"
        instructions += f"🏆 Supported leagues: {', '.join(self.supported_leagues.keys())}\n"

        return instructions


# Global configuration instance
config = AppConfig.load_from_env()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration from environment variables."""
    global config
    config = AppConfig.load_from_env()