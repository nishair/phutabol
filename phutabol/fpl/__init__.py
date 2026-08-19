"""Fantasy Premier League team selection."""

from .client import FPLClient
from .projections import ProjectedPlayer, build_projections
from .optimizer import Squad, optimize_squad

__all__ = [
    "FPLClient",
    "ProjectedPlayer",
    "build_projections",
    "Squad",
    "optimize_squad",
]
