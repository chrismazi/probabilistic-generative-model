"""
Feature engineering module.

Provides:
- Rolling window team features
- Elo rating system
- Feature orchestration
"""

from src.features.rolling import (
    TeamRollingFeatures,
    MatchFeatures,
    RollingFeatureBuilder,
    get_feature_builder,
)
from src.features.elo import (
    EloRating,
    EloRatingSystem,
    get_elo_system,
)
from src.features.builder import (
    FeatureOrchestrator,
    get_orchestrator,
)

__all__ = [
    # Rolling features
    "TeamRollingFeatures",
    "MatchFeatures",
    "RollingFeatureBuilder",
    "get_feature_builder",
    # Elo
    "EloRating",
    "EloRatingSystem",
    "get_elo_system",
    # Orchestration
    "FeatureOrchestrator",
    "get_orchestrator",
]
