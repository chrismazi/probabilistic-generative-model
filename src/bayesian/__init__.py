"""
Bayesian models module.

Provides:
- Hierarchical Poisson/NegBin models for half goals
- Prior specifications
- Prediction generation
"""

from src.bayesian.priors import (
    HalfGoalPriors,
    ModelConfig,
    DEFAULT_POISSON_CONFIG,
    DEFAULT_NEGBIN_CONFIG,
)
from src.bayesian.model import (
    MatchData,
    TrainingData,
    HalfGoalModel,
)
from src.bayesian.prediction import (
    MatchPrediction,
    Predictor,
)

__all__ = [
    # Priors
    "HalfGoalPriors",
    "ModelConfig",
    "DEFAULT_POISSON_CONFIG",
    "DEFAULT_NEGBIN_CONFIG",
    # Model
    "MatchData",
    "TrainingData",
    "HalfGoalModel",
    # Prediction
    "MatchPrediction",
    "Predictor",
]
