"""
Evaluation module.

Provides:
- Metrics (Brier score, calibration, log loss)
- Baseline comparisons
- Model comparison reports
- Walk-forward backtesting
"""

from src.evaluation.metrics import (
    BrierScore,
    CalibrationBin,
    CalibrationResult,
    log_loss,
    accuracy_at_threshold,
)
from src.evaluation.baselines import (
    BaselinePrediction,
    AlwaysSecondHalfBaseline,
    RandomBaseline,
    LeaguePriorBaseline,
    ClimatologyBaseline,
    get_all_baselines,
)
from src.evaluation.comparison import (
    ModelEvaluation,
    ComparisonResult,
    ModelEvaluator,
    compute_outcomes_from_scores,
    load_evaluation_data,
)
from src.evaluation.backtesting import (
    BacktestFold,
    BacktestResult,
    WalkForwardBacktester,
)

__all__ = [
    # Metrics
    "BrierScore",
    "CalibrationBin",
    "CalibrationResult",
    "log_loss",
    "accuracy_at_threshold",
    # Baselines
    "BaselinePrediction",
    "AlwaysSecondHalfBaseline",
    "RandomBaseline",
    "LeaguePriorBaseline",
    "ClimatologyBaseline",
    "get_all_baselines",
    # Comparison
    "ModelEvaluation",
    "ComparisonResult",
    "ModelEvaluator",
    "compute_outcomes_from_scores",
    "load_evaluation_data",
    # Backtesting
    "BacktestFold",
    "BacktestResult",
    "WalkForwardBacktester",
]
