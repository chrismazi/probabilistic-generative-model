"""
Model comparison and evaluation report generation.

Provides:
- Side-by-side model comparison
- Per-league breakdown
- Statistical significance tests
- Evaluation reports
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
from sqlalchemy import text

from src.db import get_session
from src.evaluation.metrics import (
    BrierScore,
    CalibrationResult,
    log_loss,
    accuracy_at_threshold,
)
from src.evaluation.baselines import (
    AlwaysSecondHalfBaseline,
    RandomBaseline,
    LeaguePriorBaseline,
    ClimatologyBaseline,
    get_all_baselines,
)
from src.utils import get_logger

logger = get_logger("evaluation.comparison")


@dataclass
class ModelEvaluation:
    """Evaluation metrics for a single model."""
    
    model_name: str
    n_matches: int
    
    # Core metrics
    brier_score: BrierScore
    log_loss: float
    accuracy: float
    
    # Calibration
    calibration: CalibrationResult
    
    # Optional: per-league breakdown
    per_league: Dict[str, "ModelEvaluation"] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "n_matches": self.n_matches,
            "brier_score": self.brier_score.score,
            "brier_skill_score": self.brier_score.skill_score,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "calibration": self.calibration.to_dict(),
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple models."""
    
    evaluations: Dict[str, ModelEvaluation]
    best_model: str
    improvement_over_baseline: Dict[str, float]  # % improvement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_model": self.best_model,
            "improvement_over_baseline": self.improvement_over_baseline,
            "models": {
                name: eval.to_dict() 
                for name, eval in self.evaluations.items()
            },
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 60, "MODEL COMPARISON REPORT", "=" * 60, ""]
        
        # Sort by Brier score (lower is better)
        sorted_models = sorted(
            self.evaluations.items(),
            key=lambda x: x[1].brier_score.score
        )
        
        lines.append(f"{'Model':<20} {'Brier':>10} {'Log Loss':>10} {'Acc':>8} {'ECE':>8}")
        lines.append("-" * 60)
        
        for name, eval in sorted_models:
            lines.append(
                f"{name:<20} "
                f"{eval.brier_score.score:>10.4f} "
                f"{eval.log_loss:>10.4f} "
                f"{eval.accuracy:>8.1%} "
                f"{eval.calibration.expected_calibration_error:>8.4f}"
            )
        
        lines.append("")
        lines.append(f"Best model: {self.best_model}")
        
        if self.improvement_over_baseline:
            lines.append("")
            lines.append("Improvement over baselines:")
            for baseline, improvement in self.improvement_over_baseline.items():
                lines.append(f"  vs {baseline}: {improvement:+.1%}")
        
        return "\n".join(lines)


class ModelEvaluator:
    """
    Evaluates model predictions against outcomes.
    
    Compares model against baselines and computes
    comprehensive metrics.
    """
    
    def __init__(self):
        self.baselines = get_all_baselines()
    
    def evaluate_model(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        model_name: str = "model",
    ) -> ModelEvaluation:
        """
        Evaluate a single model.
        
        Args:
            predictions: Predicted P(G2 > G1)
            outcomes: Actual outcomes (1 if G2 > G1, else 0)
            model_name: Name for this model
            
        Returns:
            ModelEvaluation with full metrics
        """
        n = len(predictions)
        
        brier = BrierScore.compute(predictions, outcomes)
        ll = log_loss(predictions, outcomes)
        acc = accuracy_at_threshold(predictions, outcomes, 0.5)
        calibration = CalibrationResult.compute(predictions, outcomes)
        
        return ModelEvaluation(
            model_name=model_name,
            n_matches=n,
            brier_score=brier,
            log_loss=ll,
            accuracy=acc,
            calibration=calibration,
        )
    
    def evaluate_with_baselines(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        league_ids: Optional[np.ndarray] = None,
        model_name: str = "model",
    ) -> ComparisonResult:
        """
        Evaluate model and compare against all baselines.
        
        Args:
            predictions: Model predictions
            outcomes: Actual outcomes
            league_ids: Optional league IDs for league prior baseline
            model_name: Name for the model
            
        Returns:
            ComparisonResult with all metrics
        """
        evaluations = {}
        
        # Evaluate main model
        evaluations[model_name] = self.evaluate_model(
            predictions, outcomes, model_name
        )
        
        # Evaluate baselines
        n = len(predictions)
        
        # Always 2H
        always_2h_preds = np.ones(n)
        evaluations["always_2h"] = self.evaluate_model(
            always_2h_preds, outcomes, "always_2h"
        )
        
        # Random 50/50
        random_preds = np.full(n, 0.5)
        evaluations["random"] = self.evaluate_model(
            random_preds, outcomes, "random"
        )
        
        # Climatology (use outcome mean as proxy)
        climatology_rate = outcomes.mean()
        climatology_preds = np.full(n, climatology_rate)
        evaluations["climatology"] = self.evaluate_model(
            climatology_preds, outcomes, "climatology"
        )
        
        # League prior (if league IDs provided)
        if league_ids is not None:
            league_prior = LeaguePriorBaseline()
            league_preds = league_prior.predict_batch(league_ids)
            evaluations["league_prior"] = self.evaluate_model(
                league_preds, outcomes, "league_prior"
            )
        
        # Find best model
        best_model = min(
            evaluations.keys(),
            key=lambda k: evaluations[k].brier_score.score
        )
        
        # Calculate improvements
        model_brier = evaluations[model_name].brier_score.score
        improvements = {}
        
        for baseline_name in ["always_2h", "random", "climatology"]:
            if baseline_name in evaluations:
                baseline_brier = evaluations[baseline_name].brier_score.score
                if baseline_brier > 0:
                    improvement = (baseline_brier - model_brier) / baseline_brier
                    improvements[baseline_name] = improvement
        
        return ComparisonResult(
            evaluations=evaluations,
            best_model=best_model,
            improvement_over_baseline=improvements,
        )
    
    def evaluate_per_league(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        league_ids: np.ndarray,
        model_name: str = "model",
    ) -> Dict[int, ModelEvaluation]:
        """
        Evaluate model per league.
        
        Args:
            predictions: Model predictions
            outcomes: Actual outcomes
            league_ids: League ID for each match
            model_name: Model name
            
        Returns:
            Dictionary of league_id -> ModelEvaluation
        """
        results = {}
        
        for league_id in np.unique(league_ids):
            mask = league_ids == league_id
            n = np.sum(mask)
            
            if n >= 10:  # Minimum sample size
                results[int(league_id)] = self.evaluate_model(
                    predictions[mask],
                    outcomes[mask],
                    f"{model_name}_league_{league_id}",
                )
        
        return results


def compute_outcomes_from_scores(
    ht_home: np.ndarray,
    ht_away: np.ndarray,
    ft_home: np.ndarray,
    ft_away: np.ndarray,
) -> np.ndarray:
    """
    Compute G2 > G1 outcomes from scores.
    
    Args:
        ht_home: Half-time home goals
        ht_away: Half-time away goals
        ft_home: Full-time home goals
        ft_away: Full-time away goals
        
    Returns:
        Array of outcomes (1 if G2 > G1, else 0)
    """
    g1 = ht_home + ht_away
    g2 = (ft_home - ht_home) + (ft_away - ht_away)
    return (g2 > g1).astype(int)


def load_evaluation_data(
    league_id: Optional[int] = None,
    season: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Load match data for evaluation.
    
    Args:
        league_id: Optional league filter
        season: Optional season filter
        
    Returns:
        Dictionary with match_ids, scores, outcomes, league_ids
    """
    query = """
        SELECT 
            m.id as match_id,
            m.league_id,
            s.ht_home,
            s.ht_away,
            s.ft_home,
            s.ft_away
        FROM matches m
        JOIN scores s ON m.id = s.match_id
        WHERE m.status = 'FINISHED'
          AND s.ht_home IS NOT NULL
    """
    
    params: dict = {}
    
    if league_id:
        query += " AND m.league_id = :league_id"
        params["league_id"] = league_id
    
    if season:
        query += " AND m.season = :season"
        params["season"] = season
    
    query += " ORDER BY m.kickoff_utc"
    
    with get_session() as session:
        result = session.execute(text(query), params).fetchall()
    
    if not result:
        return {
            "match_ids": np.array([]),
            "league_ids": np.array([]),
            "outcomes": np.array([]),
        }
    
    data = np.array(result)
    
    match_ids = data[:, 0].astype(int)
    league_ids = data[:, 1].astype(int)
    ht_home = data[:, 2].astype(int)
    ht_away = data[:, 3].astype(int)
    ft_home = data[:, 4].astype(int)
    ft_away = data[:, 5].astype(int)
    
    outcomes = compute_outcomes_from_scores(ht_home, ht_away, ft_home, ft_away)
    
    return {
        "match_ids": match_ids,
        "league_ids": league_ids,
        "ht_home": ht_home,
        "ht_away": ht_away,
        "ft_home": ft_home,
        "ft_away": ft_away,
        "outcomes": outcomes,
    }
