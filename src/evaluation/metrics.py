"""
Evaluation metrics for model performance.

Provides:
- Brier score
- Log loss
- Calibration metrics
- Baseline comparisons
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class BrierScore:
    """Brier score breakdown."""
    
    score: float  # Lower is better, 0 = perfect
    reliability: float  # Calibration component
    resolution: float  # Discrimination component
    uncertainty: float  # Irreducible uncertainty
    
    @classmethod
    def compute(cls, predictions: np.ndarray, outcomes: np.ndarray) -> "BrierScore":
        """
        Compute Brier score with decomposition.
        
        Args:
            predictions: Predicted probabilities P(event)
            outcomes: Actual outcomes (0 or 1)
            
        Returns:
            BrierScore with components
        """
        n = len(predictions)
        if n == 0:
            return cls(score=0.0, reliability=0.0, resolution=0.0, uncertainty=0.0)
        
        # Brier score
        score = float(np.mean((predictions - outcomes) ** 2))
        
        # Base rate (climatology)
        base_rate = float(np.mean(outcomes))
        
        # Uncertainty (irreducible)
        uncertainty = base_rate * (1 - base_rate)
        
        # For decomposition, bin predictions
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        reliability = 0.0
        resolution = 0.0
        
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if i == n_bins - 1:  # Include 1.0 in last bin
                mask = mask | (predictions == 1.0)
            
            n_k = np.sum(mask)
            if n_k > 0:
                p_k = np.mean(predictions[mask])
                o_k = np.mean(outcomes[mask])
                
                reliability += n_k * (p_k - o_k) ** 2
                resolution += n_k * (o_k - base_rate) ** 2
        
        reliability = reliability / n
        resolution = resolution / n
        
        return cls(
            score=score,
            reliability=reliability,
            resolution=resolution,
            uncertainty=uncertainty,
        )
    
    @property
    def skill_score(self) -> float:
        """
        Brier Skill Score vs climatology.
        
        BSS = 1 - BS_model / BS_climatology
        
        Where BS_climatology = base_rate * (1 - base_rate) = uncertainty.
        
        Interpretation:
            1 = perfect
            0 = no better than climatology
            <0 = worse than climatology
        """
        if self.uncertainty == 0:
            return 0.0
        return 1 - self.score / self.uncertainty
    
    def skill_score_vs_reference(self, reference_brier: float) -> float:
        """
        Brier Skill Score vs arbitrary reference.
        
        BSS = 1 - BS_model / BS_reference
        
        Args:
            reference_brier: Reference model's Brier score
            
        Returns:
            Skill score vs that reference
        """
        if reference_brier == 0:
            return 0.0
        return 1 - self.score / reference_brier


def log_loss(predictions: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute log loss (cross-entropy).
    
    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes (0 or 1)
        eps: Small value to avoid log(0)
        
    Returns:
        Log loss (lower is better)
    """
    p = np.clip(predictions, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def accuracy_at_threshold(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute accuracy at a decision threshold.
    
    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        threshold: Decision threshold
        
    Returns:
        Accuracy (0-1)
    """
    predicted_class = (predictions >= threshold).astype(int)
    return float(np.mean(predicted_class == outcomes))


@dataclass
class CalibrationBin:
    """Single calibration bin."""
    
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_observed: float
    count: int
    
    @property
    def calibration_error(self) -> float:
        """Absolute calibration error for this bin."""
        return abs(self.mean_predicted - self.mean_observed)


@dataclass
class CalibrationResult:
    """Calibration analysis result."""
    
    bins: List[CalibrationBin]
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    
    @classmethod
    def compute(
        cls,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
        min_bin_count: int = 5,
    ) -> "CalibrationResult":
        """
        Compute calibration metrics.
        
        Args:
            predictions: Predicted probabilities
            outcomes: Actual outcomes
            n_bins: Number of calibration bins
            min_bin_count: Minimum samples per bin for reliable estimate
            
        Returns:
            CalibrationResult with bins and summary metrics
            
        Note:
            Bins with fewer than min_bin_count samples are still included
            but flagged in the output. Use caution with small samples.
        """
        n = len(predictions)
        if n == 0:
            return cls(bins=[], expected_calibration_error=0.0, maximum_calibration_error=0.0)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        
        weighted_errors = []
        max_error = 0.0
        
        for i in range(n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            
            mask = (predictions >= lower) & (predictions < upper)
            if i == n_bins - 1:
                mask = mask | (predictions == 1.0)
            
            count = int(np.sum(mask))
            
            if count > 0:
                mean_pred = float(np.mean(predictions[mask]))
                mean_obs = float(np.mean(outcomes[mask]))
                error = abs(mean_pred - mean_obs)
                
                bins.append(CalibrationBin(
                    bin_lower=lower,
                    bin_upper=upper,
                    mean_predicted=mean_pred,
                    mean_observed=mean_obs,
                    count=count,
                ))
                
                weighted_errors.append(count * error)
                max_error = max(max_error, error)
        
        ece = sum(weighted_errors) / n if n > 0 else 0.0
        
        return cls(
            bins=bins,
            expected_calibration_error=ece,
            maximum_calibration_error=max_error,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "ece": self.expected_calibration_error,
            "mce": self.maximum_calibration_error,
            "bins": [
                {
                    "range": f"{b.bin_lower:.1f}-{b.bin_upper:.1f}",
                    "predicted": b.mean_predicted,
                    "observed": b.mean_observed,
                    "count": b.count,
                    "error": b.calibration_error,
                }
                for b in self.bins
            ],
        }
