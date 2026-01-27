"""
Tests for evaluation metrics and comparison.

Tests for:
- Brier score computation
- Calibration metrics
- Baseline models
- Model comparison
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    BrierScore,
    CalibrationResult,
    log_loss,
    accuracy_at_threshold,
)
from src.evaluation.baselines import (
    AlwaysSecondHalfBaseline,
    RandomBaseline,
)
from src.evaluation.comparison import (
    ModelEvaluation,
    ModelEvaluator,
    compute_outcomes_from_scores,
)


class TestBrierScore:
    """Test Brier score computation."""
    
    def test_perfect_predictions(self):
        """Perfect predictions give Brier score of 0."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        
        brier = BrierScore.compute(predictions, outcomes)
        
        assert brier.score == pytest.approx(0.0, abs=1e-10)
    
    def test_worst_predictions(self):
        """Completely wrong predictions give Brier score of 1."""
        predictions = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1, 0, 1, 0])
        
        brier = BrierScore.compute(predictions, outcomes)
        
        assert brier.score == pytest.approx(1.0, abs=1e-10)
    
    def test_random_predictions(self):
        """0.5 predictions give Brier score of 0.25."""
        predictions = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        
        brier = BrierScore.compute(predictions, outcomes)
        
        assert brier.score == pytest.approx(0.25, abs=1e-10)
    
    def test_skill_score_perfect(self):
        """Perfect model has skill score of 1."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        
        brier = BrierScore.compute(predictions, outcomes)
        
        assert brier.skill_score == pytest.approx(1.0, abs=1e-10)
    
    def test_skill_score_climatology(self):
        """Climatology predictions have skill score near 0."""
        outcomes = np.array([1, 0, 1, 0])
        climatology = outcomes.mean()
        predictions = np.full(4, climatology)
        
        brier = BrierScore.compute(predictions, outcomes)
        
        # Skill score should be 0 for climatology
        assert brier.skill_score == pytest.approx(0.0, abs=0.01)
    
    def test_empty_arrays(self):
        """Handle empty arrays gracefully."""
        brier = BrierScore.compute(np.array([]), np.array([]))
        
        assert brier.score == 0.0


class TestLogLoss:
    """Test log loss computation."""
    
    def test_perfect_predictions(self):
        """Near-perfect predictions have low log loss."""
        predictions = np.array([0.999, 0.001, 0.999])
        outcomes = np.array([1, 0, 1])
        
        ll = log_loss(predictions, outcomes)
        
        assert ll < 0.01
    
    def test_random_predictions(self):
        """0.5 predictions have log loss of ~0.693."""
        predictions = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        
        ll = log_loss(predictions, outcomes)
        
        assert ll == pytest.approx(0.693, abs=0.01)
    
    def test_clipping(self):
        """Extreme predictions are clipped to avoid log(0)."""
        predictions = np.array([1.0, 0.0])
        outcomes = np.array([1, 0])
        
        ll = log_loss(predictions, outcomes)
        
        assert np.isfinite(ll)


class TestAccuracy:
    """Test accuracy computation."""
    
    def test_perfect_accuracy(self):
        """Perfect discrimination gives accuracy of 1."""
        predictions = np.array([0.9, 0.1, 0.8, 0.2])
        outcomes = np.array([1, 0, 1, 0])
        
        acc = accuracy_at_threshold(predictions, outcomes, 0.5)
        
        assert acc == 1.0
    
    def test_random_accuracy(self):
        """Random predictions give ~50% accuracy."""
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, 100)
        outcomes = (predictions > 0.5).astype(int)  # Self-consistent
        
        acc = accuracy_at_threshold(predictions, outcomes, 0.5)
        
        assert acc == 1.0  # Self-consistent should be perfect


class TestCalibration:
    """Test calibration computation."""
    
    def test_perfectly_calibrated(self):
        """Perfectly calibrated model has ECE of 0."""
        # Predictions match observed frequencies
        predictions = np.array([0.1] * 10 + [0.9] * 10)
        outcomes = np.array([0] * 9 + [1] * 1 + [1] * 9 + [0] * 1)
        
        calibration = CalibrationResult.compute(predictions, outcomes)
        
        # Should be close to 0 (not exactly due to binning)
        assert calibration.expected_calibration_error < 0.1
    
    def test_overconfident(self):
        """Overconfident model has high ECE."""
        # Always predict 0.9 but only 50% actually happen
        predictions = np.array([0.9] * 100)
        outcomes = np.array([1, 0] * 50)
        
        calibration = CalibrationResult.compute(predictions, outcomes)
        
        # Large calibration error expected
        assert calibration.expected_calibration_error > 0.3
    
    def test_bins_non_empty(self):
        """Calibration bins are computed correctly."""
        predictions = np.random.uniform(0, 1, 100)
        outcomes = (np.random.random(100) > 0.5).astype(int)
        
        calibration = CalibrationResult.compute(predictions, outcomes, n_bins=10)
        
        total_count = sum(b.count for b in calibration.bins)
        assert total_count == 100


class TestBaselines:
    """Test baseline models."""
    
    def test_always_2h_baseline(self):
        """Always 2H baseline predicts 1.0."""
        baseline = AlwaysSecondHalfBaseline()
        
        pred = baseline.predict(123)
        
        assert pred.p_2h_gt_1h == 1.0
        assert pred.is_constant is True
    
    def test_always_2h_batch(self):
        """Always 2H batch prediction."""
        baseline = AlwaysSecondHalfBaseline()
        
        preds = baseline.predict_batch(10)
        
        assert len(preds) == 10
        assert all(p == 1.0 for p in preds)
    
    def test_random_baseline(self):
        """Random baseline predicts 0.5."""
        baseline = RandomBaseline()
        
        pred = baseline.predict(123)
        
        assert pred.p_2h_gt_1h == 0.5


class TestComputeOutcomes:
    """Test outcome computation from scores."""
    
    def test_second_half_more_goals(self):
        """Correctly identify when 2H has more goals."""
        ht_home = np.array([0, 1, 0])
        ht_away = np.array([0, 0, 1])
        ft_home = np.array([2, 2, 1])
        ft_away = np.array([1, 0, 1])
        
        outcomes = compute_outcomes_from_scores(ht_home, ht_away, ft_home, ft_away)
        
        # Match 0: HT 0-0, FT 2-1 → G1=0, G2=3 → 1
        # Match 1: HT 1-0, FT 2-0 → G1=1, G2=1 → 0
        # Match 2: HT 0-1, FT 1-1 → G1=1, G2=1 → 0
        assert outcomes[0] == 1
        assert outcomes[1] == 0
        assert outcomes[2] == 0


class TestModelEvaluator:
    """Test model evaluator."""
    
    def test_evaluate_model(self):
        """Test basic model evaluation."""
        evaluator = ModelEvaluator()
        
        predictions = np.array([0.7, 0.3, 0.6, 0.4])
        outcomes = np.array([1, 0, 1, 0])
        
        eval_result = evaluator.evaluate_model(predictions, outcomes, "test_model")
        
        assert eval_result.model_name == "test_model"
        assert eval_result.n_matches == 4
        assert 0 <= eval_result.brier_score.score <= 1
        assert 0 <= eval_result.accuracy <= 1
    
    def test_evaluate_with_baselines(self):
        """Test comparison with baselines."""
        evaluator = ModelEvaluator()
        
        # Model predictions better than random
        predictions = np.array([0.8, 0.2, 0.7, 0.3] * 10)
        outcomes = np.array([1, 0, 1, 0] * 10)
        
        result = evaluator.evaluate_with_baselines(predictions, outcomes, model_name="good_model")
        
        assert "good_model" in result.evaluations
        assert "always_2h" in result.evaluations
        assert "random" in result.evaluations
        
        # Good model should beat random
        assert result.evaluations["good_model"].brier_score.score < \
               result.evaluations["random"].brier_score.score
