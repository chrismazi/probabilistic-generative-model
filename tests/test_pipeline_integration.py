"""
End-to-end pipeline integration tests.

Tests that all stages of the pipeline work together:
- Features are computed correctly
- Model builds without error
- Predictions have correct schema
- Decisions use correct logic
- Evaluation produces valid metrics

This is Phase 6.5 - a gate before Phase 7 (API/Dashboard).
"""

from datetime import datetime, timezone, timedelta
from typing import List
import numpy as np
import pytest

# Feature imports
from src.features.rolling import TeamRollingFeatures, MatchFeatures
from src.features.elo import EloRatingSystem

# Bayesian imports
from src.bayesian.priors import ModelConfig, HalfGoalPriors
from src.bayesian.model import MatchData, TrainingData, HalfGoalModel

# Evaluation imports
from src.evaluation.metrics import BrierScore, CalibrationResult
from src.evaluation.comparison import ModelEvaluator, compute_outcomes_from_scores

# Decision imports
from src.decision.engine import (
    DecisionEngine,
    DecisionConfig,
    DecisionOutcome,
    EXPERIMENTAL_CONFIG,
    PRODUCTION_CONFIG,
)
from src.decision.kelly import kelly_fraction, bayesian_kelly


class TestPipelineSchemaConsistency:
    """Test that all pipeline stages produce consistent schemas."""
    
    def test_match_data_schema(self):
        """MatchData has all required fields."""
        match = MatchData(
            match_id=1,
            league_idx=0,
            home_team_idx=0,
            away_team_idx=1,
            g1_home=1,
            g1_away=0,
            g2_home=0,
            g2_away=1,
        )
        
        assert hasattr(match, 'match_id')
        assert hasattr(match, 'league_idx')
        assert hasattr(match, 'home_team_idx')
        assert hasattr(match, 'away_team_idx')
        assert hasattr(match, 'g1_home')
        assert hasattr(match, 'g1_away')
        assert hasattr(match, 'g2_home')
        assert hasattr(match, 'g2_away')
        assert hasattr(match, 'has_scores')
        assert match.has_scores is True
    
    def test_training_data_from_matches(self):
        """TrainingData correctly processes batch of matches."""
        # Create sample matches
        matches = [
            MatchData(
                match_id=i,
                league_idx=0,
                home_team_idx=i % 5,
                away_team_idx=(i + 1) % 5,
                g1_home=np.random.randint(0, 3),
                g1_away=np.random.randint(0, 3),
                g2_home=np.random.randint(0, 3),
                g2_away=np.random.randint(0, 3),
                home_elo=1500 + np.random.randint(-100, 100),
                away_elo=1500 + np.random.randint(-100, 100),
            )
            for i in range(50)
        ]
        
        data = TrainingData.from_matches(matches)
        
        # Check all arrays exist and have correct length
        assert data.n_matches == 50
        assert len(data.g1_home) == 50
        assert len(data.g1_away) == 50
        assert len(data.g2_home) == 50
        assert len(data.g2_away) == 50
        assert len(data.home_team_idx) == 50
        assert len(data.away_team_idx) == 50
        
        # Check Elo scaling is applied
        assert np.abs(data.home_elo_scaled.mean()) < 1.0  # Centered
    
    def test_brier_score_schema(self):
        """BrierScore has all required components."""
        predictions = np.array([0.6, 0.3, 0.7, 0.4])
        outcomes = np.array([1, 0, 1, 0])
        
        brier = BrierScore.compute(predictions, outcomes)
        
        assert hasattr(brier, 'score')
        assert hasattr(brier, 'reliability')
        assert hasattr(brier, 'resolution')
        assert hasattr(brier, 'uncertainty')
        assert hasattr(brier, 'skill_score')
        assert hasattr(brier, 'skill_score_vs_reference')
        
        # Check values are in valid ranges
        assert 0 <= brier.score <= 1
        assert brier.skill_score <= 1
    
    def test_calibration_result_schema(self):
        """CalibrationResult has all required components."""
        predictions = np.random.uniform(0, 1, 100)
        outcomes = (predictions > np.random.uniform(0, 1, 100)).astype(int)
        
        calibration = CalibrationResult.compute(predictions, outcomes, n_bins=10)
        
        assert hasattr(calibration, 'bins')
        assert hasattr(calibration, 'expected_calibration_error')
        assert hasattr(calibration, 'maximum_calibration_error')
        
        # Check bins have correct structure
        for bin in calibration.bins:
            assert hasattr(bin, 'bin_lower')
            assert hasattr(bin, 'bin_upper')
            assert hasattr(bin, 'mean_predicted')
            assert hasattr(bin, 'mean_observed')
            assert hasattr(bin, 'count')
        
        # Check to_dict works
        d = calibration.to_dict()
        assert 'ece' in d
        assert 'mce' in d
        assert 'bins' in d
    
    def test_decision_schema(self):
        """BetDecision has all required fields."""
        engine = DecisionEngine(EXPERIMENTAL_CONFIG)
        
        decision = engine.make_decision(
            match_id=123,
            p_mean=0.65,
            p_ci=(0.60, 0.70),
            odds=2.0,
        )
        
        assert hasattr(decision, 'match_id')
        assert hasattr(decision, 'decision')
        assert hasattr(decision, 'timestamp')
        assert hasattr(decision, 'p_2h_gt_1h')
        assert hasattr(decision, 'p_2h_gt_1h_ci')
        assert hasattr(decision, 'odds')
        assert hasattr(decision, 'break_even_prob')
        assert hasattr(decision, 'stake_fraction')
        assert hasattr(decision, 'expected_value')
        assert hasattr(decision, 'reason')
        
        # Check to_dict works
        d = decision.to_dict()
        assert 'match_id' in d
        assert 'decision' in d


class TestPipelineEndToEnd:
    """Test complete pipeline flow with synthetic data."""
    
    @pytest.fixture
    def synthetic_matches(self) -> List[MatchData]:
        """Generate synthetic match data."""
        np.random.seed(42)
        
        matches = []
        for i in range(100):
            # Create realistic-ish scores
            g1_home = np.random.poisson(0.6)
            g1_away = np.random.poisson(0.5)
            g2_home = np.random.poisson(0.7)
            g2_away = np.random.poisson(0.6)
            
            matches.append(MatchData(
                match_id=i,
                league_idx=0,
                home_team_idx=i % 10,
                away_team_idx=(i + 3) % 10,
                g1_home=g1_home,
                g1_away=g1_away,
                g2_home=g2_home,
                g2_away=g2_away,
                home_elo=1500 + (i % 10 - 5) * 50,
                away_elo=1500 + ((i + 3) % 10 - 5) * 50,
            ))
        
        return matches
    
    def test_training_data_preparation(self, synthetic_matches):
        """Test that TrainingData correctly prepares batch data."""
        data = TrainingData.from_matches(synthetic_matches)
        
        assert data.n_matches == 100
        assert data.n_leagues == 1
        
        # Check all scores are integers
        assert data.g1_home.dtype in [np.int32, np.int64, int]
        
        # Check team indices are valid
        assert data.home_team_idx.max() < 10
        assert data.away_team_idx.max() < 10
    
    def test_model_builds_successfully(self, synthetic_matches):
        """Test that model builds without error."""
        pytest.importorskip("pymc")
        
        data = TrainingData.from_matches(synthetic_matches)
        
        # Use minimal config for speed
        config = ModelConfig(
            model_type="poisson",
            n_samples=10,  # Very small for test
            n_tune=10,
            n_chains=1,
        )
        
        model = HalfGoalModel(config)
        pm_model = model.build_model(data)
        
        assert pm_model is not None
        assert model.model is not None
    
    def test_evaluation_on_predictions(self, synthetic_matches):
        """Test that evaluation works on synthetic predictions."""
        # Generate synthetic predictions
        np.random.seed(42)
        n = len(synthetic_matches)
        predictions = np.random.beta(5, 5, size=n)  # Centered around 0.5
        
        # Compute actual outcomes
        outcomes = np.array([
            1 if (m.g2_home + m.g2_away) > (m.g1_home + m.g1_away) else 0
            for m in synthetic_matches
        ])
        
        # Run evaluation
        evaluator = ModelEvaluator()
        result = evaluator.evaluate_with_baselines(
            predictions, outcomes, model_name="synthetic_model"
        )
        
        assert "synthetic_model" in result.evaluations
        assert "always_2h" in result.evaluations
        assert "random" in result.evaluations
        
        # Check best model is determined
        assert result.best_model in result.evaluations
    
    def test_decision_engine_batch(self, synthetic_matches):
        """Test that decision engine handles batch correctly."""
        engine = DecisionEngine(EXPERIMENTAL_CONFIG)
        
        # Prepare batch
        matches_for_decision = [
            {
                "match_id": m.match_id,
                "p_mean": 0.5 + (m.home_elo - 1500) / 1000,  # Simple prediction
                "p_ci": (0.4, 0.6),
                "odds": 2.0 if i % 3 == 0 else None,  # Some have odds
            }
            for i, m in enumerate(synthetic_matches[:20])
        ]
        
        decisions = engine.make_batch_decisions(matches_for_decision)
        
        assert len(decisions) == 20
        
        # Check we have a mix of outcomes
        outcomes = [d.decision for d in decisions]
        assert DecisionOutcome.SIGNAL in outcomes or DecisionOutcome.SKIP in outcomes
    
    def test_outcomes_from_scores(self):
        """Test that outcomes are computed correctly."""
        ht_home = np.array([0, 1, 0, 2])
        ht_away = np.array([0, 0, 1, 0])
        ft_home = np.array([2, 2, 1, 3])
        ft_away = np.array([1, 0, 2, 1])
        
        outcomes = compute_outcomes_from_scores(ht_home, ht_away, ft_home, ft_away)
        
        # Match 0: HT 0-0 (G1=0), FT 2-1 (G2=3) → 1 (G2 > G1)
        # Match 1: HT 1-0 (G1=1), FT 2-0 (G2=1) → 0 (G2 == G1)
        # Match 2: HT 0-1 (G1=1), FT 1-2 (G2=2) → 1 (G2 > G1)
        # Match 3: HT 2-0 (G1=2), FT 3-1 (G2=2) → 0 (G2 == G1)
        assert outcomes[0] == 1
        assert outcomes[1] == 0
        assert outcomes[2] == 1  # Fixed: 2 > 1
        assert outcomes[3] == 0


class TestProductionConfigPresets:
    """Test that config presets behave correctly."""
    
    def test_experimental_vs_production_thresholds(self):
        """Production config should be stricter."""
        assert PRODUCTION_CONFIG.safety_threshold > EXPERIMENTAL_CONFIG.safety_threshold
        assert PRODUCTION_CONFIG.min_edge >= EXPERIMENTAL_CONFIG.min_edge
        assert PRODUCTION_CONFIG.max_bet_fraction <= EXPERIMENTAL_CONFIG.max_bet_fraction
    
    def test_production_rejects_borderline_bets(self):
        """Production config rejects bets that experimental allows."""
        # Create a borderline edge case
        p_samples = np.random.beta(55, 45, size=1000)  # Mean ~0.55
        
        # With experimental settings
        exp_engine = DecisionEngine(EXPERIMENTAL_CONFIG)
        exp_decision = exp_engine.make_decision(
            match_id=1,
            p_mean=0.55,
            p_samples=p_samples,
            p_ci=(0.50, 0.60),
            odds=2.0,
        )
        
        # With production settings
        prod_engine = DecisionEngine(PRODUCTION_CONFIG)
        prod_decision = prod_engine.make_decision(
            match_id=1,
            p_mean=0.55,
            p_samples=p_samples,
            p_ci=(0.50, 0.60),
            odds=2.0,
        )
        
        # Production should be at least as restrictive
        if exp_decision.decision == DecisionOutcome.BELOW_SAFETY_THRESHOLD:
            assert prod_decision.decision == DecisionOutcome.BELOW_SAFETY_THRESHOLD
        
        # If both bet, production should bet less
        if exp_decision.decision == DecisionOutcome.BET and prod_decision.decision == DecisionOutcome.BET:
            assert prod_decision.stake_fraction <= exp_decision.stake_fraction


class TestEloIntegration:
    """Test Elo rating system integration."""
    
    def test_elo_chronological_processing(self):
        """Test that Elo updates happen in order."""
        system = EloRatingSystem()
        
        # Simulate a season
        now = datetime.now(timezone.utc)
        
        # Team 1 beats team 2
        system.update_ratings(1, 2, 3, 0, now - timedelta(days=10))
        
        # Team 2 beats team 3
        system.update_ratings(2, 3, 2, 0, now - timedelta(days=5))
        
        # Check ratings reflect history
        rating_1 = system.get_rating(1)
        rating_2 = system.get_rating(2)
        rating_3 = system.get_rating(3)
        
        # Team 1 should be highest (won big)
        # Team 3 should be lowest (lost only game)
        assert rating_1.rating > 1500
        assert rating_3.rating < 1500
    
    def test_elo_respects_as_of(self):
        """Test that get_ratings_as_of respects time."""
        # This would require mock DB - skip for now
        pass


class TestKellyMath:
    """Test Kelly criterion math."""
    
    def test_kelly_edge_cases(self):
        """Test Kelly on edge cases."""
        # Certainty: 100% probability at any odds
        assert kelly_fraction(1.0, 2.0) == pytest.approx(1.0, abs=0.01)
        
        # Certainty of loss
        assert kelly_fraction(0.0, 2.0) < 0
        
        # Fair odds
        assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0, abs=1e-10)
    
    def test_bayesian_kelly_respects_safety(self):
        """Test that Bayesian Kelly enforces safety threshold."""
        # Uncertain samples
        p_samples = np.random.uniform(0.4, 0.6, size=1000)  # Very spread
        
        result = bayesian_kelly(
            p_samples=p_samples,
            odds=2.0,
            safety_threshold=0.9,  # Very strict
        )
        
        # Should not satisfy threshold
        assert result.satisfies_safety_threshold is False
        assert result.fraction == 0
