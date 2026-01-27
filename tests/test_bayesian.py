"""
Tests for Bayesian model components.

Tests for:
- Prior configurations
- Model data structures  
- Prediction logic (without full MCMC)
"""

from datetime import datetime, timezone
import numpy as np
import pytest

from src.bayesian.priors import HalfGoalPriors, ModelConfig, DEFAULT_POISSON_CONFIG
from src.bayesian.model import MatchData, TrainingData, HalfGoalModel
from src.bayesian.prediction import MatchPrediction, compute_entropy


class TestHalfGoalPriors:
    """Test prior configurations."""
    
    def test_default_priors(self):
        """Test default prior values are sensible."""
        priors = HalfGoalPriors()
        
        # Intercept should give ~0.6-1.2 goals per team per half
        expected_goals = np.exp(priors.intercept_mean)
        assert 0.4 < expected_goals < 1.5
        
        # Home advantage should be positive
        assert priors.home_advantage_mean > 0
        
        # Attack/defense SD should allow reasonable variation
        assert 0.1 < priors.attack_sd < 1.0
    
    def test_second_half_effect(self):
        """Test 2H typically has more goals."""
        priors = HalfGoalPriors()
        
        # 2H effect should be positive (more goals)
        assert priors.half2_effect_mean > 0


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_poisson_config(self):
        """Test Poisson config defaults."""
        config = DEFAULT_POISSON_CONFIG
        
        assert config.model_type == "poisson"
        assert config.n_chains == 4
        assert config.target_accept >= 0.9
    
    def test_config_with_custom_priors(self):
        """Test config accepts custom priors."""
        custom = HalfGoalPriors(intercept_mean=0.0)
        config = ModelConfig(priors=custom)
        
        assert config.priors.intercept_mean == 0.0


class TestMatchData:
    """Test match data structure."""
    
    def test_has_scores_true(self):
        """Test has_scores when all scores present."""
        match = MatchData(
            match_id=1,
            league_idx=0,
            home_team_idx=0,
            away_team_idx=1,
            g1_home=1,
            g1_away=0,
            g2_home=2,
            g2_away=1,
        )
        
        assert match.has_scores is True
    
    def test_has_scores_false(self):
        """Test has_scores when scores missing."""
        match = MatchData(
            match_id=1,
            league_idx=0,
            home_team_idx=0,
            away_team_idx=1,
            g1_home=1,
            g1_away=None,
            g2_home=2,
            g2_away=1,
        )
        
        assert match.has_scores is False
    
    def test_default_elo(self):
        """Test default Elo is 1500."""
        match = MatchData(
            match_id=1,
            league_idx=0,
            home_team_idx=0,
            away_team_idx=1,
        )
        
        assert match.home_elo == 1500.0
        assert match.away_elo == 1500.0


class TestTrainingData:
    """Test training data preparation."""
    
    def test_from_matches_filters_incomplete(self):
        """Test that matches without scores are filtered."""
        matches = [
            MatchData(1, 0, 0, 1, g1_home=1, g1_away=0, g2_home=2, g2_away=1),
            MatchData(2, 0, 0, 2, g1_home=None, g1_away=0, g2_home=1, g2_away=0),  # Incomplete
            MatchData(3, 0, 1, 2, g1_home=0, g1_away=0, g2_home=1, g2_away=1),
        ]
        
        data = TrainingData.from_matches(matches)
        
        assert data.n_matches == 2  # Only complete matches
    
    def test_elo_scaling(self):
        """Test Elo is scaled to ~N(0,1)."""
        matches = [
            MatchData(1, 0, 0, 1, g1_home=1, g1_away=0, g2_home=2, g2_away=1,
                     home_elo=1600, away_elo=1400),
        ]
        
        data = TrainingData.from_matches(matches)
        
        # 1600 -> (1600-1500)/200 = 0.5
        assert abs(data.home_elo_scaled[0] - 0.5) < 0.01
        assert abs(data.away_elo_scaled[0] - (-0.5)) < 0.01
    
    def test_empty_matches_raises(self):
        """Test error when no matches have scores."""
        matches = [
            MatchData(1, 0, 0, 1),  # No scores
        ]
        
        with pytest.raises(ValueError, match="No matches with scores"):
            TrainingData.from_matches(matches)


class TestEntropyComputation:
    """Test entropy calculation."""
    
    def test_entropy_at_half(self):
        """Entropy is maximum at p=0.5."""
        entropy_half = compute_entropy(0.5)
        entropy_other = compute_entropy(0.7)
        
        assert entropy_half > entropy_other
        assert abs(entropy_half - 1.0) < 0.01  # Binary entropy max is 1
    
    def test_entropy_at_extremes(self):
        """Entropy is 0 at p=0 and p=1."""
        assert compute_entropy(0.0) == 0.0
        assert compute_entropy(1.0) == 0.0
    
    def test_entropy_near_extremes(self):
        """Entropy is low near extremes."""
        assert compute_entropy(0.01) < 0.1
        assert compute_entropy(0.99) < 0.1


class TestMatchPrediction:
    """Test prediction structure."""
    
    def test_to_dict(self):
        """Test serialization."""
        pred = MatchPrediction(
            match_id=123,
            model_version="v1.0.0",
            created_at=datetime.now(),
            p_2h_gt_1h=0.55,
            p_2h_gt_1h_ci_low=0.45,
            p_2h_gt_1h_ci_high=0.65,
            expected_g1=1.2,
            expected_g2=1.4,
        )
        
        d = pred.to_dict()
        
        assert d["match_id"] == 123
        assert d["p_2h_gt_1h"] == 0.55
        assert len(d["p_2h_gt_1h_ci"]) == 2


class TestHalfGoalModelStructure:
    """Test model structure (without full MCMC)."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = HalfGoalModel()
        
        assert model.config.model_type == "poisson"
        assert model.trace is None
        assert model.model is None
    
    def test_model_with_negbin_config(self):
        """Test NegBin model config."""
        config = ModelConfig(model_type="negbin")
        model = HalfGoalModel(config=config)
        
        assert model.config.model_type == "negbin"
    
    def test_build_model_creates_pymc_model(self):
        """Test model builds PyMC model."""
        # Skip if PyMC not installed
        pytest.importorskip("pymc")
        
        matches = [
            MatchData(i, 0, i % 5, (i + 1) % 5, 
                     g1_home=1, g1_away=0, g2_home=1, g2_away=1)
            for i in range(20)
        ]
        data = TrainingData.from_matches(matches)
        
        model = HalfGoalModel()
        pm_model = model.build_model(data)
        
        assert pm_model is not None
        assert model.model is not None
