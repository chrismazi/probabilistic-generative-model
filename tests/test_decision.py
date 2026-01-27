"""
Tests for decision module.

Tests for:
- Kelly criterion calculations
- Decision engine logic
- Bayesian safety check
"""

import numpy as np
import pytest

from src.decision.kelly import (
    kelly_fraction,
    fractional_kelly,
    break_even_probability,
    expected_value,
    bayesian_kelly,
)
from src.decision.engine import (
    DecisionOutcome,
    DecisionConfig,
    BetDecision,
    DecisionEngine,
)


class TestKellyFraction:
    """Test Kelly fraction calculation."""
    
    def test_even_money_50_percent(self):
        """50% probability at even money = no edge."""
        f = kelly_fraction(0.5, 2.0)
        assert f == pytest.approx(0.0, abs=1e-10)
    
    def test_positive_edge(self):
        """60% probability at even money = positive edge."""
        f = kelly_fraction(0.6, 2.0)
        # f = (0.6 * 1 - 0.4) / 1 = 0.2
        assert f == pytest.approx(0.2, abs=1e-10)
    
    def test_negative_edge(self):
        """40% probability at even money = negative edge."""
        f = kelly_fraction(0.4, 2.0)
        # f = (0.4 * 1 - 0.6) / 1 = -0.2
        assert f == pytest.approx(-0.2, abs=1e-10)
    
    def test_high_odds_low_probability(self):
        """Long shot with fair odds."""
        # 10% probability at 10.0 odds = break-even
        f = kelly_fraction(0.1, 10.0)
        # f = (0.1 * 9 - 0.9) / 9 = 0
        assert f == pytest.approx(0.0, abs=1e-10)
    
    def test_high_odds_slight_edge(self):
        """Long shot with slight edge."""
        # 12% probability at 10.0 odds
        f = kelly_fraction(0.12, 10.0)
        # f = (0.12 * 9 - 0.88) / 9 = (1.08 - 0.88) / 9 = 0.022
        assert f > 0
    
    def test_invalid_odds(self):
        """Odds <= 1 returns 0."""
        assert kelly_fraction(0.5, 1.0) == 0.0
        assert kelly_fraction(0.5, 0.5) == 0.0


class TestFractionalKelly:
    """Test fractional Kelly with caps."""
    
    def test_quarter_kelly(self):
        """Quarter Kelly reduces bet by 75%."""
        full = kelly_fraction(0.6, 2.0)
        quarter = fractional_kelly(0.6, 2.0, kelly_fraction_mult=0.25)
        
        assert quarter == pytest.approx(full * 0.25, abs=1e-10)
    
    def test_respects_max_cap(self):
        """Bet is capped at max_bet_fraction."""
        # Very high edge
        fraction = fractional_kelly(0.9, 2.0, kelly_fraction_mult=1.0, max_bet_fraction=0.05)
        
        assert fraction == 0.05
    
    def test_negative_edge_returns_zero(self):
        """Negative edge = no bet."""
        fraction = fractional_kelly(0.4, 2.0)
        assert fraction == 0.0


class TestBreakEvenProbability:
    """Test break-even probability calculation."""
    
    def test_even_money(self):
        """Even money (2.0) needs 50%."""
        p_be = break_even_probability(2.0)
        assert p_be == pytest.approx(0.5, abs=1e-10)
    
    def test_short_odds(self):
        """1.5 odds needs 66.7%."""
        p_be = break_even_probability(1.5)
        assert p_be == pytest.approx(0.667, abs=0.001)
    
    def test_long_odds(self):
        """4.0 odds needs 25%."""
        p_be = break_even_probability(4.0)
        assert p_be == pytest.approx(0.25, abs=1e-10)


class TestExpectedValue:
    """Test expected value calculation."""
    
    def test_positive_ev(self):
        """60% at 2.0 has positive EV."""
        ev = expected_value(0.6, 2.0, stake=1.0)
        # EV = 1 * (0.6 * 2 - 1) = 0.2
        assert ev == pytest.approx(0.2, abs=1e-10)
    
    def test_negative_ev(self):
        """40% at 2.0 has negative EV."""
        ev = expected_value(0.4, 2.0, stake=1.0)
        assert ev == pytest.approx(-0.2, abs=1e-10)
    
    def test_break_even(self):
        """50% at 2.0 has zero EV."""
        ev = expected_value(0.5, 2.0, stake=1.0)
        assert ev == pytest.approx(0.0, abs=1e-10)


class TestBayesianKelly:
    """Test Bayesian Kelly with safety check."""
    
    def test_high_confidence_positive_edge(self):
        """High confidence in positive edge = bet."""
        # Samples all above break-even
        p_samples = np.random.beta(60, 40, size=1000)  # Mean ~0.6
        
        result = bayesian_kelly(
            p_samples=p_samples,
            odds=2.0,
            safety_threshold=0.6,
        )
        
        assert result.prob_exceeds_breakeven > 0.8
        assert result.satisfies_safety_threshold is True
        assert result.fraction > 0
    
    def test_uncertain_edge(self):
        """Uncertain edge below safety threshold = no bet."""
        # Samples centered around break-even
        p_samples = np.random.beta(50, 50, size=1000)  # Mean ~0.5
        
        result = bayesian_kelly(
            p_samples=p_samples,
            odds=2.0,
            safety_threshold=0.6,
        )
        
        # About 50% should exceed break-even
        assert result.prob_exceeds_breakeven < 0.6
        assert result.satisfies_safety_threshold is False
        assert result.fraction == 0
    
    def test_negative_edge(self):
        """Clear negative edge = no bet."""
        p_samples = np.random.beta(40, 60, size=1000)  # Mean ~0.4
        
        result = bayesian_kelly(
            p_samples=p_samples,
            odds=2.0,
            safety_threshold=0.6,
        )
        
        assert result.prob_exceeds_breakeven < 0.2
        assert result.edge < 0


class TestDecisionEngine:
    """Test decision engine."""
    
    @pytest.fixture
    def engine(self):
        return DecisionEngine(DecisionConfig(
            min_edge=0.02,
            safety_threshold=0.6,
            kelly_fraction=0.25,
            max_bet_fraction=0.05,
        ))
    
    def test_bet_with_positive_edge(self, engine):
        """Strong positive edge = BET."""
        p_samples = np.random.beta(70, 30, size=1000)  # Mean ~0.7
        
        decision = engine.make_decision(
            match_id=1,
            p_mean=0.7,
            p_samples=p_samples,
            p_ci=(0.65, 0.75),
            odds=2.0,
        )
        
        assert decision.decision == DecisionOutcome.BET
        assert decision.stake_fraction > 0
        assert decision.expected_value > 0
    
    def test_skip_below_safety(self, engine):
        """Uncertain edge = BELOW_SAFETY_THRESHOLD."""
        p_samples = np.random.beta(50, 50, size=1000)  # Mean ~0.5
        
        decision = engine.make_decision(
            match_id=2,
            p_mean=0.52,
            p_samples=p_samples,
            p_ci=(0.45, 0.58),
            odds=2.0,
        )
        
        assert decision.decision == DecisionOutcome.BELOW_SAFETY_THRESHOLD
        assert decision.stake_fraction == 0
    
    def test_no_odds_strong_signal(self, engine):
        """Strong signal without odds = SIGNAL (not BET)."""
        decision = engine.make_decision(
            match_id=3,
            p_mean=0.7,
            p_ci=(0.65, 0.75),
            odds=None,
        )
        
        assert decision.decision == DecisionOutcome.SIGNAL
        assert "no odds" in decision.reason.lower()
    
    def test_invalid_probability(self, engine):
        """Invalid probability = INVALID_PREDICTION."""
        decision = engine.make_decision(
            match_id=4,
            p_mean=1.5,  # Invalid
            p_ci=(0.0, 1.0),
            odds=2.0,
        )
        
        assert decision.decision == DecisionOutcome.INVALID_PREDICTION


class TestBetDecision:
    """Test BetDecision dataclass."""
    
    def test_to_dict(self):
        """Test serialization."""
        from datetime import datetime
        
        decision = BetDecision(
            match_id=123,
            decision=DecisionOutcome.BET,
            timestamp=datetime(2024, 8, 17, 14, 0),
            p_2h_gt_1h=0.65,
            p_2h_gt_1h_ci=(0.60, 0.70),
            odds=2.0,
            break_even_prob=0.5,
            stake_fraction=0.03,
            expected_value=0.3,
            prob_exceeds_breakeven=0.85,
            reason="Good edge",
        )
        
        d = decision.to_dict()
        
        assert d["match_id"] == 123
        assert d["decision"] == "bet"
        assert d["p_2h_gt_1h"] == 0.65
        assert d["stake_fraction"] == 0.03
