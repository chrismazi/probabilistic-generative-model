"""
Decision engine for bet/no-bet decisions.

Implements:
- Break-even probability check
- Bayesian safety check (P(p > p_be) > τ)
- Kelly sizing with caps
- Edge thresholds
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np

from src.decision.kelly import (
    break_even_probability,
    bayesian_kelly,
    expected_value,
    BayesianKellyResult,
)
from src.utils import get_logger

logger = get_logger("decision.engine")


class DecisionOutcome(str, Enum):
    """Decision outcome types."""
    BET = "bet"
    SKIP = "skip"
    INSUFFICIENT_EDGE = "insufficient_edge"
    BELOW_SAFETY_THRESHOLD = "below_safety_threshold"
    MISSING_ODDS = "missing_odds"
    INVALID_PREDICTION = "invalid_prediction"


@dataclass
class DecisionConfig:
    """Configuration for decision engine."""
    
    # Minimum edge required to consider betting
    min_edge: float = 0.02  # 2% minimum edge
    
    # Bayesian safety threshold
    # P(p > p_be) must exceed this to bet
    safety_threshold: float = 0.6
    
    # Kelly parameters
    kelly_fraction: float = 0.25  # Quarter Kelly
    max_bet_fraction: float = 0.05  # 5% of bankroll max
    
    # Feature importance (without odds)
    min_probability_gap: float = 0.05  # Gap between p and 0.5 to consider
    
    # Risk limits
    max_daily_bets: int = 10
    max_daily_exposure: float = 0.20  # 20% of bankroll


@dataclass
class BetDecision:
    """A single bet decision."""
    
    match_id: int
    decision: DecisionOutcome
    timestamp: datetime
    
    # Prediction
    p_2h_gt_1h: float
    p_2h_gt_1h_ci: tuple[float, float]
    
    # Odds (if available)
    odds: Optional[float] = None
    break_even_prob: Optional[float] = None
    
    # Sizing
    stake_fraction: float = 0.0
    expected_value: float = 0.0
    
    # Safety
    prob_exceeds_breakeven: float = 0.0
    
    # Explanation
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "match_id": self.match_id,
            "decision": self.decision.value,
            "timestamp": self.timestamp.isoformat(),
            "p_2h_gt_1h": self.p_2h_gt_1h,
            "p_ci_low": self.p_2h_gt_1h_ci[0],
            "p_ci_high": self.p_2h_gt_1h_ci[1],
            "odds": self.odds,
            "break_even_prob": self.break_even_prob,
            "stake_fraction": self.stake_fraction,
            "expected_value": self.expected_value,
            "prob_exceeds_breakeven": self.prob_exceeds_breakeven,
            "reason": self.reason,
        }


class DecisionEngine:
    """
    Decision engine for bet/no-bet decisions.
    
    Implements:
    - Bayesian safety check
    - Kelly sizing with caps
    - Edge thresholds
    """
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        """
        Initialize decision engine.
        
        Args:
            config: Decision configuration
        """
        self.config = config or DecisionConfig()
    
    def make_decision(
        self,
        match_id: int,
        p_mean: float,
        p_samples: Optional[np.ndarray] = None,
        p_ci: tuple[float, float] = (0.0, 1.0),
        odds: Optional[float] = None,
    ) -> BetDecision:
        """
        Make bet/no-bet decision for a match.
        
        Args:
            match_id: Match identifier
            p_mean: Mean predicted probability P(G2 > G1)
            p_samples: Posterior samples (for Bayesian safety check)
            p_ci: Credible interval
            odds: Decimal odds (if available)
            
        Returns:
            BetDecision with sizing and explanation
        """
        timestamp = datetime.now()
        
        # Validate prediction
        if not (0 < p_mean < 1):
            return BetDecision(
                match_id=match_id,
                decision=DecisionOutcome.INVALID_PREDICTION,
                timestamp=timestamp,
                p_2h_gt_1h=p_mean,
                p_2h_gt_1h_ci=p_ci,
                reason=f"Invalid probability: {p_mean}",
            )
        
        # Case 1: No odds available - use threshold-based decision
        if odds is None:
            return self._decision_without_odds(match_id, p_mean, p_ci, timestamp)
        
        # Case 2: Odds available - full Bayesian Kelly
        return self._decision_with_odds(
            match_id=match_id,
            p_mean=p_mean,
            p_samples=p_samples,
            p_ci=p_ci,
            odds=odds,
            timestamp=timestamp,
        )
    
    def _decision_without_odds(
        self,
        match_id: int,
        p_mean: float,
        p_ci: tuple[float, float],
        timestamp: datetime,
    ) -> BetDecision:
        """
        Make decision without odds (threshold-based).
        
        Without odds, we can only:
        - Flag matches with strong predictions
        - Report confidence intervals
        """
        # Check if prediction is confident enough
        gap = abs(p_mean - 0.5)
        
        if gap < self.config.min_probability_gap:
            return BetDecision(
                match_id=match_id,
                decision=DecisionOutcome.INSUFFICIENT_EDGE,
                timestamp=timestamp,
                p_2h_gt_1h=p_mean,
                p_2h_gt_1h_ci=p_ci,
                odds=None,
                reason=f"Probability too close to 0.5 (gap={gap:.3f})",
            )
        
        # Strong prediction - flag for potential bet
        direction = "2H > 1H" if p_mean > 0.5 else "1H > 2H"
        
        return BetDecision(
            match_id=match_id,
            decision=DecisionOutcome.SKIP,  # Can't bet without odds
            timestamp=timestamp,
            p_2h_gt_1h=p_mean,
            p_2h_gt_1h_ci=p_ci,
            odds=None,
            reason=f"Strong signal for {direction} but no odds available",
        )
    
    def _decision_with_odds(
        self,
        match_id: int,
        p_mean: float,
        p_samples: Optional[np.ndarray],
        p_ci: tuple[float, float],
        odds: float,
        timestamp: datetime,
    ) -> BetDecision:
        """
        Make decision with odds (full Bayesian Kelly).
        """
        p_be = break_even_probability(odds)
        
        # Generate samples if not provided
        if p_samples is None:
            # Approximate with beta distribution
            n_samples = 1000
            # Use CI to estimate alpha/beta
            p_samples = np.random.beta(
                a=max(1, p_mean * 10),
                b=max(1, (1 - p_mean) * 10),
                size=n_samples,
            )
        
        # Bayesian Kelly calculation
        kelly_result = bayesian_kelly(
            p_samples=p_samples,
            odds=odds,
            kelly_fraction_mult=self.config.kelly_fraction,
            max_bet_fraction=self.config.max_bet_fraction,
            safety_threshold=self.config.safety_threshold,
        )
        
        # Check safety threshold
        if not kelly_result.satisfies_safety_threshold:
            return BetDecision(
                match_id=match_id,
                decision=DecisionOutcome.BELOW_SAFETY_THRESHOLD,
                timestamp=timestamp,
                p_2h_gt_1h=p_mean,
                p_2h_gt_1h_ci=p_ci,
                odds=odds,
                break_even_prob=p_be,
                prob_exceeds_breakeven=kelly_result.prob_exceeds_breakeven,
                reason=f"P(p > p_be) = {kelly_result.prob_exceeds_breakeven:.2f} < {self.config.safety_threshold}",
            )
        
        # Check edge threshold
        if kelly_result.edge < self.config.min_edge:
            return BetDecision(
                match_id=match_id,
                decision=DecisionOutcome.INSUFFICIENT_EDGE,
                timestamp=timestamp,
                p_2h_gt_1h=p_mean,
                p_2h_gt_1h_ci=p_ci,
                odds=odds,
                break_even_prob=p_be,
                expected_value=kelly_result.edge,
                prob_exceeds_breakeven=kelly_result.prob_exceeds_breakeven,
                reason=f"Edge {kelly_result.edge:.3f} < min {self.config.min_edge}",
            )
        
        # Bet!
        ev = expected_value(p_mean, odds, stake=1.0)
        
        return BetDecision(
            match_id=match_id,
            decision=DecisionOutcome.BET,
            timestamp=timestamp,
            p_2h_gt_1h=p_mean,
            p_2h_gt_1h_ci=p_ci,
            odds=odds,
            break_even_prob=p_be,
            stake_fraction=kelly_result.fraction,
            expected_value=ev,
            prob_exceeds_breakeven=kelly_result.prob_exceeds_breakeven,
            reason=f"Edge={kelly_result.edge:.3f}, stake={kelly_result.fraction:.3f}",
        )
    
    def make_batch_decisions(
        self,
        matches: List[Dict[str, Any]],
    ) -> List[BetDecision]:
        """
        Make decisions for multiple matches.
        
        Args:
            matches: List of dicts with match_id, p_mean, p_samples, p_ci, odds
            
        Returns:
            List of BetDecision
        """
        decisions = []
        
        for match in matches:
            decision = self.make_decision(
                match_id=match["match_id"],
                p_mean=match["p_mean"],
                p_samples=match.get("p_samples"),
                p_ci=match.get("p_ci", (0.0, 1.0)),
                odds=match.get("odds"),
            )
            decisions.append(decision)
        
        return decisions


def get_decision_engine(config: Optional[DecisionConfig] = None) -> DecisionEngine:
    """Get a configured decision engine."""
    return DecisionEngine(config=config)
