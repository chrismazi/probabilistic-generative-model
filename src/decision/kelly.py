"""
Kelly criterion and bankroll management.

Provides:
- Full Kelly
- Fractional Kelly (recommended)
- Bayesian edge with credible intervals
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class KellyResult:
    """Result of Kelly calculation."""
    
    fraction: float  # Optimal fraction of bankroll
    edge: float  # Expected edge (EV - 1)
    is_bet: bool  # Whether to bet at all
    
    # Uncertainty from posterior
    fraction_ci: Optional[Tuple[float, float]] = None
    edge_ci: Optional[Tuple[float, float]] = None
    
    # Safety metrics
    prob_positive_edge: float = 0.0  # P(edge > 0)


def kelly_fraction(p: float, odds: float) -> float:
    """
    Calculate Kelly fraction.
    
    f* = (p * (odds - 1) - (1 - p)) / (odds - 1)
       = (p * odds - 1) / (odds - 1)
    
    Args:
        p: Probability of winning
        odds: Decimal odds (e.g., 2.0 for even money)
        
    Returns:
        Optimal fraction of bankroll (can be negative = don't bet)
    """
    if odds <= 1:
        return 0.0
    
    b = odds - 1  # Net profit if win
    q = 1 - p  # Probability of losing
    
    f = (p * b - q) / b
    return f


def fractional_kelly(
    p: float,
    odds: float,
    kelly_fraction_mult: float = 0.25,
    max_bet_fraction: float = 0.05,
) -> float:
    """
    Calculate fractional Kelly with safety limits.
    
    Fractional Kelly reduces variance at cost of growth.
    Common values: 0.25 (quarter Kelly), 0.5 (half Kelly)
    
    Args:
        p: Probability of winning
        odds: Decimal odds
        kelly_fraction_mult: Fraction of full Kelly (0.25 = quarter)
        max_bet_fraction: Maximum bet as fraction of bankroll
        
    Returns:
        Fraction of bankroll to bet (0 if no bet)
    """
    full_kelly = kelly_fraction(p, odds)
    
    if full_kelly <= 0:
        return 0.0
    
    # Apply fraction
    fractional = full_kelly * kelly_fraction_mult
    
    # Apply maximum cap
    return min(fractional, max_bet_fraction)


def break_even_probability(odds: float) -> float:
    """
    Calculate break-even probability for given odds.
    
    p_be = 1 / odds
    
    Args:
        odds: Decimal odds
        
    Returns:
        Probability needed to break even
    """
    if odds <= 0:
        return 1.0
    return 1.0 / odds


def expected_value(p: float, odds: float, stake: float = 1.0) -> float:
    """
    Calculate expected value.
    
    EV = p * (odds - 1) * stake - (1 - p) * stake
       = stake * (p * odds - 1)
    
    Args:
        p: Win probability
        odds: Decimal odds
        stake: Bet amount
        
    Returns:
        Expected value (positive = +EV)
    """
    return stake * (p * odds - 1)


@dataclass
class BayesianKellyResult(KellyResult):
    """Kelly result with Bayesian safety check."""
    
    # P(p > p_be) - probability our estimate exceeds break-even
    prob_exceeds_breakeven: float = 0.0
    satisfies_safety_threshold: bool = False


def bayesian_kelly(
    p_samples: np.ndarray,
    odds: float,
    kelly_fraction_mult: float = 0.25,
    max_bet_fraction: float = 0.05,
    safety_threshold: float = 0.6,
) -> BayesianKellyResult:
    """
    Calculate Kelly with Bayesian safety check.
    
    Uses posterior samples to compute:
    - P(p > p_be) - probability our estimate exceeds break-even
    - Only bets if P(p > p_be) > safety_threshold
    
    Args:
        p_samples: Posterior samples of win probability
        odds: Decimal odds
        kelly_fraction_mult: Fraction of full Kelly
        max_bet_fraction: Maximum bet fraction
        safety_threshold: Minimum P(p > p_be) required to bet
        
    Returns:
        BayesianKellyResult with safety check
    """
    p_mean = float(np.mean(p_samples))
    p_be = break_even_probability(odds)
    
    # Probability that true p exceeds break-even
    prob_exceeds = float(np.mean(p_samples > p_be))
    
    # Calculate edge for each sample
    edges = p_samples * odds - 1
    edge_mean = float(np.mean(edges))
    edge_ci = (float(np.percentile(edges, 5)), float(np.percentile(edges, 95)))
    
    # Kelly for each sample
    kelly_samples = np.array([kelly_fraction(p, odds) for p in p_samples])
    kelly_mean = float(np.mean(kelly_samples))
    kelly_ci = (
        float(np.percentile(kelly_samples, 5)),
        float(np.percentile(kelly_samples, 95)),
    )
    
    # Apply safety check
    satisfies_safety = prob_exceeds > safety_threshold
    
    # Final fraction
    if satisfies_safety and edge_mean > 0:
        fraction = fractional_kelly(p_mean, odds, kelly_fraction_mult, max_bet_fraction)
    else:
        fraction = 0.0
    
    return BayesianKellyResult(
        fraction=fraction,
        edge=edge_mean,
        is_bet=fraction > 0,
        fraction_ci=kelly_ci,
        edge_ci=edge_ci,
        prob_positive_edge=float(np.mean(edges > 0)),
        prob_exceeds_breakeven=prob_exceeds,
        satisfies_safety_threshold=satisfies_safety,
    )
