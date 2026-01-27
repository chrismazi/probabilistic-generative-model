"""
Prior specifications for Bayesian models.

Contains weakly informative priors for:
- Attack/defense parameters
- League intercepts
- Overdispersion (NegBin)

These priors prevent sampler disasters while remaining flexible.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HalfGoalPriors:
    """
    Prior specifications for half-goal model.
    
    All values are on log scale unless specified.
    """
    
    # League-level intercepts (expected log-goals per half)
    # log(0.6) ≈ -0.5, log(1.2) ≈ 0.2
    intercept_mean: float = -0.2  # ~0.8 goals per team per half
    intercept_sd: float = 0.3
    
    # Second half modifier (2H typically has slightly more goals)
    half2_effect_mean: float = 0.1  # ~10% more goals in 2H
    half2_effect_sd: float = 0.2
    
    # Home advantage PRIOR (on log scale)
    # Prior centered at exp(0.2) ≈ 1.22
    # Actual value is estimated from data - see get_posterior_summary()
    home_advantage_mean: float = 0.2
    home_advantage_sd: float = 0.15
    
    # Team attack/defense (deviations from league mean)
    # SD of 0.3 → most teams within ±30% of league average
    attack_sd: float = 0.3
    defense_sd: float = 0.3
    
    # Hyperpriors for hierarchical structure
    # How much teams vary within a league
    attack_sd_hyper_sd: float = 0.2
    defense_sd_hyper_sd: float = 0.2
    
    # League-level variation around global mean
    league_intercept_sd: float = 0.2
    
    # Negative Binomial overdispersion prior
    # Half-normal with this SD (higher = more overdispersion possible)
    negbin_phi_sd: float = 2.0
    
    # Feature coefficients (Elo, rolling features)
    feature_coef_sd: float = 0.1


@dataclass 
class ModelConfig:
    """
    Configuration for model training.
    """
    
    # Model type
    model_type: str = "poisson"  # "poisson" or "negbin"
    
    # Priors
    priors: HalfGoalPriors = None
    
    # Sampling parameters
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 4
    target_accept: float = 0.9
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Feature inclusion
    use_elo: bool = True
    use_rolling_features: bool = True
    
    # Regularization
    sum_to_zero_constraint: bool = True
    
    def __post_init__(self):
        if self.priors is None:
            self.priors = HalfGoalPriors()


# Default configurations
DEFAULT_POISSON_CONFIG = ModelConfig(model_type="poisson")
DEFAULT_NEGBIN_CONFIG = ModelConfig(model_type="negbin")
