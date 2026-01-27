"""
Hierarchical Poisson/NegBin model for half-goal prediction.

Model v1: Simple, stable, calibrated.

Targets:
    G1_home, G1_away (first half goals)
    G2_home, G2_away (second half goals)

Structure:
    log(μ_h,1) = α_1,L + a_i,1 - d_j,1 + β'x_m
    log(μ_h,2) = α_2,L + a_i,2 - d_j,2 + β'x_m + half2_effect

Where:
    α_L = league intercept
    a_i = team attack strength (sum-to-zero per league)
    d_j = team defense strength (sum-to-zero per league)
    x_m = match features (Elo, rolling stats)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import arviz as az

from src.bayesian.priors import ModelConfig, HalfGoalPriors, DEFAULT_POISSON_CONFIG
from src.utils import get_logger

logger = get_logger("bayesian.model")


@dataclass
class MatchData:
    """Training data for a single match."""
    
    match_id: int
    league_idx: int  # League index (0-based)
    home_team_idx: int  # Team index within league
    away_team_idx: int
    
    # Scores (training targets)
    g1_home: Optional[int] = None
    g1_away: Optional[int] = None
    g2_home: Optional[int] = None
    g2_away: Optional[int] = None
    
    # Features
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_attack_strength: float = 1.0
    home_defense_strength: float = 1.0
    away_attack_strength: float = 1.0
    away_defense_strength: float = 1.0
    
    @property
    def has_scores(self) -> bool:
        """Check if all scores are available."""
        return all(
            v is not None 
            for v in [self.g1_home, self.g1_away, self.g2_home, self.g2_away]
        )


@dataclass
class TrainingData:
    """Prepared data for model training."""
    
    n_matches: int
    n_leagues: int
    n_teams_per_league: Dict[int, int]
    
    # Indices (0-based)
    league_idx: np.ndarray  # (n_matches,)
    home_team_idx: np.ndarray  # (n_matches,)
    away_team_idx: np.ndarray  # (n_matches,)
    
    # Targets (for training)
    g1_home: np.ndarray  # (n_matches,)
    g1_away: np.ndarray
    g2_home: np.ndarray
    g2_away: np.ndarray
    
    # Features
    home_elo_scaled: np.ndarray  # Scaled to ~N(0,1)
    away_elo_scaled: np.ndarray
    home_attack: np.ndarray
    home_defense: np.ndarray
    away_attack: np.ndarray
    away_defense: np.ndarray
    
    # Mappings for prediction
    team_id_to_idx: Dict[int, int]
    league_id_to_idx: Dict[int, int]
    
    @classmethod
    def from_matches(cls, matches: List[MatchData]) -> "TrainingData":
        """Create training data from match list."""
        
        # Filter to matches with scores
        scored = [m for m in matches if m.has_scores]
        n = len(scored)
        
        if n == 0:
            raise ValueError("No matches with scores")
        
        # Build team/league indices
        leagues = sorted(set(m.league_idx for m in scored))
        league_id_to_idx = {lid: i for i, lid in enumerate(leagues)}
        
        # Count teams per league
        teams_per_league: Dict[int, set] = {l: set() for l in leagues}
        for m in scored:
            teams_per_league[m.league_idx].add(m.home_team_idx)
            teams_per_league[m.league_idx].add(m.away_team_idx)
        
        n_teams = {l: len(teams) for l, teams in teams_per_league.items()}
        
        # Build arrays
        league_idx = np.array([league_id_to_idx[m.league_idx] for m in scored])
        home_team_idx = np.array([m.home_team_idx for m in scored])
        away_team_idx = np.array([m.away_team_idx for m in scored])
        
        g1_home = np.array([m.g1_home for m in scored])
        g1_away = np.array([m.g1_away for m in scored])
        g2_home = np.array([m.g2_home for m in scored])
        g2_away = np.array([m.g2_away for m in scored])
        
        # Scale Elo (center and scale)
        home_elo = np.array([m.home_elo for m in scored])
        away_elo = np.array([m.away_elo for m in scored])
        elo_mean = 1500.0
        elo_sd = 200.0
        home_elo_scaled = (home_elo - elo_mean) / elo_sd
        away_elo_scaled = (away_elo - elo_mean) / elo_sd
        
        # Attack/defense features
        home_attack = np.array([m.home_attack_strength for m in scored])
        home_defense = np.array([m.home_defense_strength for m in scored])
        away_attack = np.array([m.away_attack_strength for m in scored])
        away_defense = np.array([m.away_defense_strength for m in scored])
        
        return cls(
            n_matches=n,
            n_leagues=len(leagues),
            n_teams_per_league=n_teams,
            league_idx=league_idx,
            home_team_idx=home_team_idx,
            away_team_idx=away_team_idx,
            g1_home=g1_home,
            g1_away=g1_away,
            g2_home=g2_home,
            g2_away=g2_away,
            home_elo_scaled=home_elo_scaled,
            away_elo_scaled=away_elo_scaled,
            home_attack=home_attack,
            home_defense=home_defense,
            away_attack=away_attack,
            away_defense=away_defense,
            team_id_to_idx={},  # TODO: populate
            league_id_to_idx=league_id_to_idx,
        )


class HalfGoalModel:
    """
    Hierarchical Poisson/NegBin model for half goals.
    
    Model v1: Simple hierarchical structure with:
    - League intercepts
    - Team attack/defense with sum-to-zero constraints
    - Second half effect
    - Home advantage
    - Optional Elo and rolling features
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize model.
        
        Args:
            config: Model configuration
        """
        self.config = config or DEFAULT_POISSON_CONFIG
        self.trace = None
        self.model = None
        self.data = None
    
    def build_model(self, data: TrainingData):
        """
        Build PyMC model.
        
        Args:
            data: Training data
        """
        import pymc as pm
        
        self.data = data
        priors = self.config.priors
        
        logger.info(f"Building {self.config.model_type} model with {data.n_matches} matches")
        
        with pm.Model() as model:
            # =========================================================
            # Hyperpriors (league-level)
            # =========================================================
            
            # Global intercept per half
            alpha_1 = pm.Normal(
                "alpha_1",
                mu=priors.intercept_mean,
                sigma=priors.intercept_sd,
            )
            alpha_2 = pm.Normal(
                "alpha_2",
                mu=priors.intercept_mean + priors.half2_effect_mean,
                sigma=priors.intercept_sd,
            )
            
            # Home advantage
            home_adv = pm.Normal(
                "home_advantage",
                mu=priors.home_advantage_mean,
                sigma=priors.home_advantage_sd,
            )
            
            # =========================================================
            # Team effects (hierarchical per league)
            # =========================================================
            
            # Team attack/defense SD (shared across leagues for now)
            attack_sd = pm.HalfNormal("attack_sd", sigma=priors.attack_sd)
            defense_sd = pm.HalfNormal("defense_sd", sigma=priors.defense_sd)
            
            # Total number of unique teams
            max_team_idx = int(max(data.home_team_idx.max(), data.away_team_idx.max()) + 1)
            
            # Team attack strengths (raw, will constrain via centering)
            attack_raw = pm.Normal(
                "attack_raw",
                mu=0,
                sigma=attack_sd,
                shape=max_team_idx,
            )
            
            # Team defense strengths
            defense_raw = pm.Normal(
                "defense_raw",
                mu=0,
                sigma=defense_sd,
                shape=max_team_idx,
            )
            
            # Center to approximate sum-to-zero
            attack = attack_raw - pm.math.mean(attack_raw)
            defense = defense_raw - pm.math.mean(defense_raw)
            
            # =========================================================
            # Feature coefficients
            # =========================================================
            
            if self.config.use_elo:
                beta_elo = pm.Normal(
                    "beta_elo",
                    mu=0,
                    sigma=priors.feature_coef_sd,
                )
            else:
                beta_elo = 0.0
            
            # =========================================================
            # Linear predictors (log scale)
            # =========================================================
            
            # First half - home team
            log_mu_1_home = (
                alpha_1
                + home_adv
                + attack[data.home_team_idx]
                - defense[data.away_team_idx]
                + beta_elo * data.home_elo_scaled
            )
            
            # First half - away team
            log_mu_1_away = (
                alpha_1
                + attack[data.away_team_idx]
                - defense[data.home_team_idx]
                + beta_elo * data.away_elo_scaled
            )
            
            # Second half - home team
            log_mu_2_home = (
                alpha_2
                + home_adv
                + attack[data.home_team_idx]
                - defense[data.away_team_idx]
                + beta_elo * data.home_elo_scaled
            )
            
            # Second half - away team
            log_mu_2_away = (
                alpha_2
                + attack[data.away_team_idx]
                - defense[data.home_team_idx]
                + beta_elo * data.away_elo_scaled
            )
            
            # Rates (must be positive)
            mu_1_home = pm.math.exp(log_mu_1_home)
            mu_1_away = pm.math.exp(log_mu_1_away)
            mu_2_home = pm.math.exp(log_mu_2_home)
            mu_2_away = pm.math.exp(log_mu_2_away)
            
            # =========================================================
            # Likelihood
            # =========================================================
            
            if self.config.model_type == "poisson":
                # Poisson likelihood
                g1_home = pm.Poisson("g1_home", mu=mu_1_home, observed=data.g1_home)
                g1_away = pm.Poisson("g1_away", mu=mu_1_away, observed=data.g1_away)
                g2_home = pm.Poisson("g2_home", mu=mu_2_home, observed=data.g2_home)
                g2_away = pm.Poisson("g2_away", mu=mu_2_away, observed=data.g2_away)
                
            elif self.config.model_type == "negbin":
                # Negative Binomial with shared overdispersion
                phi = pm.HalfNormal("phi", sigma=priors.negbin_phi_sd)
                
                g1_home = pm.NegativeBinomial(
                    "g1_home", mu=mu_1_home, alpha=phi, observed=data.g1_home
                )
                g1_away = pm.NegativeBinomial(
                    "g1_away", mu=mu_1_away, alpha=phi, observed=data.g1_away
                )
                g2_home = pm.NegativeBinomial(
                    "g2_home", mu=mu_2_home, alpha=phi, observed=data.g2_home
                )
                g2_away = pm.NegativeBinomial(
                    "g2_away", mu=mu_2_away, alpha=phi, observed=data.g2_away
                )
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
            
            # Store for later
            pm.Deterministic("mu_1_home", mu_1_home)
            pm.Deterministic("mu_1_away", mu_1_away)
            pm.Deterministic("mu_2_home", mu_2_home)
            pm.Deterministic("mu_2_away", mu_2_away)
        
        self.model = model
        logger.info("Model built successfully")
        return model
    
    def fit(self, data: TrainingData) -> az.InferenceData:
        """
        Fit model using NUTS.
        
        Args:
            data: Training data
            
        Returns:
            ArviZ InferenceData with posterior samples
        """
        import pymc as pm
        
        if self.model is None:
            self.build_model(data)
        
        logger.info(
            f"Sampling with {self.config.n_chains} chains, "
            f"{self.config.n_samples} samples, "
            f"target_accept={self.config.target_accept}"
        )
        
        with self.model:
            self.trace = pm.sample(
                draws=self.config.n_samples,
                tune=self.config.n_tune,
                chains=self.config.n_chains,
                target_accept=self.config.target_accept,
                random_seed=self.config.random_seed,
                return_inferencedata=True,
            )
        
        # Check diagnostics
        self._check_diagnostics()
        
        return self.trace
    
    def _check_diagnostics(self) -> Dict[str, Any]:
        """Check MCMC diagnostics and log warnings."""
        
        if self.trace is None:
            return {}
        
        summary = az.summary(self.trace)
        
        # Get actual values
        max_rhat = float(summary["r_hat"].max())
        min_ess_bulk = float(summary["ess_bulk"].min())
        min_ess_tail = float(summary["ess_tail"].min()) if "ess_tail" in summary else min_ess_bulk
        
        # Check R-hat
        rhat_issues = summary[summary["r_hat"] > 1.05]
        if len(rhat_issues) > 0:
            logger.warning(f"R-hat > 1.05 for {len(rhat_issues)} parameters (max={max_rhat:.3f})")
        
        # Check ESS
        ess_issues = summary[summary["ess_bulk"] < 400]
        if len(ess_issues) > 0:
            logger.warning(f"ESS < 400 for {len(ess_issues)} parameters (min={min_ess_bulk:.0f})")
        
        # Check divergences
        n_divergences = 0
        if hasattr(self.trace, "sample_stats"):
            divergences = self.trace.sample_stats.get("diverging", None)
            if divergences is not None:
                n_divergences = int(np.sum(divergences.values))
                if n_divergences > 0:
                    logger.warning(f"Found {n_divergences} divergent samples")
        
        diagnostics = {
            "n_divergences": n_divergences,
            "max_rhat": max_rhat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "n_rhat_issues": len(rhat_issues),
            "n_ess_issues": len(ess_issues),
            "is_healthy": n_divergences == 0 and max_rhat < 1.05 and min_ess_bulk >= 400,
        }
        
        return diagnostics
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get MCMC diagnostics for model evaluation.
        
        Returns:
            Dictionary with:
                - n_divergences: Number of divergent samples
                - max_rhat: Maximum R-hat across parameters
                - min_ess_bulk: Minimum bulk ESS
                - min_ess_tail: Minimum tail ESS  
                - is_healthy: True if no issues detected
        """
        return self._check_diagnostics()
    
    def get_posterior_summary(self) -> Dict[str, Any]:
        """
        Get posterior summary including home advantage credible interval.
        
        Returns:
            Dictionary with parameter summaries
        """
        if self.trace is None:
            raise ValueError("Model not fitted")
        
        posterior = self.trace.posterior
        
        # Home advantage (this is the posterior, not a fixed value!)
        home_adv = posterior["home_advantage"].values.flatten()
        
        # Convert from log scale to multiplicative effect
        home_adv_mult = np.exp(home_adv)
        
        return {
            "home_advantage": {
                "log_scale": {
                    "mean": float(np.mean(home_adv)),
                    "median": float(np.median(home_adv)),
                    "ci_5": float(np.percentile(home_adv, 5)),
                    "ci_95": float(np.percentile(home_adv, 95)),
                },
                "multiplicative": {
                    "mean": float(np.mean(home_adv_mult)),
                    "median": float(np.median(home_adv_mult)),
                    "ci_5": float(np.percentile(home_adv_mult, 5)),
                    "ci_95": float(np.percentile(home_adv_mult, 95)),
                    "interpretation": "Home team scores ~{:.0f}% more goals (median)".format(
                        (np.median(home_adv_mult) - 1) * 100
                    ),
                },
            },
            "alpha_1": {
                "mean": float(posterior["alpha_1"].values.mean()),
                "expected_goals_per_team": float(np.exp(posterior["alpha_1"].values.mean())),
            },
            "alpha_2": {
                "mean": float(posterior["alpha_2"].values.mean()),
                "expected_goals_per_team": float(np.exp(posterior["alpha_2"].values.mean())),
            },
        }
    
    def predict_match(
        self,
        home_team_idx: int,
        away_team_idx: int,
        home_elo_scaled: float = 0.0,
        away_elo_scaled: float = 0.0,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Predict match outcome using posterior samples.
        
        Args:
            home_team_idx: Home team index
            away_team_idx: Away team index
            home_elo_scaled: Scaled home Elo
            away_elo_scaled: Scaled away Elo
            n_samples: Number of Monte Carlo samples per posterior
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        if self.trace is None:
            raise ValueError("Model not fitted")
        
        # Extract posterior samples
        posterior = self.trace.posterior
        
        alpha_1 = posterior["alpha_1"].values.flatten()
        alpha_2 = posterior["alpha_2"].values.flatten()
        home_adv = posterior["home_advantage"].values.flatten()
        attack = posterior["attack_raw"].values  # (chains, draws, teams)
        defense = posterior["defense_raw"].values
        
        # Reshape for easier indexing
        attack = attack.reshape(-1, attack.shape[-1])  # (n_samples, n_teams)
        defense = defense.reshape(-1, defense.shape[-1])
        
        # Center (approximate sum-to-zero)
        attack = attack - attack.mean(axis=1, keepdims=True)
        defense = defense - defense.mean(axis=1, keepdims=True)
        
        # Get Elo coefficient if used
        if self.config.use_elo and "beta_elo" in posterior:
            beta_elo = posterior["beta_elo"].values.flatten()
        else:
            beta_elo = np.zeros_like(alpha_1)
        
        # Compute rates for each posterior sample
        log_mu_1_home = (
            alpha_1 + home_adv 
            + attack[:, home_team_idx] - defense[:, away_team_idx]
            + beta_elo * home_elo_scaled
        )
        log_mu_1_away = (
            alpha_1
            + attack[:, away_team_idx] - defense[:, home_team_idx]
            + beta_elo * away_elo_scaled
        )
        log_mu_2_home = (
            alpha_2 + home_adv
            + attack[:, home_team_idx] - defense[:, away_team_idx]
            + beta_elo * home_elo_scaled
        )
        log_mu_2_away = (
            alpha_2
            + attack[:, away_team_idx] - defense[:, home_team_idx]
            + beta_elo * away_elo_scaled
        )
        
        mu_1_home = np.exp(log_mu_1_home)
        mu_1_away = np.exp(log_mu_1_away)
        mu_2_home = np.exp(log_mu_2_home)
        mu_2_away = np.exp(log_mu_2_away)
        
        # Monte Carlo simulation for P(G2 > G1)
        p_2h_gt_1h_samples = []
        
        for i in range(len(mu_1_home)):
            # Sample goals from Poisson
            g1_h = np.random.poisson(mu_1_home[i], n_samples)
            g1_a = np.random.poisson(mu_1_away[i], n_samples)
            g2_h = np.random.poisson(mu_2_home[i], n_samples)
            g2_a = np.random.poisson(mu_2_away[i], n_samples)
            
            g1_total = g1_h + g1_a
            g2_total = g2_h + g2_a
            
            p = (g2_total > g1_total).mean()
            p_2h_gt_1h_samples.append(p)
        
        p_samples = np.array(p_2h_gt_1h_samples)
        
        return {
            "p_2h_gt_1h": float(np.mean(p_samples)),
            "p_2h_gt_1h_ci": (
                float(np.percentile(p_samples, 5)),
                float(np.percentile(p_samples, 95)),
            ),
            "p_2h_gt_1h_std": float(np.std(p_samples)),
            "mu_1_home": float(np.mean(mu_1_home)),
            "mu_1_away": float(np.mean(mu_1_away)),
            "mu_2_home": float(np.mean(mu_2_home)),
            "mu_2_away": float(np.mean(mu_2_away)),
            "expected_g1": float(np.mean(mu_1_home + mu_1_away)),
            "expected_g2": float(np.mean(mu_2_home + mu_2_away)),
        }
    
    def compute_loo(self) -> az.ELPDData:
        """
        Compute PSIS-LOO for model comparison.
        
        Returns:
            ArviZ ELPD data
        """
        if self.trace is None:
            raise ValueError("Model not fitted")
        
        with self.model:
            loo = az.loo(self.trace, pointwise=True)
        
        return loo
    
    def save(self, path: str) -> None:
        """Save fitted model to disk."""
        if self.trace is not None:
            self.trace.to_netcdf(path)
            logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, config: Optional[ModelConfig] = None) -> "HalfGoalModel":
        """Load fitted model from disk."""
        model = cls(config=config)
        model.trace = az.from_netcdf(path)
        logger.info(f"Model loaded from {path}")
        return model
