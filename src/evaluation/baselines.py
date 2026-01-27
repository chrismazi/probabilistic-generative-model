"""
Baseline models for comparison.

Provides naive baselines that the model MUST beat:
1. "Always 2H" - Always predict P(G2 > G1) = 1
2. "League prior" - Use historical league rate
3. "50/50" - Random baseline

These establish the minimum performance bar.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sqlalchemy import text

from src.db import get_session
from src.utils import get_logger

logger = get_logger("evaluation.baselines")


@dataclass
class BaselinePrediction:
    """Prediction from a baseline model."""
    
    model_name: str
    p_2h_gt_1h: float
    
    # For constant baselines
    is_constant: bool = True


class AlwaysSecondHalfBaseline:
    """
    Baseline: Always predict 2H has more goals.
    
    Predicts P(G2 > G1) = 1.0 for all matches.
    This is what a naive bettor might do.
    """
    
    NAME = "always_2h"
    
    def predict(self, match_id: int) -> BaselinePrediction:
        """Predict P(G2 > G1) = 1.0."""
        return BaselinePrediction(
            model_name=self.NAME,
            p_2h_gt_1h=1.0,
            is_constant=True,
        )
    
    def predict_batch(self, n_matches: int) -> np.ndarray:
        """Return array of 1.0 predictions."""
        return np.ones(n_matches)


class RandomBaseline:
    """
    Baseline: Random 50/50.
    
    Predicts P(G2 > G1) = 0.5 for all matches.
    This is the "no skill" baseline.
    """
    
    NAME = "random_50_50"
    
    def predict(self, match_id: int) -> BaselinePrediction:
        """Predict P(G2 > G1) = 0.5."""
        return BaselinePrediction(
            model_name=self.NAME,
            p_2h_gt_1h=0.5,
            is_constant=True,
        )
    
    def predict_batch(self, n_matches: int) -> np.ndarray:
        """Return array of 0.5 predictions."""
        return np.full(n_matches, 0.5)


class LeaguePriorBaseline:
    """
    Baseline: Use historical league rate.
    
    For each league, predicts P(G2 > G1) = historical rate.
    This uses aggregate information but no match-specific features.
    """
    
    NAME = "league_prior"
    
    def __init__(self):
        self._cache: Dict[int, float] = {}
    
    def get_league_prior(self, league_id: int, season: Optional[str] = None) -> float:
        """
        Get historical P(G2 > G1) for a league.
        
        Args:
            league_id: League ID
            season: Optional season filter
            
        Returns:
            Historical rate (0-1)
        """
        cache_key = (league_id, season)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        query = """
            SELECT 
                AVG(CASE WHEN 
                    (s.ft_home - s.ht_home) + (s.ft_away - s.ht_away) > s.ht_home + s.ht_away 
                    THEN 1.0 ELSE 0.0 END) as p_2h_gt_1h
            FROM matches m
            JOIN scores s ON m.id = s.match_id
            WHERE m.league_id = :league_id
              AND m.status = 'FINISHED'
              AND s.ht_home IS NOT NULL
        """
        
        params: dict = {"league_id": league_id}
        if season:
            query += " AND m.season = :season"
            params["season"] = season
        
        with get_session() as session:
            result = session.execute(text(query), params).fetchone()
        
        rate = float(result[0]) if result and result[0] else 0.5
        self._cache[cache_key] = rate
        
        return rate
    
    def predict(self, match_id: int, league_id: int) -> BaselinePrediction:
        """Predict using league prior."""
        prior = self.get_league_prior(league_id)
        return BaselinePrediction(
            model_name=self.NAME,
            p_2h_gt_1h=prior,
            is_constant=False,  # Varies by league
        )
    
    def predict_batch(self, league_ids: np.ndarray) -> np.ndarray:
        """Return array of league prior predictions."""
        return np.array([self.get_league_prior(lid) for lid in league_ids])


class ClimatologyBaseline:
    """
    Baseline: Global climatology.
    
    Predicts P(G2 > G1) = overall historical rate across all leagues.
    """
    
    NAME = "climatology"
    
    def __init__(self):
        self._global_rate: Optional[float] = None
    
    def get_global_rate(self) -> float:
        """Get global P(G2 > G1) rate."""
        if self._global_rate is not None:
            return self._global_rate
        
        query = """
            SELECT 
                AVG(CASE WHEN 
                    (s.ft_home - s.ht_home) + (s.ft_away - s.ht_away) > s.ht_home + s.ht_away 
                    THEN 1.0 ELSE 0.0 END) as p_2h_gt_1h
            FROM matches m
            JOIN scores s ON m.id = s.match_id
            WHERE m.status = 'FINISHED'
              AND s.ht_home IS NOT NULL
        """
        
        with get_session() as session:
            result = session.execute(text(query)).fetchone()
        
        self._global_rate = float(result[0]) if result and result[0] else 0.5
        return self._global_rate
    
    def predict(self, match_id: int) -> BaselinePrediction:
        """Predict using global climatology."""
        return BaselinePrediction(
            model_name=self.NAME,
            p_2h_gt_1h=self.get_global_rate(),
            is_constant=True,
        )
    
    def predict_batch(self, n_matches: int) -> np.ndarray:
        """Return array of climatology predictions."""
        return np.full(n_matches, self.get_global_rate())


def get_all_baselines() -> Dict[str, object]:
    """Get all baseline models."""
    return {
        "always_2h": AlwaysSecondHalfBaseline(),
        "random": RandomBaseline(),
        "league_prior": LeaguePriorBaseline(),
        "climatology": ClimatologyBaseline(),
    }
