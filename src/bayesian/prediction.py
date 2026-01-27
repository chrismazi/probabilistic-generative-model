"""
Prediction module for generating match predictions.

Converts model posteriors into actionable predictions:
- P(G2 > G1)
- Credible intervals
- Expected goals per half
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import numpy as np
from sqlalchemy import text

from src.bayesian.model import HalfGoalModel, MatchData, TrainingData
from src.bayesian.priors import ModelConfig
from src.db import get_session
from src.features import get_feature_builder, get_elo_system
from src.utils import get_logger, AsOfDate

logger = get_logger("bayesian.prediction")


@dataclass
class MatchPrediction:
    """Prediction for a single match."""
    
    match_id: int
    model_version: str
    created_at: datetime
    
    # Core prediction
    p_2h_gt_1h: float
    p_2h_gt_1h_ci_low: float
    p_2h_gt_1h_ci_high: float
    
    # Other probabilities
    p_1h_gt_2h: Optional[float] = None
    p_equal: Optional[float] = None
    
    # Expected values
    expected_g1: float = 0.0
    expected_g2: float = 0.0
    
    # Uncertainty
    entropy: float = 0.0
    
    # Metadata
    is_valid: bool = True
    invalid_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "model_version": self.model_version,
            "p_2h_gt_1h": self.p_2h_gt_1h,
            "p_2h_gt_1h_ci": [self.p_2h_gt_1h_ci_low, self.p_2h_gt_1h_ci_high],
            "expected_g1": self.expected_g1,
            "expected_g2": self.expected_g2,
            "is_valid": self.is_valid,
        }


def compute_entropy(p: float) -> float:
    """Compute binary entropy for probability."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class Predictor:
    """
    Generates predictions from fitted model.
    
    Responsibilities:
    - Load model
    - Prepare match features
    - Generate predictions with uncertainty
    - Store predictions
    """
    
    MODEL_VERSION = "v1.0.0"
    
    def __init__(
        self,
        model: HalfGoalModel,
        feature_builder=None,
        elo_system=None,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Fitted HalfGoalModel
            feature_builder: Feature builder for match features
            elo_system: Elo rating system
        """
        self.model = model
        self.feature_builder = feature_builder or get_feature_builder()
        self.elo_system = elo_system or get_elo_system()
    
    def predict_match(
        self,
        match_id: int,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        kickoff_utc: datetime,
        home_team_idx: int,
        away_team_idx: int,
    ) -> MatchPrediction:
        """
        Generate prediction for a single match.
        
        Args:
            match_id: Match ID
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: League ID
            kickoff_utc: Kickoff time
            home_team_idx: Home team index in model
            away_team_idx: Away team index in model
            
        Returns:
            MatchPrediction with probabilities and uncertainty
        """
        # Get features
        as_of = AsOfDate(kickoff_utc)
        features = self.feature_builder.build_match_features(
            match_id=match_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id,
            kickoff_utc=kickoff_utc,
        )
        
        # Check validity
        if not features.is_valid:
            return MatchPrediction(
                match_id=match_id,
                model_version=self.MODEL_VERSION,
                created_at=datetime.now(),
                p_2h_gt_1h=features.league_p_2h_gt_1h,  # Fallback to league prior
                p_2h_gt_1h_ci_low=0.3,
                p_2h_gt_1h_ci_high=0.7,
                is_valid=False,
                invalid_reason=features.invalid_reason,
            )
        
        # Get Elo-scaled values
        home_elo = features.home_features.matches_in_window  # Placeholder
        away_elo = features.away_features.matches_in_window
        home_elo_scaled = (home_elo - 1500) / 200 if home_elo else 0.0
        away_elo_scaled = (away_elo - 1500) / 200 if away_elo else 0.0
        
        # Get model prediction
        try:
            pred = self.model.predict_match(
                home_team_idx=home_team_idx,
                away_team_idx=away_team_idx,
                home_elo_scaled=home_elo_scaled,
                away_elo_scaled=away_elo_scaled,
            )
        except Exception as e:
            logger.error(f"Prediction failed for match {match_id}: {e}")
            return MatchPrediction(
                match_id=match_id,
                model_version=self.MODEL_VERSION,
                created_at=datetime.now(),
                p_2h_gt_1h=features.league_p_2h_gt_1h,
                p_2h_gt_1h_ci_low=0.3,
                p_2h_gt_1h_ci_high=0.7,
                is_valid=False,
                invalid_reason=str(e),
            )
        
        # Compute entropy
        entropy = compute_entropy(pred["p_2h_gt_1h"])
        
        return MatchPrediction(
            match_id=match_id,
            model_version=self.MODEL_VERSION,
            created_at=datetime.now(),
            p_2h_gt_1h=pred["p_2h_gt_1h"],
            p_2h_gt_1h_ci_low=pred["p_2h_gt_1h_ci"][0],
            p_2h_gt_1h_ci_high=pred["p_2h_gt_1h_ci"][1],
            expected_g1=pred["expected_g1"],
            expected_g2=pred["expected_g2"],
            entropy=entropy,
            is_valid=True,
        )
    
    def store_prediction(self, prediction: MatchPrediction) -> None:
        """Store prediction to database."""
        
        with get_session() as session:
            # Get or create model version
            model_version_id = self._get_or_create_model_version(session)
            
            session.execute(
                text("""
                    INSERT INTO predictions (
                        match_id, model_version_id, created_at,
                        p_2h_gt_1h, p_2h_gt_1h_ci_low, p_2h_gt_1h_ci_high,
                        entropy, explanation
                    )
                    VALUES (
                        :match_id, :model_version_id, :created_at,
                        :p, :ci_low, :ci_high,
                        :entropy, :explanation
                    )
                    ON CONFLICT (match_id, model_version_id) DO UPDATE SET
                        p_2h_gt_1h = EXCLUDED.p_2h_gt_1h,
                        p_2h_gt_1h_ci_low = EXCLUDED.p_2h_gt_1h_ci_low,
                        p_2h_gt_1h_ci_high = EXCLUDED.p_2h_gt_1h_ci_high,
                        entropy = EXCLUDED.entropy,
                        created_at = EXCLUDED.created_at
                """),
                {
                    "match_id": prediction.match_id,
                    "model_version_id": model_version_id,
                    "created_at": prediction.created_at,
                    "p": prediction.p_2h_gt_1h,
                    "ci_low": prediction.p_2h_gt_1h_ci_low,
                    "ci_high": prediction.p_2h_gt_1h_ci_high,
                    "entropy": prediction.entropy,
                    "explanation": json.dumps(prediction.to_dict()),
                }
            )
    
    def _get_or_create_model_version(self, session) -> int:
        """Get or create model version record."""
        
        result = session.execute(
            text("SELECT id FROM model_versions WHERE version = :version"),
            {"version": self.MODEL_VERSION}
        ).fetchone()
        
        if result:
            return result[0]
        
        result = session.execute(
            text("""
                INSERT INTO model_versions (version, model_type, config)
                VALUES (:version, :model_type, :config)
                RETURNING id
            """),
            {
                "version": self.MODEL_VERSION,
                "model_type": self.model.config.model_type,
                "config": json.dumps({
                    "model_type": self.model.config.model_type,
                    "n_samples": self.model.config.n_samples,
                    "n_chains": self.model.config.n_chains,
                }),
            }
        )
        return result.fetchone()[0]
