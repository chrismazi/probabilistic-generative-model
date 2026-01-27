"""
Feature orchestration module.

Coordinates feature building for matches:
- Rolling window features
- Elo ratings  
- League baselines
- Minimum history enforcement
- Storage to database
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import text

from src.config import settings
from src.db import get_session
from src.features.rolling import (
    RollingFeatureBuilder,
    MatchFeatures,
    get_feature_builder,
)
from src.features.elo import EloRatingSystem, get_elo_system
from src.utils import get_logger, AsOfDate
import json

logger = get_logger("features.builder")


class FeatureOrchestrator:
    """
    Orchestrates feature building for matches.
    
    Responsibilities:
    - Build features for upcoming matches
    - Store computed features
    - Enforce minimum history requirements
    - Track feature versions
    """
    
    FEATURE_VERSION = "v1.0"
    
    def __init__(
        self,
        rolling_builder: Optional[RollingFeatureBuilder] = None,
        elo_system: Optional[EloRatingSystem] = None,
        window_size: int = 10,
    ):
        """
        Initialize orchestrator.
        
        Args:
            rolling_builder: Rolling feature builder
            elo_system: Elo rating system
            window_size: Rolling window size for features
        """
        self.rolling_builder = rolling_builder or get_feature_builder()
        self.elo_system = elo_system or get_elo_system()
        self.window_size = window_size
    
    def build_features_for_match(
        self,
        match_id: int,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        kickoff_utc: datetime,
    ) -> MatchFeatures:
        """
        Build all features for a single match.
        
        Args:
            match_id: Match ID
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: League ID
            kickoff_utc: Kickoff time
            
        Returns:
            MatchFeatures with all computed values
        """
        logger.debug(f"Building features for match {match_id}")
        
        # Build rolling features
        features = self.rolling_builder.build_match_features(
            match_id=match_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id,
            kickoff_utc=kickoff_utc,
            window_size=self.window_size,
        )
        
        return features
    
    def build_features_for_league(
        self,
        league_id: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        only_upcoming: bool = False,
    ) -> List[MatchFeatures]:
        """
        Build features for all matches in a league.
        
        Args:
            league_id: League ID
            date_from: Start date filter
            date_to: End date filter
            only_upcoming: Only process scheduled matches
            
        Returns:
            List of MatchFeatures
        """
        # Build query
        query = """
            SELECT 
                m.id,
                m.home_team_id,
                m.away_team_id,
                m.kickoff_utc
            FROM matches m
            WHERE m.league_id = :league_id
        """
        
        params: dict = {"league_id": league_id}
        
        if only_upcoming:
            query += " AND m.status IN ('SCHEDULED', 'TIMED')"
        
        if date_from:
            query += " AND m.kickoff_utc >= :date_from"
            params["date_from"] = date_from
        
        if date_to:
            query += " AND m.kickoff_utc <= :date_to"
            params["date_to"] = date_to
        
        query += " ORDER BY m.kickoff_utc ASC"
        
        with get_session() as session:
            result = session.execute(text(query), params).fetchall()
        
        logger.info(f"Building features for {len(result)} matches in league {league_id}")
        
        features_list = []
        
        for row in result:
            match_id, home_id, away_id, kickoff = row
            
            try:
                features = self.build_features_for_match(
                    match_id=match_id,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    league_id=league_id,
                    kickoff_utc=kickoff,
                )
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"Failed to build features for match {match_id}: {e}")
        
        valid_count = sum(1 for f in features_list if f.is_valid)
        logger.info(f"Built {len(features_list)} features, {valid_count} valid")
        
        return features_list
    
    def store_features(self, features: MatchFeatures) -> None:
        """
        Store computed features to database.
        
        Args:
            features: MatchFeatures to store
        """
        feature_dict = features.to_dict()
        
        with get_session() as session:
            session.execute(
                text("""
                    INSERT INTO match_features (
                        match_id, feature_version, computed_at,
                        home_features, away_features, league_features,
                        home_match_count, away_match_count, is_valid
                    )
                    VALUES (
                        :match_id, :version, NOW(),
                        :home_features, :away_features, :league_features,
                        :home_count, :away_count, :is_valid
                    )
                    ON CONFLICT (match_id, feature_version) DO UPDATE SET
                        home_features = EXCLUDED.home_features,
                        away_features = EXCLUDED.away_features,
                        league_features = EXCLUDED.league_features,
                        home_match_count = EXCLUDED.home_match_count,
                        away_match_count = EXCLUDED.away_match_count,
                        is_valid = EXCLUDED.is_valid,
                        computed_at = NOW()
                """),
                {
                    "match_id": features.match_id,
                    "version": self.FEATURE_VERSION,
                    "home_features": json.dumps(feature_dict["home_features"]),
                    "away_features": json.dumps(feature_dict["away_features"]),
                    "league_features": json.dumps(feature_dict["league"]),
                    "home_count": features.home_features.matches_in_window,
                    "away_count": features.away_features.matches_in_window,
                    "is_valid": features.is_valid,
                }
            )
    
    def build_and_store_for_league(
        self,
        league_id: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        only_upcoming: bool = False,
    ) -> dict:
        """
        Build and store features for a league.
        
        Args:
            league_id: League ID
            date_from: Start date
            date_to: End date
            only_upcoming: Only upcoming matches
            
        Returns:
            Summary statistics
        """
        features_list = self.build_features_for_league(
            league_id=league_id,
            date_from=date_from,
            date_to=date_to,
            only_upcoming=only_upcoming,
        )
        
        stored = 0
        errors = 0
        
        for features in features_list:
            try:
                self.store_features(features)
                stored += 1
            except Exception as e:
                logger.error(f"Failed to store features for match {features.match_id}: {e}")
                errors += 1
        
        return {
            "total": len(features_list),
            "stored": stored,
            "valid": sum(1 for f in features_list if f.is_valid),
            "invalid": sum(1 for f in features_list if not f.is_valid),
            "errors": errors,
        }


def get_orchestrator(window_size: int = 10) -> FeatureOrchestrator:
    """Get a configured feature orchestrator."""
    return FeatureOrchestrator(window_size=window_size)
