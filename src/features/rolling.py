"""
Rolling window feature computation.

Computes team-level features using ONLY historical data (strict as-of logic).
No data leakage is allowed - all features use matches before the current kickoff.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
from sqlalchemy import text

from src.config import settings
from src.constants import ROLLING_WINDOWS, MIN_MATCHES_DEFAULT
from src.db import get_session
from src.utils import get_logger, AsOfDate, safe_divide

logger = get_logger("features.rolling")


@dataclass
class TeamRollingFeatures:
    """Rolling window features for a team."""
    
    team_id: int
    as_of_date: date
    window_size: int
    matches_in_window: int
    
    # Goals scored/conceded
    goals_scored_avg: float = 0.0
    goals_conceded_avg: float = 0.0
    
    # Half-specific patterns (key for P(G2 > G1))
    goals_1h_scored_avg: float = 0.0
    goals_1h_conceded_avg: float = 0.0
    goals_2h_scored_avg: float = 0.0
    goals_2h_conceded_avg: float = 0.0
    
    # Rates
    clean_sheet_rate: float = 0.0
    failed_to_score_rate: float = 0.0
    
    # Half comparison rates
    rate_2h_gt_1h: float = 0.0  # Rate where team's matches had 2H > 1H
    rate_0_0_at_ht: float = 0.0  # Rate of 0-0 at half-time
    
    # Home/Away splits
    home_goals_avg: float = 0.0
    away_goals_avg: float = 0.0
    
    # Win/Draw/Loss rates
    win_rate: float = 0.0
    draw_rate: float = 0.0
    loss_rate: float = 0.0
    
    @property
    def has_minimum_history(self) -> bool:
        """Check if team has minimum required matches."""
        return self.matches_in_window >= settings.min_matches_required


@dataclass
class MatchFeatures:
    """Computed features for a single match."""
    
    match_id: int
    kickoff_utc: datetime
    league_id: int
    home_team_id: int
    away_team_id: int
    
    # Home team features
    home_features: TeamRollingFeatures
    
    # Away team features
    away_features: TeamRollingFeatures
    
    # League baseline features
    league_p_2h_gt_1h: float = 0.5
    league_avg_goals_1h: float = 0.0
    league_avg_goals_2h: float = 0.0
    
    # Derived features
    home_attack_strength: float = 1.0
    home_defense_strength: float = 1.0
    away_attack_strength: float = 1.0
    away_defense_strength: float = 1.0
    
    # Validity
    is_valid: bool = True
    invalid_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "match_id": self.match_id,
            "home_features": {
                "goals_scored_avg": self.home_features.goals_scored_avg,
                "goals_conceded_avg": self.home_features.goals_conceded_avg,
                "goals_1h_scored_avg": self.home_features.goals_1h_scored_avg,
                "goals_2h_scored_avg": self.home_features.goals_2h_scored_avg,
                "rate_2h_gt_1h": self.home_features.rate_2h_gt_1h,
                "matches_in_window": self.home_features.matches_in_window,
            },
            "away_features": {
                "goals_scored_avg": self.away_features.goals_scored_avg,
                "goals_conceded_avg": self.away_features.goals_conceded_avg,
                "goals_1h_scored_avg": self.away_features.goals_1h_scored_avg,
                "goals_2h_scored_avg": self.away_features.goals_2h_scored_avg,
                "rate_2h_gt_1h": self.away_features.rate_2h_gt_1h,
                "matches_in_window": self.away_features.matches_in_window,
            },
            "league": {
                "p_2h_gt_1h": self.league_p_2h_gt_1h,
                "avg_goals_1h": self.league_avg_goals_1h,
                "avg_goals_2h": self.league_avg_goals_2h,
            },
            "derived": {
                "home_attack": self.home_attack_strength,
                "home_defense": self.home_defense_strength,
                "away_attack": self.away_attack_strength,
                "away_defense": self.away_defense_strength,
            },
            "is_valid": self.is_valid,
            "invalid_reason": self.invalid_reason,
        }


class RollingFeatureBuilder:
    """
    Builds rolling window features for teams.
    
    CRITICAL: All features are computed using ONLY matches before the as_of date.
    This prevents data leakage and ensures honest backtesting.
    """
    
    def __init__(self, window_days: int = 180):
        """
        Initialize feature builder.
        
        Args:
            window_days: Number of days to look back for features
        """
        self.window_days = window_days
    
    def get_team_features(
        self,
        team_id: int,
        league_id: int,
        as_of: AsOfDate,
        window_size: int = 10,
    ) -> TeamRollingFeatures:
        """
        Compute rolling features for a team.
        
        Args:
            team_id: Team ID
            league_id: League ID (for league-specific features)
            as_of: As-of date (features use only matches before this)
            window_size: Number of recent matches to consider
            
        Returns:
            TeamRollingFeatures with computed values
        """
        with get_session() as session:
            # Get recent matches for this team BEFORE as_of date
            result = session.execute(
                text("""
                    SELECT 
                        m.id,
                        m.kickoff_utc,
                        m.home_team_id,
                        m.away_team_id,
                        s.ht_home,
                        s.ht_away,
                        s.ft_home,
                        s.ft_away
                    FROM matches m
                    JOIN scores s ON m.id = s.match_id
                    WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
                      AND m.league_id = :league_id
                      AND m.status = 'FINISHED'
                      AND m.kickoff_utc < :as_of
                      AND s.ht_home IS NOT NULL
                      AND s.ft_home IS NOT NULL
                    ORDER BY m.kickoff_utc DESC
                    LIMIT :window_size
                """),
                {
                    "team_id": team_id,
                    "league_id": league_id,
                    "as_of": as_of.datetime,
                    "window_size": window_size,
                }
            ).fetchall()
        
        matches = list(result)
        n = len(matches)
        
        if n == 0:
            return TeamRollingFeatures(
                team_id=team_id,
                as_of_date=as_of.date,
                window_size=window_size,
                matches_in_window=0,
            )
        
        # Compute aggregates
        goals_scored = []
        goals_conceded = []
        goals_1h_scored = []
        goals_1h_conceded = []
        goals_2h_scored = []
        goals_2h_conceded = []
        clean_sheets = []
        failed_to_score = []
        total_g1 = []
        total_g2 = []
        ht_0_0 = []
        wins = []
        draws = []
        losses = []
        home_goals = []
        away_goals = []
        
        for row in matches:
            match_id, kickoff, home_id, away_id, ht_h, ht_a, ft_h, ft_a = row
            
            is_home = (home_id == team_id)
            
            if is_home:
                scored = ft_h
                conceded = ft_a
                scored_1h = ht_h
                conceded_1h = ht_a
                scored_2h = ft_h - ht_h
                conceded_2h = ft_a - ht_a
                home_goals.append(scored)
            else:
                scored = ft_a
                conceded = ft_h
                scored_1h = ht_a
                conceded_1h = ht_h
                scored_2h = ft_a - ht_a
                conceded_2h = ft_h - ht_h
                away_goals.append(scored)
            
            goals_scored.append(scored)
            goals_conceded.append(conceded)
            goals_1h_scored.append(scored_1h)
            goals_1h_conceded.append(conceded_1h)
            goals_2h_scored.append(scored_2h)
            goals_2h_conceded.append(conceded_2h)
            
            clean_sheets.append(1 if conceded == 0 else 0)
            failed_to_score.append(1 if scored == 0 else 0)
            
            # Total match goals by half
            g1 = ht_h + ht_a
            g2 = (ft_h - ht_h) + (ft_a - ht_a)
            total_g1.append(g1)
            total_g2.append(g2)
            
            ht_0_0.append(1 if ht_h == 0 and ht_a == 0 else 0)
            
            # Win/draw/loss
            if scored > conceded:
                wins.append(1)
                draws.append(0)
                losses.append(0)
            elif scored == conceded:
                wins.append(0)
                draws.append(1)
                losses.append(0)
            else:
                wins.append(0)
                draws.append(0)
                losses.append(1)
        
        # Compute averages with safe division
        return TeamRollingFeatures(
            team_id=team_id,
            as_of_date=as_of.date,
            window_size=window_size,
            matches_in_window=n,
            goals_scored_avg=np.mean(goals_scored),
            goals_conceded_avg=np.mean(goals_conceded),
            goals_1h_scored_avg=np.mean(goals_1h_scored),
            goals_1h_conceded_avg=np.mean(goals_1h_conceded),
            goals_2h_scored_avg=np.mean(goals_2h_scored),
            goals_2h_conceded_avg=np.mean(goals_2h_conceded),
            clean_sheet_rate=np.mean(clean_sheets),
            failed_to_score_rate=np.mean(failed_to_score),
            rate_2h_gt_1h=np.mean([1 if g2 > g1 else 0 for g1, g2 in zip(total_g1, total_g2)]),
            rate_0_0_at_ht=np.mean(ht_0_0),
            home_goals_avg=np.mean(home_goals) if home_goals else 0.0,
            away_goals_avg=np.mean(away_goals) if away_goals else 0.0,
            win_rate=np.mean(wins),
            draw_rate=np.mean(draws),
            loss_rate=np.mean(losses),
        )
    
    def get_league_baseline(
        self,
        league_id: int,
        as_of: AsOfDate,
        lookback_days: int = 365,
    ) -> dict:
        """
        Compute league baseline features.
        
        Args:
            league_id: League ID
            as_of: As-of date
            lookback_days: Days to look back
            
        Returns:
            Dictionary with league baselines
        """
        cutoff = as_of.datetime - timedelta(days=lookback_days)
        
        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT 
                        AVG(s.ht_home + s.ht_away) as avg_goals_1h,
                        AVG((s.ft_home - s.ht_home) + (s.ft_away - s.ht_away)) as avg_goals_2h,
                        AVG(CASE WHEN 
                            (s.ft_home - s.ht_home) + (s.ft_away - s.ht_away) > s.ht_home + s.ht_away 
                            THEN 1.0 ELSE 0.0 END) as p_2h_gt_1h,
                        COUNT(*) as match_count
                    FROM matches m
                    JOIN scores s ON m.id = s.match_id
                    WHERE m.league_id = :league_id
                      AND m.status = 'FINISHED'
                      AND m.kickoff_utc < :as_of
                      AND m.kickoff_utc >= :cutoff
                      AND s.ht_home IS NOT NULL
                """),
                {
                    "league_id": league_id,
                    "as_of": as_of.datetime,
                    "cutoff": cutoff,
                }
            ).fetchone()
        
        if result and result[3] > 0:  # match_count > 0
            return {
                "avg_goals_1h": float(result[0] or 0),
                "avg_goals_2h": float(result[1] or 0),
                "p_2h_gt_1h": float(result[2] or 0.5),
                "match_count": int(result[3]),
            }
        
        # Default values if no data
        return {
            "avg_goals_1h": 1.2,
            "avg_goals_2h": 1.4,
            "p_2h_gt_1h": 0.5,
            "match_count": 0,
        }
    
    def build_match_features(
        self,
        match_id: int,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        kickoff_utc: datetime,
        window_size: int = 10,
    ) -> MatchFeatures:
        """
        Build all features for a match.
        
        Args:
            match_id: Match ID
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: League ID
            kickoff_utc: Kickoff time (features computed before this)
            window_size: Rolling window size
            
        Returns:
            MatchFeatures with all computed values
        """
        as_of = AsOfDate(kickoff_utc)
        
        # Get team features
        home_features = self.get_team_features(
            home_team_id, league_id, as_of, window_size
        )
        away_features = self.get_team_features(
            away_team_id, league_id, as_of, window_size
        )
        
        # Get league baseline
        league = self.get_league_baseline(league_id, as_of)
        
        # Check validity
        is_valid = True
        invalid_reason = None
        
        if not home_features.has_minimum_history:
            is_valid = False
            invalid_reason = f"Home team has only {home_features.matches_in_window} matches"
        elif not away_features.has_minimum_history:
            is_valid = False
            invalid_reason = f"Away team has only {away_features.matches_in_window} matches"
        
        # Compute attack/defense strengths relative to league average
        league_avg_scored = league["avg_goals_1h"] + league["avg_goals_2h"]
        if league_avg_scored > 0:
            home_attack = safe_divide(
                home_features.goals_scored_avg, 
                league_avg_scored / 2,  # Per team average
                default=1.0
            )
            away_attack = safe_divide(
                away_features.goals_scored_avg,
                league_avg_scored / 2,
                default=1.0
            )
            home_defense = safe_divide(
                home_features.goals_conceded_avg,
                league_avg_scored / 2,
                default=1.0
            )
            away_defense = safe_divide(
                away_features.goals_conceded_avg,
                league_avg_scored / 2,
                default=1.0
            )
        else:
            home_attack = away_attack = home_defense = away_defense = 1.0
        
        return MatchFeatures(
            match_id=match_id,
            kickoff_utc=kickoff_utc,
            league_id=league_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_features=home_features,
            away_features=away_features,
            league_p_2h_gt_1h=league["p_2h_gt_1h"],
            league_avg_goals_1h=league["avg_goals_1h"],
            league_avg_goals_2h=league["avg_goals_2h"],
            home_attack_strength=home_attack,
            home_defense_strength=home_defense,
            away_attack_strength=away_attack,
            away_defense_strength=away_defense,
            is_valid=is_valid,
            invalid_reason=invalid_reason,
        )


def get_feature_builder(window_days: int = 180) -> RollingFeatureBuilder:
    """Get a configured feature builder."""
    return RollingFeatureBuilder(window_days=window_days)
