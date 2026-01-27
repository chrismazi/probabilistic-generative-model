"""
Elo rating system for team strength estimation.

Implements a soccer-specific Elo rating with:
- Home advantage adjustment
- Goal difference scaling
- Decay for time since last match
- Strict temporal ordering (no future leakage)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
from sqlalchemy import text

from src.db import get_session
from src.utils import get_logger, AsOfDate

logger = get_logger("features.elo")


# Elo parameters
INITIAL_RATING = 1500.0
K_FACTOR = 32.0  # Learning rate
HOME_ADVANTAGE = 100.0  # Points advantage for home team
GOAL_DIFF_WEIGHT = 11.0  # For goal difference adjustment
GOAL_DIFF_POWER = 0.5  # Diminishing returns on goal difference


@dataclass
class EloRating:
    """Team Elo rating at a point in time."""
    
    team_id: int
    rating: float
    matches_played: int
    last_match_date: Optional[datetime] = None
    
    @property
    def is_established(self) -> bool:
        """Rating is reliable after sufficient matches."""
        return self.matches_played >= 5


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for team A against team B.
    
    Returns probability of team A winning (0 to 1).
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def goal_difference_multiplier(goal_diff: int) -> float:
    """
    Multiplier based on goal difference.
    
    Larger victories = larger rating changes, with diminishing returns.
    """
    if goal_diff <= 1:
        return 1.0
    return (GOAL_DIFF_WEIGHT + goal_diff) ** GOAL_DIFF_POWER


def actual_score(goals_for: int, goals_against: int) -> float:
    """
    Convert match result to score (1 = win, 0.5 = draw, 0 = loss).
    """
    if goals_for > goals_against:
        return 1.0
    elif goals_for == goals_against:
        return 0.5
    return 0.0


class EloRatingSystem:
    """
    Elo rating system for soccer teams.
    
    Ratings are computed chronologically with no future data leakage.
    """
    
    def __init__(
        self,
        k_factor: float = K_FACTOR,
        home_advantage: float = HOME_ADVANTAGE,
        initial_rating: float = INITIAL_RATING,
    ):
        """
        Initialize Elo system.
        
        Args:
            k_factor: Learning rate for rating updates
            home_advantage: Home team rating boost
            initial_rating: Starting rating for new teams
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        
        # In-memory rating cache: team_id -> EloRating
        self._ratings: Dict[int, EloRating] = {}
    
    def get_rating(self, team_id: int) -> EloRating:
        """Get current rating for a team (or initialize if new)."""
        if team_id not in self._ratings:
            self._ratings[team_id] = EloRating(
                team_id=team_id,
                rating=self.initial_rating,
                matches_played=0,
            )
        return self._ratings[team_id]
    
    def update_ratings(
        self,
        home_team_id: int,
        away_team_id: int,
        home_goals: int,
        away_goals: int,
        match_date: datetime,
    ) -> Tuple[float, float]:
        """
        Update ratings after a match.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_date: Match datetime
            
        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        home = self.get_rating(home_team_id)
        away = self.get_rating(away_team_id)
        
        # Adjust home rating for home advantage
        home_adjusted = home.rating + self.home_advantage
        
        # Expected scores
        home_expected = expected_score(home_adjusted, away.rating)
        away_expected = 1.0 - home_expected
        
        # Actual scores
        home_actual = actual_score(home_goals, away_goals)
        away_actual = 1.0 - home_actual
        
        # Goal difference multiplier
        goal_diff = abs(home_goals - away_goals)
        gd_mult = goal_difference_multiplier(goal_diff)
        
        # Rating changes
        home_change = self.k_factor * gd_mult * (home_actual - home_expected)
        away_change = self.k_factor * gd_mult * (away_actual - away_expected)
        
        # Update ratings
        new_home_rating = home.rating + home_change
        new_away_rating = away.rating + away_change
        
        self._ratings[home_team_id] = EloRating(
            team_id=home_team_id,
            rating=new_home_rating,
            matches_played=home.matches_played + 1,
            last_match_date=match_date,
        )
        
        self._ratings[away_team_id] = EloRating(
            team_id=away_team_id,
            rating=new_away_rating,
            matches_played=away.matches_played + 1,
            last_match_date=match_date,
        )
        
        return new_home_rating, new_away_rating
    
    def process_matches_chronologically(
        self,
        league_id: int,
        up_to: Optional[AsOfDate] = None,
    ) -> None:
        """
        Process all matches for a league in chronological order.
        
        This builds the rating history up to a point in time.
        
        Args:
            league_id: League ID
            up_to: Process matches up to this date (exclusive)
        """
        query = """
            SELECT 
                m.id,
                m.kickoff_utc,
                m.home_team_id,
                m.away_team_id,
                s.ft_home,
                s.ft_away
            FROM matches m
            JOIN scores s ON m.id = s.match_id
            WHERE m.league_id = :league_id
              AND m.status = 'FINISHED'
              AND s.ft_home IS NOT NULL
        """
        
        params: dict = {"league_id": league_id}
        
        if up_to:
            query += " AND m.kickoff_utc < :up_to"
            params["up_to"] = up_to.datetime
        
        query += " ORDER BY m.kickoff_utc ASC"
        
        with get_session() as session:
            result = session.execute(text(query), params).fetchall()
        
        logger.info(f"Processing {len(result)} matches for Elo ratings")
        
        for row in result:
            match_id, kickoff, home_id, away_id, ft_home, ft_away = row
            
            self.update_ratings(
                home_team_id=home_id,
                away_team_id=away_id,
                home_goals=ft_home,
                away_goals=ft_away,
                match_date=kickoff,
            )
    
    def get_ratings_as_of(
        self,
        team_ids: list[int],
        league_id: int,
        as_of: AsOfDate,
    ) -> Dict[int, EloRating]:
        """
        Get ratings for teams as of a specific date.
        
        Reprocesses matches up to the date to ensure accuracy.
        
        Args:
            team_ids: List of team IDs
            league_id: League ID
            as_of: As-of date
            
        Returns:
            Dictionary of team_id -> EloRating
        """
        # Clear cache and reprocess
        self._ratings = {}
        self.process_matches_chronologically(league_id, up_to=as_of)
        
        # Return ratings for requested teams
        return {tid: self.get_rating(tid) for tid in team_ids}
    
    def predict_match(
        self,
        home_team_id: int,
        away_team_id: int,
    ) -> Dict[str, float]:
        """
        Predict match outcome based on current ratings.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            
        Returns:
            Dictionary with home_win, draw, away_win probabilities
        """
        home = self.get_rating(home_team_id)
        away = self.get_rating(away_team_id)
        
        # Adjust for home advantage
        home_adjusted = home.rating + self.home_advantage
        
        # Win probability
        home_win_prob = expected_score(home_adjusted, away.rating)
        
        # Approximate draw probability (empirical formula)
        rating_diff = abs(home_adjusted - away.rating)
        draw_prob = 0.28 * np.exp(-rating_diff / 400)
        
        # Adjust win probabilities
        home_win = home_win_prob * (1 - draw_prob)
        away_win = (1 - home_win_prob) * (1 - draw_prob)
        
        return {
            "home_win": home_win,
            "draw": draw_prob,
            "away_win": away_win,
            "home_rating": home.rating,
            "away_rating": away.rating,
            "home_established": home.is_established,
            "away_established": away.is_established,
        }


def get_elo_system(
    k_factor: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
) -> EloRatingSystem:
    """Get a configured Elo rating system."""
    return EloRatingSystem(
        k_factor=k_factor,
        home_advantage=home_advantage,
    )
