"""
Tests for feature engineering.

Tests for:
- Rolling window features
- Elo rating system
- As-of date logic (no leakage)
- Minimum history requirements
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from src.features.rolling import (
    TeamRollingFeatures,
    MatchFeatures,
    RollingFeatureBuilder,
)
from src.features.elo import (
    EloRating,
    EloRatingSystem,
    expected_score,
    actual_score,
    goal_difference_multiplier,
)
from src.utils import AsOfDate


class TestTeamRollingFeatures:
    """Test TeamRollingFeatures dataclass."""
    
    def test_has_minimum_history_true(self):
        """Team with enough matches has minimum history."""
        features = TeamRollingFeatures(
            team_id=1,
            as_of_date=datetime.now().date(),
            window_size=10,
            matches_in_window=6,
        )
        
        # Default min is 5
        assert features.has_minimum_history is True
    
    def test_has_minimum_history_false(self):
        """Team with few matches lacks minimum history."""
        features = TeamRollingFeatures(
            team_id=1,
            as_of_date=datetime.now().date(),
            window_size=10,
            matches_in_window=3,
        )
        
        assert features.has_minimum_history is False
    
    def test_empty_features(self):
        """Test default values for empty features."""
        features = TeamRollingFeatures(
            team_id=1,
            as_of_date=datetime.now().date(),
            window_size=10,
            matches_in_window=0,
        )
        
        assert features.goals_scored_avg == 0.0
        assert features.rate_2h_gt_1h == 0.0


class TestMatchFeatures:
    """Test MatchFeatures dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        home = TeamRollingFeatures(
            team_id=1,
            as_of_date=datetime.now().date(),
            window_size=10,
            matches_in_window=8,
            goals_scored_avg=1.5,
        )
        away = TeamRollingFeatures(
            team_id=2,
            as_of_date=datetime.now().date(),
            window_size=10,
            matches_in_window=7,
            goals_scored_avg=1.2,
        )
        
        features = MatchFeatures(
            match_id=123,
            kickoff_utc=datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc),
            league_id=1,
            home_team_id=1,
            away_team_id=2,
            home_features=home,
            away_features=away,
        )
        
        d = features.to_dict()
        
        assert d["match_id"] == 123
        assert d["home_features"]["goals_scored_avg"] == 1.5
        assert d["away_features"]["goals_scored_avg"] == 1.2
        assert d["is_valid"] is True
    
    def test_invalid_reason(self):
        """Test invalid match features."""
        home = TeamRollingFeatures(
            team_id=1,
            as_of_date=datetime.now().date(),
            window_size=10,
            matches_in_window=2,  # Too few
        )
        away = TeamRollingFeatures(
            team_id=2,
            as_of_date=datetime.now().date(),
            window_size=10,
            matches_in_window=8,
        )
        
        features = MatchFeatures(
            match_id=123,
            kickoff_utc=datetime.now(timezone.utc),
            league_id=1,
            home_team_id=1,
            away_team_id=2,
            home_features=home,
            away_features=away,
            is_valid=False,
            invalid_reason="Home team has only 2 matches",
        )
        
        assert features.is_valid is False
        assert "Home team" in features.invalid_reason


class TestEloHelperFunctions:
    """Test Elo helper functions."""
    
    def test_expected_score_equal_ratings(self):
        """Equal ratings should give 50% win probability."""
        prob = expected_score(1500, 1500)
        assert abs(prob - 0.5) < 0.001
    
    def test_expected_score_higher_rating(self):
        """Higher rated team should have higher win probability."""
        prob = expected_score(1600, 1400)
        assert prob > 0.5
        assert prob < 1.0
    
    def test_expected_score_much_higher_rating(self):
        """Much higher rating should give near-certain win."""
        prob = expected_score(1800, 1200)
        assert prob > 0.9
    
    def test_actual_score_win(self):
        """Win returns 1.0."""
        assert actual_score(3, 1) == 1.0
    
    def test_actual_score_draw(self):
        """Draw returns 0.5."""
        assert actual_score(1, 1) == 0.5
    
    def test_actual_score_loss(self):
        """Loss returns 0.0."""
        assert actual_score(0, 2) == 0.0
    
    def test_goal_diff_multiplier_close(self):
        """Close games have multiplier of 1."""
        assert goal_difference_multiplier(0) == 1.0
        assert goal_difference_multiplier(1) == 1.0
    
    def test_goal_diff_multiplier_blowout(self):
        """Large goal differences have higher multiplier."""
        mult_3 = goal_difference_multiplier(3)
        mult_5 = goal_difference_multiplier(5)
        
        assert mult_3 > 1.0
        assert mult_5 > mult_3


class TestEloRatingSystem:
    """Test EloRatingSystem."""
    
    def test_initial_rating(self):
        """New teams start at 1500."""
        system = EloRatingSystem()
        rating = system.get_rating(1)
        
        assert rating.rating == 1500.0
        assert rating.matches_played == 0
        assert rating.is_established is False
    
    def test_update_after_win(self):
        """Winner's rating increases."""
        system = EloRatingSystem()
        
        # Team 1 beats Team 2
        new_home, new_away = system.update_ratings(
            home_team_id=1,
            away_team_id=2,
            home_goals=2,
            away_goals=0,
            match_date=datetime.now(timezone.utc),
        )
        
        assert new_home > 1500.0  # Winner gains
        assert new_away < 1500.0  # Loser drops
    
    def test_update_after_draw(self):
        """Draw causes small changes based on expected result."""
        system = EloRatingSystem()
        
        new_home, new_away = system.update_ratings(
            home_team_id=1,
            away_team_id=2,
            home_goals=1,
            away_goals=1,
            match_date=datetime.now(timezone.utc),
        )
        
        # Home team underperformed (had advantage), away team overperformed
        assert new_home < 1500.0
        assert new_away > 1500.0
    
    def test_matches_played_increments(self):
        """Matches played counter increases."""
        system = EloRatingSystem()
        
        system.update_ratings(1, 2, 1, 0, datetime.now(timezone.utc))
        system.update_ratings(1, 3, 2, 1, datetime.now(timezone.utc))
        
        rating = system.get_rating(1)
        assert rating.matches_played == 2
    
    def test_established_after_5_matches(self):
        """Rating is established after 5 matches."""
        system = EloRatingSystem()
        
        for i in range(5):
            system.update_ratings(1, i + 10, 1, 0, datetime.now(timezone.utc))
        
        rating = system.get_rating(1)
        assert rating.is_established is True
    
    def test_predict_match(self):
        """Test match prediction."""
        system = EloRatingSystem()
        
        # Set up some ratings
        system.update_ratings(1, 2, 3, 0, datetime.now(timezone.utc))
        system.update_ratings(1, 3, 2, 0, datetime.now(timezone.utc))
        
        pred = system.predict_match(1, 2)
        
        assert "home_win" in pred
        assert "draw" in pred
        assert "away_win" in pred
        assert pred["home_win"] + pred["draw"] + pred["away_win"] == pytest.approx(1.0, abs=0.01)


class TestAsOfDateLogic:
    """Test that features respect as-of date."""
    
    def test_asof_date_creation(self):
        """Test AsOfDate creation."""
        as_of = AsOfDate(datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc))
        
        assert as_of.date == datetime(2024, 8, 17).date()
    
    def test_is_before_true(self):
        """Test matches before as-of are detected."""
        as_of = AsOfDate(datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc))
        
        earlier = datetime(2024, 8, 10, 14, 0, tzinfo=timezone.utc)
        assert as_of.is_before(earlier) is True
    
    def test_is_before_false(self):
        """Test matches after as-of are excluded."""
        as_of = AsOfDate(datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc))
        
        later = datetime(2024, 8, 20, 14, 0, tzinfo=timezone.utc)
        assert as_of.is_before(later) is False
    
    def test_lookback(self):
        """Test lookback creates earlier as-of."""
        as_of = AsOfDate(datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc))
        lookback = as_of.lookback(7)
        
        assert lookback.date == datetime(2024, 8, 10).date()
