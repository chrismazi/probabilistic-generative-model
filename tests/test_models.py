"""
Tests for data models.
"""

from datetime import datetime

import pytest

from src.data.models import (
    ApiMatch,
    ApiScore,
    ApiFullScore,
    ApiTeam,
    ApiCompetition,
    LeagueCoverage,
    MatchStatus,
    Score,
)


class TestScore:
    """Test Score model computed properties."""
    
    def test_g1_total(self):
        """Test first half total goals."""
        score = Score(match_id=1, ht_home=1, ht_away=2, ft_home=2, ft_away=3)
        assert score.g1_total == 3
    
    def test_g1_total_none_when_missing(self):
        """Test g1_total is None when HT missing."""
        score = Score(match_id=1, ft_home=2, ft_away=3)
        assert score.g1_total is None
    
    def test_g2_total(self):
        """Test second half total goals."""
        score = Score(match_id=1, ht_home=1, ht_away=1, ft_home=2, ft_away=3)
        # G2 = (2-1) + (3-1) = 1 + 2 = 3
        assert score.g2_total == 3
    
    def test_g2_total_none_when_missing(self):
        """Test g2_total is None when any score missing."""
        score = Score(match_id=1, ht_home=1, ht_away=1, ft_home=2)
        assert score.g2_total is None
    
    def test_is_2h_gt_1h_true(self):
        """Test 2H > 1H detection."""
        # G1 = 0+0 = 0, G2 = (2-0) + (1-0) = 3
        score = Score(match_id=1, ht_home=0, ht_away=0, ft_home=2, ft_away=1)
        assert score.is_2h_gt_1h is True
    
    def test_is_2h_gt_1h_false(self):
        """Test 2H <= 1H detection."""
        # G1 = 2+1 = 3, G2 = (2-2) + (2-1) = 1
        score = Score(match_id=1, ht_home=2, ht_away=1, ft_home=2, ft_away=2)
        assert score.is_2h_gt_1h is False
    
    def test_is_2h_gt_1h_equal(self):
        """Test equal halves."""
        # G1 = 1+1 = 2, G2 = (2-1) + (2-1) = 2
        score = Score(match_id=1, ht_home=1, ht_away=1, ft_home=2, ft_away=2)
        assert score.is_2h_gt_1h is False


class TestLeagueCoverage:
    """Test LeagueCoverage computation."""
    
    def test_compute_full_coverage(self):
        """Test 100% coverage."""
        cov = LeagueCoverage.compute("PL", "2024-25", total=380, with_ht=380)
        
        assert cov.ht_coverage_pct == 100.0
        assert cov.is_reliable is True
    
    def test_compute_partial_coverage(self):
        """Test partial coverage."""
        cov = LeagueCoverage.compute("PL", "2024-25", total=380, with_ht=340)
        
        assert 89 < cov.ht_coverage_pct < 90
        assert cov.is_reliable is False  # Below 90%
    
    def test_compute_ninety_percent_threshold(self):
        """Test 90% threshold."""
        cov = LeagueCoverage.compute("PL", "2024-25", total=100, with_ht=90)
        
        assert cov.ht_coverage_pct == 90.0
        assert cov.is_reliable is True
    
    def test_compute_empty_season(self):
        """Test empty season doesn't crash."""
        cov = LeagueCoverage.compute("PL", "2024-25", total=0, with_ht=0)
        
        assert cov.ht_coverage_pct == 0.0
        assert cov.is_reliable is False


class TestApiModels:
    """Test API response model parsing."""
    
    def test_api_match_datetime_parsing(self):
        """Test datetime parsing from API."""
        data = {
            "id": 123,
            "competition": {"id": 1, "name": "PL", "code": "PL"},
            "season": {"startDate": "2024-08-01", "endDate": "2025-05-31"},
            "utcDate": "2024-08-17T14:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"id": 1, "name": "Team A"},
            "awayTeam": {"id": 2, "name": "Team B"},
        }
        
        match = ApiMatch.model_validate(data)
        
        assert match.id == 123
        assert match.utc_date.year == 2024
        assert match.utc_date.month == 8
        assert match.status == MatchStatus.FINISHED
    
    def test_match_status_enum(self):
        """Test match status values."""
        assert MatchStatus.FINISHED.value == "FINISHED"
        assert MatchStatus.SCHEDULED.value == "SCHEDULED"
        assert MatchStatus.POSTPONED.value == "POSTPONED"
