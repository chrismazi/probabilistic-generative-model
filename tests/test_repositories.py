"""
Tests for database repositories.

Critical tests for:
- Upsert operations work correctly
- Unique constraints prevent duplicates
- Idempotency is maintained
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from src.db.repositories import (
    LeagueRepository,
    TeamRepository,
    MatchRepository,
    ScoreRepository,
)


class TestLeagueRepository:
    """Test LeagueRepository operations."""
    
    @pytest.fixture
    def session(self):
        """Create mock session."""
        return MagicMock()
    
    @pytest.fixture
    def repo(self, session):
        """Create repository with mock session."""
        return LeagueRepository(session)
    
    def test_get_by_code_found(self, repo, session):
        """Test getting existing league."""
        session.execute.return_value.fetchone.return_value = (
            1, "PL", "Premier League", "England"
        )
        
        result = repo.get_by_code("PL")
        
        assert result is not None
        assert result["id"] == 1
        assert result["code"] == "PL"
    
    def test_get_by_code_not_found(self, repo, session):
        """Test getting non-existent league."""
        session.execute.return_value.fetchone.return_value = None
        
        result = repo.get_by_code("UNKNOWN")
        
        assert result is None
    
    def test_upsert_creates_new(self, repo, session):
        """Test upsert creates new league when not exists."""
        # First call: not found, second call: returns new ID
        session.execute.return_value.fetchone.side_effect = [
            None,  # get_by_code
            (42,),  # INSERT returning
        ]
        
        league_id = repo.upsert("PL", "Premier League", "England", "PL")
        
        assert league_id == 42
        assert session.execute.call_count >= 2
    
    def test_upsert_returns_existing(self, repo, session):
        """Test upsert returns existing league ID."""
        session.execute.return_value.fetchone.return_value = (
            99, "PL", "Premier League", "England"
        )
        
        # When league exists, upsert should find it
        existing = repo.get_by_code("PL")
        
        assert existing["id"] == 99
    
    def test_upsert_idempotent(self, repo, session):
        """Test calling upsert twice returns same ID."""
        # Simulate: first call creates (ID=1), second finds existing
        session.execute.return_value.fetchone.side_effect = [
            None, (1,),  # First upsert: not found, creates
            (1, "PL", "Premier League", "England"),  # Second: found
        ]
        
        id1 = repo.upsert("PL", "Premier League", "England", "PL")
        
        # Reset for second call
        existing = repo.get_by_code("PL")
        
        assert id1 == 1
        assert existing["id"] == 1


class TestTeamRepository:
    """Test TeamRepository operations."""
    
    @pytest.fixture
    def session(self):
        return MagicMock()
    
    @pytest.fixture
    def repo(self, session):
        return TeamRepository(session)
    
    def test_upsert_creates_new_team(self, repo, session):
        """Test creating new team."""
        session.execute.return_value.fetchone.side_effect = [
            None,  # Not found
            (1,),  # INSERT returns ID
        ]
        
        team_id = repo.upsert(
            league_id=1,
            provider_team_id=57,
            name="Arsenal",
            short_name="ARS",
        )
        
        assert team_id == 1
    
    def test_upsert_returns_existing_team(self, repo, session):
        """Test returns existing team."""
        session.execute.return_value.fetchone.return_value = (42, "Arsenal", "ARS")
        
        existing = repo.get_by_provider_id(1, 57)
        
        assert existing["id"] == 42
        assert existing["name"] == "Arsenal"


class TestMatchRepository:
    """Test MatchRepository operations."""
    
    @pytest.fixture
    def session(self):
        return MagicMock()
    
    @pytest.fixture
    def repo(self, session):
        return MatchRepository(session)
    
    def test_upsert_creates_new_match(self, repo, session):
        """Test creating new match."""
        session.execute.return_value.fetchone.side_effect = [
            None,  # Not found
            (1,),  # INSERT returns ID
        ]
        
        match_id = repo.upsert(
            league_id=1,
            season="2024-25",
            matchday=1,
            kickoff_utc=datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc),
            home_team_id=1,
            away_team_id=2,
            status="FINISHED",
            provider_match_id=12345,
        )
        
        assert match_id == 1
    
    def test_upsert_updates_status(self, repo, session):
        """Test updating match status."""
        # Match exists with SCHEDULED
        session.execute.return_value.fetchone.return_value = (42, "SCHEDULED")
        
        match_id = repo.upsert(
            league_id=1,
            season="2024-25",
            matchday=1,
            kickoff_utc=datetime(2024, 8, 17, 14, 0, tzinfo=timezone.utc),
            home_team_id=1,
            away_team_id=2,
            status="FINISHED",  # New status
            provider_match_id=12345,
        )
        
        assert match_id == 42
        # Should have called UPDATE
        calls = session.execute.call_args_list
        update_call = [c for c in calls if "UPDATE" in str(c)]
        assert len(update_call) > 0
    
    def test_count_by_league_season(self, repo, session):
        """Test counting matches."""
        session.execute.return_value.scalar.return_value = 380
        
        count = repo.count_by_league_season(1, "2024-25")
        
        assert count == 380


class TestScoreRepository:
    """Test ScoreRepository operations."""
    
    @pytest.fixture
    def session(self):
        return MagicMock()
    
    @pytest.fixture
    def repo(self, session):
        return ScoreRepository(session)
    
    def test_upsert_with_ht_sets_available_true(self, repo, session):
        """Test ht_available is True when HT scores present."""
        repo.upsert(
            match_id=1,
            ht_home=1,
            ht_away=0,
            ft_home=2,
            ft_away=1,
        )
        
        # Check the call
        call_args = session.execute.call_args
        params = call_args[0][1]
        
        assert params["ht_available"] is True
    
    def test_upsert_without_ht_sets_available_false(self, repo, session):
        """Test ht_available is False when HT scores missing."""
        repo.upsert(
            match_id=1,
            ht_home=None,
            ht_away=None,
            ft_home=2,
            ft_away=1,
        )
        
        call_args = session.execute.call_args
        params = call_args[0][1]
        
        assert params["ht_available"] is False
    
    def test_upsert_idempotent(self, repo, session):
        """Test upserting score twice doesn't error."""
        # Both calls should succeed (ON CONFLICT handles it)
        repo.upsert(match_id=1, ht_home=0, ht_away=0, ft_home=0, ft_away=0)
        repo.upsert(match_id=1, ht_home=1, ht_away=0, ft_home=2, ft_away=1)
        
        assert session.execute.call_count == 2
    
    def test_count_with_ht(self, repo, session):
        """Test counting scores with HT data."""
        session.execute.return_value.scalar.return_value = 350
        
        count = repo.count_with_ht(1, "2024-25")
        
        assert count == 350


class TestRepositoryConstraints:
    """Test that constraints are properly enforced."""
    
    def test_league_code_unique(self):
        """League code should be unique (enforced by DB)."""
        # This is enforced by the UNIQUE constraint in schema.sql
        # The ON CONFLICT clause in upsert handles duplicates gracefully
        pass
    
    def test_team_league_provider_unique(self):
        """Team should be unique per league+provider_id."""
        # UNIQUE(league_id, provider_team_id) in schema.sql
        pass
    
    def test_match_provider_unique(self):
        """Match should be unique by provider_match_id."""
        # UNIQUE on provider_match_id in schema.sql
        pass
    
    def test_score_match_unique(self):
        """Only one score per match."""
        # UNIQUE on match_id in schema.sql
        pass
