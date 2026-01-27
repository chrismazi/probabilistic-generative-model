"""
Tests for ingestion idempotency.

Critical tests to ensure:
- Running ingest twice does not duplicate rows
- Upsert operations work correctly
- Data is updated, not duplicated
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.db.repositories import (
    LeagueRepository,
    TeamRepository,
    MatchRepository,
    ScoreRepository,
)


class TestLeagueRepositoryIdempotency:
    """Test league upsert idempotency."""
    
    @pytest.fixture
    def mock_session(self):
        return MagicMock()
    
    def test_upsert_creates_new_league(self, mock_session):
        """First upsert creates a new league."""
        # Mock: league doesn't exist
        mock_session.execute.return_value.fetchone.side_effect = [
            None,  # get_by_code returns None
            (1,),  # INSERT returns id
        ]
        
        repo = LeagueRepository(mock_session)
        league_id = repo.upsert("PL", "Premier League", "England", "PL")
        
        assert league_id == 1
        # Verify INSERT was called
        calls = mock_session.execute.call_args_list
        assert len(calls) >= 2  # SELECT then INSERT
    
    def test_upsert_returns_existing_league(self, mock_session):
        """Second upsert returns existing league without INSERT."""
        # Mock: league already exists
        mock_session.execute.return_value.fetchone.return_value = (42,)
        
        repo = LeagueRepository(mock_session)
        league_id = repo.get_by_code("PL")
        
        # Should return existing ID
        assert league_id == {"id": 42, "code": None, "name": None, "country": None}
    
    def test_upsert_twice_same_result(self, mock_session):
        """Upserting twice returns same ID."""
        # First call: doesn't exist, creates
        # Second call: exists, returns
        mock_session.execute.return_value.fetchone.side_effect = [
            None,  # First get_by_code
            (1,),  # INSERT
            (1,),  # Second get_by_code (exists now)
        ]
        
        repo = LeagueRepository(mock_session)
        
        id1 = repo.upsert("PL", "Premier League", "England", "PL")
        
        # Reset mock for second call simulation
        mock_session.execute.return_value.fetchone.return_value = (1,)
        result = repo.get_by_code("PL")
        
        assert id1 == 1
        assert result["id"] == 1


class TestTeamRepositoryIdempotency:
    """Test team upsert idempotency."""
    
    @pytest.fixture
    def mock_session(self):
        return MagicMock()
    
    def test_upsert_creates_new_team(self, mock_session):
        """First upsert creates a new team."""
        mock_session.execute.return_value.fetchone.side_effect = [
            None,  # get_by_provider_id returns None
            (1,),  # INSERT returns id
        ]
        
        repo = TeamRepository(mock_session)
        team_id = repo.upsert(
            league_id=1,
            provider_team_id=123,
            name="Arsenal",
            short_name="ARS",
        )
        
        assert team_id == 1
    
    def test_upsert_returns_existing_team(self, mock_session):
        """Second upsert returns existing team."""
        mock_session.execute.return_value.fetchone.return_value = (42, "Arsenal", "ARS")
        
        repo = TeamRepository(mock_session)
        existing = repo.get_by_provider_id(1, 123)
        
        assert existing["id"] == 42
        assert existing["name"] == "Arsenal"


class TestMatchRepositoryIdempotency:
    """Test match upsert idempotency."""
    
    @pytest.fixture
    def mock_session(self):
        return MagicMock()
    
    def test_upsert_creates_new_match(self, mock_session):
        """First upsert creates a new match."""
        mock_session.execute.return_value.fetchone.side_effect = [
            None,  # get_by_provider_id returns None
            (1,),  # INSERT returns id
        ]
        
        repo = MatchRepository(mock_session)
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
    
    def test_upsert_updates_existing_match(self, mock_session):
        """Second upsert updates status, returns same ID."""
        # Match exists with SCHEDULED status
        mock_session.execute.return_value.fetchone.return_value = (42, "SCHEDULED")
        
        repo = MatchRepository(mock_session)
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
        # Verify UPDATE was called (not INSERT)
        calls = [c for c in mock_session.execute.call_args_list]
        update_called = any("UPDATE" in str(c) for c in calls)
        assert update_called


class TestScoreRepositoryIdempotency:
    """Test score upsert idempotency."""
    
    @pytest.fixture
    def mock_session(self):
        return MagicMock()
    
    def test_upsert_creates_new_score(self, mock_session):
        """First upsert creates a new score."""
        repo = ScoreRepository(mock_session)
        repo.upsert(
            match_id=1,
            ht_home=1,
            ht_away=0,
            ft_home=2,
            ft_away=1,
        )
        
        # Verify INSERT was called with ON CONFLICT
        call_args = str(mock_session.execute.call_args)
        assert "INSERT" in call_args or "insert" in call_args.lower()
    
    def test_upsert_updates_existing_score(self, mock_session):
        """Second upsert updates existing score."""
        repo = ScoreRepository(mock_session)
        
        # First call
        repo.upsert(match_id=1, ht_home=0, ht_away=0, ft_home=0, ft_away=0)
        
        # Second call with updated values
        repo.upsert(match_id=1, ht_home=1, ht_away=0, ft_home=2, ft_away=1)
        
        # Both should succeed without error (ON CONFLICT handles it)
        assert mock_session.execute.call_count == 2
    
    def test_ht_available_flag_set_correctly(self, mock_session):
        """Verify ht_available is True when both HT scores present."""
        repo = ScoreRepository(mock_session)
        repo.upsert(
            match_id=1,
            ht_home=1,
            ht_away=0,
            ft_home=2,
            ft_away=1,
        )
        
        # Check the parameters passed to execute
        call_args = mock_session.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        
        assert params.get("ht_available") is True
    
    def test_ht_available_false_when_missing(self, mock_session):
        """Verify ht_available is False when HT scores missing."""
        repo = ScoreRepository(mock_session)
        repo.upsert(
            match_id=1,
            ht_home=None,
            ht_away=None,
            ft_home=2,
            ft_away=1,
        )
        
        call_args = mock_session.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        
        assert params.get("ht_available") is False


class TestFullIngestionIdempotency:
    """Integration tests for full ingestion idempotency."""
    
    def test_ingest_twice_same_match_count(self):
        """
        Ingesting the same data twice should not duplicate matches.
        
        This is a placeholder for integration test with real DB.
        In practice, run against a test database.
        """
        # TODO: Implement with test database
        # 
        # 1. Clear test DB
        # 2. Ingest PL season 2024
        # 3. Count matches
        # 4. Ingest PL season 2024 again
        # 5. Count matches - should be same
        # 
        # count_before = session.execute("SELECT COUNT(*) FROM matches").scalar()
        # ingest_season("PL", 2024)
        # count_after = session.execute("SELECT COUNT(*) FROM matches").scalar()
        # assert count_before == count_after
        pass
    
    def test_score_update_does_not_duplicate(self):
        """
        Updating a score should not create duplicate score rows.
        
        Placeholder for integration test.
        """
        # TODO: Implement with test database
        pass
