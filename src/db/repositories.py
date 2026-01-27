"""
Repository pattern for database operations.

Provides clean abstractions for CRUD operations on core entities.
All upsert operations are idempotent.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.constants import MatchStatusEnum
from src.db.connection import get_session


class LeagueRepository:
    """Repository for league operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_code(self, code: str) -> Optional[dict]:
        """Get league by code."""
        result = self.session.execute(
            text("SELECT id, code, name, country FROM leagues WHERE code = :code"),
            {"code": code}
        ).fetchone()
        
        if result:
            return {"id": result[0], "code": result[1], "name": result[2], "country": result[3]}
        return None
    
    def upsert(self, code: str, name: str, country: str, provider_key: Optional[str] = None) -> int:
        """
        Insert or update league. Returns league ID.
        
        Idempotent: safe to call multiple times.
        """
        existing = self.get_by_code(code)
        if existing:
            return existing["id"]
        
        result = self.session.execute(
            text("""
                INSERT INTO leagues (code, name, country, provider_key)
                VALUES (:code, :name, :country, :provider_key)
                ON CONFLICT (code) DO UPDATE SET
                    name = EXCLUDED.name,
                    country = EXCLUDED.country,
                    updated_at = NOW()
                RETURNING id
            """),
            {"code": code, "name": name, "country": country, "provider_key": provider_key}
        )
        return result.fetchone()[0]


class TeamRepository:
    """Repository for team operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_provider_id(self, league_id: int, provider_team_id: int) -> Optional[dict]:
        """Get team by provider ID."""
        result = self.session.execute(
            text("""
                SELECT id, name, short_name 
                FROM teams 
                WHERE league_id = :league_id AND provider_team_id = :provider_id
            """),
            {"league_id": league_id, "provider_id": provider_team_id}
        ).fetchone()
        
        if result:
            return {"id": result[0], "name": result[1], "short_name": result[2]}
        return None
    
    def upsert(
        self,
        league_id: int,
        provider_team_id: int,
        name: str,
        short_name: Optional[str] = None,
    ) -> int:
        """
        Insert or update team. Returns team ID.
        
        Idempotent: safe to call multiple times.
        """
        existing = self.get_by_provider_id(league_id, provider_team_id)
        if existing:
            return existing["id"]
        
        result = self.session.execute(
            text("""
                INSERT INTO teams (league_id, name, short_name, provider_team_id)
                VALUES (:league_id, :name, :short_name, :provider_id)
                ON CONFLICT (league_id, provider_team_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    short_name = EXCLUDED.short_name,
                    updated_at = NOW()
                RETURNING id
            """),
            {
                "league_id": league_id,
                "name": name,
                "short_name": short_name,
                "provider_id": provider_team_id,
            }
        )
        return result.fetchone()[0]


class MatchRepository:
    """Repository for match operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_provider_id(self, provider_match_id: int) -> Optional[dict]:
        """Get match by provider ID."""
        result = self.session.execute(
            text("SELECT id, status FROM matches WHERE provider_match_id = :provider_id"),
            {"provider_id": provider_match_id}
        ).fetchone()
        
        if result:
            return {"id": result[0], "status": result[1]}
        return None
    
    def upsert(
        self,
        league_id: int,
        season: str,
        matchday: Optional[int],
        kickoff_utc: datetime,
        home_team_id: int,
        away_team_id: int,
        status: str,
        provider_match_id: int,
    ) -> int:
        """
        Insert or update match. Returns match ID.
        
        Idempotent: safe to call multiple times.
        """
        existing = self.get_by_provider_id(provider_match_id)
        
        if existing:
            # Update status if changed
            self.session.execute(
                text("""
                    UPDATE matches 
                    SET status = :status, updated_at = NOW()
                    WHERE id = :id
                """),
                {"id": existing["id"], "status": status}
            )
            return existing["id"]
        
        result = self.session.execute(
            text("""
                INSERT INTO matches (
                    league_id, season, matchday, kickoff_utc,
                    home_team_id, away_team_id, status, provider_match_id
                )
                VALUES (
                    :league_id, :season, :matchday, :kickoff_utc,
                    :home_team_id, :away_team_id, :status, :provider_match_id
                )
                ON CONFLICT (provider_match_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = NOW()
                RETURNING id
            """),
            {
                "league_id": league_id,
                "season": season,
                "matchday": matchday,
                "kickoff_utc": kickoff_utc,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "status": status,
                "provider_match_id": provider_match_id,
            }
        )
        return result.fetchone()[0]
    
    def count_by_league_season(self, league_id: int, season: str) -> int:
        """Count matches for a league/season."""
        result = self.session.execute(
            text("""
                SELECT COUNT(*) FROM matches 
                WHERE league_id = :league_id AND season = :season
            """),
            {"league_id": league_id, "season": season}
        )
        return result.scalar() or 0


class ScoreRepository:
    """Repository for score operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_match_id(self, match_id: int) -> Optional[dict]:
        """Get score by match ID."""
        result = self.session.execute(
            text("""
                SELECT ht_home, ht_away, ft_home, ft_away, ht_available
                FROM scores WHERE match_id = :match_id
            """),
            {"match_id": match_id}
        ).fetchone()
        
        if result:
            return {
                "ht_home": result[0],
                "ht_away": result[1],
                "ft_home": result[2],
                "ft_away": result[3],
                "ht_available": result[4],
            }
        return None
    
    def upsert(
        self,
        match_id: int,
        ht_home: Optional[int],
        ht_away: Optional[int],
        ft_home: Optional[int],
        ft_away: Optional[int],
    ) -> None:
        """
        Insert or update score.
        
        Idempotent: safe to call multiple times.
        """
        ht_available = ht_home is not None and ht_away is not None
        
        self.session.execute(
            text("""
                INSERT INTO scores (match_id, ht_home, ht_away, ft_home, ft_away, ht_available)
                VALUES (:match_id, :ht_home, :ht_away, :ft_home, :ft_away, :ht_available)
                ON CONFLICT (match_id) DO UPDATE SET
                    ht_home = EXCLUDED.ht_home,
                    ht_away = EXCLUDED.ht_away,
                    ft_home = EXCLUDED.ft_home,
                    ft_away = EXCLUDED.ft_away,
                    ht_available = EXCLUDED.ht_available,
                    updated_at = NOW()
            """),
            {
                "match_id": match_id,
                "ht_home": ht_home,
                "ht_away": ht_away,
                "ft_home": ft_home,
                "ft_away": ft_away,
                "ht_available": ht_available,
            }
        )
    
    def count_with_ht(self, league_id: int, season: str) -> int:
        """Count scores with HT data for a league/season."""
        result = self.session.execute(
            text("""
                SELECT COUNT(*) FROM scores s
                JOIN matches m ON s.match_id = m.id
                WHERE m.league_id = :league_id 
                  AND m.season = :season
                  AND s.ht_available = TRUE
            """),
            {"league_id": league_id, "season": season}
        )
        return result.scalar() or 0
