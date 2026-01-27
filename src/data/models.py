"""
Pydantic models for football data.

These models define the structure of data from the football-data.org API
and our internal representations.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class MatchStatus(str, Enum):
    """Match status from football-data.org."""
    SCHEDULED = "SCHEDULED"
    TIMED = "TIMED"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    SUSPENDED = "SUSPENDED"
    CANCELLED = "CANCELLED"
    AWARDED = "AWARDED"


# =============================================================================
# API Response Models (from football-data.org)
# =============================================================================

class ApiTeam(BaseModel):
    """Team from API response."""
    id: int
    name: str
    short_name: Optional[str] = Field(None, alias="shortName")
    tla: Optional[str] = None  # Three-letter abbreviation
    
    model_config = {"populate_by_name": True}


class ApiScore(BaseModel):
    """Score breakdown from API response."""
    home: Optional[int] = None
    away: Optional[int] = None


class ApiFullScore(BaseModel):
    """Full score object from API response."""
    winner: Optional[str] = None
    duration: Optional[str] = None
    full_time: Optional[ApiScore] = Field(None, alias="fullTime")
    half_time: Optional[ApiScore] = Field(None, alias="halfTime")
    
    model_config = {"populate_by_name": True}


class ApiCompetition(BaseModel):
    """Competition/league from API response."""
    id: int
    name: str
    code: str
    type: Optional[str] = None
    emblem: Optional[str] = None


class ApiMatch(BaseModel):
    """Match from API response."""
    id: int
    competition: ApiCompetition
    season: dict
    utc_date: datetime = Field(alias="utcDate")
    status: MatchStatus
    matchday: Optional[int] = None
    stage: Optional[str] = None
    home_team: ApiTeam = Field(alias="homeTeam")
    away_team: ApiTeam = Field(alias="awayTeam")
    score: Optional[ApiFullScore] = None
    
    model_config = {"populate_by_name": True}
    
    @field_validator("utc_date", mode="before")
    @classmethod
    def parse_datetime(cls, v: str | datetime) -> datetime:
        """Parse ISO datetime string."""
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(v.replace("Z", "+00:00"))


class ApiMatchResponse(BaseModel):
    """Response from matches endpoint."""
    count: int = Field(alias="resultSet", default=0)
    matches: list[ApiMatch] = []
    
    model_config = {"populate_by_name": True}
    
    @field_validator("count", mode="before")
    @classmethod
    def extract_count(cls, v: dict | int) -> int:
        """Extract count from resultSet object."""
        if isinstance(v, dict):
            return v.get("count", 0)
        return v


# =============================================================================
# Internal Models
# =============================================================================

class League(BaseModel):
    """Internal league representation."""
    id: Optional[int] = None
    code: str
    name: str
    country: str
    provider_key: Optional[str] = None
    is_active: bool = True


class Team(BaseModel):
    """Internal team representation."""
    id: Optional[int] = None
    league_id: int
    name: str
    short_name: Optional[str] = None
    provider_team_id: int


class Match(BaseModel):
    """Internal match representation."""
    id: Optional[int] = None
    league_id: int
    season: str
    matchday: Optional[int] = None
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    status: MatchStatus
    provider_match_id: int


class Score(BaseModel):
    """Internal score representation."""
    match_id: int
    ht_home: Optional[int] = None
    ht_away: Optional[int] = None
    ft_home: Optional[int] = None
    ft_away: Optional[int] = None
    ht_available: bool = False
    
    @property
    def g1_total(self) -> Optional[int]:
        """Total goals in first half."""
        if self.ht_home is None or self.ht_away is None:
            return None
        return self.ht_home + self.ht_away
    
    @property
    def g2_total(self) -> Optional[int]:
        """Total goals in second half."""
        if any(v is None for v in [self.ht_home, self.ht_away, self.ft_home, self.ft_away]):
            return None
        return (self.ft_home - self.ht_home) + (self.ft_away - self.ht_away)
    
    @property
    def is_2h_gt_1h(self) -> Optional[bool]:
        """True if second half has more goals than first half."""
        g1, g2 = self.g1_total, self.g2_total
        if g1 is None or g2 is None:
            return None
        return g2 > g1


class MatchWithScore(BaseModel):
    """Match combined with score for convenience."""
    match: Match
    score: Optional[Score] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    league_code: Optional[str] = None


# =============================================================================
# Coverage Report Models
# =============================================================================

class LeagueCoverage(BaseModel):
    """Coverage statistics for a league/season."""
    league_code: str
    season: str
    total_matches: int
    matches_with_ht: int
    ht_coverage_pct: float
    avg_delay_hours: Optional[float] = None
    is_reliable: bool = False  # True if coverage >= 90%
    
    @classmethod
    def compute(
        cls,
        league_code: str,
        season: str,
        total: int,
        with_ht: int,
        avg_delay: Optional[float] = None,
    ) -> "LeagueCoverage":
        """Compute coverage statistics."""
        pct = (with_ht / total * 100) if total > 0 else 0.0
        return cls(
            league_code=league_code,
            season=season,
            total_matches=total,
            matches_with_ht=with_ht,
            ht_coverage_pct=pct,
            avg_delay_hours=avg_delay,
            is_reliable=pct >= 90.0,
        )
