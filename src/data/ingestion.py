"""
Data ingestion pipeline for fetching and storing match data.

Handles:
- Fetching fixtures and results from football-data.org
- Upserting to database via Repository pattern
- Data quality validation
- Coverage reporting
"""

from datetime import date, datetime, timedelta
from typing import Optional

from src.config import settings
from src.constants import MatchStatusEnum, SUPPORTED_LEAGUES
from src.data.client import FootballDataClient, get_client
from src.data.models import (
    ApiMatch,
    LeagueCoverage,
)
from src.db import (
    get_session,
    LeagueRepository,
    TeamRepository,
    MatchRepository,
    ScoreRepository,
)
from src.utils import get_logger

logger = get_logger("ingestion")


class IngestionPipeline:
    """
    Pipeline for ingesting match data from football-data.org.
    
    Responsibilities:
    - Fetch matches for specified leagues and date ranges
    - Upsert leagues, teams, matches, and scores via Repository pattern
    - Track coverage statistics
    - Run data quality checks
    
    All upsert operations are idempotent - safe to run multiple times.
    """
    
    def __init__(
        self,
        client: Optional[FootballDataClient] = None,
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            client: API client (uses default if not provided)
        """
        self.client = client or get_client()
    
    def _format_season(self, api_match: ApiMatch) -> str:
        """Format season string (e.g., '2024-25')."""
        start_year = api_match.season.get("startDate", "")[:4]
        end_year = api_match.season.get("endDate", "")[:4]
        
        if start_year and end_year:
            return f"{start_year}-{end_year[-2:]}"
        return start_year or "unknown"
    
    def ingest_matches(
        self,
        league_code: str,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        season: Optional[int] = None,
    ) -> dict:
        """
        Ingest matches for a league.
        
        All database operations are idempotent - running twice
        will not duplicate data.
        
        Args:
            league_code: League code (e.g., "PL")
            date_from: Start date
            date_to: End date
            season: Season year
            
        Returns:
            Summary of ingestion results
        """
        logger.info(f"Ingesting {league_code} matches")
        
        # Fetch from API
        api_matches = self.client.get_matches(
            league_code=league_code,
            date_from=date_from,
            date_to=date_to,
            season=season,
        )
        
        stats = {
            "total_fetched": len(api_matches),
            "matches_created": 0,
            "matches_updated": 0,
            "scores_added": 0,
            "errors": [],
        }
        
        if not api_matches:
            logger.warning(f"No matches found for {league_code}")
            return stats
        
        with get_session() as session:
            # Initialize repositories
            league_repo = LeagueRepository(session)
            team_repo = TeamRepository(session)
            match_repo = MatchRepository(session)
            score_repo = ScoreRepository(session)
            
            # Get or create league
            league_info = SUPPORTED_LEAGUES.get(league_code, (league_code, "Unknown"))
            league_id = league_repo.upsert(
                code=league_code,
                name=league_info[0],
                country=league_info[1],
                provider_key=league_code,
            )
            
            for api_match in api_matches:
                try:
                    # Upsert teams (idempotent)
                    home_team_id = team_repo.upsert(
                        league_id=league_id,
                        provider_team_id=api_match.home_team.id,
                        name=api_match.home_team.name,
                        short_name=api_match.home_team.short_name,
                    )
                    away_team_id = team_repo.upsert(
                        league_id=league_id,
                        provider_team_id=api_match.away_team.id,
                        name=api_match.away_team.name,
                        short_name=api_match.away_team.short_name,
                    )
                    
                    # Upsert match (idempotent)
                    season_str = self._format_season(api_match)
                    match_id = match_repo.upsert(
                        league_id=league_id,
                        season=season_str,
                        matchday=api_match.matchday,
                        kickoff_utc=api_match.utc_date,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        status=api_match.status.value,
                        provider_match_id=api_match.id,
                    )
                    
                    stats["matches_created"] += 1
                    
                    # Upsert score if match is finished (idempotent)
                    if api_match.status == MatchStatusEnum.FINISHED and api_match.score:
                        ht = api_match.score.half_time
                        ft = api_match.score.full_time
                        
                        score_repo.upsert(
                            match_id=match_id,
                            ht_home=ht.home if ht else None,
                            ht_away=ht.away if ht else None,
                            ft_home=ft.home if ft else None,
                            ft_away=ft.away if ft else None,
                        )
                        stats["scores_added"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing match {api_match.id}: {e}")
                    stats["errors"].append({
                        "match_id": api_match.id,
                        "error": str(e),
                    })
        
        logger.info(
            f"Ingested {league_code}: "
            f"{stats['matches_created']} matches, "
            f"{stats['scores_added']} scores, "
            f"{len(stats['errors'])} errors"
        )
        
        return stats
    
    def ingest_yesterday_and_today(
        self,
        league_codes: Optional[list[str]] = None,
    ) -> dict:
        """
        Ingest matches from yesterday and today.
        
        Convenience method for daily ingestion jobs.
        Idempotent - safe to run multiple times per day.
        
        Args:
            league_codes: List of league codes (defaults to all supported)
            
        Returns:
            Summary of ingestion results per league
        """
        if league_codes is None:
            league_codes = list(SUPPORTED_LEAGUES.keys())
        
        yesterday = date.today() - timedelta(days=1)
        today = date.today()
        
        logger.info(f"Daily ingestion: {yesterday} to {today} for {len(league_codes)} leagues")
        
        results = {}
        
        for code in league_codes:
            try:
                results[code] = self.ingest_matches(
                    league_code=code,
                    date_from=yesterday,
                    date_to=today,
                )
            except Exception as e:
                logger.error(f"Failed to ingest {code}: {e}")
                results[code] = {"error": str(e)}
        
        return results
    
    def compute_coverage(
        self,
        league_code: str,
        season: str,
    ) -> LeagueCoverage:
        """
        Compute coverage statistics for a league/season.
        
        Args:
            league_code: League code
            season: Season string (e.g., "2024-25")
            
        Returns:
            Coverage statistics
        """
        from sqlalchemy import text
        
        with get_session() as session:
            # Get league ID
            league_repo = LeagueRepository(session)
            league = league_repo.get_by_code(league_code)
            
            if not league:
                return LeagueCoverage.compute(
                    league_code=league_code,
                    season=season,
                    total=0,
                    with_ht=0,
                )
            
            league_id = league["id"]
            
            # Get match repository for counting
            match_repo = MatchRepository(session)
            score_repo = ScoreRepository(session)
            
            total = match_repo.count_by_league_season(league_id, season)
            with_ht = score_repo.count_with_ht(league_id, season)
            
            return LeagueCoverage.compute(
                league_code=league_code,
                season=season,
                total=total,
                with_ht=with_ht,
            )


def get_pipeline() -> IngestionPipeline:
    """Get a configured ingestion pipeline instance."""
    return IngestionPipeline()
