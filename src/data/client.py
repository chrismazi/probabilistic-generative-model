"""
Football-data.org API client with resilience features.

Features:
- Retry with exponential backoff
- Response caching (per day/endpoint)
- Rate-limit handling
- Proper error handling
"""

import hashlib
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.data.models import (
    ApiMatch,
    ApiMatchResponse,
    League,
    MatchStatus,
)


class FootballDataAPIError(Exception):
    """Base exception for API errors."""
    pass


class RateLimitError(FootballDataAPIError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")


class APIResponseError(FootballDataAPIError):
    """Invalid API response."""
    pass


class FootballDataClient:
    """
    Client for football-data.org API with resilience features.
    
    Features:
    - Automatic retry with exponential backoff
    - Response caching to reduce API calls
    - Rate limit handling
    - Proper timeout handling
    """
    
    # Supported leagues with their country mappings
    SUPPORTED_LEAGUES = {
        "PL": ("Premier League", "England"),
        "BL1": ("Bundesliga", "Germany"),
        "SA": ("Serie A", "Italy"),
        "PD": ("La Liga", "Spain"),
        "FL1": ("Ligue 1", "France"),
        "ELC": ("Championship", "England"),
        "DED": ("Eredivisie", "Netherlands"),
        "PPL": ("Primeira Liga", "Portugal"),
        "CL": ("Champions League", "Europe"),
        "EC": ("European Championship", "Europe"),
        "WC": ("World Cup", "World"),
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize the API client.
        
        Args:
            api_key: API key for football-data.org
            base_url: Base URL for the API
            cache_dir: Directory for caching responses
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.api_key = api_key or settings.football_data_api_key
        self.base_url = base_url or settings.football_data_base_url
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track rate limiting
        self._last_request_time: float = 0
        self._min_request_interval = 60.0 / settings.api_requests_per_minute
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key."""
        return {
            "X-Auth-Token": self.api_key,
            "Content-Type": "application/json",
        }
    
    def _get_cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        """Generate cache key for endpoint + params."""
        param_str = json.dumps(params, sort_keys=True)
        key_str = f"{endpoint}:{param_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached response."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _read_cache(self, cache_key: str) -> Optional[dict]:
        """Read cached response if valid."""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            
            # Check if cache is still valid
            cached_at = datetime.fromisoformat(cached["cached_at"])
            if datetime.now() - cached_at > self.cache_ttl:
                cache_path.unlink()  # Remove stale cache
                return None
            
            return cached["data"]
        except (json.JSONDecodeError, KeyError):
            cache_path.unlink()
            return None
    
    def _write_cache(self, cache_key: str, data: dict) -> None:
        """Write response to cache."""
        cache_path = self._get_cache_path(cache_key)
        
        with open(cache_path, "w") as f:
            json.dump({
                "cached_at": datetime.now().isoformat(),
                "data": data,
            }, f)
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
    
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(settings.api_retry_max_attempts),
        wait=wait_exponential(
            min=settings.api_retry_min_wait,
            max=settings.api_retry_max_wait,
        ),
    )
    def _make_request(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> dict:
        """
        Make API request with retry and caching.
        
        Args:
            endpoint: API endpoint (e.g., "/competitions/PL/matches")
            params: Query parameters
            use_cache: Whether to use caching
            
        Returns:
            API response as dictionary
        """
        params = params or {}
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(endpoint, params)
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached
        
        # Respect rate limits
        self._wait_for_rate_limit()
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                url,
                params=params,
                headers=self._get_headers(),
            )
        
        self._last_request_time = time.time()
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("X-RequestCounter-Reset", 60))
            raise RateLimitError(retry_after)
        
        # Handle other errors
        if response.status_code != 200:
            raise APIResponseError(
                f"API error {response.status_code}: {response.text}"
            )
        
        data = response.json()
        
        # Cache successful response
        if use_cache:
            self._write_cache(cache_key, data)
        
        return data
    
    # =========================================================================
    # Public API Methods
    # =========================================================================
    
    def get_leagues(self) -> list[League]:
        """Get list of supported leagues."""
        return [
            League(
                code=code,
                name=name,
                country=country,
                provider_key=code,
            )
            for code, (name, country) in self.SUPPORTED_LEAGUES.items()
        ]
    
    def get_matches(
        self,
        league_code: str,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        season: Optional[int] = None,
        status: Optional[MatchStatus] = None,
        matchday: Optional[int] = None,
    ) -> list[ApiMatch]:
        """
        Get matches for a league.
        
        Args:
            league_code: League code (e.g., "PL", "BL1")
            date_from: Start date filter
            date_to: End date filter
            season: Season year (e.g., 2024 for 2024-25)
            status: Filter by match status
            matchday: Filter by matchday
            
        Returns:
            List of matches
        """
        params: dict[str, Any] = {}
        
        if date_from:
            params["dateFrom"] = date_from.isoformat()
        if date_to:
            params["dateTo"] = date_to.isoformat()
        if season:
            params["season"] = season
        if status:
            params["status"] = status.value
        if matchday:
            params["matchday"] = matchday
        
        endpoint = f"/competitions/{league_code}/matches"
        data = self._make_request(endpoint, params)
        
        response = ApiMatchResponse.model_validate(data)
        return response.matches
    
    def get_match(self, match_id: int) -> ApiMatch:
        """
        Get a single match by ID.
        
        Args:
            match_id: Match ID from football-data.org
            
        Returns:
            Match details
        """
        endpoint = f"/matches/{match_id}"
        data = self._make_request(endpoint)
        return ApiMatch.model_validate(data)
    
    def get_finished_matches(
        self,
        league_code: str,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        season: Optional[int] = None,
    ) -> list[ApiMatch]:
        """
        Get finished matches (with scores).
        
        Convenience method that filters for FINISHED status only.
        """
        return self.get_matches(
            league_code=league_code,
            date_from=date_from,
            date_to=date_to,
            season=season,
            status=MatchStatus.FINISHED,
        )
    
    def get_upcoming_matches(
        self,
        league_code: str,
        days_ahead: int = 7,
    ) -> list[ApiMatch]:
        """
        Get upcoming scheduled matches.
        
        Args:
            league_code: League code
            days_ahead: How many days to look ahead
            
        Returns:
            List of upcoming matches
        """
        today = date.today()
        future = today + timedelta(days=days_ahead)
        
        matches = self.get_matches(
            league_code=league_code,
            date_from=today,
            date_to=future,
        )
        
        # Filter for scheduled/timed only
        return [
            m for m in matches 
            if m.status in (MatchStatus.SCHEDULED, MatchStatus.TIMED)
        ]
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear cached responses.
        
        Args:
            older_than_hours: Only clear cache older than this. If None, clear all.
            
        Returns:
            Number of cache files removed
        """
        removed = 0
        cutoff = None
        
        if older_than_hours is not None:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
        
        for cache_file in self.cache_dir.glob("*.json"):
            should_remove = True
            
            if cutoff is not None:
                try:
                    with open(cache_file, "r") as f:
                        cached = json.load(f)
                    cached_at = datetime.fromisoformat(cached["cached_at"])
                    should_remove = cached_at < cutoff
                except (json.JSONDecodeError, KeyError):
                    should_remove = True
            
            if should_remove:
                cache_file.unlink()
                removed += 1
        
        return removed


# Convenience function for quick access
def get_client() -> FootballDataClient:
    """Get a configured API client instance."""
    return FootballDataClient()
