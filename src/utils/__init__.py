"""
Shared utility functions.

Includes:
- Logging setup
- Time helpers (as-of logic)
- Retry helpers
- Caching wrappers
"""

# Re-export logging utilities for convenience
from src.utils.logging import setup_logging, get_logger, LogContext

import functools
import hashlib
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from src.config import settings

T = TypeVar("T")


# =============================================================================
# Logging
# =============================================================================

def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logs
        format_string: Custom format string
        
    Returns:
        Configured root logger
    """
    level = level or settings.log_level
    log_file = log_file or settings.log_file
    
    format_string = format_string or (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
        force=True,
    )
    
    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    return logging.getLogger("betting_model")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(f"betting_model.{name}")


# =============================================================================
# Time Helpers
# =============================================================================

def now_utc() -> datetime:
    """Get current time in UTC."""
    return datetime.now(timezone.utc)


def today_utc() -> date:
    """Get current date in UTC."""
    return now_utc().date()


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.
    
    If naive, assumes UTC. If aware, converts.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_season(start_year: int) -> str:
    """
    Format season string (e.g., 2024 -> '2024-25').
    """
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def parse_season(season_str: str) -> int:
    """
    Parse season string to start year (e.g., '2024-25' -> 2024).
    """
    return int(season_str.split("-")[0])


class AsOfDate:
    """
    Helper for 'as-of' date filtering.
    
    Ensures feature computation only uses data from before the as-of date.
    This is critical for preventing data leakage.
    """
    
    def __init__(self, as_of: date | datetime):
        if isinstance(as_of, datetime):
            self.date = as_of.date()
            self.datetime = to_utc(as_of)
        else:
            self.date = as_of
            self.datetime = datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc)
    
    def is_before(self, dt: date | datetime) -> bool:
        """Check if given date/datetime is before the as-of date."""
        if isinstance(dt, datetime):
            return to_utc(dt) < self.datetime
        return dt < self.date
    
    def filter_matches(self, kickoff_times: list[datetime]) -> list[bool]:
        """Return mask of matches that are strictly before as-of."""
        return [self.is_before(kt) for kt in kickoff_times]
    
    def lookback(self, days: int) -> "AsOfDate":
        """Get as-of date N days prior."""
        return AsOfDate(self.date - timedelta(days=days))


# =============================================================================
# Caching Helpers
# =============================================================================

def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


def file_cache(
    cache_dir: Optional[Path] = None,
    ttl_hours: int = 24,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for file-based caching.
    
    Args:
        cache_dir: Directory for cache files
        ttl_hours: Time-to-live in hours
        
    Returns:
        Decorator function
    """
    cache_dir = cache_dir or settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = f"{func.__name__}_{cache_key(*args, **kwargs)}"
            cache_path = cache_dir / f"{key}.json"
            
            # Check cache
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        cached = json.load(f)
                    
                    cached_at = datetime.fromisoformat(cached["cached_at"])
                    if datetime.now() - cached_at < timedelta(hours=ttl_hours):
                        return cached["data"]
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # Compute and cache
            result = func(*args, **kwargs)
            
            with open(cache_path, "w") as f:
                json.dump({
                    "cached_at": datetime.now().isoformat(),
                    "data": result,
                }, f, default=str)
            
            return result
        
        return wrapper
    
    return decorator


# =============================================================================
# Math Helpers
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def clip(value: float, low: float, high: float) -> float:
    """Clip value to range [low, high]."""
    return max(low, min(high, value))
