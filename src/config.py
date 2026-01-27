"""
Configuration management for the probabilistic generative model.

Uses pydantic-settings for type-safe, validated configuration from environment.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DecisionMode(str, Enum):
    """Decision layer operating mode."""
    PROBABILITY_ONLY = "probability_only"  # Default: output probs, no stakes
    STAKE_ENABLED = "stake_enabled"        # Requires explicit opt-in


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # =========================================================================
    # Database
    # =========================================================================
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/betting_model",
        description="PostgreSQL connection string",
    )
    database_pool_size: int = Field(default=5, ge=1, le=20)
    
    # =========================================================================
    # football-data.org API
    # =========================================================================
    football_data_api_key: str = Field(
        default="",
        description="API key for football-data.org",
    )
    football_data_base_url: str = Field(
        default="https://api.football-data.org/v4",
    )
    
    # =========================================================================
    # API Rate Limiting & Resilience
    # =========================================================================
    api_requests_per_minute: int = Field(default=10, ge=1)
    api_retry_max_attempts: int = Field(default=3, ge=1, le=10)
    api_retry_min_wait: float = Field(default=1.0, ge=0.1)
    api_retry_max_wait: float = Field(default=60.0, ge=1.0)
    
    # =========================================================================
    # Caching
    # =========================================================================
    cache_dir: Path = Field(default=Path(".cache"))
    cache_ttl_hours: int = Field(default=24, ge=1)
    
    # =========================================================================
    # Model Settings
    # =========================================================================
    min_matches_required: int = Field(
        default=5,
        ge=1,
        description="Minimum matches for team to get predictions (else fallback to league prior)",
    )
    random_seed: int = Field(default=42)
    
    # =========================================================================
    # Decision Layer
    # =========================================================================
    decision_mode: DecisionMode = Field(
        default=DecisionMode.PROBABILITY_ONLY,
        description="Operating mode: probability_only (safe) or stake_enabled",
    )
    edge_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Minimum edge (p - p_be) required to recommend",
    )
    confidence_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="P(p > p_be) must exceed this for recommendation",
    )
    kelly_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Fractional Kelly multiplier (0.25 = quarter Kelly)",
    )
    
    # =========================================================================
    # Logging
    # =========================================================================
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = Field(default=None)
    
    # =========================================================================
    # Validation
    # =========================================================================
    @field_validator("cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, v: str | Path) -> Path:
        """Convert string to Path and create if needed."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator("log_file", mode="before")
    @classmethod
    def validate_log_file(cls, v: Optional[str | Path]) -> Optional[Path]:
        """Convert string to Path and create parent dir if needed."""
        if v is None:
            return None
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience exports
settings = get_settings()
