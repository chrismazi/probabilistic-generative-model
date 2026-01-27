"""
Constants and enums for the betting model.

Centralizes provider-specific field names, statuses, and configuration
to make parsing stable when providers change.
"""

from enum import Enum
from typing import Dict, Tuple


# =============================================================================
# Match Status Enum
# =============================================================================

class MatchStatusEnum(str, Enum):
    """
    Match status values.
    
    Aligned with football-data.org statuses.
    """
    SCHEDULED = "SCHEDULED"
    TIMED = "TIMED"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    SUSPENDED = "SUSPENDED"
    CANCELLED = "CANCELLED"
    AWARDED = "AWARDED"
    
    @classmethod
    def terminal_statuses(cls) -> set["MatchStatusEnum"]:
        """Statuses where the match is complete."""
        return {cls.FINISHED, cls.CANCELLED, cls.AWARDED}
    
    @classmethod
    def active_statuses(cls) -> set["MatchStatusEnum"]:
        """Statuses where the match is in progress."""
        return {cls.IN_PLAY, cls.PAUSED}
    
    @classmethod
    def pending_statuses(cls) -> set["MatchStatusEnum"]:
        """Statuses where the match hasn't started."""
        return {cls.SCHEDULED, cls.TIMED}


class DecisionRuleEnum(str, Enum):
    """Decision rule that triggered an action."""
    EDGE_THRESHOLD = "edge_threshold"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    UNCERTAINTY_FILTER = "uncertainty_filter"
    MINIMUM_HISTORY = "minimum_history"
    NO_ACTION = "no_action"


class DecisionOutcomeEnum(str, Enum):
    """Outcome of a decision."""
    WIN = "WIN"
    LOSE = "LOSE"
    VOID = "VOID"
    PENDING = "PENDING"


# =============================================================================
# Supported Leagues
# =============================================================================

# Format: code -> (name, country)
SUPPORTED_LEAGUES: Dict[str, Tuple[str, str]] = {
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

# Leagues with reliable HT data (>90% coverage historically)
RELIABLE_LEAGUES = {"PL", "BL1", "SA", "PD", "FL1", "CL"}


# =============================================================================
# Provider Field Mappings (football-data.org v4)
# =============================================================================

class FootballDataFields:
    """Field names from football-data.org API."""
    
    # Match object
    MATCH_ID = "id"
    UTC_DATE = "utcDate"
    STATUS = "status"
    MATCHDAY = "matchday"
    
    # Team object
    TEAM_ID = "id"
    TEAM_NAME = "name"
    TEAM_SHORT_NAME = "shortName"
    TEAM_TLA = "tla"
    
    # Score object
    FULL_TIME = "fullTime"
    HALF_TIME = "halfTime"
    SCORE_HOME = "home"
    SCORE_AWAY = "away"
    
    # Competition object
    COMPETITION_CODE = "code"
    COMPETITION_NAME = "name"
    
    # Season object
    SEASON_START = "startDate"
    SEASON_END = "endDate"


# =============================================================================
# Feature Engineering Constants
# =============================================================================

# Minimum matches required for team predictions
MIN_MATCHES_DEFAULT = 5

# Rolling window sizes for feature computation
ROLLING_WINDOWS = [5, 10, 20]

# Time decay lambda for exponential weighting (days)
TIME_DECAY_LAMBDA = 0.01


# =============================================================================
# Model Constants
# =============================================================================

# Prior hyperparameters for Bayesian model
PRIOR_ATTACK_MEAN = 0.0
PRIOR_ATTACK_SD = 1.0
PRIOR_DEFENSE_MEAN = 0.0
PRIOR_DEFENSE_SD = 1.0
PRIOR_HOME_ADVANTAGE_MEAN = 0.3
PRIOR_HOME_ADVANTAGE_SD = 0.2

# Half-time to full-time intensity ratio (typical)
HALF_INTENSITY_RATIO = 0.48  # First half typically has ~48% of goals


# =============================================================================
# Evaluation Constants
# =============================================================================

# Confidence levels for credible intervals
CREDIBLE_INTERVAL_LEVELS = [0.50, 0.80, 0.90, 0.95]

# Calibration bins
CALIBRATION_BINS = 10

# Minimum sample size for reliable calibration estimate
MIN_CALIBRATION_SAMPLES = 30


# =============================================================================
# Decision Layer Constants
# =============================================================================

# Default edge threshold (p - p_be must exceed this)
DEFAULT_EDGE_THRESHOLD = 0.05

# Default confidence threshold for P(p > p_be)
DEFAULT_CONFIDENCE_THRESHOLD = 0.95

# Default Kelly fraction (quarter Kelly)
DEFAULT_KELLY_FRACTION = 0.25
