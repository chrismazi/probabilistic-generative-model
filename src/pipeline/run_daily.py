"""
Daily pipeline runner.

Entry point for the complete daily workflow:
    ingest → quality checks → feature build → predictions

Usage:
    python -m src.pipeline.run_daily
    python -m src.pipeline.run_daily --leagues PL BL1
    python -m src.pipeline.run_daily --skip-features
"""

import argparse
import sys
from datetime import date, timedelta
from typing import Optional

from src.config import settings
from src.constants import SUPPORTED_LEAGUES, RELIABLE_LEAGUES
from src.pipeline import (
    Pipeline,
    PipelineResult,
    IngestStep,
    QualityCheckStep,
    FeatureBuildStep,
)
from src.utils.logging import setup_logging, get_logger

logger = get_logger("pipeline.daily")


def run_daily_pipeline(
    leagues: Optional[list[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    skip_features: bool = False,
    skip_predictions: bool = False,
    fail_on_quality_errors: bool = False,
) -> PipelineResult:
    """
    Run the complete daily pipeline.
    
    Steps:
        1. Ingest matches from yesterday and today
        2. Run data quality checks
        3. Build features (if not skipped)
        4. Generate predictions (if not skipped)
    
    Args:
        leagues: League codes to process (defaults to reliable leagues)
        date_from: Start date (defaults to yesterday)
        date_to: End date (defaults to today)
        skip_features: Skip feature building step
        skip_predictions: Skip prediction step
        fail_on_quality_errors: Stop pipeline on quality errors
        
    Returns:
        Pipeline result with all step outcomes
    """
    # Defaults
    leagues = leagues or list(RELIABLE_LEAGUES)
    date_from = date_from or (date.today() - timedelta(days=1))
    date_to = date_to or date.today()
    
    logger.info(f"Starting daily pipeline for {len(leagues)} leagues")
    logger.info(f"Date range: {date_from} to {date_to}")
    
    # Build pipeline
    pipeline = Pipeline(name="daily")
    
    # Step 1: Ingest
    pipeline.add_step(IngestStep(
        leagues=leagues,
        date_from=date_from,
        date_to=date_to,
    ))
    
    # Step 2: Quality checks
    pipeline.add_step(QualityCheckStep(
        fail_on_errors=fail_on_quality_errors,
    ))
    
    # Step 3: Feature building
    if not skip_features:
        pipeline.add_step(FeatureBuildStep(
            only_upcoming=True,
            window_size=10,
        ))
    
    # Step 4: Predictions (placeholder - to be implemented in Phase 4)
    if not skip_predictions:
        # TODO: Add PredictStep when Bayesian model is implemented
        pass
    
    # Execute
    result = pipeline.run(stop_on_error=True)
    
    # Log summary
    summary = result.summary()
    logger.info(f"Pipeline completed: {'SUCCESS' if result.success else 'FAILED'}")
    logger.info(f"Total duration: {summary['duration_seconds']:.2f}s")
    
    for step in summary["steps"]:
        status = "✓" if step["success"] else "✗"
        logger.info(
            f"  {status} {step['stage']}: "
            f"{step['records']} records, "
            f"{step['duration']:.2f}s"
        )
    
    return result


def run_season_backfill(
    leagues: list[str],
    season: int,
) -> PipelineResult:
    """
    Backfill a full season of data.
    
    Use this for initial data load or catching up on missed data.
    
    Args:
        leagues: League codes to process
        season: Season year (e.g., 2024 for 2024-25)
        
    Returns:
        Pipeline result
    """
    logger.info(f"Starting season backfill for {season}")
    
    pipeline = Pipeline(name=f"backfill_{season}")
    
    pipeline.add_step(IngestStep(
        leagues=leagues,
        season=season,
    ))
    
    pipeline.add_step(QualityCheckStep(fail_on_errors=False))
    
    return pipeline.run(stop_on_error=False)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run daily betting model pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.pipeline.run_daily
    python -m src.pipeline.run_daily --leagues PL BL1 SA
    python -m src.pipeline.run_daily --skip-features --skip-predictions
    python -m src.pipeline.run_daily --backfill 2024
        """,
    )
    
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="League codes to process (default: reliable leagues)",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature building step",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Skip prediction step",
    )
    parser.add_argument(
        "--fail-on-quality",
        action="store_true",
        help="Stop pipeline on quality check errors",
    )
    parser.add_argument(
        "--backfill",
        type=int,
        default=None,
        metavar="SEASON",
        help="Backfill a full season (e.g., 2024)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Use JSON log format",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        level="DEBUG" if args.debug else "INFO",
        json_format=args.json_logs,
    )
    
    # Validate API key
    if not settings.football_data_api_key:
        logger.error("FOOTBALL_DATA_API_KEY not set")
        sys.exit(1)
    
    # Run appropriate pipeline
    if args.backfill:
        leagues = args.leagues or list(RELIABLE_LEAGUES)
        result = run_season_backfill(leagues, args.backfill)
    else:
        result = run_daily_pipeline(
            leagues=args.leagues,
            date_from=date.fromisoformat(args.date_from) if args.date_from else None,
            date_to=date.fromisoformat(args.date_to) if args.date_to else None,
            skip_features=args.skip_features,
            skip_predictions=args.skip_predictions,
            fail_on_quality_errors=args.fail_on_quality,
        )
    
    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
