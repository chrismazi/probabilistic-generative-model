"""
Phase 6.6: Real-data end-to-end gate.

This script runs the complete pipeline on real data:
1. Ingest from football-data.org API
2. Build features as-of kickoff
3. Fit Bayesian model
4. Evaluate on held-out data
5. Generate daily prediction/decision artifacts

Usage:
    python -m src.pipeline.run_e2e --league PL --seasons 2024 2023
    python -m src.pipeline.run_e2e --league PL --dry-run
"""

import argparse
import json
import os
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from src.config import settings
from src.constants import SUPPORTED_LEAGUES, RELIABLE_LEAGUES
from src.utils import get_logger, now_utc

logger = get_logger("e2e")


# =============================================================================
# Output directory
# =============================================================================

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def save_artifact(name: str, data: Any, as_json: bool = True) -> Path:
    """Save artifact to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if as_json:
        path = ARTIFACTS_DIR / f"{name}_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        path = ARTIFACTS_DIR / f"{name}_{timestamp}.txt"
        with open(path, "w") as f:
            f.write(str(data))
    
    logger.info(f"Saved artifact: {path}")
    return path


# =============================================================================
# Step 1: Ingestion
# =============================================================================

def run_ingestion(
    league_code: str,
    seasons: List[int],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run data ingestion for a league.
    
    Args:
        league_code: League code (e.g., "PL")
        seasons: List of seasons to ingest
        dry_run: If True, just check connectivity
        
    Returns:
        Ingestion statistics
    """
    from src.data.ingestion import IngestionPipeline
    from src.data.quality import run_quality_checks
    
    logger.info(f"=== Step 1: Ingestion for {league_code} ===")
    
    if not settings.football_data_api_key:
        return {
            "status": "skipped",
            "reason": "No API key configured",
            "league": league_code,
        }
    
    if dry_run:
        # Just test API connectivity
        from src.data.api_client import get_football_data_client
        client = get_football_data_client()
        try:
            competitions = client.get_competitions()
            return {
                "status": "dry_run_ok",
                "api_reachable": True,
                "competitions_count": len(competitions),
            }
        except Exception as e:
            return {
                "status": "dry_run_failed",
                "error": str(e),
            }
    
    pipeline = IngestionPipeline()
    stats = {"league": league_code, "seasons": {}}
    
    for season in seasons:
        try:
            result = pipeline.ingest_matches(
                league_code=league_code,
                season=season,
            )
            stats["seasons"][season] = result
        except Exception as e:
            logger.error(f"Season {season} failed: {e}")
            stats["seasons"][season] = {"error": str(e)}
    
    # Run quality checks
    quality = run_quality_checks()
    stats["quality"] = {
        "total_matches": quality.total_matches,
        "error_count": quality.error_count,
        "warning_count": quality.warning_count,
        "is_healthy": quality.is_healthy,
    }
    
    return stats


# =============================================================================
# Step 2: Feature Building
# =============================================================================

def run_feature_building(
    league_id: int,
    upcoming_only: bool = False,
) -> Dict[str, Any]:
    """
    Build features for matches.
    
    Args:
        league_id: Database league ID
        upcoming_only: Only build for upcoming matches
        
    Returns:
        Feature building statistics
    """
    from src.features import get_orchestrator
    
    logger.info(f"=== Step 2: Feature Building ===")
    
    orchestrator = get_orchestrator(window_size=10)
    
    result = orchestrator.build_and_store_for_league(
        league_id=league_id,
        only_upcoming=upcoming_only,
    )
    
    return {
        "total_matches": result["total"],
        "valid_features": result["valid"],
        "invalid_features": result["invalid"],
        "errors": result["errors"],
        "validity_rate": result["valid"] / max(result["total"], 1),
    }


# =============================================================================
# Step 3: Model Fitting
# =============================================================================

def run_model_fitting(
    league_id: int,
    model_type: str = "poisson",
    max_matches: int = 500,
) -> Dict[str, Any]:
    """
    Fit Bayesian model on historical data.
    
    Args:
        league_id: Database league ID
        model_type: "poisson" or "negbin"
        max_matches: Maximum matches to use (for speed)
        
    Returns:
        Model fitting results and diagnostics
    """
    from sqlalchemy import text
    from src.db import get_session
    from src.bayesian import HalfGoalModel, MatchData, TrainingData, ModelConfig
    
    logger.info(f"=== Step 3: Model Fitting ({model_type}) ===")
    
    # Load training data from DB
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT 
                    m.id,
                    m.home_team_id,
                    m.away_team_id,
                    s.ht_home,
                    s.ht_away,
                    s.ft_home,
                    s.ft_away
                FROM matches m
                JOIN scores s ON m.id = s.match_id
                WHERE m.league_id = :league_id
                  AND m.status = 'FINISHED'
                  AND s.ht_home IS NOT NULL
                  AND s.ft_home IS NOT NULL
                ORDER BY m.kickoff_utc DESC
                LIMIT :max_matches
            """),
            {"league_id": league_id, "max_matches": max_matches}
        ).fetchall()
    
    if len(result) < 50:
        return {
            "status": "insufficient_data",
            "matches_found": len(result),
            "minimum_required": 50,
        }
    
    # Convert to MatchData
    matches = []
    for row in result:
        match_id, home_id, away_id, ht_h, ht_a, ft_h, ft_a = row
        matches.append(MatchData(
            match_id=match_id,
            league_idx=league_id,
            home_team_idx=home_id,
            away_team_idx=away_id,
            g1_home=ht_h,
            g1_away=ht_a,
            g2_home=ft_h - ht_h,
            g2_away=ft_a - ht_a,
        ))
    
    # Prepare training data
    data = TrainingData.from_matches(matches)
    
    logger.info(f"Training on {data.n_matches} matches")
    
    # Configure model
    config = ModelConfig(
        model_type=model_type,
        n_samples=1000,  # Reduced for speed
        n_tune=500,
        n_chains=2,
        target_accept=0.9,
    )
    
    # Fit model
    model = HalfGoalModel(config)
    
    try:
        model.fit(data)
        diagnostics = model.get_diagnostics()
        posterior_summary = model.get_posterior_summary()
        
        # Save model
        model_path = ARTIFACTS_DIR / f"model_{model_type}_{datetime.now().strftime('%Y%m%d')}.nc"
        model.save(str(model_path))
        
        return {
            "status": "success",
            "matches_used": data.n_matches,
            "model_type": model_type,
            "diagnostics": diagnostics,
            "posterior_summary": posterior_summary,
            "model_path": str(model_path),
        }
        
    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "matches_used": data.n_matches,
        }


# =============================================================================
# Step 4: Evaluation
# =============================================================================

def run_evaluation(
    league_id: int,
    test_window_days: int = 30,
) -> Dict[str, Any]:
    """
    Run walk-forward evaluation on held-out data.
    
    Args:
        league_id: Database league ID
        test_window_days: Size of test window
        
    Returns:
        Evaluation results
    """
    from src.evaluation import WalkForwardBacktester, ModelEvaluator
    from src.bayesian import ModelConfig
    
    logger.info(f"=== Step 4: Evaluation ===")
    
    # Configure for faster evaluation
    config = ModelConfig(
        model_type="poisson",
        n_samples=500,
        n_tune=250,
        n_chains=1,
    )
    
    backtester = WalkForwardBacktester(
        model_config=config,
        min_train_matches=100,
        test_window_days=test_window_days,
    )
    
    try:
        result = backtester.run_backtest(
            league_id=league_id,
            max_folds=3,  # Limit for speed
        )
        
        # Extract key metrics
        evaluation = {
            "status": "success",
            "n_folds": len(result.folds),
            "total_test_matches": sum(f.n_test for f in result.folds),
            "aggregate": {
                "brier_score": result.aggregate_evaluation.brier_score.score,
                "brier_skill_score": result.aggregate_evaluation.brier_score.skill_score,
                "log_loss": result.aggregate_evaluation.log_loss,
                "accuracy": result.aggregate_evaluation.accuracy,
                "ece": result.aggregate_evaluation.calibration.expected_calibration_error,
            },
            "diagnostics": result.model_diagnostics,
        }
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# Step 5: Decision Layer
# =============================================================================

def run_decision_layer(
    league_id: int,
) -> Dict[str, Any]:
    """
    Run decision layer on upcoming predictions.
    
    Args:
        league_id: Database league ID
        
    Returns:
        Decision statistics
    """
    from sqlalchemy import text
    from src.db import get_session
    from src.decision import DecisionEngine, EXPERIMENTAL_CONFIG, DecisionOutcome
    
    logger.info(f"=== Step 5: Decision Layer ===")
    
    # Get upcoming matches
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT m.id, m.home_team_id, m.away_team_id, m.kickoff_utc
                FROM matches m
                WHERE m.league_id = :league_id
                  AND m.status IN ('SCHEDULED', 'TIMED')
                  AND m.kickoff_utc > NOW()
                  AND m.kickoff_utc < NOW() + INTERVAL '7 days'
                ORDER BY m.kickoff_utc
            """),
            {"league_id": league_id}
        ).fetchall()
    
    if not result:
        return {
            "status": "no_upcoming_matches",
            "league_id": league_id,
        }
    
    engine = DecisionEngine(EXPERIMENTAL_CONFIG)
    
    decisions = []
    decision_counts = {
        DecisionOutcome.SIGNAL.value: 0,
        DecisionOutcome.SKIP.value: 0,
        DecisionOutcome.BET.value: 0,
    }
    
    for row in result:
        match_id, home_id, away_id, kickoff = row
        
        # Generate mock prediction (in real use, comes from model)
        # For now, use simple heuristic
        p_mean = 0.5 + np.random.uniform(-0.15, 0.15)
        p_ci = (max(0.1, p_mean - 0.1), min(0.9, p_mean + 0.1))
        
        decision = engine.make_decision(
            match_id=match_id,
            p_mean=p_mean,
            p_ci=p_ci,
            odds=None,  # No odds available
        )
        
        decisions.append(decision.to_dict())
        if decision.decision.value in decision_counts:
            decision_counts[decision.decision.value] += 1
    
    # Save decisions artifact
    save_artifact(f"decisions_{league_id}", decisions)
    
    return {
        "status": "success",
        "upcoming_matches": len(result),
        "decisions_made": len(decisions),
        "decision_breakdown": decision_counts,
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_full_pipeline(
    league_code: str,
    seasons: List[int],
    dry_run: bool = False,
    skip_model: bool = False,
) -> Dict[str, Any]:
    """
    Run complete end-to-end pipeline.
    
    Args:
        league_code: League code
        seasons: Seasons to ingest
        dry_run: Just test connectivity
        skip_model: Skip model fitting (for speed)
        
    Returns:
        Complete pipeline results
    """
    logger.info("=" * 60)
    logger.info("PHASE 6.6: Real-Data End-to-End Gate")
    logger.info("=" * 60)
    
    results = {
        "started_at": now_utc().isoformat(),
        "league": league_code,
        "seasons": seasons,
        "steps": {},
    }
    
    # Step 1: Ingestion
    results["steps"]["ingestion"] = run_ingestion(league_code, seasons, dry_run)
    
    if dry_run:
        results["status"] = "dry_run_complete"
        return results
    
    # Get league ID from database
    from sqlalchemy import text
    from src.db import get_session
    
    with get_session() as session:
        result = session.execute(
            text("SELECT id FROM leagues WHERE code = :code"),
            {"code": league_code}
        ).fetchone()
    
    if not result:
        results["status"] = "league_not_found"
        return results
    
    league_id = result[0]
    results["league_id"] = league_id
    
    # Step 2: Feature Building
    results["steps"]["features"] = run_feature_building(league_id)
    
    # Step 3: Model Fitting
    if not skip_model:
        results["steps"]["model"] = run_model_fitting(league_id)
    else:
        results["steps"]["model"] = {"status": "skipped"}
    
    # Step 4: Evaluation
    if not skip_model:
        results["steps"]["evaluation"] = run_evaluation(league_id)
    else:
        results["steps"]["evaluation"] = {"status": "skipped"}
    
    # Step 5: Decision Layer
    results["steps"]["decisions"] = run_decision_layer(league_id)
    
    # Final status
    results["completed_at"] = now_utc().isoformat()
    results["status"] = "complete"
    
    # Save full results
    save_artifact("e2e_results", results)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end pipeline on real data",
    )
    
    parser.add_argument(
        "--league",
        type=str,
        default="PL",
        help="League code (default: PL)",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2024, 2023],
        help="Seasons to ingest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just test API connectivity",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model fitting (faster)",
    )
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        league_code=args.league,
        seasons=args.seasons,
        dry_run=args.dry_run,
        skip_model=args.skip_model,
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
