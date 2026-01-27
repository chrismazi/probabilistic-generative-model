"""
Backtesting framework for walk-forward evaluation.

Provides:
- Time-based train/test splits
- Walk-forward validation
- Out-of-sample evaluation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sqlalchemy import text

from src.db import get_session
from src.bayesian.model import HalfGoalModel, MatchData, TrainingData
from src.bayesian.priors import ModelConfig
from src.evaluation.metrics import BrierScore, CalibrationResult
from src.evaluation.comparison import ModelEvaluator, ModelEvaluation
from src.utils import get_logger, AsOfDate

logger = get_logger("evaluation.backtesting")


@dataclass
class BacktestFold:
    """Single fold of backtest."""
    
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    n_train: int = 0
    n_test: int = 0
    
    # Results
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    outcomes: np.ndarray = field(default_factory=lambda: np.array([]))
    evaluation: Optional[ModelEvaluation] = None


@dataclass
class BacktestResult:
    """Complete backtest result."""
    
    folds: List[BacktestFold]
    aggregate_evaluation: ModelEvaluation
    model_diagnostics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_folds": len(self.folds),
            "total_test_matches": sum(f.n_test for f in self.folds),
            "aggregate": self.aggregate_evaluation.to_dict(),
            "diagnostics": self.model_diagnostics,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_period": f"{f.train_start.date()} to {f.train_end.date()}",
                    "test_period": f"{f.test_start.date()} to {f.test_end.date()}",
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    "brier_score": f.evaluation.brier_score.score if f.evaluation else None,
                }
                for f in self.folds
            ],
        }


class WalkForwardBacktester:
    """
    Walk-forward backtesting.
    
    Trains model on historical data, predicts on future window,
    then advances in time. This simulates real deployment.
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        min_train_matches: int = 200,
        test_window_days: int = 30,
        retrain_frequency_days: int = 30,
    ):
        """
        Initialize backtester.
        
        Args:
            model_config: Model configuration
            min_train_matches: Minimum training matches required
            test_window_days: Size of test window in days
            retrain_frequency_days: How often to retrain
        """
        self.model_config = model_config or ModelConfig()
        self.min_train_matches = min_train_matches
        self.test_window_days = test_window_days
        self.retrain_frequency_days = retrain_frequency_days
        
        self.evaluator = ModelEvaluator()
    
    def generate_folds(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[BacktestFold]:
        """
        Generate walk-forward folds.
        
        Args:
            start_date: Earliest training data date
            end_date: Latest test date
            
        Returns:
            List of BacktestFold
        """
        folds = []
        fold_id = 0
        
        current_date = start_date + timedelta(days=180)  # Initial training period
        
        while current_date < end_date:
            train_end = current_date
            test_start = current_date
            test_end = min(
                current_date + timedelta(days=self.test_window_days),
                end_date
            )
            
            folds.append(BacktestFold(
                fold_id=fold_id,
                train_start=start_date,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
            
            fold_id += 1
            current_date += timedelta(days=self.retrain_frequency_days)
        
        return folds
    
    def run_backtest(
        self,
        league_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_folds: int = 10,
    ) -> BacktestResult:
        """
        Run full walk-forward backtest.
        
        Args:
            league_id: League to backtest
            start_date: Start of training data
            end_date: End of test data
            max_folds: Maximum number of folds
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Starting backtest for league {league_id}")
        
        # Get date range from data if not specified
        if start_date is None or end_date is None:
            dates = self._get_date_range(league_id)
            start_date = start_date or dates[0]
            end_date = end_date or dates[1]
        
        # Generate folds
        folds = self.generate_folds(start_date, end_date)[:max_folds]
        logger.info(f"Generated {len(folds)} folds")
        
        all_predictions = []
        all_outcomes = []
        diagnostics_list = []
        
        for fold in folds:
            logger.info(f"Running fold {fold.fold_id}")
            
            # Load training data
            train_matches = self._load_matches(
                league_id,
                fold.train_start,
                fold.train_end,
            )
            
            if len(train_matches) < self.min_train_matches:
                logger.warning(f"Fold {fold.fold_id}: insufficient training data")
                continue
            
            # Load test data
            test_matches = self._load_matches(
                league_id,
                fold.test_start,
                fold.test_end,
            )
            
            if len(test_matches) == 0:
                logger.warning(f"Fold {fold.fold_id}: no test data")
                continue
            
            fold.n_train = len(train_matches)
            fold.n_test = len(test_matches)
            
            # Train model
            model = HalfGoalModel(self.model_config)
            train_data = TrainingData.from_matches(train_matches)
            
            try:
                model.fit(train_data)
                diagnostics = model.get_diagnostics()
                diagnostics_list.append(diagnostics)
            except Exception as e:
                logger.error(f"Fold {fold.fold_id} training failed: {e}")
                continue
            
            # Generate predictions
            predictions = []
            outcomes = []
            
            for match in test_matches:
                try:
                    pred = model.predict_match(
                        home_team_idx=match.home_team_idx,
                        away_team_idx=match.away_team_idx,
                    )
                    predictions.append(pred["p_2h_gt_1h"])
                    
                    # Compute actual outcome
                    g1 = (match.g1_home or 0) + (match.g1_away or 0)
                    g2 = (match.g2_home or 0) + (match.g2_away or 0)
                    outcomes.append(1 if g2 > g1 else 0)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for match {match.match_id}: {e}")
            
            if predictions:
                fold.predictions = np.array(predictions)
                fold.outcomes = np.array(outcomes)
                fold.evaluation = self.evaluator.evaluate_model(
                    fold.predictions, fold.outcomes, f"fold_{fold.fold_id}"
                )
                
                all_predictions.extend(predictions)
                all_outcomes.extend(outcomes)
        
        # Aggregate results
        if all_predictions:
            aggregate_eval = self.evaluator.evaluate_model(
                np.array(all_predictions),
                np.array(all_outcomes),
                "aggregate",
            )
        else:
            aggregate_eval = ModelEvaluation(
                model_name="aggregate",
                n_matches=0,
                brier_score=BrierScore(0, 0, 0, 0),
                log_loss=0,
                accuracy=0,
                calibration=CalibrationResult([], 0, 0),
            )
        
        # Aggregate diagnostics
        agg_diagnostics = {
            "n_folds_completed": len([f for f in folds if f.evaluation]),
            "total_divergences": sum(d.get("n_divergences", 0) for d in diagnostics_list),
            "max_rhat": max((d.get("max_rhat", 1.0) for d in diagnostics_list), default=1.0),
            "min_ess": min((d.get("min_ess_bulk", 0) for d in diagnostics_list), default=0),
            "all_healthy": all(d.get("is_healthy", False) for d in diagnostics_list),
        }
        
        return BacktestResult(
            folds=folds,
            aggregate_evaluation=aggregate_eval,
            model_diagnostics=agg_diagnostics,
        )
    
    def _get_date_range(self, league_id: int) -> Tuple[datetime, datetime]:
        """Get date range for a league."""
        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT MIN(kickoff_utc), MAX(kickoff_utc)
                    FROM matches
                    WHERE league_id = :league_id AND status = 'FINISHED'
                """),
                {"league_id": league_id}
            ).fetchone()
        
        if result and result[0] and result[1]:
            return result[0], result[1]
        
        return datetime(2020, 1, 1), datetime.now()
    
    def _load_matches(
        self,
        league_id: int,
        start_date: datetime,
        end_date: datetime,
    ) -> List[MatchData]:
        """Load matches for training/testing."""
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
                      AND m.kickoff_utc >= :start_date
                      AND m.kickoff_utc < :end_date
                      AND s.ht_home IS NOT NULL
                    ORDER BY m.kickoff_utc
                """),
                {
                    "league_id": league_id,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            ).fetchall()
        
        matches = []
        for row in result:
            matches.append(MatchData(
                match_id=row[0],
                league_idx=league_id,
                home_team_idx=row[1],
                away_team_idx=row[2],
                g1_home=row[3],
                g1_away=row[4],
                g2_home=row[5] - row[3],  # 2H goals = FT - HT
                g2_away=row[6] - row[4],
            ))
        
        return matches
