"""
Decision audit logging.

Logs every decision for:
- Accountability
- Backtesting
- Performance tracking
- Debugging bad decisions
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from sqlalchemy import text

from src.db import get_session
from src.decision.engine import BetDecision, DecisionOutcome
from src.utils import get_logger

logger = get_logger("decision.audit")


@dataclass
class AuditEntry:
    """Single audit log entry."""
    
    id: Optional[int]
    match_id: int
    decision_type: str
    decision_time: datetime
    
    # Prediction at decision time
    p_predicted: float
    p_ci_low: float
    p_ci_high: float
    
    # Odds and sizing
    odds: Optional[float]
    stake_fraction: float
    expected_value: float
    
    # Safety metrics
    prob_exceeds_breakeven: float
    
    # Outcome (filled later)
    actual_outcome: Optional[int] = None  # 1 if G2 > G1, else 0
    realized_pnl: Optional[float] = None
    
    # Metadata
    model_version: Optional[str] = None
    explanation: Optional[str] = None


class AuditLogger:
    """
    Audit logger for bet decisions.
    
    Responsibilities:
    - Log every decision
    - Join with actual outcomes
    - Compute realized P&L
    """
    
    def __init__(self):
        pass
    
    def log_decision(self, decision: BetDecision, model_version: str = "v1") -> int:
        """
        Log a bet decision to the database.
        
        Args:
            decision: The bet decision
            model_version: Model version string
            
        Returns:
            The audit log ID
        """
        with get_session() as session:
            result = session.execute(
                text("""
                    INSERT INTO decisions (
                        match_id, decision_type, decision_time,
                        p_predicted, p_ci_low, p_ci_high,
                        odds, stake_fraction, expected_value,
                        prob_exceeds_breakeven, model_version, explanation
                    )
                    VALUES (
                        :match_id, :decision_type, :decision_time,
                        :p_predicted, :p_ci_low, :p_ci_high,
                        :odds, :stake_fraction, :expected_value,
                        :prob_exceeds_breakeven, :model_version, :explanation
                    )
                    RETURNING id
                """),
                {
                    "match_id": decision.match_id,
                    "decision_type": decision.decision.value,
                    "decision_time": decision.timestamp,
                    "p_predicted": decision.p_2h_gt_1h,
                    "p_ci_low": decision.p_2h_gt_1h_ci[0],
                    "p_ci_high": decision.p_2h_gt_1h_ci[1],
                    "odds": decision.odds,
                    "stake_fraction": decision.stake_fraction,
                    "expected_value": decision.expected_value,
                    "prob_exceeds_breakeven": decision.prob_exceeds_breakeven,
                    "model_version": model_version,
                    "explanation": decision.reason,
                }
            )
            audit_id = result.fetchone()[0]
        
        logger.info(f"Logged decision for match {decision.match_id}: {decision.decision.value}")
        return audit_id
    
    def log_batch(self, decisions: List[BetDecision], model_version: str = "v1") -> List[int]:
        """Log multiple decisions."""
        return [self.log_decision(d, model_version) for d in decisions]
    
    def update_outcome(
        self,
        match_id: int,
        actual_outcome: int,
        realized_pnl: Optional[float] = None,
    ) -> None:
        """
        Update decision with actual outcome.
        
        Args:
            match_id: Match ID
            actual_outcome: 1 if G2 > G1, else 0
            realized_pnl: Realized P&L (if bet was placed)
        """
        with get_session() as session:
            session.execute(
                text("""
                    UPDATE decisions
                    SET actual_outcome = :outcome,
                        realized_pnl = :pnl
                    WHERE match_id = :match_id
                """),
                {
                    "match_id": match_id,
                    "outcome": actual_outcome,
                    "pnl": realized_pnl,
                }
            )
    
    def get_performance_summary(
        self,
        model_version: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get performance summary for decisions.
        
        Args:
            model_version: Filter by model version
            from_date: Start date
            to_date: End date
            
        Returns:
            Performance metrics
        """
        query = """
            SELECT 
                decision_type,
                COUNT(*) as count,
                SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) as with_outcome,
                AVG(CASE WHEN actual_outcome = 1 THEN 1.0 ELSE 0.0 END) as hit_rate,
                SUM(realized_pnl) as total_pnl,
                AVG(p_predicted) as avg_p,
                AVG(actual_outcome::float) as avg_outcome
            FROM decisions
            WHERE 1=1
        """
        
        params: dict = {}
        
        if model_version:
            query += " AND model_version = :version"
            params["version"] = model_version
        
        if from_date:
            query += " AND decision_time >= :from_date"
            params["from_date"] = from_date
        
        if to_date:
            query += " AND decision_time <= :to_date"
            params["to_date"] = to_date
        
        query += " GROUP BY decision_type"
        
        with get_session() as session:
            result = session.execute(text(query), params).fetchall()
        
        summary = {
            "by_decision_type": {},
            "total_decisions": 0,
            "total_bets": 0,
            "total_pnl": 0.0,
        }
        
        for row in result:
            decision_type = row[0]
            summary["by_decision_type"][decision_type] = {
                "count": row[1],
                "with_outcome": row[2],
                "hit_rate": float(row[3]) if row[3] else None,
                "total_pnl": float(row[4]) if row[4] else 0.0,
                "avg_p": float(row[5]) if row[5] else None,
                "avg_outcome": float(row[6]) if row[6] else None,
            }
            summary["total_decisions"] += row[1]
            
            if decision_type == "bet":
                summary["total_bets"] = row[1]
                summary["total_pnl"] = float(row[4]) if row[4] else 0.0
        
        return summary
    
    def get_calibration_by_bucket(
        self,
        model_version: Optional[str] = None,
        n_buckets: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get calibration by probability bucket.
        
        Args:
            model_version: Filter by model version
            n_buckets: Number of buckets
            
        Returns:
            List of bucket stats
        """
        bucket_size = 1.0 / n_buckets
        
        query = """
            SELECT 
                FLOOR(p_predicted / :bucket_size) * :bucket_size as bucket,
                COUNT(*) as count,
                AVG(p_predicted) as avg_p,
                AVG(actual_outcome::float) as avg_outcome
            FROM decisions
            WHERE actual_outcome IS NOT NULL
        """
        
        params: dict = {"bucket_size": bucket_size}
        
        if model_version:
            query += " AND model_version = :version"
            params["version"] = model_version
        
        query += " GROUP BY bucket ORDER BY bucket"
        
        with get_session() as session:
            result = session.execute(text(query), params).fetchall()
        
        return [
            {
                "bucket": f"{row[0]:.1f}-{row[0] + bucket_size:.1f}",
                "count": row[1],
                "avg_predicted": float(row[2]) if row[2] else None,
                "avg_actual": float(row[3]) if row[3] else None,
            }
            for row in result
        ]


def get_audit_logger() -> AuditLogger:
    """Get an audit logger instance."""
    return AuditLogger()
