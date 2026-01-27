"""
Decision module.

Provides:
- Kelly criterion and bankroll management
- Bet/no-bet decision engine
- Audit logging
"""

from src.decision.kelly import (
    kelly_fraction,
    fractional_kelly,
    break_even_probability,
    expected_value,
    bayesian_kelly,
    KellyResult,
    BayesianKellyResult,
)
from src.decision.engine import (
    DecisionOutcome,
    DecisionConfig,
    BetDecision,
    DecisionEngine,
    get_decision_engine,
    EXPERIMENTAL_CONFIG,
    PRODUCTION_CONFIG,
)
from src.decision.audit import (
    AuditEntry,
    AuditLogger,
    get_audit_logger,
)

__all__ = [
    # Kelly
    "kelly_fraction",
    "fractional_kelly",
    "break_even_probability",
    "expected_value",
    "bayesian_kelly",
    "KellyResult",
    "BayesianKellyResult",
    # Engine
    "DecisionOutcome",
    "DecisionConfig",
    "BetDecision",
    "DecisionEngine",
    "get_decision_engine",
    # Audit
    "AuditEntry",
    "AuditLogger",
    "get_audit_logger",
]
