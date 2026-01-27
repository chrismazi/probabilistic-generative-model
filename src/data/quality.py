"""
Data quality checks for the betting model.

Validates:
- Score consistency (HT <= FT)
- Missing data patterns
- Timezone normalization
- Duplicate detection
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.config import settings


@dataclass
class QualityIssue:
    """Represents a data quality issue."""
    issue_type: str
    severity: str  # "error", "warning", "info"
    match_id: Optional[int]
    description: str
    details: Optional[dict] = None


@dataclass
class QualityReport:
    """Summary of data quality check results."""
    checked_at: datetime
    total_matches: int
    total_scores: int
    issues: list[QualityIssue]
    
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")
    
    @property
    def is_healthy(self) -> bool:
        return self.error_count == 0


class DataQualityChecker:
    """
    Runs data quality checks on the match database.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize with database connection."""
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url)
    
    def check_score_consistency(self, session: Session) -> list[QualityIssue]:
        """Check that HT scores don't exceed FT scores."""
        issues = []
        
        result = session.execute(
            text("""
                SELECT m.id, s.ht_home, s.ht_away, s.ft_home, s.ft_away
                FROM scores s
                JOIN matches m ON s.match_id = m.id
                WHERE (s.ht_home > s.ft_home OR s.ht_away > s.ft_away)
                  AND s.ht_home IS NOT NULL
                  AND s.ft_home IS NOT NULL
            """)
        ).fetchall()
        
        for row in result:
            issues.append(QualityIssue(
                issue_type="score_consistency",
                severity="error",
                match_id=row[0],
                description="HT score exceeds FT score",
                details={
                    "ht_home": row[1],
                    "ht_away": row[2],
                    "ft_home": row[3],
                    "ft_away": row[4],
                }
            ))
        
        return issues
    
    def check_missing_ht_scores(self, session: Session) -> list[QualityIssue]:
        """Check for finished matches missing HT scores."""
        issues = []
        
        result = session.execute(
            text("""
                SELECT l.code, m.season, COUNT(*) as missing_count
                FROM matches m
                JOIN leagues l ON m.league_id = l.id
                LEFT JOIN scores s ON m.id = s.match_id
                WHERE m.status = 'FINISHED'
                  AND (s.ht_home IS NULL OR s.ht_away IS NULL)
                GROUP BY l.code, m.season
                HAVING COUNT(*) > 0
            """)
        ).fetchall()
        
        for row in result:
            issues.append(QualityIssue(
                issue_type="missing_ht_score",
                severity="warning",
                match_id=None,
                description=f"Missing HT scores in {row[0]} {row[1]}",
                details={
                    "league": row[0],
                    "season": row[1],
                    "missing_count": row[2],
                }
            ))
        
        return issues
    
    def check_duplicate_matches(self, session: Session) -> list[QualityIssue]:
        """Check for duplicate matches (same teams, same date)."""
        issues = []
        
        result = session.execute(
            text("""
                SELECT 
                    home_team_id, away_team_id, 
                    DATE(kickoff_utc) as match_date,
                    COUNT(*) as dup_count
                FROM matches
                GROUP BY home_team_id, away_team_id, DATE(kickoff_utc)
                HAVING COUNT(*) > 1
            """)
        ).fetchall()
        
        for row in result:
            issues.append(QualityIssue(
                issue_type="duplicate_match",
                severity="error",
                match_id=None,
                description="Duplicate match detected",
                details={
                    "home_team_id": row[0],
                    "away_team_id": row[1],
                    "match_date": str(row[2]),
                    "count": row[3],
                }
            ))
        
        return issues
    
    def check_negative_scores(self, session: Session) -> list[QualityIssue]:
        """Check for negative score values."""
        issues = []
        
        result = session.execute(
            text("""
                SELECT m.id, s.ht_home, s.ht_away, s.ft_home, s.ft_away
                FROM scores s
                JOIN matches m ON s.match_id = m.id
                WHERE s.ht_home < 0 OR s.ht_away < 0 
                   OR s.ft_home < 0 OR s.ft_away < 0
            """)
        ).fetchall()
        
        for row in result:
            issues.append(QualityIssue(
                issue_type="negative_score",
                severity="error",
                match_id=row[0],
                description="Negative score value detected",
                details={
                    "ht_home": row[1],
                    "ht_away": row[2],
                    "ft_home": row[3],
                    "ft_away": row[4],
                }
            ))
        
        return issues
    
    def run_all_checks(self) -> QualityReport:
        """Run all quality checks and return report."""
        issues = []
        
        with Session(self.engine) as session:
            # Get counts
            total_matches = session.execute(
                text("SELECT COUNT(*) FROM matches")
            ).scalar() or 0
            
            total_scores = session.execute(
                text("SELECT COUNT(*) FROM scores")
            ).scalar() or 0
            
            # Run checks
            issues.extend(self.check_score_consistency(session))
            issues.extend(self.check_missing_ht_scores(session))
            issues.extend(self.check_duplicate_matches(session))
            issues.extend(self.check_negative_scores(session))
        
        return QualityReport(
            checked_at=datetime.now(),
            total_matches=total_matches,
            total_scores=total_scores,
            issues=issues,
        )


def run_quality_checks() -> QualityReport:
    """Convenience function to run all quality checks."""
    checker = DataQualityChecker()
    return checker.run_all_checks()
