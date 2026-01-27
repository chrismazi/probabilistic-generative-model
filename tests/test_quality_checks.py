"""
Tests for data quality checks.

Critical tests for:
- HT ≤ FT constraint enforcement
- Missing data handling
- Duplicate detection
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data.quality import (
    DataQualityChecker,
    QualityIssue,
    QualityReport,
)


class TestQualityIssue:
    """Test QualityIssue dataclass."""
    
    def test_create_error_issue(self):
        """Test creating an error-level issue."""
        issue = QualityIssue(
            issue_type="score_consistency",
            severity="error",
            match_id=123,
            description="HT score exceeds FT score",
            details={"ht_home": 3, "ft_home": 2},
        )
        
        assert issue.severity == "error"
        assert issue.match_id == 123
        assert issue.details["ht_home"] == 3
    
    def test_create_warning_issue(self):
        """Test creating a warning-level issue."""
        issue = QualityIssue(
            issue_type="missing_ht_score",
            severity="warning",
            match_id=None,
            description="Missing HT scores in PL 2024-25",
        )
        
        assert issue.severity == "warning"
        assert issue.match_id is None


class TestQualityReport:
    """Test QualityReport dataclass."""
    
    def test_empty_report_is_healthy(self):
        """Report with no issues is healthy."""
        report = QualityReport(
            checked_at=datetime.now(),
            total_matches=100,
            total_scores=100,
            issues=[],
        )
        
        assert report.is_healthy is True
        assert report.error_count == 0
        assert report.warning_count == 0
    
    def test_report_with_errors_not_healthy(self):
        """Report with errors is not healthy."""
        report = QualityReport(
            checked_at=datetime.now(),
            total_matches=100,
            total_scores=100,
            issues=[
                QualityIssue(
                    issue_type="score_consistency",
                    severity="error",
                    match_id=1,
                    description="test error",
                ),
            ],
        )
        
        assert report.is_healthy is False
        assert report.error_count == 1
    
    def test_report_with_only_warnings_is_healthy(self):
        """Report with only warnings is still healthy."""
        report = QualityReport(
            checked_at=datetime.now(),
            total_matches=100,
            total_scores=100,
            issues=[
                QualityIssue(
                    issue_type="missing_ht",
                    severity="warning",
                    match_id=None,
                    description="test warning",
                ),
            ],
        )
        
        assert report.is_healthy is True
        assert report.warning_count == 1


class TestScoreConsistencyCheck:
    """Test HT ≤ FT constraint checking."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return MagicMock()
    
    @pytest.fixture
    def checker(self):
        """Create checker with mocked DB."""
        with patch("src.data.quality.create_engine"):
            return DataQualityChecker("postgresql://test:test@localhost/test")
    
    def test_detects_ht_exceeds_ft_home(self, checker, mock_session):
        """Detect when HT home score exceeds FT home score."""
        # Mock query result: ht_home=3, ft_home=2 (invalid)
        mock_session.execute.return_value.fetchall.return_value = [
            (1, 3, 0, 2, 1)  # match_id, ht_home, ht_away, ft_home, ft_away
        ]
        
        issues = checker.check_score_consistency(mock_session)
        
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].issue_type == "score_consistency"
        assert issues[0].match_id == 1
    
    def test_detects_ht_exceeds_ft_away(self, checker, mock_session):
        """Detect when HT away score exceeds FT away score."""
        mock_session.execute.return_value.fetchall.return_value = [
            (2, 0, 2, 1, 1)  # ht_away=2, ft_away=1 (invalid)
        ]
        
        issues = checker.check_score_consistency(mock_session)
        
        assert len(issues) == 1
        assert issues[0].match_id == 2
    
    def test_valid_scores_no_issues(self, checker, mock_session):
        """Valid scores produce no issues."""
        mock_session.execute.return_value.fetchall.return_value = []
        
        issues = checker.check_score_consistency(mock_session)
        
        assert len(issues) == 0
    
    def test_multiple_invalid_scores(self, checker, mock_session):
        """Detect multiple invalid scores."""
        mock_session.execute.return_value.fetchall.return_value = [
            (1, 3, 0, 2, 1),
            (2, 0, 2, 1, 1),
            (3, 2, 2, 1, 1),
        ]
        
        issues = checker.check_score_consistency(mock_session)
        
        assert len(issues) == 3


class TestNegativeScoreCheck:
    """Test negative score detection."""
    
    @pytest.fixture
    def checker(self):
        with patch("src.data.quality.create_engine"):
            return DataQualityChecker("postgresql://test:test@localhost/test")
    
    def test_detects_negative_ht_home(self, checker):
        """Detect negative HT home score."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = [
            (1, -1, 0, 2, 1)
        ]
        
        issues = checker.check_negative_scores(mock_session)
        
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].issue_type == "negative_score"


class TestDuplicateCheck:
    """Test duplicate match detection."""
    
    @pytest.fixture
    def checker(self):
        with patch("src.data.quality.create_engine"):
            return DataQualityChecker("postgresql://test:test@localhost/test")
    
    def test_detects_duplicate_matches(self, checker):
        """Detect duplicate matches on same date."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = [
            (1, 2, "2024-08-17", 2)  # home_id, away_id, date, count
        ]
        
        issues = checker.check_duplicate_matches(mock_session)
        
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].issue_type == "duplicate_match"
        assert issues[0].details["count"] == 2
