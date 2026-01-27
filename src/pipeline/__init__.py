"""
Pipeline orchestration module.

Assembles steps: ingest → quality checks → feature build → predict → write outputs
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Optional

from src.utils import get_logger, now_utc

logger = get_logger("pipeline")


class PipelineStage(str, Enum):
    """Pipeline stages."""
    INGEST = "ingest"
    QUALITY_CHECK = "quality_check"
    FEATURE_BUILD = "feature_build"
    PREDICT = "predict"
    DECISION = "decision"
    OUTPUT = "output"


@dataclass
class StepResult:
    """Result from a pipeline step."""
    stage: PipelineStage
    success: bool
    started_at: datetime
    completed_at: datetime
    records_processed: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


@dataclass
class PipelineResult:
    """Result from full pipeline run."""
    started_at: datetime
    completed_at: Optional[datetime] = None
    steps: list[StepResult] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return all(s.success for s in self.steps)
    
    @property
    def total_duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()
    
    def add_step(self, step: StepResult) -> None:
        self.steps.append(step)
    
    def summary(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "duration_seconds": self.total_duration_seconds,
            "steps": [
                {
                    "stage": s.stage.value,
                    "success": s.success,
                    "duration": s.duration_seconds,
                    "records": s.records_processed,
                    "errors": len(s.errors),
                }
                for s in self.steps
            ],
        }


class PipelineStep(ABC):
    """Base class for pipeline steps."""
    
    stage: PipelineStage
    
    @abstractmethod
    def run(self, context: dict[str, Any]) -> StepResult:
        """Execute the step."""
        pass


class Pipeline:
    """
    Orchestrates pipeline execution.
    
    Usage:
        pipeline = Pipeline()
        pipeline.add_step(IngestStep(leagues=["PL", "BL1"]))
        pipeline.add_step(QualityCheckStep())
        pipeline.add_step(FeatureBuildStep())
        pipeline.add_step(PredictStep())
        result = pipeline.run()
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.steps: list[PipelineStep] = []
        self.context: dict[str, Any] = {}
    
    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Add a step to the pipeline. Returns self for chaining."""
        self.steps.append(step)
        return self
    
    def set_context(self, key: str, value: Any) -> "Pipeline":
        """Set context value. Returns self for chaining."""
        self.context[key] = value
        return self
    
    def run(self, stop_on_error: bool = True) -> PipelineResult:
        """
        Execute all pipeline steps.
        
        Args:
            stop_on_error: If True, stop on first error
            
        Returns:
            Pipeline result with all step outcomes
        """
        result = PipelineResult(started_at=now_utc())
        
        logger.info(f"Starting pipeline '{self.name}' with {len(self.steps)} steps")
        
        for step in self.steps:
            logger.info(f"Running step: {step.stage.value}")
            
            try:
                step_result = step.run(self.context)
            except Exception as e:
                step_result = StepResult(
                    stage=step.stage,
                    success=False,
                    started_at=now_utc(),
                    completed_at=now_utc(),
                    errors=[str(e)],
                )
                logger.error(f"Step {step.stage.value} failed: {e}")
            
            result.add_step(step_result)
            
            if not step_result.success and stop_on_error:
                logger.error(f"Pipeline stopped due to error in {step.stage.value}")
                break
            
            logger.info(
                f"Step {step.stage.value} completed: "
                f"{step_result.records_processed} records in {step_result.duration_seconds:.2f}s"
            )
        
        result.completed_at = now_utc()
        logger.info(
            f"Pipeline '{self.name}' completed: "
            f"{'SUCCESS' if result.success else 'FAILED'} "
            f"in {result.total_duration_seconds:.2f}s"
        )
        
        return result


# =============================================================================
# Built-in Steps (to be expanded in later phases)
# =============================================================================

class IngestStep(PipelineStep):
    """Ingest matches from API."""
    
    stage = PipelineStage.INGEST
    
    def __init__(
        self,
        leagues: list[str],
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        season: Optional[int] = None,
    ):
        self.leagues = leagues
        self.date_from = date_from
        self.date_to = date_to
        self.season = season
    
    def run(self, context: dict[str, Any]) -> StepResult:
        from src.data.ingestion import IngestionPipeline
        
        started = now_utc()
        pipeline = IngestionPipeline()
        
        total_fetched = 0
        errors = []
        
        for league in self.leagues:
            try:
                stats = pipeline.ingest_matches(
                    league_code=league,
                    date_from=self.date_from,
                    date_to=self.date_to,
                    season=self.season,
                )
                total_fetched += stats.get("total_fetched", 0)
                errors.extend(stats.get("errors", []))
            except Exception as e:
                errors.append(f"{league}: {e}")
        
        # Store in context for next steps
        context["leagues"] = self.leagues
        context["season"] = self.season
        
        return StepResult(
            stage=self.stage,
            success=len(errors) == 0,
            started_at=started,
            completed_at=now_utc(),
            records_processed=total_fetched,
            errors=[str(e) for e in errors],
        )


class QualityCheckStep(PipelineStep):
    """Run data quality checks."""
    
    stage = PipelineStage.QUALITY_CHECK
    
    def __init__(self, fail_on_errors: bool = True):
        self.fail_on_errors = fail_on_errors
    
    def run(self, context: dict[str, Any]) -> StepResult:
        from src.data.quality import run_quality_checks
        
        started = now_utc()
        report = run_quality_checks()
        
        context["quality_report"] = report
        
        success = report.is_healthy if self.fail_on_errors else True
        
        return StepResult(
            stage=self.stage,
            success=success,
            started_at=started,
            completed_at=now_utc(),
            records_processed=report.total_matches,
            errors=[f"{i.issue_type}: {i.description}" for i in report.issues if i.severity == "error"],
            metadata={
                "total_matches": report.total_matches,
                "error_count": report.error_count,
                "warning_count": report.warning_count,
            },
        )
