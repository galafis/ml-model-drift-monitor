"""
Batch monitoring module.

Processes historical data batches for drift analysis, enabling
retrospective comparison between time periods and generation of
comprehensive comparison reports.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.settings import AppSettings
from src.detectors.concept_drift import ConceptDriftDetector
from src.detectors.data_drift import DataDriftDetector
from src.detectors.performance_drift import PerformanceDriftDetector
from src.utils.logger import get_logger

logger = get_logger("batch_monitor")


@dataclass
class BatchInfo:
    """Metadata for a data batch."""

    batch_id: str
    timestamp: str
    sample_count: int
    feature_count: int
    has_labels: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchComparisonReport:
    """Report comparing two data batches."""

    report_id: str
    timestamp: str
    reference_batch: BatchInfo
    current_batch: BatchInfo
    data_drift_summary: Dict[str, Any]
    concept_drift_summary: Dict[str, Any]
    performance_comparison: Optional[Dict[str, Any]]
    overall_assessment: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "reference_batch": {
                "batch_id": self.reference_batch.batch_id,
                "timestamp": self.reference_batch.timestamp,
                "sample_count": self.reference_batch.sample_count,
            },
            "current_batch": {
                "batch_id": self.current_batch.batch_id,
                "timestamp": self.current_batch.timestamp,
                "sample_count": self.current_batch.sample_count,
            },
            "data_drift_summary": self.data_drift_summary,
            "concept_drift_summary": self.concept_drift_summary,
            "performance_comparison": self.performance_comparison,
            "overall_assessment": self.overall_assessment,
            "details": self.details,
        }


class BatchMonitor:
    """
    Processes historical data batches for comparative drift analysis.

    Enables retrospective monitoring by comparing feature distributions,
    prediction patterns, and performance metrics across time-ordered batches.

    Attributes:
        settings: Application settings.
        data_drift_detector: Data drift detector.
        concept_drift_detector: Concept drift detector.
        performance_detector: Performance drift detector.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        task_type: str = "classification",
    ):
        """
        Initialize the batch monitor.

        Args:
            settings: Application settings.
            task_type: "classification" or "regression".
        """
        self.settings = settings or AppSettings()
        self.task_type = task_type

        drift_config = self.settings.drift
        self.data_drift_detector = DataDriftDetector(config=drift_config)
        self.concept_drift_detector = ConceptDriftDetector(config=drift_config)
        self.performance_detector = PerformanceDriftDetector(
            config=drift_config, task_type=task_type
        )

        self._comparison_reports: List[BatchComparisonReport] = []
        logger.info("BatchMonitor initialized (task_type=%s)", task_type)

    def compare_batches(
        self,
        reference_features: np.ndarray,
        reference_predictions: np.ndarray,
        current_features: np.ndarray,
        current_predictions: np.ndarray,
        reference_labels: Optional[np.ndarray] = None,
        current_labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        reference_batch_id: Optional[str] = None,
        current_batch_id: Optional[str] = None,
    ) -> BatchComparisonReport:
        """
        Compare two data batches across all drift dimensions.

        Args:
            reference_features: Features from the reference batch.
            reference_predictions: Predictions from the reference batch.
            current_features: Features from the current batch.
            current_predictions: Predictions from the current batch.
            reference_labels: Labels from the reference batch (optional).
            current_labels: Labels from the current batch (optional).
            feature_names: Feature column names (optional).
            reference_batch_id: Identifier for the reference batch.
            current_batch_id: Identifier for the current batch.

        Returns:
            BatchComparisonReport with detailed comparison results.
        """
        reference_features = np.atleast_2d(reference_features)
        current_features = np.atleast_2d(current_features)

        ref_batch = BatchInfo(
            batch_id=reference_batch_id or str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_count=reference_features.shape[0],
            feature_count=reference_features.shape[1],
            has_labels=reference_labels is not None,
        )
        cur_batch = BatchInfo(
            batch_id=current_batch_id or str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_count=current_features.shape[0],
            feature_count=current_features.shape[1],
            has_labels=current_labels is not None,
        )

        data_drift_result = self.data_drift_detector.detect(
            reference_features, current_features, feature_names
        )
        data_drift_summary = {
            "is_drifted": data_drift_result.is_drifted,
            "drift_score": data_drift_result.drift_score,
            "severity": data_drift_result.severity.value,
            "drifted_features": data_drift_result.drifted_feature_count,
            "total_features": data_drift_result.total_feature_count,
            "method_scores": data_drift_result.method_scores,
        }

        concept_result = self.concept_drift_detector.detect(
            np.asarray(reference_predictions).flatten(),
            np.asarray(current_predictions).flatten(),
            reference_labels=(
                np.asarray(reference_labels).flatten()
                if reference_labels is not None
                else None
            ),
            current_labels=(
                np.asarray(current_labels).flatten()
                if current_labels is not None
                else None
            ),
        )
        concept_drift_summary = {
            "is_drifted": concept_result.is_drifted,
            "drift_type": concept_result.drift_type.value,
            "drift_score": concept_result.drift_score,
            "prediction_shift": concept_result.prediction_shift_score,
            "label_drift": concept_result.label_drift_score,
        }

        performance_comparison: Optional[Dict[str, Any]] = None
        if reference_labels is not None and current_labels is not None:
            ref_labels = np.asarray(reference_labels).flatten()
            cur_labels = np.asarray(current_labels).flatten()
            ref_preds = np.asarray(reference_predictions).flatten()
            cur_preds = np.asarray(current_predictions).flatten()

            ref_metrics = self.performance_detector._compute_metrics(
                ref_labels, ref_preds
            )
            perf_result = self.performance_detector.detect(
                cur_labels, cur_preds, reference_metrics=ref_metrics
            )
            performance_comparison = {
                "is_degraded": perf_result.is_degraded,
                "degradation_level": perf_result.degradation_level.value,
                "reference_metrics": perf_result.reference_metrics,
                "current_metrics": perf_result.current_metrics,
                "changes": perf_result.metric_changes,
            }

        overall = self._assess_overall(
            data_drift_summary, concept_drift_summary, performance_comparison
        )

        report = BatchComparisonReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            reference_batch=ref_batch,
            current_batch=cur_batch,
            data_drift_summary=data_drift_summary,
            concept_drift_summary=concept_drift_summary,
            performance_comparison=performance_comparison,
            overall_assessment=overall,
        )

        self._comparison_reports.append(report)
        logger.info(
            "Batch comparison complete: ref=%s vs cur=%s, assessment=%s",
            ref_batch.batch_id[:8],
            cur_batch.batch_id[:8],
            overall,
        )

        return report

    def process_batch_sequence(
        self,
        batches: List[Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None,
    ) -> List[BatchComparisonReport]:
        """
        Process a sequence of batches, comparing each to the first (reference).

        Each batch dict should have keys: "features", "predictions",
        and optionally "labels".

        Args:
            batches: List of batch dictionaries.
            feature_names: Feature names (optional).

        Returns:
            List of comparison reports (one per subsequent batch).
        """
        if len(batches) < 2:
            logger.warning("Need at least 2 batches for comparison.")
            return []

        reference = batches[0]
        reports: List[BatchComparisonReport] = []

        for i, batch in enumerate(batches[1:], start=1):
            report = self.compare_batches(
                reference_features=reference["features"],
                reference_predictions=reference["predictions"],
                current_features=batch["features"],
                current_predictions=batch["predictions"],
                reference_labels=reference.get("labels"),
                current_labels=batch.get("labels"),
                feature_names=feature_names,
                reference_batch_id=f"batch_0",
                current_batch_id=f"batch_{i}",
            )
            reports.append(report)

        logger.info(
            "Processed %d batch comparisons from sequence of %d batches.",
            len(reports),
            len(batches),
        )
        return reports

    def get_reports(self, limit: int = 10) -> List[BatchComparisonReport]:
        """Retrieve recent comparison reports."""
        return list(reversed(self._comparison_reports))[:limit]

    @staticmethod
    def _assess_overall(
        data_drift: Dict[str, Any],
        concept_drift: Dict[str, Any],
        performance: Optional[Dict[str, Any]],
    ) -> str:
        """Determine overall assessment from all summaries."""
        score = 0

        if data_drift.get("is_drifted"):
            severity = data_drift.get("severity", "low")
            score += {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(
                severity, 1
            )

        if concept_drift.get("is_drifted"):
            score += 2

        if performance and performance.get("is_degraded"):
            level = performance.get("degradation_level", "warning")
            score += {"warning": 1, "degraded": 2, "critical": 3}.get(level, 1)

        if score == 0:
            return "stable"
        elif score <= 2:
            return "minor_changes"
        elif score <= 4:
            return "significant_drift"
        else:
            return "critical_degradation"
