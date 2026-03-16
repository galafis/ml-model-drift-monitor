"""
Model monitoring orchestrator.

Coordinates all three drift detectors (data, concept, performance)
to provide a unified monitoring interface. Manages scheduling,
state, and reporting.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.settings import AppSettings, DriftConfig, MonitoringConfig
from src.detectors.concept_drift import ConceptDriftDetector, ConceptDriftResult
from src.detectors.data_drift import DataDriftDetector, DriftResult, DriftSeverity
from src.detectors.performance_drift import (
    PerformanceDriftDetector,
    PerformanceDriftResult,
)
from src.utils.logger import get_logger

logger = get_logger("model_monitor")


class MonitoringStatus(str, Enum):
    """Status of the monitoring system."""

    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class MonitoringReport:
    """Complete monitoring report combining all detector outputs."""

    report_id: str
    model_name: str
    timestamp: str
    status: str
    data_drift: Optional[Dict[str, Any]] = None
    concept_drift: Optional[Dict[str, Any]] = None
    performance_drift: Optional[Dict[str, Any]] = None
    overall_health: str = "healthy"
    alerts_triggered: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a serializable dictionary."""
        return {
            "report_id": self.report_id,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "status": self.status,
            "data_drift": self.data_drift,
            "concept_drift": self.concept_drift,
            "performance_drift": self.performance_drift,
            "overall_health": self.overall_health,
            "alerts_triggered": self.alerts_triggered,
            "metadata": self.metadata,
        }


class ModelMonitor:
    """
    Orchestrates drift detection across data, concept, and performance dimensions.

    Maintains reference data, coordinates detector execution, generates
    unified reports, and supports scheduled monitoring.

    Attributes:
        model_name: Identifier for the monitored model.
        settings: Application settings.
        data_drift_detector: Data drift detector instance.
        concept_drift_detector: Concept drift detector instance.
        performance_drift_detector: Performance drift detector instance.
        status: Current monitoring status.
    """

    def __init__(
        self,
        model_name: str,
        settings: Optional[AppSettings] = None,
        task_type: str = "classification",
    ):
        """
        Initialize the model monitor.

        Args:
            model_name: Name/identifier of the model being monitored.
            settings: Application settings. Uses defaults if not provided.
            task_type: "classification" or "regression".
        """
        self.model_name = model_name
        self.settings = settings or AppSettings()
        self.task_type = task_type

        drift_config = self.settings.drift
        monitoring_config = self.settings.monitoring

        self.data_drift_detector = DataDriftDetector(config=drift_config)
        self.concept_drift_detector = ConceptDriftDetector(config=drift_config)
        self.performance_drift_detector = PerformanceDriftDetector(
            config=drift_config,
            task_type=task_type,
            window_size=monitoring_config.window_size,
        )

        self._reference_features: Optional[np.ndarray] = None
        self._reference_predictions: Optional[np.ndarray] = None
        self._reference_labels: Optional[np.ndarray] = None
        self._reference_metrics: Optional[Dict[str, float]] = None
        self._feature_names: Optional[List[str]] = None

        self.status = MonitoringStatus.IDLE
        self._reports: List[MonitoringReport] = []
        self._check_count = 0
        self._last_check_time: Optional[str] = None
        self._scheduled_task: Optional[asyncio.Task] = None

        logger.info(
            "ModelMonitor initialized for '%s' (task=%s)", model_name, task_type
        )

    def set_reference(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Set reference (baseline) data for drift comparison.

        Args:
            features: Reference feature matrix (n_samples, n_features).
            predictions: Reference model predictions.
            labels: Reference ground truth labels (optional).
            feature_names: Feature column names (optional).
        """
        self._reference_features = np.atleast_2d(features)
        self._reference_predictions = np.asarray(predictions).flatten()
        self._reference_labels = (
            np.asarray(labels).flatten() if labels is not None else None
        )
        self._feature_names = feature_names

        if self._reference_labels is not None:
            self._reference_metrics = (
                self.performance_drift_detector._compute_metrics(
                    self._reference_labels, self._reference_predictions
                )
            )

        logger.info(
            "Reference data set: %d samples, %d features",
            self._reference_features.shape[0],
            self._reference_features.shape[1],
        )

    def run_check(
        self,
        current_features: np.ndarray,
        current_predictions: np.ndarray,
        current_labels: Optional[np.ndarray] = None,
    ) -> MonitoringReport:
        """
        Execute a single monitoring check across all detectors.

        Args:
            current_features: Current feature data.
            current_predictions: Current model predictions.
            current_labels: Current ground truth labels (optional).

        Returns:
            MonitoringReport with results from all detectors.

        Raises:
            RuntimeError: If reference data has not been set.
        """
        if self._reference_features is None:
            raise RuntimeError(
                "Reference data not set. Call set_reference() first."
            )

        self.status = MonitoringStatus.RUNNING
        self._check_count += 1
        report_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        current_features = np.atleast_2d(current_features)
        current_predictions = np.asarray(current_predictions).flatten()
        current_labels_arr = (
            np.asarray(current_labels).flatten()
            if current_labels is not None
            else None
        )

        data_drift_result: Optional[DriftResult] = None
        concept_drift_result: Optional[ConceptDriftResult] = None
        perf_drift_result: Optional[PerformanceDriftResult] = None
        alerts: List[str] = []

        try:
            data_drift_result = self.data_drift_detector.detect(
                self._reference_features,
                current_features,
                feature_names=self._feature_names,
            )
            if data_drift_result.is_drifted:
                alerts.append(
                    f"Data drift detected: {data_drift_result.drifted_feature_count}/"
                    f"{data_drift_result.total_feature_count} features drifted "
                    f"(severity={data_drift_result.severity.value})"
                )
        except Exception as exc:
            logger.error("Data drift detection failed: %s", exc)

        try:
            current_errors = None
            if current_labels_arr is not None:
                current_errors = (
                    current_predictions != current_labels_arr
                ).astype(float)

            concept_drift_result = self.concept_drift_detector.detect(
                self._reference_predictions,
                current_predictions,
                reference_labels=self._reference_labels,
                current_labels=current_labels_arr,
                current_errors=current_errors,
            )
            if concept_drift_result.is_drifted:
                alerts.append(
                    f"Concept drift detected: type={concept_drift_result.drift_type.value}, "
                    f"score={concept_drift_result.drift_score:.4f}"
                )
        except Exception as exc:
            logger.error("Concept drift detection failed: %s", exc)

        if current_labels_arr is not None:
            try:
                perf_drift_result = self.performance_drift_detector.detect(
                    current_labels_arr,
                    current_predictions,
                    reference_metrics=self._reference_metrics,
                )
                if perf_drift_result.is_degraded:
                    alerts.append(
                        f"Performance degradation: level="
                        f"{perf_drift_result.degradation_level.value}, "
                        f"score={perf_drift_result.overall_score:.4f}"
                    )
            except Exception as exc:
                logger.error("Performance drift detection failed: %s", exc)

        overall_health = self._assess_health(
            data_drift_result, concept_drift_result, perf_drift_result
        )

        report = MonitoringReport(
            report_id=report_id,
            model_name=self.model_name,
            timestamp=timestamp,
            status="completed",
            data_drift=self._serialize_data_drift(data_drift_result),
            concept_drift=self._serialize_concept_drift(concept_drift_result),
            performance_drift=self._serialize_perf_drift(perf_drift_result),
            overall_health=overall_health,
            alerts_triggered=alerts,
            metadata={
                "check_number": self._check_count,
                "current_samples": current_features.shape[0],
                "task_type": self.task_type,
                "has_labels": current_labels_arr is not None,
            },
        )

        self._reports.append(report)
        max_reports = self.settings.monitoring.max_reports_stored
        if len(self._reports) > max_reports:
            self._reports = self._reports[-max_reports:]

        self._last_check_time = timestamp
        self.status = MonitoringStatus.IDLE

        logger.info(
            "Monitoring check #%d completed: health=%s, alerts=%d",
            self._check_count,
            overall_health,
            len(alerts),
        )

        return report

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status information."""
        return {
            "model_name": self.model_name,
            "status": self.status.value,
            "check_count": self._check_count,
            "last_check_time": self._last_check_time,
            "has_reference": self._reference_features is not None,
            "reference_samples": (
                self._reference_features.shape[0]
                if self._reference_features is not None
                else 0
            ),
            "stored_reports": len(self._reports),
            "task_type": self.task_type,
        }

    def get_reports(
        self, limit: int = 10, offset: int = 0
    ) -> List[MonitoringReport]:
        """Retrieve stored monitoring reports with pagination."""
        reports = list(reversed(self._reports))
        return reports[offset : offset + limit]

    def get_report_by_id(self, report_id: str) -> Optional[MonitoringReport]:
        """Retrieve a specific report by ID."""
        for report in self._reports:
            if report.report_id == report_id:
                return report
        return None

    async def start_scheduled_monitoring(
        self,
        data_callback,
        interval: Optional[int] = None,
    ) -> None:
        """
        Start scheduled monitoring with periodic checks.

        Args:
            data_callback: Async callable returning (features, predictions, labels).
            interval: Check interval in seconds. Uses config default if None.
        """
        if interval is None:
            interval = self.settings.monitoring.check_interval_seconds

        self.status = MonitoringStatus.RUNNING
        logger.info(
            "Starting scheduled monitoring (interval=%ds)", interval
        )

        async def _run_loop():
            while self.status == MonitoringStatus.RUNNING:
                try:
                    features, predictions, labels = await data_callback()
                    self.run_check(features, predictions, labels)
                except Exception as exc:
                    logger.error("Scheduled check failed: %s", exc)
                    self.status = MonitoringStatus.ERROR

                await asyncio.sleep(interval)

        self._scheduled_task = asyncio.create_task(_run_loop())

    def stop_scheduled_monitoring(self) -> None:
        """Stop scheduled monitoring."""
        self.status = MonitoringStatus.STOPPED
        if self._scheduled_task and not self._scheduled_task.done():
            self._scheduled_task.cancel()
        logger.info("Scheduled monitoring stopped.")

    @staticmethod
    def _assess_health(
        data_drift: Optional[DriftResult],
        concept_drift: Optional[ConceptDriftResult],
        perf_drift: Optional[PerformanceDriftResult],
    ) -> str:
        """Determine overall model health from detector results."""
        issues = 0

        if data_drift and data_drift.is_drifted:
            if data_drift.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
                issues += 2
            else:
                issues += 1

        if concept_drift and concept_drift.is_drifted:
            issues += 2

        if perf_drift and perf_drift.is_degraded:
            if perf_drift.degradation_level.value == "critical":
                issues += 3
            elif perf_drift.degradation_level.value == "degraded":
                issues += 2
            else:
                issues += 1

        if issues == 0:
            return "healthy"
        elif issues <= 2:
            return "warning"
        elif issues <= 4:
            return "degraded"
        else:
            return "critical"

    @staticmethod
    def _serialize_data_drift(
        result: Optional[DriftResult],
    ) -> Optional[Dict[str, Any]]:
        """Serialize data drift result to dictionary."""
        if result is None:
            return None
        return {
            "is_drifted": result.is_drifted,
            "drift_score": result.drift_score,
            "severity": result.severity.value,
            "drifted_feature_count": result.drifted_feature_count,
            "total_feature_count": result.total_feature_count,
            "drifted_feature_fraction": result.drifted_feature_fraction,
            "method_scores": result.method_scores,
            "feature_results": [
                {
                    "feature_name": fr.feature_name,
                    "is_drifted": fr.is_drifted,
                    "ks_statistic": fr.ks_statistic,
                    "ks_p_value": fr.ks_p_value,
                    "psi_value": fr.psi_value,
                    "js_divergence": fr.js_divergence,
                    "severity": fr.severity.value,
                }
                for fr in result.feature_results
            ],
        }

    @staticmethod
    def _serialize_concept_drift(
        result: Optional[ConceptDriftResult],
    ) -> Optional[Dict[str, Any]]:
        """Serialize concept drift result to dictionary."""
        if result is None:
            return None
        return {
            "is_drifted": result.is_drifted,
            "drift_type": result.drift_type.value,
            "drift_score": result.drift_score,
            "prediction_shift_score": result.prediction_shift_score,
            "label_drift_score": result.label_drift_score,
            "adwin_detected": result.adwin_detected,
            "window_means": result.window_means,
        }

    @staticmethod
    def _serialize_perf_drift(
        result: Optional[PerformanceDriftResult],
    ) -> Optional[Dict[str, Any]]:
        """Serialize performance drift result to dictionary."""
        if result is None:
            return None
        return {
            "is_degraded": result.is_degraded,
            "degradation_level": result.degradation_level.value,
            "overall_score": result.overall_score,
            "current_metrics": result.current_metrics,
            "reference_metrics": result.reference_metrics,
            "metric_changes": result.metric_changes,
            "trends": [
                {
                    "metric_name": t.metric_name,
                    "slope": t.slope,
                    "r_squared": t.r_squared,
                    "p_value": t.p_value,
                    "is_degrading": t.is_degrading,
                    "trend_direction": t.trend_direction,
                }
                for t in result.trends
            ],
        }
