"""
Performance drift detection module.

Implements sliding window performance monitoring with degradation
detection and trend analysis:
- Sliding window metric computation (accuracy, precision, recall, F1, MAE, RMSE)
- Degradation detection using threshold-based comparison
- Trend analysis using linear regression over historical windows

References:
    - Gama et al. (2014), "A survey on concept drift adaptation"
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from src.config.settings import DriftConfig
from src.utils.logger import get_logger

logger = get_logger("performance_drift")


class MetricType(str, Enum):
    """Supported performance metric types."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    MAE = "mae"
    RMSE = "rmse"


class DegradationLevel(str, Enum):
    """Performance degradation severity levels."""

    NONE = "none"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class MetricWindow:
    """A single metric computation window."""

    window_index: int
    metric_name: str
    value: float
    sample_count: int
    timestamp: str


@dataclass
class TrendAnalysis:
    """Result of linear regression trend analysis over metric windows."""

    metric_name: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    is_degrading: bool
    trend_direction: str
    windows_analyzed: int


@dataclass
class PerformanceDriftResult:
    """Comprehensive performance drift detection result."""

    timestamp: str
    is_degraded: bool
    degradation_level: DegradationLevel
    overall_score: float
    current_metrics: Dict[str, float]
    reference_metrics: Dict[str, float]
    metric_changes: Dict[str, float]
    trends: List[TrendAnalysis]
    window_history: List[MetricWindow]
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceDriftDetector:
    """
    Detects performance degradation in ML models using sliding windows.

    Maintains a history of metric computations over sliding windows,
    compares current metrics against reference baselines, and performs
    trend analysis using linear regression to detect gradual degradation.

    Attributes:
        config: Drift configuration with thresholds.
        task_type: Either "classification" or "regression".
        window_size: Number of samples per evaluation window.
        metric_history: Historical metric values per metric type.
    """

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        task_type: str = "classification",
        window_size: int = 200,
    ):
        """
        Initialize the performance drift detector.

        Args:
            config: Drift configuration.
            task_type: "classification" or "regression".
            window_size: Samples per sliding window.
        """
        self.config = config or DriftConfig()
        self.task_type = task_type
        self.window_size = window_size
        self.metric_history: Dict[str, List[float]] = {}
        self._window_records: List[MetricWindow] = []
        self._window_counter = 0

        logger.info(
            "PerformanceDriftDetector initialized (task=%s, window_size=%d, "
            "degradation_threshold=%.3f)",
            self.task_type,
            self.window_size,
            self.config.performance_degradation_threshold,
        )

    def detect(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reference_metrics: Optional[Dict[str, float]] = None,
        y_true_reference: Optional[np.ndarray] = None,
        y_pred_reference: Optional[np.ndarray] = None,
    ) -> PerformanceDriftResult:
        """
        Run performance drift detection.

        Computes current metrics, compares against reference, and
        analyzes trends over historical windows.

        Args:
            y_true: Current ground truth labels/values.
            y_pred: Current model predictions.
            reference_metrics: Pre-computed reference metrics. If not
                               provided, computed from reference data.
            y_true_reference: Reference ground truth (used if
                              reference_metrics not provided).
            y_pred_reference: Reference predictions (used if
                              reference_metrics not provided).

        Returns:
            PerformanceDriftResult with full analysis.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        current_metrics = self._compute_metrics(y_true, y_pred)

        if reference_metrics is None:
            if y_true_reference is not None and y_pred_reference is not None:
                reference_metrics = self._compute_metrics(
                    np.asarray(y_true_reference).flatten(),
                    np.asarray(y_pred_reference).flatten(),
                )
            else:
                reference_metrics = {k: v for k, v in current_metrics.items()}

        self._record_window(current_metrics, len(y_true))

        metric_changes = self._compute_changes(reference_metrics, current_metrics)

        is_degraded, degradation_level = self._check_degradation(
            metric_changes
        )

        trends = self._analyze_trends()

        trend_degrading = any(t.is_degrading for t in trends)
        if trend_degrading and degradation_level == DegradationLevel.NONE:
            degradation_level = DegradationLevel.WARNING
            is_degraded = True

        overall_score = self._compute_overall_score(
            metric_changes, trends, is_degraded
        )

        logger.info(
            "Performance drift: degraded=%s, level=%s, score=%.4f, "
            "metrics=%s",
            is_degraded,
            degradation_level.value,
            overall_score,
            {k: f"{v:.4f}" for k, v in current_metrics.items()},
        )

        return PerformanceDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_degraded=is_degraded,
            degradation_level=degradation_level,
            overall_score=float(overall_score),
            current_metrics=current_metrics,
            reference_metrics=reference_metrics,
            metric_changes=metric_changes,
            trends=trends,
            window_history=list(self._window_records[-20:]),
            details={
                "task_type": self.task_type,
                "window_size": self.window_size,
                "total_windows": self._window_counter,
                "current_sample_count": len(y_true),
            },
        )

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute performance metrics based on task type.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            Dictionary of metric name to value.
        """
        metrics: Dict[str, float] = {}

        if self.task_type == "classification":
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

            n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
            average = "binary" if n_classes <= 2 else "weighted"

            try:
                metrics["precision"] = float(
                    precision_score(y_true, y_pred, average=average, zero_division=0)
                )
                metrics["recall"] = float(
                    recall_score(y_true, y_pred, average=average, zero_division=0)
                )
                metrics["f1"] = float(
                    f1_score(y_true, y_pred, average=average, zero_division=0)
                )
            except Exception as exc:
                logger.warning("Error computing classification metrics: %s", exc)
                metrics.setdefault("precision", 0.0)
                metrics.setdefault("recall", 0.0)
                metrics.setdefault("f1", 0.0)

        elif self.task_type == "regression":
            y_true_f = y_true.astype(float)
            y_pred_f = y_pred.astype(float)
            metrics["mae"] = float(mean_absolute_error(y_true_f, y_pred_f))
            metrics["rmse"] = float(
                np.sqrt(mean_squared_error(y_true_f, y_pred_f))
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return metrics

    def _record_window(
        self, metrics: Dict[str, float], sample_count: int
    ) -> None:
        """Store metric values in history for trend analysis."""
        self._window_counter += 1
        ts = datetime.now(timezone.utc).isoformat()

        for name, value in metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = []
            self.metric_history[name].append(value)

            max_history = self.config.trend_lookback_windows * 2
            if len(self.metric_history[name]) > max_history:
                self.metric_history[name] = self.metric_history[name][-max_history:]

            self._window_records.append(
                MetricWindow(
                    window_index=self._window_counter,
                    metric_name=name,
                    value=value,
                    sample_count=sample_count,
                    timestamp=ts,
                )
            )

        max_records = self.config.trend_lookback_windows * len(metrics) * 2
        if len(self._window_records) > max_records:
            self._window_records = self._window_records[-max_records:]

    def _compute_changes(
        self,
        reference: Dict[str, float],
        current: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute relative changes between reference and current metrics.

        For metrics where higher is better (accuracy, precision, recall, F1),
        a negative change means degradation.
        For error metrics (MAE, RMSE), a positive change means degradation.

        Returns:
            Dictionary of metric_name -> relative_change.
        """
        changes: Dict[str, float] = {}
        for name in current:
            ref_val = reference.get(name, current[name])
            cur_val = current[name]

            if abs(ref_val) > 1e-10:
                relative_change = (cur_val - ref_val) / abs(ref_val)
            else:
                relative_change = cur_val - ref_val

            changes[name] = float(relative_change)

        return changes

    def _check_degradation(
        self, changes: Dict[str, float]
    ) -> Tuple[bool, DegradationLevel]:
        """
        Determine if performance has degraded beyond threshold.

        Args:
            changes: Relative metric changes.

        Returns:
            Tuple of (is_degraded, degradation_level).
        """
        threshold = self.config.performance_degradation_threshold
        higher_is_better = {"accuracy", "precision", "recall", "f1"}
        lower_is_better = {"mae", "rmse"}

        worst_degradation = 0.0

        for name, change in changes.items():
            if name in higher_is_better:
                degradation = max(0.0, -change)
            elif name in lower_is_better:
                degradation = max(0.0, change)
            else:
                degradation = abs(change)

            worst_degradation = max(worst_degradation, degradation)

        if worst_degradation < threshold * 0.5:
            return False, DegradationLevel.NONE
        elif worst_degradation < threshold:
            return True, DegradationLevel.WARNING
        elif worst_degradation < threshold * 2:
            return True, DegradationLevel.DEGRADED
        else:
            return True, DegradationLevel.CRITICAL

    def _analyze_trends(self) -> List[TrendAnalysis]:
        """
        Perform linear regression trend analysis on each metric.

        Uses the most recent windows (up to trend_lookback_windows)
        and fits a linear model to detect degradation trends.

        Returns:
            List of TrendAnalysis results, one per metric.
        """
        trends: List[TrendAnalysis] = []
        lookback = self.config.trend_lookback_windows

        higher_is_better = {"accuracy", "precision", "recall", "f1"}

        for name, history in self.metric_history.items():
            if len(history) < 3:
                continue

            values = np.array(history[-lookback:])
            x = np.arange(len(values), dtype=float)

            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                x, values
            )

            r_squared = r_value ** 2

            if name in higher_is_better:
                is_degrading = slope < 0 and p_value < 0.1
                trend_dir = "improving" if slope > 0 else "declining"
            else:
                is_degrading = slope > 0 and p_value < 0.1
                trend_dir = "improving" if slope < 0 else "worsening"

            trends.append(
                TrendAnalysis(
                    metric_name=name,
                    slope=float(slope),
                    intercept=float(intercept),
                    r_squared=float(r_squared),
                    p_value=float(p_value),
                    is_degrading=is_degrading,
                    trend_direction=trend_dir,
                    windows_analyzed=len(values),
                )
            )

        return trends

    @staticmethod
    def _compute_overall_score(
        changes: Dict[str, float],
        trends: List[TrendAnalysis],
        is_degraded: bool,
    ) -> float:
        """
        Compute an overall performance drift score combining changes and trends.

        Returns:
            Score in [0, 1] range where higher means worse degradation.
        """
        if not changes:
            return 0.0

        higher_is_better = {"accuracy", "precision", "recall", "f1"}
        degradation_scores: List[float] = []

        for name, change in changes.items():
            if name in higher_is_better:
                degradation_scores.append(max(0.0, -change))
            else:
                degradation_scores.append(max(0.0, change))

        change_score = float(np.mean(degradation_scores)) if degradation_scores else 0.0

        trend_score = 0.0
        if trends:
            degrading_count = sum(1 for t in trends if t.is_degrading)
            trend_score = degrading_count / len(trends)

        overall = 0.7 * min(change_score, 1.0) + 0.3 * trend_score
        return float(min(overall, 1.0))

    def reset(self) -> None:
        """Reset all history and state."""
        self.metric_history.clear()
        self._window_records.clear()
        self._window_counter = 0
        logger.info("PerformanceDriftDetector state reset.")
