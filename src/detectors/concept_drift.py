"""
Concept drift detection module.

Implements detection of changes in the relationship between input features
and model predictions/labels:
- ADWIN-inspired adaptive windowing for streaming drift detection
- Prediction distribution shift analysis
- Label drift detection comparing reference and current label distributions

References:
    - ADWIN: Bifet & Gavalda (2007), "Learning from Time-Changing Data
      with Adaptive Windowing"
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.config.settings import DriftConfig
from src.utils.logger import get_logger

logger = get_logger("concept_drift")


class ConceptDriftType(str, Enum):
    """Types of concept drift detected."""

    NONE = "none"
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"


@dataclass
class ConceptDriftResult:
    """Result of concept drift detection."""

    timestamp: str
    is_drifted: bool
    drift_type: ConceptDriftType
    drift_score: float
    prediction_shift_score: float
    label_drift_score: float
    adwin_detected: bool
    window_means: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class ADWINWindow:
    """
    ADWIN-inspired adaptive window for streaming data.

    Maintains a window of observations and detects significant changes
    in the data distribution by comparing sub-windows using the Hoeffding
    bound. When drift is detected, old data is dropped.

    This is a simplified implementation suitable for monitoring scenarios
    where data arrives in batches.

    Attributes:
        delta: Confidence parameter for the Hoeffding bound.
        max_window_size: Maximum elements to retain in the window.
    """

    def __init__(self, delta: float = 0.002, max_window_size: int = 5000):
        """
        Initialize the ADWIN window.

        Args:
            delta: Sensitivity parameter. Smaller values mean less
                   sensitivity to change (fewer false positives).
            max_window_size: Maximum number of observations stored.
        """
        self.delta = delta
        self.max_window_size = max_window_size
        self._window: List[float] = []
        self._drift_detected = False
        self._cut_point: Optional[int] = None

    @property
    def size(self) -> int:
        """Current window size."""
        return len(self._window)

    @property
    def mean(self) -> float:
        """Current window mean."""
        if not self._window:
            return 0.0
        return float(np.mean(self._window))

    def add_batch(self, values: np.ndarray) -> bool:
        """
        Add a batch of observations and check for drift.

        Args:
            values: Array of new observations.

        Returns:
            True if drift was detected after adding these values.
        """
        values_list = values.flatten().tolist()
        self._window.extend(values_list)

        if len(self._window) > self.max_window_size:
            excess = len(self._window) - self.max_window_size
            self._window = self._window[excess:]

        self._drift_detected = False
        self._cut_point = None

        if len(self._window) < 10:
            return False

        self._drift_detected, self._cut_point = self._check_drift()

        if self._drift_detected and self._cut_point is not None:
            self._window = self._window[self._cut_point:]

        return self._drift_detected

    def _check_drift(self) -> Tuple[bool, Optional[int]]:
        """
        Check for drift using sub-window comparison with Hoeffding bound.

        Iterates over possible cut points and compares the mean of the
        left sub-window (older data) with the right sub-window (newer data).

        Returns:
            Tuple of (drift_detected, cut_point_index).
        """
        n = len(self._window)
        window_arr = np.array(self._window)
        total_sum = np.sum(window_arr)

        min_subwindow = max(5, n // 10)

        step = max(1, n // 50)

        best_cut = None
        best_epsilon_diff = 0.0

        left_sum = np.sum(window_arr[:min_subwindow])
        for i in range(min_subwindow, n - min_subwindow, step):
            left_sum = np.sum(window_arr[:i])
            right_sum = total_sum - left_sum

            n_left = i
            n_right = n - i

            mean_left = left_sum / n_left
            mean_right = right_sum / n_right

            m_inv = 1.0 / n_left + 1.0 / n_right
            epsilon = np.sqrt(0.5 * m_inv * np.log(4.0 * n / self.delta))

            diff = abs(mean_left - mean_right)

            if diff > epsilon and diff > best_epsilon_diff:
                best_epsilon_diff = diff
                best_cut = i

        if best_cut is not None:
            return True, best_cut

        return False, None

    def reset(self) -> None:
        """Clear the window state."""
        self._window.clear()
        self._drift_detected = False
        self._cut_point = None


class ConceptDriftDetector:
    """
    Detects concept drift in ML model predictions and labels.

    Combines three approaches:
    1. ADWIN-based streaming detection on model error rates.
    2. Prediction distribution shift (comparing predicted class/value
       distributions).
    3. Label drift (comparing actual label distributions when available).

    Attributes:
        config: Drift configuration with thresholds.
        adwin: ADWIN adaptive window instance.
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        """
        Initialize the concept drift detector.

        Args:
            config: Drift configuration. Uses defaults if not provided.
        """
        self.config = config or DriftConfig()
        self.adwin = ADWINWindow(
            delta=self.config.adwin_delta,
            max_window_size=5000,
        )
        self._error_history: List[float] = []
        logger.info(
            "ConceptDriftDetector initialized (threshold=%.3f, adwin_delta=%.4f)",
            self.config.concept_drift_threshold,
            self.config.adwin_delta,
        )

    def detect(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        reference_labels: Optional[np.ndarray] = None,
        current_labels: Optional[np.ndarray] = None,
        current_errors: Optional[np.ndarray] = None,
    ) -> ConceptDriftResult:
        """
        Run concept drift detection.

        Args:
            reference_predictions: Model predictions on reference data.
            current_predictions: Model predictions on current data.
            reference_labels: Ground truth labels for reference data (optional).
            current_labels: Ground truth labels for current data (optional).
            current_errors: Binary error indicators (1=error, 0=correct)
                            for ADWIN detection (optional).

        Returns:
            ConceptDriftResult with detection outcomes.
        """
        reference_predictions = np.asarray(reference_predictions).flatten()
        current_predictions = np.asarray(current_predictions).flatten()

        pred_shift_score = self._prediction_distribution_shift(
            reference_predictions, current_predictions
        )

        label_drift_score = 0.0
        if reference_labels is not None and current_labels is not None:
            label_drift_score = self._label_drift(
                np.asarray(reference_labels).flatten(),
                np.asarray(current_labels).flatten(),
            )

        adwin_detected = False
        if current_errors is not None:
            errors = np.asarray(current_errors).flatten()
            adwin_detected = self.adwin.add_batch(errors)
            self._error_history.extend(errors.tolist())

        drift_score = self._aggregate_score(
            pred_shift_score, label_drift_score, adwin_detected
        )

        is_drifted = drift_score > self.config.concept_drift_threshold

        drift_type = self._classify_drift_type(
            pred_shift_score, label_drift_score, adwin_detected
        )

        window_means = {
            "adwin_window_mean": self.adwin.mean,
            "adwin_window_size": float(self.adwin.size),
        }

        if len(self._error_history) > 0:
            recent_n = min(100, len(self._error_history))
            window_means["recent_error_rate"] = float(
                np.mean(self._error_history[-recent_n:])
            )

        logger.info(
            "Concept drift detection: drifted=%s, type=%s, score=%.4f, "
            "pred_shift=%.4f, label_drift=%.4f, adwin=%s",
            is_drifted,
            drift_type.value,
            drift_score,
            pred_shift_score,
            label_drift_score,
            adwin_detected,
        )

        return ConceptDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_drifted=is_drifted,
            drift_type=drift_type,
            drift_score=float(drift_score),
            prediction_shift_score=float(pred_shift_score),
            label_drift_score=float(label_drift_score),
            adwin_detected=adwin_detected,
            window_means=window_means,
            details={
                "reference_prediction_mean": float(np.mean(reference_predictions)),
                "current_prediction_mean": float(np.mean(current_predictions)),
                "reference_prediction_std": float(np.std(reference_predictions)),
                "current_prediction_std": float(np.std(current_predictions)),
            },
        )

    def _prediction_distribution_shift(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """
        Measure shift in the prediction distribution.

        Uses a combination of KS test and mean/std shift to quantify
        how much the prediction distribution has changed.

        Args:
            reference: Reference predictions.
            current: Current predictions.

        Returns:
            Shift score in [0, 1] range.
        """
        if len(reference) < 2 or len(current) < 2:
            return 0.0

        ks_stat, ks_p = stats.ks_2samp(reference, current)

        ref_mean, cur_mean = np.mean(reference), np.mean(current)
        ref_std = np.std(reference)

        if ref_std > 0:
            normalized_mean_shift = abs(cur_mean - ref_mean) / ref_std
        else:
            normalized_mean_shift = abs(cur_mean - ref_mean)

        ks_component = ks_stat
        mean_component = min(normalized_mean_shift / 3.0, 1.0)

        shift_score = 0.6 * ks_component + 0.4 * mean_component
        return float(min(shift_score, 1.0))

    @staticmethod
    def _label_drift(
        reference_labels: np.ndarray, current_labels: np.ndarray
    ) -> float:
        """
        Detect drift in label distributions.

        For classification: compares class frequency distributions.
        For regression: uses KS test on label values.

        Args:
            reference_labels: Reference ground truth labels.
            current_labels: Current ground truth labels.

        Returns:
            Label drift score in [0, 1] range.
        """
        if len(reference_labels) < 2 or len(current_labels) < 2:
            return 0.0

        n_unique_ref = len(np.unique(reference_labels))
        n_unique_cur = len(np.unique(current_labels))

        if n_unique_ref <= 20 and n_unique_cur <= 20:
            all_labels = np.unique(np.concatenate([reference_labels, current_labels]))
            ref_freq = np.zeros(len(all_labels))
            cur_freq = np.zeros(len(all_labels))

            for i, label in enumerate(all_labels):
                ref_freq[i] = np.sum(reference_labels == label) / len(reference_labels)
                cur_freq[i] = np.sum(current_labels == label) / len(current_labels)

            eps = 1e-10
            ref_freq = ref_freq + eps
            cur_freq = cur_freq + eps
            ref_freq = ref_freq / ref_freq.sum()
            cur_freq = cur_freq / cur_freq.sum()

            from scipy.spatial.distance import jensenshannon

            js_dist = jensenshannon(ref_freq, cur_freq, base=2)
            return float(js_dist ** 2)
        else:
            ks_stat, _ = stats.ks_2samp(
                reference_labels.astype(float), current_labels.astype(float)
            )
            return float(ks_stat)

    @staticmethod
    def _aggregate_score(
        pred_shift: float, label_drift: float, adwin_detected: bool
    ) -> float:
        """Combine sub-scores into an overall concept drift score."""
        adwin_score = 0.5 if adwin_detected else 0.0

        if label_drift > 0:
            score = 0.35 * pred_shift + 0.35 * label_drift + 0.30 * adwin_score
        else:
            score = 0.60 * pred_shift + 0.40 * adwin_score

        return float(min(score, 1.0))

    @staticmethod
    def _classify_drift_type(
        pred_shift: float, label_drift: float, adwin_detected: bool
    ) -> ConceptDriftType:
        """
        Classify the type of concept drift observed.

        Heuristic classification:
        - Sudden: ADWIN fires with high prediction shift.
        - Gradual: Moderate shift without ADWIN trigger.
        - Incremental: Low but consistent shift signals.
        """
        if adwin_detected and pred_shift > 0.3:
            return ConceptDriftType.SUDDEN
        elif pred_shift > 0.15 or label_drift > 0.15:
            return ConceptDriftType.GRADUAL
        elif pred_shift > 0.05 or label_drift > 0.05:
            return ConceptDriftType.INCREMENTAL
        return ConceptDriftType.NONE

    def reset(self) -> None:
        """Reset detector state."""
        self.adwin.reset()
        self._error_history.clear()
        logger.info("ConceptDriftDetector state reset.")
