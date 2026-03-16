"""
Unit tests for PerformanceDriftDetector.

Tests sliding window metrics, degradation detection, and trend analysis.
"""

import numpy as np
import pytest

from src.detectors.performance_drift import (
    DegradationLevel,
    PerformanceDriftDetector,
    PerformanceDriftResult,
    TrendAnalysis,
)


class TestPerformanceDriftDetector:
    """Tests for the PerformanceDriftDetector class."""

    def test_classification_metrics_computed(self, drift_config, rng):
        """Classification metrics should be computed correctly."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="classification"
        )
        y_true = rng.choice([0, 1], size=200)
        y_pred = y_true.copy()

        result = detector.detect(y_true, y_pred)

        assert isinstance(result, PerformanceDriftResult)
        assert "accuracy" in result.current_metrics
        assert "precision" in result.current_metrics
        assert "recall" in result.current_metrics
        assert "f1" in result.current_metrics
        assert result.current_metrics["accuracy"] == 1.0

    def test_regression_metrics_computed(self, drift_config, rng):
        """Regression metrics should be computed correctly."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="regression"
        )
        y_true = rng.standard_normal(200)
        y_pred = y_true + rng.standard_normal(200) * 0.1

        result = detector.detect(y_true, y_pred)

        assert "mae" in result.current_metrics
        assert "rmse" in result.current_metrics
        assert result.current_metrics["mae"] < 0.5
        assert result.current_metrics["rmse"] < 0.5

    def test_no_degradation_perfect_match(self, drift_config, rng):
        """Perfect predictions should show no degradation."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="classification"
        )
        y_true = rng.choice([0, 1], size=200)

        result = detector.detect(y_true, y_true)

        assert result.is_degraded is False
        assert result.degradation_level == DegradationLevel.NONE

    def test_detect_degradation(self, drift_config, rng):
        """Degraded predictions should trigger detection."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="classification"
        )
        y_true = rng.choice([0, 1], size=300)
        y_pred_good = y_true.copy()

        ref_metrics = detector._compute_metrics(y_true, y_pred_good)

        y_pred_bad = rng.choice([0, 1], size=300)

        result = detector.detect(y_true, y_pred_bad, reference_metrics=ref_metrics)

        assert result.is_degraded is True
        assert result.degradation_level != DegradationLevel.NONE

    def test_metric_changes_computed(self, drift_config, rng):
        """Metric changes should reflect differences from reference."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="classification"
        )

        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 25)
        y_pred_ref = y_true.copy()
        y_pred_cur = np.roll(y_true, 1)

        ref_metrics = detector._compute_metrics(y_true, y_pred_ref)
        result = detector.detect(y_true, y_pred_cur, reference_metrics=ref_metrics)

        assert "accuracy" in result.metric_changes
        assert result.metric_changes["accuracy"] < 0

    def test_trend_analysis_with_history(self, drift_config, rng):
        """Trend analysis should work after accumulating window history."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="classification"
        )

        y_true = rng.choice([0, 1], size=100)
        for i in range(5):
            noise_level = 0.05 + i * 0.05
            y_pred = y_true.copy()
            flip_count = int(len(y_true) * noise_level)
            flip_idx = rng.choice(len(y_true), size=flip_count, replace=False)
            y_pred[flip_idx] = 1 - y_pred[flip_idx]
            detector.detect(y_true, y_pred)

        result = detector.detect(
            y_true,
            rng.choice([0, 1], size=100),
        )

        assert len(result.trends) > 0
        for trend in result.trends:
            assert isinstance(trend, TrendAnalysis)
            assert trend.windows_analyzed >= 3

    def test_regression_degradation(self, drift_config, rng):
        """Regression performance degradation should be detected."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="regression"
        )
        y_true = rng.standard_normal(200)
        y_pred_good = y_true + rng.standard_normal(200) * 0.01
        y_pred_bad = y_true + rng.standard_normal(200) * 5.0

        ref_metrics = detector._compute_metrics(y_true, y_pred_good)
        result = detector.detect(y_true, y_pred_bad, reference_metrics=ref_metrics)

        assert result.is_degraded is True

    def test_window_history_stored(self, drift_config, rng):
        """Window records should accumulate over multiple checks."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="classification"
        )
        y_true = rng.choice([0, 1], size=50)

        for _ in range(3):
            detector.detect(y_true, y_true)

        assert len(detector._window_records) > 0
        assert detector._window_counter == 3

    def test_reset_clears_state(self, drift_config, rng):
        """Reset should clear all history."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="classification"
        )
        y_true = rng.choice([0, 1], size=50)
        detector.detect(y_true, y_true)

        detector.reset()

        assert len(detector.metric_history) == 0
        assert len(detector._window_records) == 0
        assert detector._window_counter == 0

    def test_overall_score_range(self, drift_config, rng):
        """Overall score should be in [0, 1] range."""
        detector = PerformanceDriftDetector(
            config=drift_config, task_type="classification"
        )
        y_true = rng.choice([0, 1], size=200)
        y_pred = rng.choice([0, 1], size=200)

        result = detector.detect(y_true, y_pred)
        assert 0.0 <= result.overall_score <= 1.0
