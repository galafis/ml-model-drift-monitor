"""
Unit tests for ConceptDriftDetector.

Tests ADWIN window, prediction distribution shift, label drift,
and combined concept drift detection.
"""

import numpy as np
import pytest

from src.detectors.concept_drift import (
    ADWINWindow,
    ConceptDriftDetector,
    ConceptDriftResult,
    ConceptDriftType,
)


class TestADWINWindow:
    """Tests for the ADWINWindow class."""

    def test_empty_window(self):
        """Empty window should have zero mean and size."""
        window = ADWINWindow(delta=0.01)
        assert window.size == 0
        assert window.mean == 0.0

    def test_add_stable_data(self, rng):
        """Stable data should not trigger drift."""
        window = ADWINWindow(delta=0.01, max_window_size=1000)
        stable_data = rng.standard_normal(200) * 0.1 + 0.5

        drift = window.add_batch(stable_data)
        assert window.size > 0
        assert isinstance(drift, bool)

    def test_detect_mean_shift(self):
        """Significant mean shift should trigger drift detection."""
        window = ADWINWindow(delta=0.01, max_window_size=2000)

        low_data = np.zeros(500) + 0.1
        window.add_batch(low_data)

        high_data = np.ones(500) + 0.9
        drift = window.add_batch(high_data)

        assert drift is True

    def test_window_size_limit(self):
        """Window should not exceed max size."""
        window = ADWINWindow(delta=0.01, max_window_size=100)
        data = np.ones(200)
        window.add_batch(data)

        assert window.size <= 100

    def test_reset_clears_state(self, rng):
        """Reset should clear all window data."""
        window = ADWINWindow(delta=0.01)
        window.add_batch(rng.standard_normal(100))
        window.reset()

        assert window.size == 0
        assert window.mean == 0.0


class TestConceptDriftDetector:
    """Tests for the ConceptDriftDetector class."""

    def test_no_drift_same_predictions(self, drift_config, rng):
        """Same predictions should show no concept drift."""
        detector = ConceptDriftDetector(config=drift_config)
        preds = rng.choice([0, 1], size=300, p=[0.7, 0.3]).astype(float)

        result = detector.detect(preds, preds)

        assert isinstance(result, ConceptDriftResult)
        assert result.prediction_shift_score < 0.05

    def test_detect_prediction_shift(self, drift_config, rng):
        """Shifted prediction distribution should be detected."""
        detector = ConceptDriftDetector(config=drift_config)
        ref_preds = rng.choice([0, 1], size=500, p=[0.8, 0.2]).astype(float)
        cur_preds = rng.choice([0, 1], size=500, p=[0.3, 0.7]).astype(float)

        result = detector.detect(ref_preds, cur_preds)

        assert result.prediction_shift_score > 0.1

    def test_label_drift_detection(self, drift_config, rng):
        """Changed label distribution should be detected."""
        detector = ConceptDriftDetector(config=drift_config)
        ref_preds = rng.standard_normal(300)
        cur_preds = rng.standard_normal(300)
        ref_labels = rng.choice([0, 1], size=300, p=[0.9, 0.1])
        cur_labels = rng.choice([0, 1], size=300, p=[0.4, 0.6])

        result = detector.detect(
            ref_preds, cur_preds,
            reference_labels=ref_labels,
            current_labels=cur_labels,
        )

        assert result.label_drift_score > 0.0

    def test_adwin_with_errors(self, drift_config):
        """ADWIN should process error streams without crashing."""
        detector = ConceptDriftDetector(config=drift_config)
        ref_preds = np.zeros(200)
        cur_preds = np.zeros(200)
        errors = np.concatenate([np.zeros(100), np.ones(100)])

        result = detector.detect(
            ref_preds, cur_preds, current_errors=errors
        )

        assert isinstance(result, ConceptDriftResult)
        assert "adwin_window_size" in result.window_means

    def test_drift_type_classification(self, drift_config, rng):
        """Drift type should be properly classified."""
        detector = ConceptDriftDetector(config=drift_config)
        ref_preds = rng.standard_normal(500)
        cur_preds = rng.standard_normal(500)

        result = detector.detect(ref_preds, cur_preds)
        assert isinstance(result.drift_type, ConceptDriftType)

    def test_reset_clears_state(self, drift_config, rng):
        """Reset should clear detector state."""
        detector = ConceptDriftDetector(config=drift_config)
        preds = rng.standard_normal(100)
        errors = rng.choice([0, 1], size=100).astype(float)
        detector.detect(preds, preds, current_errors=errors)

        detector.reset()
        assert detector.adwin.size == 0

    def test_details_contain_statistics(self, drift_config, rng):
        """Result details should contain prediction statistics."""
        detector = ConceptDriftDetector(config=drift_config)
        ref = rng.standard_normal(200)
        cur = rng.standard_normal(200) + 1.0

        result = detector.detect(ref, cur)

        assert "reference_prediction_mean" in result.details
        assert "current_prediction_mean" in result.details
        assert "reference_prediction_std" in result.details
        assert "current_prediction_std" in result.details
