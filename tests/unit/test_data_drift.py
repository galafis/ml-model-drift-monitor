"""
Unit tests for DataDriftDetector.

Tests KS test, PSI, Jensen-Shannon divergence, and feature-level
drift analysis with known distributions.
"""

import numpy as np
import pytest

from src.detectors.data_drift import (
    DataDriftDetector,
    DriftResult,
    DriftSeverity,
    FeatureDriftResult,
)


class TestDataDriftDetector:
    """Tests for the DataDriftDetector class."""

    def test_no_drift_same_distribution(self, drift_config, rng):
        """Identical distributions should report no drift."""
        detector = DataDriftDetector(config=drift_config)
        data = rng.standard_normal((500, 3))

        result = detector.detect(data, data, feature_names=["a", "b", "c"])

        assert isinstance(result, DriftResult)
        assert result.is_drifted is False
        assert result.drift_score < 0.05
        assert result.drifted_feature_count == 0

    def test_detect_significant_drift(self, drift_config, rng):
        """Shifted distributions should be detected as drifted."""
        detector = DataDriftDetector(config=drift_config)
        reference = rng.standard_normal((500, 3))
        current = rng.standard_normal((500, 3)) + 3.0

        result = detector.detect(reference, current)

        assert result.is_drifted is True
        assert result.drift_score > 0.05
        assert result.drifted_feature_count > 0
        assert result.severity != DriftSeverity.NONE

    def test_feature_results_populated(self, drift_config, rng):
        """Each feature should have individual drift results."""
        detector = DataDriftDetector(config=drift_config)
        reference = rng.standard_normal((300, 4))
        current = rng.standard_normal((300, 4))
        current[:, 0] += 5.0

        names = ["f0", "f1", "f2", "f3"]
        result = detector.detect(reference, current, feature_names=names)

        assert len(result.feature_results) == 4
        assert result.feature_results[0].feature_name == "f0"
        assert result.feature_results[0].is_drifted is True

        for fr in result.feature_results:
            assert isinstance(fr, FeatureDriftResult)
            assert fr.ks_statistic >= 0.0
            assert fr.ks_p_value >= 0.0
            assert fr.psi_value >= 0.0
            assert fr.js_divergence >= 0.0

    def test_ks_test_known_values(self, drift_config):
        """KS test should reject H0 for clearly different distributions."""
        detector = DataDriftDetector(config=drift_config)
        ref = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        cur = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=float)

        stat, p_val = detector._ks_test(ref, cur)
        assert stat > 0.5
        assert p_val < 0.05

    def test_psi_no_change(self, drift_config, rng):
        """PSI should be near zero for same distribution."""
        detector = DataDriftDetector(config=drift_config)
        data = rng.standard_normal(1000)

        psi = detector._compute_psi(data, data)
        assert psi < 0.05

    def test_psi_significant_change(self, drift_config, rng):
        """PSI should exceed threshold for shifted distributions."""
        detector = DataDriftDetector(config=drift_config)
        ref = rng.standard_normal(1000)
        cur = rng.standard_normal(1000) + 2.0

        psi = detector._compute_psi(ref, cur)
        assert psi > 0.1

    def test_js_divergence_bounds(self, drift_config, rng):
        """JS divergence should be in [0, 1] range."""
        detector = DataDriftDetector(config=drift_config)
        ref = rng.standard_normal(500)
        cur = rng.standard_normal(500) + 5.0

        js = detector._compute_js_divergence(ref, cur)
        assert 0.0 <= js <= 1.0

    def test_js_divergence_same_data(self, drift_config, rng):
        """JS divergence should be near zero for identical data."""
        detector = DataDriftDetector(config=drift_config)
        data = rng.standard_normal(500)

        js = detector._compute_js_divergence(data, data)
        assert js < 0.02

    def test_shape_mismatch_raises(self, drift_config, rng):
        """Mismatched feature counts should raise ValueError."""
        detector = DataDriftDetector(config=drift_config)
        ref = rng.standard_normal((100, 3))
        cur = rng.standard_normal((100, 5))

        with pytest.raises(ValueError, match="Feature count mismatch"):
            detector.detect(ref, cur)

    def test_feature_names_mismatch_raises(self, drift_config, rng):
        """Wrong number of feature names should raise ValueError."""
        detector = DataDriftDetector(config=drift_config)
        data = rng.standard_normal((100, 3))

        with pytest.raises(ValueError, match="Feature names count"):
            detector.detect(data, data, feature_names=["a", "b"])

    def test_constant_feature_handled(self, drift_config):
        """Constant features should not cause errors."""
        detector = DataDriftDetector(config=drift_config)
        ref = np.ones((100, 1))
        cur = np.ones((100, 1))

        result = detector.detect(ref, cur)
        assert result.drifted_feature_count == 0

    def test_method_scores_present(self, drift_config, rng):
        """Method scores should be populated in the result."""
        detector = DataDriftDetector(config=drift_config)
        data = rng.standard_normal((200, 2))
        result = detector.detect(data, data)

        assert "ks_mean_statistic" in result.method_scores
        assert "psi_mean" in result.method_scores
        assert "js_mean" in result.method_scores

    def test_metadata_contains_sample_info(self, drift_config, rng):
        """Result metadata should include sample counts."""
        detector = DataDriftDetector(config=drift_config)
        ref = rng.standard_normal((300, 2))
        cur = rng.standard_normal((200, 2))

        result = detector.detect(ref, cur)
        assert result.metadata["reference_samples"] == 300
        assert result.metadata["current_samples"] == 200
