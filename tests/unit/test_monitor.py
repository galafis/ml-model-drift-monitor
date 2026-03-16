"""
Unit tests for ModelMonitor orchestrator.

Tests monitoring execution, report generation, state management,
and health assessment.
"""

import numpy as np
import pytest

from src.monitors.model_monitor import ModelMonitor, MonitoringReport, MonitoringStatus


class TestModelMonitor:
    """Tests for the ModelMonitor class."""

    def test_initialization(self, app_settings):
        """Monitor should initialize with correct defaults."""
        monitor = ModelMonitor(
            model_name="test_model",
            settings=app_settings,
            task_type="classification",
        )

        assert monitor.model_name == "test_model"
        assert monitor.status == MonitoringStatus.IDLE
        assert monitor.task_type == "classification"

    def test_set_reference(self, app_settings, reference_data):
        """Setting reference data should store it correctly."""
        monitor = ModelMonitor("test_model", settings=app_settings)
        features, predictions, labels = reference_data

        monitor.set_reference(features, predictions, labels)

        status = monitor.get_status()
        assert status["has_reference"] is True
        assert status["reference_samples"] == features.shape[0]

    def test_run_check_without_reference_raises(self, app_settings, rng):
        """Running check without reference should raise RuntimeError."""
        monitor = ModelMonitor("test_model", settings=app_settings)
        features = rng.standard_normal((100, 5))
        preds = rng.choice([0, 1], size=100)

        with pytest.raises(RuntimeError, match="Reference data not set"):
            monitor.run_check(features, preds)

    def test_run_check_no_drift(
        self, app_settings, reference_data, no_drift_data, feature_names
    ):
        """Check with similar data should report healthy status."""
        monitor = ModelMonitor("test_model", settings=app_settings)
        ref_features, ref_preds, ref_labels = reference_data
        cur_features, cur_preds, cur_labels = no_drift_data

        monitor.set_reference(
            ref_features, ref_preds, ref_labels, feature_names
        )
        report = monitor.run_check(cur_features, cur_preds, cur_labels)

        assert isinstance(report, MonitoringReport)
        assert report.status == "completed"
        assert report.model_name == "test_model"
        assert report.data_drift is not None
        assert report.concept_drift is not None

    def test_run_check_with_drift(
        self, app_settings, reference_data, drifted_data, feature_names
    ):
        """Check with drifted data should detect issues."""
        monitor = ModelMonitor("test_model", settings=app_settings)
        ref_features, ref_preds, ref_labels = reference_data
        cur_features, cur_preds, cur_labels = drifted_data

        monitor.set_reference(
            ref_features, ref_preds, ref_labels, feature_names
        )
        report = monitor.run_check(cur_features, cur_preds, cur_labels)

        assert report.data_drift is not None
        has_drift = (
            report.data_drift.get("is_drifted", False)
            or report.concept_drift.get("is_drifted", False)
            or (
                report.performance_drift is not None
                and report.performance_drift.get("is_degraded", False)
            )
        )
        assert has_drift is True

    def test_run_check_without_labels(
        self, app_settings, reference_data, feature_names
    ):
        """Check without labels should skip performance detection."""
        monitor = ModelMonitor("test_model", settings=app_settings)
        ref_features, ref_preds, _ = reference_data

        monitor.set_reference(ref_features, ref_preds, feature_names=feature_names)
        report = monitor.run_check(ref_features, ref_preds)

        assert report.performance_drift is None

    def test_report_stored(self, app_settings, reference_data, feature_names):
        """Reports should be stored and retrievable."""
        monitor = ModelMonitor("test_model", settings=app_settings)
        ref_features, ref_preds, ref_labels = reference_data
        monitor.set_reference(
            ref_features, ref_preds, ref_labels, feature_names
        )

        report = monitor.run_check(ref_features, ref_preds, ref_labels)
        retrieved = monitor.get_report_by_id(report.report_id)

        assert retrieved is not None
        assert retrieved.report_id == report.report_id

    def test_get_reports_pagination(
        self, app_settings, reference_data, feature_names
    ):
        """Report listing should support pagination."""
        monitor = ModelMonitor("test_model", settings=app_settings)
        ref_features, ref_preds, ref_labels = reference_data
        monitor.set_reference(
            ref_features, ref_preds, ref_labels, feature_names
        )

        for _ in range(5):
            monitor.run_check(ref_features, ref_preds, ref_labels)

        reports = monitor.get_reports(limit=3, offset=0)
        assert len(reports) == 3

        reports2 = monitor.get_reports(limit=3, offset=3)
        assert len(reports2) == 2

    def test_status_updates(self, app_settings, reference_data, feature_names):
        """Status should reflect monitoring state changes."""
        monitor = ModelMonitor("test_model", settings=app_settings)

        status1 = monitor.get_status()
        assert status1["check_count"] == 0
        assert status1["has_reference"] is False

        ref_features, ref_preds, ref_labels = reference_data
        monitor.set_reference(
            ref_features, ref_preds, ref_labels, feature_names
        )
        monitor.run_check(ref_features, ref_preds, ref_labels)

        status2 = monitor.get_status()
        assert status2["check_count"] == 1
        assert status2["has_reference"] is True
        assert status2["last_check_time"] is not None

    def test_report_to_dict(self, app_settings, reference_data, feature_names):
        """Report should be serializable to dictionary."""
        monitor = ModelMonitor("test_model", settings=app_settings)
        ref_features, ref_preds, ref_labels = reference_data
        monitor.set_reference(
            ref_features, ref_preds, ref_labels, feature_names
        )
        report = monitor.run_check(ref_features, ref_preds, ref_labels)

        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "report_id" in report_dict
        assert "model_name" in report_dict
        assert "data_drift" in report_dict
