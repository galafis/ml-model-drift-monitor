"""
Integration tests for the end-to-end monitoring flow.

Tests the complete pipeline from data ingestion through drift detection,
report generation, alerting, and metrics export.
"""

import numpy as np
import pytest

from src.alerting.alert_manager import AlertManager
from src.alerting.rules import AlertRule, AlertSeverity, RuleEngine, RuleType
from src.config.settings import AlertingConfig, AppSettings
from src.exporters.prometheus_exporter import PrometheusExporter
from src.monitors.batch_monitor import BatchMonitor
from src.monitors.model_monitor import ModelMonitor
from src.storage.cache import CacheManager
from src.storage.report_store import ReportStore


@pytest.fixture
def rng():
    return np.random.default_rng(seed=123)


class TestEndToEndMonitoringFlow:
    """Full pipeline integration test."""

    def test_complete_monitoring_cycle(self, rng):
        """Test reference setup -> monitoring -> alerting -> export."""
        settings = AppSettings()
        monitor = ModelMonitor(
            model_name="integration_model",
            settings=settings,
            task_type="classification",
        )

        n_features = 4
        ref_features = rng.standard_normal((500, n_features))
        ref_labels = (ref_features[:, 0] > 0).astype(int)
        ref_preds = ref_labels.copy()
        feature_names = [f"feat_{i}" for i in range(n_features)]

        monitor.set_reference(ref_features, ref_preds, ref_labels, feature_names)

        cur_features = rng.standard_normal((300, n_features))
        cur_labels = (cur_features[:, 0] > 0).astype(int)
        cur_preds = cur_labels.copy()

        report = monitor.run_check(cur_features, cur_preds, cur_labels)

        assert report.status == "completed"
        assert report.data_drift is not None
        assert report.concept_drift is not None
        assert report.performance_drift is not None

        rule_engine = RuleEngine()
        metrics = {
            "data_drift_score": report.data_drift.get("drift_score", 0.0),
            "concept_drift_score": report.concept_drift.get("drift_score", 0.0),
        }
        if report.performance_drift:
            metrics["performance_score"] = report.performance_drift.get(
                "overall_score", 0.0
            )

        evaluations = rule_engine.evaluate(metrics)
        assert isinstance(evaluations, list)

        exporter = PrometheusExporter()
        exporter.update_data_drift(
            "integration_model",
            report.data_drift.get("drift_score", 0.0),
            report.data_drift.get("is_drifted", False),
        )
        exporter.update_health("integration_model", report.overall_health)
        exporter.increment_checks("integration_model")

        metrics_text = exporter.generate_text_metrics()
        assert len(metrics_text) > 0

    def test_drift_detection_pipeline(self, rng):
        """Test that drift is properly detected end-to-end."""
        settings = AppSettings()
        monitor = ModelMonitor(
            model_name="drift_test",
            settings=settings,
            task_type="classification",
        )

        ref_features = rng.standard_normal((500, 3))
        ref_labels = (ref_features[:, 0] > 0).astype(int)
        ref_preds = ref_labels.copy()

        monitor.set_reference(ref_features, ref_preds, ref_labels)

        drifted_features = rng.standard_normal((500, 3)) + 3.0
        drifted_labels = (drifted_features[:, 0] > 0).astype(int)
        bad_preds = rng.choice([0, 1], size=500)

        report = monitor.run_check(drifted_features, bad_preds, drifted_labels)

        assert report.data_drift["is_drifted"] is True
        assert report.overall_health != "healthy"

    def test_batch_comparison_flow(self, rng):
        """Test batch monitoring with sequential batch comparison."""
        batch_monitor = BatchMonitor(task_type="classification")

        batches = []
        for i in range(4):
            n_samples = 200
            features = rng.standard_normal((n_samples, 3)) + i * 0.5
            labels = (features[:, 0] > 0).astype(int)
            preds = labels.copy()
            batches.append({
                "features": features,
                "predictions": preds,
                "labels": labels,
            })

        reports = batch_monitor.process_batch_sequence(batches)

        assert len(reports) == 3
        for report in reports:
            assert report.overall_assessment in [
                "stable",
                "minor_changes",
                "significant_drift",
                "critical_degradation",
            ]

    def test_alert_manager_integration(self, rng):
        """Test alert manager with rule evaluation."""
        config = AlertingConfig(
            enabled=True,
            cooldown_seconds=0,
            max_alerts_per_hour=100,
            channels=["log"],
        )

        rules = [
            AlertRule(
                name="test_threshold",
                description="Test threshold rule",
                rule_type=RuleType.THRESHOLD,
                severity=AlertSeverity.WARNING,
                metric_name="drift_score",
                threshold=0.1,
                direction="above",
            ),
        ]

        manager = AlertManager(config=config, rules=rules)

        alerts = manager.evaluate_and_alert(
            model_name="test_model",
            metrics={"drift_score": 0.5},
        )

        assert len(alerts) > 0
        assert alerts[0].severity == AlertSeverity.WARNING

        summary = manager.get_alert_summary()
        assert summary["total_alerts"] > 0

    def test_cache_manager_roundtrip(self):
        """Test cache store and retrieve cycle."""
        cache = CacheManager()

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cache.cache_reference_distribution("model_a", "feature_x", data)

        retrieved = cache.get_reference_distribution("model_a", "feature_x")
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, data)

    def test_report_store_json_fallback(self, tmp_path):
        """Test report storage with JSON fallback."""
        from src.config.settings import StorageConfig

        config = StorageConfig(fallback_json_dir=str(tmp_path / "reports"))
        store = ReportStore(config=config)

        report = {
            "report_id": "test-123",
            "model_name": "test_model",
            "timestamp": "2025-01-01T00:00:00Z",
            "status": "completed",
            "overall_health": "healthy",
        }

        store.save_report(report)
        retrieved = store.get_report("test-123")

        assert retrieved is not None
        assert retrieved["report_id"] == "test-123"

        reports = store.list_reports(model_name="test_model")
        assert len(reports) >= 1

        deleted = store.delete_report("test-123")
        assert deleted is True

    def test_multiple_monitoring_cycles(self, rng):
        """Test multiple sequential monitoring cycles."""
        settings = AppSettings()
        monitor = ModelMonitor("cycle_test", settings=settings)

        ref = rng.standard_normal((300, 3))
        labels = (ref[:, 0] > 0).astype(int)
        monitor.set_reference(ref, labels, labels)

        for i in range(5):
            shift = i * 0.2
            cur = rng.standard_normal((200, 3)) + shift
            cur_labels = (cur[:, 0] > shift).astype(int)
            cur_preds = cur_labels.copy()

            report = monitor.run_check(cur, cur_preds, cur_labels)
            assert report.status == "completed"

        status = monitor.get_status()
        assert status["check_count"] == 5
        assert status["stored_reports"] == 5
