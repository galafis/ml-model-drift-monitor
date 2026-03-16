"""
Unit tests for the FastAPI endpoints.

Tests health check, monitoring execution, report listing,
and Prometheus metrics endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Provide a FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root(self, client):
        """Root endpoint should return service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data


class TestMonitorEndpoints:
    """Tests for monitoring endpoints."""

    def test_monitor_run(self, client):
        """Monitor run endpoint should execute monitoring check."""
        payload = {
            "model_name": "test_api_model",
            "reference_features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] * 50,
            "reference_predictions": [0, 1, 0] * 50,
            "current_features": [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]] * 50,
            "current_predictions": [0, 1, 0] * 50,
            "reference_labels": [0, 1, 0] * 50,
            "current_labels": [0, 1, 0] * 50,
            "feature_names": ["f1", "f2"],
            "task_type": "classification",
        }

        response = client.post("/api/v1/monitor/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test_api_model"
        assert data["status"] == "completed"
        assert "data_drift" in data
        assert "concept_drift" in data
        assert "overall_health" in data

    def test_monitor_status(self, client):
        """Monitor status endpoint should return current state."""
        response = client.get(
            "/api/v1/monitor/status", params={"model_name": "nonexistent"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"

    def test_reports_list_empty(self, client):
        """Report listing should return empty for unknown model."""
        response = client.get(
            "/api/v1/reports", params={"model_name": "no_model"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0

    def test_report_not_found(self, client):
        """Getting non-existent report should return 404."""
        response = client.get("/api/v1/reports/nonexistent-id")
        assert response.status_code == 404


class TestAlertEndpoints:
    """Tests for alert endpoints."""

    def test_alerts_list(self, client):
        """Alerts endpoint should return list."""
        response = client.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "alerts" in data


class TestPrometheusEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_prometheus_metrics(self, client):
        """Prometheus endpoint should return text metrics."""
        response = client.get("/api/v1/metrics/prometheus")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
