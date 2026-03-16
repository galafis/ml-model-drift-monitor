"""
Prometheus metrics exporter.

Exposes drift detection metrics, performance indicators, and alert
counters as Prometheus gauges and counters for scraping by Prometheus
server.
"""

from typing import Any, Dict, Optional

from src.config.settings import ExporterConfig
from src.utils.logger import get_logger

logger = get_logger("exporters.prometheus")


class PrometheusExporter:
    """
    Exports monitoring metrics in Prometheus format.

    Manages Prometheus gauge and counter metrics for drift scores,
    performance metrics, and alert counts. Uses the prometheus_client
    library when available, with a text-based fallback.

    Attributes:
        config: Exporter configuration.
        prefix: Metric name prefix.
    """

    def __init__(self, config: Optional[ExporterConfig] = None):
        """
        Initialize the Prometheus exporter.

        Args:
            config: Exporter configuration.
        """
        self.config = config or ExporterConfig()
        self.prefix = self.config.metric_prefix
        self._use_prometheus_client = False
        self._gauges: Dict[str, Any] = {}
        self._counters: Dict[str, Any] = {}
        self._fallback_metrics: Dict[str, float] = {}

        self._try_prometheus_client()

    def _try_prometheus_client(self) -> None:
        """Attempt to initialize prometheus_client library."""
        try:
            from prometheus_client import Counter, Gauge, CollectorRegistry

            self._registry = CollectorRegistry()

            self._gauges["data_drift_score"] = Gauge(
                f"{self.prefix}_data_drift_score",
                "Current data drift score",
                ["model"],
                registry=self._registry,
            )
            self._gauges["concept_drift_score"] = Gauge(
                f"{self.prefix}_concept_drift_score",
                "Current concept drift score",
                ["model"],
                registry=self._registry,
            )
            self._gauges["performance_score"] = Gauge(
                f"{self.prefix}_performance_score",
                "Performance degradation score",
                ["model"],
                registry=self._registry,
            )
            self._gauges["data_drift_detected"] = Gauge(
                f"{self.prefix}_data_drift_detected",
                "Whether data drift is detected (0 or 1)",
                ["model"],
                registry=self._registry,
            )
            self._gauges["concept_drift_detected"] = Gauge(
                f"{self.prefix}_concept_drift_detected",
                "Whether concept drift is detected (0 or 1)",
                ["model"],
                registry=self._registry,
            )
            self._gauges["performance_degraded"] = Gauge(
                f"{self.prefix}_performance_degraded",
                "Whether performance is degraded (0 or 1)",
                ["model"],
                registry=self._registry,
            )
            self._gauges["model_health"] = Gauge(
                f"{self.prefix}_model_health",
                "Model health score (0=critical, 1=degraded, 2=warning, 3=healthy)",
                ["model"],
                registry=self._registry,
            )

            self._counters["alerts_total"] = Counter(
                f"{self.prefix}_alerts_total",
                "Total number of alerts triggered",
                ["model", "severity"],
                registry=self._registry,
            )
            self._counters["checks_total"] = Counter(
                f"{self.prefix}_checks_total",
                "Total monitoring checks performed",
                ["model"],
                registry=self._registry,
            )

            self._use_prometheus_client = True
            logger.info("Prometheus client initialized with registry.")

        except ImportError:
            logger.warning(
                "prometheus_client not installed. Using text-based fallback."
            )
            self._use_prometheus_client = False

    def update_data_drift(
        self, model_name: str, score: float, is_drifted: bool
    ) -> None:
        """
        Update data drift metrics.

        Args:
            model_name: Model identifier.
            score: Drift score value.
            is_drifted: Whether drift was detected.
        """
        if self._use_prometheus_client:
            self._gauges["data_drift_score"].labels(model=model_name).set(score)
            self._gauges["data_drift_detected"].labels(model=model_name).set(
                1.0 if is_drifted else 0.0
            )
        else:
            self._fallback_metrics[f"data_drift_score_{model_name}"] = score
            self._fallback_metrics[f"data_drift_detected_{model_name}"] = (
                1.0 if is_drifted else 0.0
            )

    def update_concept_drift(
        self, model_name: str, score: float, is_drifted: bool
    ) -> None:
        """
        Update concept drift metrics.

        Args:
            model_name: Model identifier.
            score: Drift score value.
            is_drifted: Whether drift was detected.
        """
        if self._use_prometheus_client:
            self._gauges["concept_drift_score"].labels(model=model_name).set(score)
            self._gauges["concept_drift_detected"].labels(model=model_name).set(
                1.0 if is_drifted else 0.0
            )
        else:
            self._fallback_metrics[f"concept_drift_score_{model_name}"] = score
            self._fallback_metrics[f"concept_drift_detected_{model_name}"] = (
                1.0 if is_drifted else 0.0
            )

    def update_performance(
        self, model_name: str, score: float, is_degraded: bool
    ) -> None:
        """
        Update performance degradation metrics.

        Args:
            model_name: Model identifier.
            score: Performance degradation score.
            is_degraded: Whether degradation was detected.
        """
        if self._use_prometheus_client:
            self._gauges["performance_score"].labels(model=model_name).set(score)
            self._gauges["performance_degraded"].labels(model=model_name).set(
                1.0 if is_degraded else 0.0
            )
        else:
            self._fallback_metrics[f"performance_score_{model_name}"] = score
            self._fallback_metrics[f"performance_degraded_{model_name}"] = (
                1.0 if is_degraded else 0.0
            )

    def update_health(self, model_name: str, health: str) -> None:
        """
        Update model health metric.

        Args:
            model_name: Model identifier.
            health: Health status string.
        """
        health_map = {
            "healthy": 3.0,
            "warning": 2.0,
            "degraded": 1.0,
            "critical": 0.0,
        }
        value = health_map.get(health, 2.0)

        if self._use_prometheus_client:
            self._gauges["model_health"].labels(model=model_name).set(value)
        else:
            self._fallback_metrics[f"model_health_{model_name}"] = value

    def increment_alerts(
        self, model_name: str, severity: str = "warning"
    ) -> None:
        """Increment the alert counter."""
        if self._use_prometheus_client:
            self._counters["alerts_total"].labels(
                model=model_name, severity=severity
            ).inc()
        else:
            key = f"alerts_total_{model_name}_{severity}"
            self._fallback_metrics[key] = (
                self._fallback_metrics.get(key, 0) + 1
            )

    def increment_checks(self, model_name: str) -> None:
        """Increment the check counter."""
        if self._use_prometheus_client:
            self._counters["checks_total"].labels(model=model_name).inc()
        else:
            key = f"checks_total_{model_name}"
            self._fallback_metrics[key] = (
                self._fallback_metrics.get(key, 0) + 1
            )

    def generate_text_metrics(self) -> str:
        """
        Generate metrics in Prometheus text exposition format.

        Returns:
            Metrics string in Prometheus format.
        """
        if self._use_prometheus_client:
            from prometheus_client import generate_latest

            return generate_latest(self._registry).decode("utf-8")

        lines: list = []
        for key, value in sorted(self._fallback_metrics.items()):
            metric_name = f"{self.prefix}_{key}"
            metric_name = metric_name.replace("-", "_")
            lines.append(f"{metric_name} {value}")

        return "\n".join(lines) + "\n"
