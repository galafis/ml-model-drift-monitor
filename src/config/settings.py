"""
Application settings and configuration management.

Provides Pydantic-based configuration classes for all system components
with environment variable support and YAML file loading.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class MonitoringConfig(BaseSettings):
    """Configuration for the monitoring scheduler and execution."""

    check_interval_seconds: int = Field(
        default=300,
        description="Interval between monitoring checks in seconds.",
    )
    window_size: int = Field(
        default=1000,
        description="Number of samples in the sliding window.",
    )
    min_samples: int = Field(
        default=100,
        description="Minimum samples required before running drift detection.",
    )
    max_reports_stored: int = Field(
        default=500,
        description="Maximum number of reports to store before rotation.",
    )
    enabled: bool = Field(default=True, description="Whether monitoring is active.")

    model_config = {"env_prefix": "MONITOR_"}


class DriftConfig(BaseSettings):
    """Configuration for drift detection thresholds and methods."""

    ks_test_threshold: float = Field(
        default=0.05,
        description="P-value threshold for Kolmogorov-Smirnov test.",
    )
    psi_threshold: float = Field(
        default=0.2,
        description="Population Stability Index threshold for drift.",
    )
    js_divergence_threshold: float = Field(
        default=0.1,
        description="Jensen-Shannon divergence threshold.",
    )
    feature_drift_fraction_threshold: float = Field(
        default=0.5,
        description="Fraction of features that must drift to trigger dataset-level alert.",
    )
    concept_drift_threshold: float = Field(
        default=0.05,
        description="Threshold for concept drift detection.",
    )
    adwin_delta: float = Field(
        default=0.002,
        description="ADWIN algorithm sensitivity parameter.",
    )
    performance_degradation_threshold: float = Field(
        default=0.1,
        description="Threshold for performance metric degradation (relative drop).",
    )
    trend_lookback_windows: int = Field(
        default=10,
        description="Number of historical windows for trend analysis.",
    )

    model_config = {"env_prefix": "DRIFT_"}


class StorageConfig(BaseSettings):
    """Configuration for data storage backends."""

    postgres_host: str = Field(default="localhost", description="PostgreSQL host.")
    postgres_port: int = Field(default=5432, description="PostgreSQL port.")
    postgres_db: str = Field(
        default="drift_monitor", description="PostgreSQL database name."
    )
    postgres_user: str = Field(
        default="drift_user", description="PostgreSQL username."
    )
    postgres_password: str = Field(
        default="drift_pass", description="PostgreSQL password."
    )
    redis_host: str = Field(default="localhost", description="Redis host.")
    redis_port: int = Field(default=6379, description="Redis port.")
    redis_db: int = Field(default=0, description="Redis database number.")
    redis_password: Optional[str] = Field(
        default=None, description="Redis password."
    )
    fallback_json_dir: str = Field(
        default="./data/reports",
        description="Directory for JSON file fallback storage.",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Default cache TTL in seconds.",
    )

    model_config = {"env_prefix": "STORAGE_"}

    @property
    def postgres_dsn(self) -> str:
        """Build PostgreSQL connection DSN."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Build Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class AlertingConfig(BaseSettings):
    """Configuration for the alerting system."""

    enabled: bool = Field(default=True, description="Whether alerting is active.")
    cooldown_seconds: int = Field(
        default=600,
        description="Minimum seconds between duplicate alerts.",
    )
    max_alerts_per_hour: int = Field(
        default=20,
        description="Maximum alerts per hour to prevent flooding.",
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for alert notifications.",
    )
    webhook_timeout: int = Field(
        default=10,
        description="Timeout for webhook requests in seconds.",
    )
    channels: List[str] = Field(
        default=["log"],
        description="Alert channels: log, webhook, email.",
    )
    email_recipients: List[str] = Field(
        default=[],
        description="Email recipients for alert notifications.",
    )

    model_config = {"env_prefix": "ALERT_"}


class ExporterConfig(BaseSettings):
    """Configuration for metrics exporters."""

    prometheus_enabled: bool = Field(
        default=True, description="Enable Prometheus metrics export."
    )
    prometheus_port: int = Field(
        default=8001, description="Port for Prometheus metrics endpoint."
    )
    metric_prefix: str = Field(
        default="ml_drift",
        description="Prefix for all Prometheus metric names.",
    )
    export_interval_seconds: int = Field(
        default=30,
        description="Interval for pushing metrics.",
    )

    model_config = {"env_prefix": "EXPORTER_"}


class AppSettings(BaseSettings):
    """Root application settings aggregating all configurations."""

    app_name: str = Field(
        default="ml-model-drift-monitor",
        description="Application name.",
    )
    app_version: str = Field(default="1.0.0", description="Application version.")
    debug: bool = Field(default=False, description="Enable debug mode.")
    log_level: str = Field(default="INFO", description="Application log level.")
    host: str = Field(default="0.0.0.0", description="API server host.")
    port: int = Field(default=8000, description="API server port.")

    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)

    model_config = {"env_prefix": "APP_"}

    @classmethod
    def from_yaml(cls, path: str) -> "AppSettings":
        """
        Load settings from a YAML configuration file.

        Values from the file serve as defaults; environment variables
        take precedence.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Configured AppSettings instance.
        """
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        monitoring_data = raw.get("monitoring", {})
        drift_data = raw.get("drift", {})
        storage_data = raw.get("storage", {})
        alerting_data = raw.get("alerting", {})
        exporter_data = raw.get("exporter", {})

        return cls(
            app_name=raw.get("app_name", "ml-model-drift-monitor"),
            app_version=raw.get("app_version", "1.0.0"),
            debug=raw.get("debug", False),
            log_level=raw.get("log_level", "INFO"),
            host=raw.get("host", "0.0.0.0"),
            port=raw.get("port", 8000),
            monitoring=MonitoringConfig(**monitoring_data),
            drift=DriftConfig(**drift_data),
            storage=StorageConfig(**storage_data),
            alerting=AlertingConfig(**alerting_data),
            exporter=ExporterConfig(**exporter_data),
        )
