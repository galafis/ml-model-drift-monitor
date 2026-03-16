"""
Alert management module.

Orchestrates alert evaluation, deduplication, cooldown management,
and multi-channel notification delivery.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from src.alerting.rules import AlertRule, AlertSeverity, RuleEngine, RuleEvaluation
from src.config.settings import AlertingConfig
from src.utils.logger import get_logger

logger = get_logger("alerting.manager")


@dataclass
class Alert:
    """A triggered alert record."""

    alert_id: str
    timestamp: str
    severity: AlertSeverity
    rule_name: str
    message: str
    model_name: str
    current_value: float
    threshold: float
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "rule_name": self.rule_name,
            "message": self.message,
            "model_name": self.model_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "acknowledged": self.acknowledged,
            "metadata": self.metadata,
        }


class AlertManager:
    """
    Manages alert lifecycle: evaluation, deduplication, and delivery.

    Features:
    - Rule-based evaluation using RuleEngine.
    - Cooldown between duplicate alerts.
    - Rate limiting per hour.
    - Multi-channel delivery (log, webhook).
    - Alert history with acknowledgment support.

    Attributes:
        config: Alerting configuration.
        rule_engine: Rule evaluation engine.
        alerts: Historical alert records.
    """

    def __init__(
        self,
        config: Optional[AlertingConfig] = None,
        rules: Optional[List[AlertRule]] = None,
    ):
        """
        Initialize the alert manager.

        Args:
            config: Alerting configuration.
            rules: Custom alert rules. Uses defaults if not provided.
        """
        self.config = config or AlertingConfig()
        self.rule_engine = RuleEngine(rules=rules)
        self.alerts: List[Alert] = []
        self._last_alert_times: Dict[str, float] = {}
        self._hourly_count = 0
        self._hour_start = time.time()

        logger.info(
            "AlertManager initialized (channels=%s, cooldown=%ds, max/hour=%d)",
            self.config.channels,
            self.config.cooldown_seconds,
            self.config.max_alerts_per_hour,
        )

    def evaluate_and_alert(
        self,
        model_name: str,
        metrics: Dict[str, float],
    ) -> List[Alert]:
        """
        Evaluate metrics against rules and fire alerts.

        Args:
            model_name: Name of the monitored model.
            metrics: Current metric values.

        Returns:
            List of newly fired alerts.
        """
        if not self.config.enabled:
            return []

        self._check_hourly_reset()

        evaluations = self.rule_engine.evaluate(metrics)
        fired_alerts: List[Alert] = []

        for evaluation in evaluations:
            if not self._check_cooldown(evaluation.rule_name):
                logger.debug(
                    "Alert '%s' suppressed by cooldown.", evaluation.rule_name
                )
                continue

            if self._hourly_count >= self.config.max_alerts_per_hour:
                logger.warning(
                    "Hourly alert limit reached (%d). Suppressing further alerts.",
                    self.config.max_alerts_per_hour,
                )
                break

            alert = Alert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=evaluation.severity,
                rule_name=evaluation.rule_name,
                message=evaluation.message,
                model_name=model_name,
                current_value=evaluation.current_value,
                threshold=evaluation.threshold,
                metadata=evaluation.metadata,
            )

            self.alerts.append(alert)
            self._last_alert_times[evaluation.rule_name] = time.time()
            self._hourly_count += 1

            self._deliver(alert)
            fired_alerts.append(alert)

        return fired_alerts

    def _check_cooldown(self, rule_name: str) -> bool:
        """Check if the cooldown period has elapsed for a rule."""
        last_time = self._last_alert_times.get(rule_name)
        if last_time is None:
            return True
        return (time.time() - last_time) >= self.config.cooldown_seconds

    def _check_hourly_reset(self) -> None:
        """Reset hourly counter if an hour has passed."""
        if time.time() - self._hour_start >= 3600:
            self._hourly_count = 0
            self._hour_start = time.time()

    def _deliver(self, alert: Alert) -> None:
        """
        Deliver alert through configured channels.

        Args:
            alert: Alert to deliver.
        """
        for channel in self.config.channels:
            try:
                if channel == "log":
                    self._deliver_log(alert)
                elif channel == "webhook":
                    self._deliver_webhook(alert)
                else:
                    logger.warning("Unknown alert channel: %s", channel)
            except Exception as exc:
                logger.error(
                    "Alert delivery failed on channel '%s': %s", channel, exc
                )

    @staticmethod
    def _deliver_log(alert: Alert) -> None:
        """Deliver alert via logging."""
        level_map = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }
        log_fn = level_map.get(alert.severity, logger.warning)
        log_fn(
            "ALERT [%s] %s: %s (model=%s, value=%.4f, threshold=%.4f)",
            alert.severity.value,
            alert.rule_name,
            alert.message,
            alert.model_name,
            alert.current_value,
            alert.threshold,
        )

    def _deliver_webhook(self, alert: Alert) -> None:
        """Deliver alert via webhook."""
        if not self.config.webhook_url:
            return

        payload = {
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "rule_name": alert.rule_name,
            "message": alert.message,
            "model_name": alert.model_name,
            "timestamp": alert.timestamp,
            "value": alert.current_value,
            "threshold": alert.threshold,
        }

        try:
            with httpx.Client(timeout=self.config.webhook_timeout) as client:
                response = client.post(self.config.webhook_url, json=payload)
                if response.status_code >= 400:
                    logger.error(
                        "Webhook delivery failed: HTTP %d", response.status_code
                    )
                else:
                    logger.debug("Webhook delivered for alert %s", alert.alert_id)
        except Exception as exc:
            logger.error("Webhook request failed: %s", exc)

    def acknowledge(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            True if alert was found and acknowledged.
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info("Alert %s acknowledged.", alert_id)
                return True
        return False

    def get_alerts(
        self,
        model_name: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> List[Alert]:
        """
        Retrieve alerts with optional filtering.

        Args:
            model_name: Filter by model name.
            severity: Filter by severity level.
            unacknowledged_only: Only return unacknowledged alerts.
            limit: Maximum alerts to return.

        Returns:
            List of matching alerts (most recent first).
        """
        filtered = list(reversed(self.alerts))

        if model_name:
            filtered = [a for a in filtered if a.model_name == model_name]

        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        if unacknowledged_only:
            filtered = [a for a in filtered if not a.acknowledged]

        return filtered[:limit]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get a summary of alert statistics."""
        total = len(self.alerts)
        by_severity: Dict[str, int] = {}
        unacknowledged = 0

        for alert in self.alerts:
            sev = alert.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            if not alert.acknowledged:
                unacknowledged += 1

        return {
            "total_alerts": total,
            "unacknowledged": unacknowledged,
            "by_severity": by_severity,
            "hourly_count": self._hourly_count,
            "hourly_limit": self.config.max_alerts_per_hour,
        }

    def clear_history(self) -> None:
        """Clear all alert history."""
        self.alerts.clear()
        self._last_alert_times.clear()
        self._hourly_count = 0
        logger.info("Alert history cleared.")
