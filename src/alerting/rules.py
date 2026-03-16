"""
Alert rule definitions and rule engine.

Provides configurable alert rules with threshold-based, trend-based,
and anomaly-based detection strategies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("alerting.rules")


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleType(str, Enum):
    """Types of alert rules."""

    THRESHOLD = "threshold"
    TREND = "trend"
    ANOMALY = "anomaly"
    COMPOSITE = "composite"


@dataclass
class AlertRule:
    """
    Definition of an alert rule.

    Attributes:
        name: Unique rule name.
        description: Human-readable description.
        rule_type: Type of rule (threshold, trend, anomaly, composite).
        severity: Alert severity when triggered.
        metric_name: Name of the metric to evaluate.
        threshold: Threshold value for threshold rules.
        direction: Comparison direction ("above" or "below").
        window_size: Number of data points for trend/anomaly rules.
        enabled: Whether the rule is active.
        cooldown_seconds: Minimum time between triggers.
        metadata: Additional rule configuration.
    """

    name: str
    description: str
    rule_type: RuleType
    severity: AlertSeverity
    metric_name: str
    threshold: float = 0.0
    direction: str = "above"
    window_size: int = 5
    enabled: bool = True
    cooldown_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleEvaluation:
    """Result of evaluating a single rule."""

    rule_name: str
    triggered: bool
    severity: AlertSeverity
    current_value: float
    threshold: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleEngine:
    """
    Evaluates alert rules against monitoring metrics.

    Supports threshold rules (metric above/below threshold),
    trend rules (sustained directional change), and anomaly rules
    (sudden spikes or drops beyond expected range).

    Attributes:
        rules: List of configured alert rules.
    """

    def __init__(self, rules: Optional[List[AlertRule]] = None):
        """
        Initialize the rule engine.

        Args:
            rules: List of alert rules. Defaults to standard rules.
        """
        self.rules = rules if rules is not None else self._default_rules()
        self._metric_history: Dict[str, List[float]] = {}
        logger.info("RuleEngine initialized with %d rules.", len(self.rules))

    def evaluate(
        self, metrics: Dict[str, float]
    ) -> List[RuleEvaluation]:
        """
        Evaluate all enabled rules against the provided metrics.

        Args:
            metrics: Dictionary of metric_name -> value.

        Returns:
            List of RuleEvaluation results for triggered rules.
        """
        for name, value in metrics.items():
            if name not in self._metric_history:
                self._metric_history[name] = []
            self._metric_history[name].append(value)
            if len(self._metric_history[name]) > 100:
                self._metric_history[name] = self._metric_history[name][-100:]

        results: List[RuleEvaluation] = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            if rule.metric_name not in metrics:
                continue

            value = metrics[rule.metric_name]

            if rule.rule_type == RuleType.THRESHOLD:
                evaluation = self._evaluate_threshold(rule, value)
            elif rule.rule_type == RuleType.TREND:
                history = self._metric_history.get(rule.metric_name, [])
                evaluation = self._evaluate_trend(rule, value, history)
            elif rule.rule_type == RuleType.ANOMALY:
                history = self._metric_history.get(rule.metric_name, [])
                evaluation = self._evaluate_anomaly(rule, value, history)
            else:
                continue

            if evaluation.triggered:
                results.append(evaluation)

        return results

    @staticmethod
    def _evaluate_threshold(
        rule: AlertRule, value: float
    ) -> RuleEvaluation:
        """Evaluate a threshold-based rule."""
        if rule.direction == "above":
            triggered = value > rule.threshold
        else:
            triggered = value < rule.threshold

        message = ""
        if triggered:
            message = (
                f"[{rule.severity.value.upper()}] {rule.name}: "
                f"{rule.metric_name}={value:.4f} is {rule.direction} "
                f"threshold {rule.threshold:.4f}"
            )

        return RuleEvaluation(
            rule_name=rule.name,
            triggered=triggered,
            severity=rule.severity,
            current_value=value,
            threshold=rule.threshold,
            message=message,
        )

    @staticmethod
    def _evaluate_trend(
        rule: AlertRule, value: float, history: List[float]
    ) -> RuleEvaluation:
        """
        Evaluate a trend-based rule.

        Checks if the metric has been consistently moving in the
        specified direction over the window.
        """
        window = rule.window_size
        if len(history) < window:
            return RuleEvaluation(
                rule_name=rule.name,
                triggered=False,
                severity=rule.severity,
                current_value=value,
                threshold=rule.threshold,
                message="",
            )

        recent = history[-window:]

        if rule.direction == "above":
            consecutive_increases = sum(
                1 for i in range(1, len(recent)) if recent[i] > recent[i - 1]
            )
            triggered = (
                consecutive_increases >= window - 2
                and value > rule.threshold
            )
        else:
            consecutive_decreases = sum(
                1 for i in range(1, len(recent)) if recent[i] < recent[i - 1]
            )
            triggered = (
                consecutive_decreases >= window - 2
                and value < rule.threshold
            )

        message = ""
        if triggered:
            message = (
                f"[{rule.severity.value.upper()}] {rule.name}: "
                f"{rule.metric_name} shows sustained trend "
                f"(current={value:.4f}, threshold={rule.threshold:.4f}, "
                f"window={window})"
            )

        return RuleEvaluation(
            rule_name=rule.name,
            triggered=triggered,
            severity=rule.severity,
            current_value=value,
            threshold=rule.threshold,
            message=message,
            metadata={"window_values": recent},
        )

    @staticmethod
    def _evaluate_anomaly(
        rule: AlertRule, value: float, history: List[float]
    ) -> RuleEvaluation:
        """
        Evaluate an anomaly-based rule.

        Detects sudden spikes or drops by comparing the current value
        against the mean +/- N standard deviations of the recent history.
        """
        import numpy as np

        window = rule.window_size
        if len(history) < window + 1:
            return RuleEvaluation(
                rule_name=rule.name,
                triggered=False,
                severity=rule.severity,
                current_value=value,
                threshold=rule.threshold,
                message="",
            )

        baseline = history[-(window + 1) : -1]
        mean = float(np.mean(baseline))
        std = float(np.std(baseline))

        n_sigma = rule.metadata.get("n_sigma", 3.0)

        if std > 1e-10:
            z_score = abs(value - mean) / std
            triggered = z_score > n_sigma
        else:
            triggered = abs(value - mean) > rule.threshold

        message = ""
        if triggered:
            message = (
                f"[{rule.severity.value.upper()}] {rule.name}: "
                f"{rule.metric_name}={value:.4f} is anomalous "
                f"(mean={mean:.4f}, std={std:.4f}, z={z_score:.2f})"
            )

        return RuleEvaluation(
            rule_name=rule.name,
            triggered=triggered,
            severity=rule.severity,
            current_value=value,
            threshold=rule.threshold,
            message=message,
            metadata={"mean": mean, "std": std},
        )

    @staticmethod
    def _default_rules() -> List[AlertRule]:
        """Create default alert rules for standard monitoring."""
        return [
            AlertRule(
                name="high_data_drift",
                description="Data drift score exceeds critical threshold",
                rule_type=RuleType.THRESHOLD,
                severity=AlertSeverity.ERROR,
                metric_name="data_drift_score",
                threshold=0.3,
                direction="above",
            ),
            AlertRule(
                name="moderate_data_drift",
                description="Data drift score exceeds warning threshold",
                rule_type=RuleType.THRESHOLD,
                severity=AlertSeverity.WARNING,
                metric_name="data_drift_score",
                threshold=0.15,
                direction="above",
            ),
            AlertRule(
                name="concept_drift_detected",
                description="Concept drift score exceeds threshold",
                rule_type=RuleType.THRESHOLD,
                severity=AlertSeverity.ERROR,
                metric_name="concept_drift_score",
                threshold=0.1,
                direction="above",
            ),
            AlertRule(
                name="performance_degradation",
                description="Performance degradation score exceeds threshold",
                rule_type=RuleType.THRESHOLD,
                severity=AlertSeverity.CRITICAL,
                metric_name="performance_score",
                threshold=0.2,
                direction="above",
            ),
            AlertRule(
                name="accuracy_drop",
                description="Model accuracy dropped below acceptable level",
                rule_type=RuleType.THRESHOLD,
                severity=AlertSeverity.ERROR,
                metric_name="accuracy",
                threshold=0.8,
                direction="below",
            ),
            AlertRule(
                name="drift_trend",
                description="Sustained increase in drift score over recent windows",
                rule_type=RuleType.TREND,
                severity=AlertSeverity.WARNING,
                metric_name="data_drift_score",
                threshold=0.1,
                direction="above",
                window_size=5,
            ),
            AlertRule(
                name="drift_anomaly",
                description="Anomalous spike in drift score",
                rule_type=RuleType.ANOMALY,
                severity=AlertSeverity.ERROR,
                metric_name="data_drift_score",
                threshold=0.5,
                window_size=10,
                metadata={"n_sigma": 3.0},
            ),
        ]

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new rule to the engine."""
        self.rules.append(rule)
        logger.info("Rule '%s' added.", rule.name)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found."""
        initial = len(self.rules)
        self.rules = [r for r in self.rules if r.name != name]
        removed = len(self.rules) < initial
        if removed:
            logger.info("Rule '%s' removed.", name)
        return removed

    def reset_history(self) -> None:
        """Clear all metric history."""
        self._metric_history.clear()
