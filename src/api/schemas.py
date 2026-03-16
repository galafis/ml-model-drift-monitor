"""
Pydantic v2 schemas for API request/response models.

Defines all data transfer objects used by the REST API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------- Health ----------

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service health status.")
    version: str = Field(description="Application version.")
    timestamp: str = Field(description="Current server timestamp.")


# ---------- Monitor ----------

class MonitorRunRequest(BaseModel):
    """Request to trigger a monitoring check."""

    model_name: str = Field(description="Name of the model to monitor.")
    reference_features: List[List[float]] = Field(
        description="Reference feature data (2D array)."
    )
    reference_predictions: List[float] = Field(
        description="Reference model predictions."
    )
    current_features: List[List[float]] = Field(
        description="Current feature data (2D array)."
    )
    current_predictions: List[float] = Field(
        description="Current model predictions."
    )
    reference_labels: Optional[List[float]] = Field(
        default=None, description="Reference ground truth labels."
    )
    current_labels: Optional[List[float]] = Field(
        default=None, description="Current ground truth labels."
    )
    feature_names: Optional[List[str]] = Field(
        default=None, description="Feature column names."
    )
    task_type: str = Field(
        default="classification",
        description="Task type: classification or regression.",
    )


class FeatureDriftDetail(BaseModel):
    """Drift result for a single feature."""

    feature_name: str
    is_drifted: bool
    ks_statistic: float
    ks_p_value: float
    psi_value: float
    js_divergence: float
    severity: str


class DataDriftResponse(BaseModel):
    """Data drift section of the monitoring report."""

    is_drifted: bool
    drift_score: float
    severity: str
    drifted_feature_count: int
    total_feature_count: int
    drifted_feature_fraction: float
    method_scores: Dict[str, float]
    feature_results: List[FeatureDriftDetail]


class ConceptDriftResponse(BaseModel):
    """Concept drift section of the monitoring report."""

    is_drifted: bool
    drift_type: str
    drift_score: float
    prediction_shift_score: float
    label_drift_score: float
    adwin_detected: bool
    window_means: Dict[str, float]


class TrendDetail(BaseModel):
    """Trend analysis for a single metric."""

    metric_name: str
    slope: float
    r_squared: float
    p_value: float
    is_degrading: bool
    trend_direction: str


class PerformanceDriftResponse(BaseModel):
    """Performance drift section of the monitoring report."""

    is_degraded: bool
    degradation_level: str
    overall_score: float
    current_metrics: Dict[str, float]
    reference_metrics: Dict[str, float]
    metric_changes: Dict[str, float]
    trends: List[TrendDetail]


class MonitoringReportResponse(BaseModel):
    """Full monitoring report response."""

    report_id: str
    model_name: str
    timestamp: str
    status: str
    overall_health: str
    data_drift: Optional[DataDriftResponse] = None
    concept_drift: Optional[ConceptDriftResponse] = None
    performance_drift: Optional[PerformanceDriftResponse] = None
    alerts_triggered: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------- Status ----------

class MonitorStatusResponse(BaseModel):
    """Monitor status response."""

    model_name: str
    status: str
    check_count: int
    last_check_time: Optional[str] = None
    has_reference: bool
    reference_samples: int
    stored_reports: int
    task_type: str


# ---------- Reports ----------

class ReportListResponse(BaseModel):
    """Paginated list of monitoring reports."""

    total: int
    reports: List[MonitoringReportResponse]


# ---------- Alerts ----------

class AlertSeverityEnum(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertResponse(BaseModel):
    """Single alert response."""

    alert_id: str
    timestamp: str
    severity: str
    rule_name: str
    message: str
    model_name: str
    acknowledged: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertListResponse(BaseModel):
    """List of alerts."""

    total: int
    alerts: List[AlertResponse]


# ---------- Prometheus ----------

class PrometheusMetricsResponse(BaseModel):
    """Prometheus metrics in text format."""

    content_type: str = "text/plain; version=0.0.4; charset=utf-8"
    metrics: str


# ---------- Error ----------

class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    status_code: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
