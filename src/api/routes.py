"""
API route definitions for the monitoring service.

Provides endpoints for health checks, monitoring execution,
report retrieval, alert management, and Prometheus metrics export.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    AlertListResponse,
    AlertResponse,
    ErrorResponse,
    HealthResponse,
    MonitoringReportResponse,
    MonitorRunRequest,
    MonitorStatusResponse,
    ReportListResponse,
)
from src.config.settings import AppSettings
from src.monitors.model_monitor import ModelMonitor
from src.utils.logger import get_logger

logger = get_logger("api.routes")

router = APIRouter()

# In-memory monitor registry for simplicity
_monitors: Dict[str, ModelMonitor] = {}
_settings = AppSettings()
_alerts: list = []


def _get_or_create_monitor(
    model_name: str, task_type: str = "classification"
) -> ModelMonitor:
    """Get existing monitor or create a new one."""
    if model_name not in _monitors:
        _monitors[model_name] = ModelMonitor(
            model_name=model_name,
            settings=_settings,
            task_type=task_type,
        )
    return _monitors[model_name]


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Service health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=_settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/monitor/run",
    response_model=MonitoringReportResponse,
    tags=["monitoring"],
    summary="Run monitoring check",
    responses={400: {"model": ErrorResponse}},
)
async def run_monitor(request: MonitorRunRequest) -> MonitoringReportResponse:
    """
    Execute a monitoring check for the specified model.

    Runs data drift, concept drift, and performance drift detection
    using the provided reference and current data.
    """
    try:
        monitor = _get_or_create_monitor(
            request.model_name, request.task_type
        )

        ref_features = np.array(request.reference_features)
        ref_predictions = np.array(request.reference_predictions)
        cur_features = np.array(request.current_features)
        cur_predictions = np.array(request.current_predictions)

        ref_labels = (
            np.array(request.reference_labels)
            if request.reference_labels is not None
            else None
        )
        cur_labels = (
            np.array(request.current_labels)
            if request.current_labels is not None
            else None
        )

        monitor.set_reference(
            features=ref_features,
            predictions=ref_predictions,
            labels=ref_labels,
            feature_names=request.feature_names,
        )

        report = monitor.run_check(
            current_features=cur_features,
            current_predictions=cur_predictions,
            current_labels=cur_labels,
        )

        for alert_msg in report.alerts_triggered:
            _alerts.append(
                {
                    "alert_id": f"alert_{len(_alerts)+1}",
                    "timestamp": report.timestamp,
                    "severity": "warning",
                    "rule_name": "drift_detected",
                    "message": alert_msg,
                    "model_name": request.model_name,
                    "acknowledged": False,
                    "metadata": {},
                }
            )

        return MonitoringReportResponse(
            report_id=report.report_id,
            model_name=report.model_name,
            timestamp=report.timestamp,
            status=report.status,
            overall_health=report.overall_health,
            data_drift=report.data_drift,
            concept_drift=report.concept_drift,
            performance_drift=report.performance_drift,
            alerts_triggered=report.alerts_triggered,
            metadata=report.metadata,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Monitor run failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/monitor/status",
    response_model=MonitorStatusResponse,
    tags=["monitoring"],
    summary="Get monitor status",
)
async def get_monitor_status(
    model_name: str = Query(default="default", description="Model name"),
) -> MonitorStatusResponse:
    """Get the current status of a model monitor."""
    if model_name in _monitors:
        status = _monitors[model_name].get_status()
        return MonitorStatusResponse(**status)

    return MonitorStatusResponse(
        model_name=model_name,
        status="idle",
        check_count=0,
        last_check_time=None,
        has_reference=False,
        reference_samples=0,
        stored_reports=0,
        task_type="classification",
    )


@router.get(
    "/reports",
    response_model=ReportListResponse,
    tags=["reports"],
    summary="List monitoring reports",
)
async def list_reports(
    model_name: str = Query(default="default", description="Model name"),
    limit: int = Query(default=10, ge=1, le=100, description="Max reports"),
    offset: int = Query(default=0, ge=0, description="Offset"),
) -> ReportListResponse:
    """Retrieve paginated list of monitoring reports."""
    if model_name not in _monitors:
        return ReportListResponse(total=0, reports=[])

    monitor = _monitors[model_name]
    reports = monitor.get_reports(limit=limit, offset=offset)

    report_responses = []
    for r in reports:
        report_responses.append(
            MonitoringReportResponse(
                report_id=r.report_id,
                model_name=r.model_name,
                timestamp=r.timestamp,
                status=r.status,
                overall_health=r.overall_health,
                data_drift=r.data_drift,
                concept_drift=r.concept_drift,
                performance_drift=r.performance_drift,
                alerts_triggered=r.alerts_triggered,
                metadata=r.metadata,
            )
        )

    return ReportListResponse(
        total=len(monitor._reports),
        reports=report_responses,
    )


@router.get(
    "/reports/{report_id}",
    response_model=MonitoringReportResponse,
    tags=["reports"],
    summary="Get report by ID",
    responses={404: {"model": ErrorResponse}},
)
async def get_report(report_id: str) -> MonitoringReportResponse:
    """Retrieve a specific monitoring report by ID."""
    for monitor in _monitors.values():
        report = monitor.get_report_by_id(report_id)
        if report:
            return MonitoringReportResponse(
                report_id=report.report_id,
                model_name=report.model_name,
                timestamp=report.timestamp,
                status=report.status,
                overall_health=report.overall_health,
                data_drift=report.data_drift,
                concept_drift=report.concept_drift,
                performance_drift=report.performance_drift,
                alerts_triggered=report.alerts_triggered,
                metadata=report.metadata,
            )

    raise HTTPException(status_code=404, detail="Report not found.")


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    tags=["alerts"],
    summary="List alerts",
)
async def list_alerts(
    model_name: Optional[str] = Query(default=None, description="Filter by model"),
    limit: int = Query(default=50, ge=1, le=200, description="Max alerts"),
) -> AlertListResponse:
    """Retrieve recent alerts with optional model filter."""
    filtered = _alerts
    if model_name:
        filtered = [a for a in _alerts if a["model_name"] == model_name]

    recent = list(reversed(filtered))[:limit]
    return AlertListResponse(
        total=len(filtered),
        alerts=[AlertResponse(**a) for a in recent],
    )


@router.get(
    "/metrics/prometheus",
    tags=["metrics"],
    summary="Prometheus metrics",
)
async def prometheus_metrics():
    """
    Export metrics in Prometheus text exposition format.

    Returns metrics for all monitored models including drift scores,
    performance metrics, and alert counts.
    """
    from fastapi.responses import PlainTextResponse

    lines: list = []
    lines.append("# HELP ml_drift_data_drift_score Data drift score")
    lines.append("# TYPE ml_drift_data_drift_score gauge")
    lines.append("# HELP ml_drift_concept_drift_score Concept drift score")
    lines.append("# TYPE ml_drift_concept_drift_score gauge")
    lines.append("# HELP ml_drift_performance_score Performance degradation score")
    lines.append("# TYPE ml_drift_performance_score gauge")
    lines.append("# HELP ml_drift_alert_count Total alerts triggered")
    lines.append("# TYPE ml_drift_alert_count counter")
    lines.append("# HELP ml_drift_check_count Total monitoring checks")
    lines.append("# TYPE ml_drift_check_count counter")

    for model_name, monitor in _monitors.items():
        reports = monitor.get_reports(limit=1)
        if reports:
            latest = reports[0]
            label = f'model="{model_name}"'

            if latest.data_drift:
                score = latest.data_drift.get("drift_score", 0.0)
                lines.append(f"ml_drift_data_drift_score{{{label}}} {score}")

            if latest.concept_drift:
                score = latest.concept_drift.get("drift_score", 0.0)
                lines.append(f"ml_drift_concept_drift_score{{{label}}} {score}")

            if latest.performance_drift:
                score = latest.performance_drift.get("overall_score", 0.0)
                lines.append(f"ml_drift_performance_score{{{label}}} {score}")

            lines.append(
                f"ml_drift_alert_count{{{label}}} {len(latest.alerts_triggered)}"
            )

        status = monitor.get_status()
        label = f'model="{model_name}"'
        lines.append(f'ml_drift_check_count{{{label}}} {status["check_count"]}')

    return PlainTextResponse(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
