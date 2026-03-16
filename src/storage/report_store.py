"""
Report storage module.

Provides persistent storage for monitoring reports with PostgreSQL
as the primary backend and JSON file fallback for environments
without database access.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import StorageConfig
from src.utils.logger import get_logger

logger = get_logger("storage.report_store")


class ReportStore:
    """
    Persistent storage for monitoring reports.

    Attempts to use PostgreSQL via psycopg2. If unavailable, falls back
    to JSON file storage in the configured directory.

    Attributes:
        config: Storage configuration.
        _use_postgres: Whether PostgreSQL is available.
        _conn: PostgreSQL connection (if available).
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize the report store.

        Args:
            config: Storage configuration.
        """
        self.config = config or StorageConfig()
        self._use_postgres = False
        self._conn = None
        self._json_dir = Path(self.config.fallback_json_dir)

        self._try_postgres()

        if not self._use_postgres:
            self._json_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Using JSON file storage at %s", self._json_dir
            )

    def _try_postgres(self) -> None:
        """Attempt to connect to PostgreSQL and create tables."""
        try:
            import psycopg2

            self._conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                dbname=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                connect_timeout=5,
            )
            self._conn.autocommit = True

            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_reports (
                        report_id VARCHAR(64) PRIMARY KEY,
                        model_name VARCHAR(256) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        overall_health VARCHAR(32),
                        report_data JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_reports_model
                        ON monitoring_reports(model_name, timestamp DESC);
                    """
                )

            self._use_postgres = True
            logger.info(
                "Connected to PostgreSQL at %s:%d/%s",
                self.config.postgres_host,
                self.config.postgres_port,
                self.config.postgres_db,
            )
        except Exception as exc:
            logger.warning(
                "PostgreSQL not available (%s). Falling back to JSON storage.",
                exc,
            )
            self._use_postgres = False
            self._conn = None

    def save_report(self, report: Dict[str, Any]) -> str:
        """
        Save a monitoring report.

        Args:
            report: Report dictionary (must contain report_id, model_name,
                    timestamp, status).

        Returns:
            The report ID.
        """
        report_id = report.get("report_id", str(uuid.uuid4()))

        if self._use_postgres:
            return self._save_postgres(report_id, report)
        return self._save_json(report_id, report)

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a report by ID.

        Args:
            report_id: The report identifier.

        Returns:
            Report dictionary or None if not found.
        """
        if self._use_postgres:
            return self._get_postgres(report_id)
        return self._get_json(report_id)

    def list_reports(
        self,
        model_name: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List reports with optional filtering and pagination.

        Args:
            model_name: Filter by model name (optional).
            limit: Maximum reports to return.
            offset: Number of reports to skip.

        Returns:
            List of report dictionaries.
        """
        if self._use_postgres:
            return self._list_postgres(model_name, limit, offset)
        return self._list_json(model_name, limit, offset)

    def delete_report(self, report_id: str) -> bool:
        """
        Delete a report by ID.

        Args:
            report_id: The report identifier.

        Returns:
            True if deleted, False if not found.
        """
        if self._use_postgres:
            return self._delete_postgres(report_id)
        return self._delete_json(report_id)

    # --- PostgreSQL backend ---

    def _save_postgres(
        self, report_id: str, report: Dict[str, Any]
    ) -> str:
        """Save report to PostgreSQL."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO monitoring_reports
                        (report_id, model_name, timestamp, status,
                         overall_health, report_data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (report_id) DO UPDATE SET
                        report_data = EXCLUDED.report_data,
                        status = EXCLUDED.status,
                        overall_health = EXCLUDED.overall_health
                    """,
                    (
                        report_id,
                        report.get("model_name", "unknown"),
                        report.get("timestamp", datetime.now(timezone.utc).isoformat()),
                        report.get("status", "completed"),
                        report.get("overall_health", "unknown"),
                        json.dumps(report),
                    ),
                )
            logger.debug("Report %s saved to PostgreSQL.", report_id)
        except Exception as exc:
            logger.error("Failed to save report to PostgreSQL: %s", exc)
            return self._save_json(report_id, report)
        return report_id

    def _get_postgres(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report from PostgreSQL."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT report_data FROM monitoring_reports WHERE report_id = %s",
                    (report_id,),
                )
                row = cur.fetchone()
                if row:
                    return row[0] if isinstance(row[0], dict) else json.loads(row[0])
        except Exception as exc:
            logger.error("Failed to get report from PostgreSQL: %s", exc)
        return None

    def _list_postgres(
        self,
        model_name: Optional[str],
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        """List reports from PostgreSQL."""
        try:
            with self._conn.cursor() as cur:
                if model_name:
                    cur.execute(
                        """
                        SELECT report_data FROM monitoring_reports
                        WHERE model_name = %s
                        ORDER BY timestamp DESC LIMIT %s OFFSET %s
                        """,
                        (model_name, limit, offset),
                    )
                else:
                    cur.execute(
                        """
                        SELECT report_data FROM monitoring_reports
                        ORDER BY timestamp DESC LIMIT %s OFFSET %s
                        """,
                        (limit, offset),
                    )
                rows = cur.fetchall()
                return [
                    r[0] if isinstance(r[0], dict) else json.loads(r[0])
                    for r in rows
                ]
        except Exception as exc:
            logger.error("Failed to list reports from PostgreSQL: %s", exc)
        return []

    def _delete_postgres(self, report_id: str) -> bool:
        """Delete report from PostgreSQL."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM monitoring_reports WHERE report_id = %s",
                    (report_id,),
                )
                return cur.rowcount > 0
        except Exception as exc:
            logger.error("Failed to delete report from PostgreSQL: %s", exc)
        return False

    # --- JSON file backend ---

    def _save_json(
        self, report_id: str, report: Dict[str, Any]
    ) -> str:
        """Save report as a JSON file."""
        filepath = self._json_dir / f"{report_id}.json"
        try:
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=str)
            logger.debug("Report %s saved to %s", report_id, filepath)
        except Exception as exc:
            logger.error("Failed to save report to JSON: %s", exc)
        return report_id

    def _get_json(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report from JSON file."""
        filepath = self._json_dir / f"{report_id}.json"
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error("Failed to read report JSON: %s", exc)
        return None

    def _list_json(
        self,
        model_name: Optional[str],
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        """List reports from JSON directory."""
        files = sorted(
            self._json_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        reports: List[Dict[str, Any]] = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if model_name and data.get("model_name") != model_name:
                    continue
                reports.append(data)
            except Exception:
                continue

        return reports[offset : offset + limit]

    def _delete_json(self, report_id: str) -> bool:
        """Delete a JSON report file."""
        filepath = self._json_dir / f"{report_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def close(self) -> None:
        """Close database connections."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
