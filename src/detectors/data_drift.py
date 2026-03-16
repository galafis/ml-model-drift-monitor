"""
Data drift detection module.

Implements statistical tests for detecting distribution shifts in input features:
- Kolmogorov-Smirnov (KS) test for continuous features
- Population Stability Index (PSI) for distribution stability
- Jensen-Shannon divergence for probability distribution comparison
- Feature-level drift analysis with aggregated reporting

References:
    - Kolmogorov-Smirnov: scipy.stats.ks_2samp
    - PSI: Siddiqi (2006), "Credit Risk Scorecards"
    - Jensen-Shannon: Lin (1991), "Divergence measures based on the Shannon entropy"
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from src.config.settings import DriftConfig
from src.utils.logger import get_logger

logger = get_logger("data_drift")


class DriftSeverity(str, Enum):
    """Severity levels for drift detection results."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureDriftResult:
    """Drift analysis result for a single feature."""

    feature_name: str
    is_drifted: bool
    ks_statistic: float
    ks_p_value: float
    psi_value: float
    js_divergence: float
    severity: DriftSeverity
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftResult:
    """Aggregated drift detection result across all features."""

    timestamp: str
    is_drifted: bool
    drift_score: float
    drifted_feature_count: int
    total_feature_count: int
    drifted_feature_fraction: float
    severity: DriftSeverity
    feature_results: List[FeatureDriftResult]
    method_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataDriftDetector:
    """
    Detects data drift between reference and current data distributions.

    Uses multiple statistical tests to provide robust drift detection
    across numerical features. Each feature is tested independently,
    and results are aggregated to determine dataset-level drift.

    Attributes:
        config: Drift detection configuration with thresholds.
        n_bins: Number of bins for histogram-based methods (PSI, JS).
    """

    def __init__(self, config: Optional[DriftConfig] = None, n_bins: int = 20):
        """
        Initialize the data drift detector.

        Args:
            config: Drift configuration. Uses defaults if not provided.
            n_bins: Number of bins for histogram-based computations.
        """
        self.config = config or DriftConfig()
        self.n_bins = n_bins
        logger.info(
            "DataDriftDetector initialized (KS threshold=%.3f, PSI threshold=%.3f, "
            "JS threshold=%.3f)",
            self.config.ks_test_threshold,
            self.config.psi_threshold,
            self.config.js_divergence_threshold,
        )

    def detect(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DriftResult:
        """
        Run drift detection on all features.

        Args:
            reference_data: Reference (training) data, shape (n_samples, n_features).
            current_data: Current (production) data, shape (n_samples, n_features).
            feature_names: Optional list of feature names.

        Returns:
            DriftResult with per-feature and aggregated results.

        Raises:
            ValueError: If input arrays have incompatible shapes.
        """
        reference_data = np.atleast_2d(reference_data)
        current_data = np.atleast_2d(current_data)

        if reference_data.shape[1] != current_data.shape[1]:
            raise ValueError(
                f"Feature count mismatch: reference has {reference_data.shape[1]}, "
                f"current has {current_data.shape[1]}."
            )

        n_features = reference_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if len(feature_names) != n_features:
            raise ValueError(
                f"Feature names count ({len(feature_names)}) does not match "
                f"data columns ({n_features})."
            )

        feature_results: List[FeatureDriftResult] = []
        ks_scores: List[float] = []
        psi_scores: List[float] = []
        js_scores: List[float] = []

        for i in range(n_features):
            ref_col = reference_data[:, i]
            cur_col = current_data[:, i]

            result = self._analyze_feature(feature_names[i], ref_col, cur_col)
            feature_results.append(result)

            ks_scores.append(result.ks_statistic)
            psi_scores.append(result.psi_value)
            js_scores.append(result.js_divergence)

        drifted_count = sum(1 for r in feature_results if r.is_drifted)
        drifted_fraction = drifted_count / n_features if n_features > 0 else 0.0

        dataset_drifted = (
            drifted_fraction >= self.config.feature_drift_fraction_threshold
        )

        drift_score = np.mean(js_scores) if js_scores else 0.0

        severity = self._compute_severity(drift_score, drifted_fraction)

        method_scores = {
            "ks_mean_statistic": float(np.mean(ks_scores)) if ks_scores else 0.0,
            "psi_mean": float(np.mean(psi_scores)) if psi_scores else 0.0,
            "js_mean": float(np.mean(js_scores)) if js_scores else 0.0,
        }

        logger.info(
            "Drift detection complete: drifted=%s, score=%.4f, "
            "drifted_features=%d/%d (%.1f%%)",
            dataset_drifted,
            drift_score,
            drifted_count,
            n_features,
            drifted_fraction * 100,
        )

        return DriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_drifted=dataset_drifted,
            drift_score=float(drift_score),
            drifted_feature_count=drifted_count,
            total_feature_count=n_features,
            drifted_feature_fraction=float(drifted_fraction),
            severity=severity,
            feature_results=feature_results,
            method_scores=method_scores,
            metadata={
                "reference_samples": reference_data.shape[0],
                "current_samples": current_data.shape[0],
                "n_bins": self.n_bins,
            },
        )

    def _analyze_feature(
        self, name: str, reference: np.ndarray, current: np.ndarray
    ) -> FeatureDriftResult:
        """
        Analyze drift for a single feature using all statistical tests.

        Args:
            name: Feature name.
            reference: Reference distribution samples.
            current: Current distribution samples.

        Returns:
            FeatureDriftResult with all test statistics.
        """
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]

        if len(ref_clean) < 2 or len(cur_clean) < 2:
            logger.warning("Feature '%s' has insufficient non-NaN samples.", name)
            return FeatureDriftResult(
                feature_name=name,
                is_drifted=False,
                ks_statistic=0.0,
                ks_p_value=1.0,
                psi_value=0.0,
                js_divergence=0.0,
                severity=DriftSeverity.NONE,
                details={"error": "insufficient_samples"},
            )

        ks_stat, ks_p = self._ks_test(ref_clean, cur_clean)
        psi = self._compute_psi(ref_clean, cur_clean)
        js_div = self._compute_js_divergence(ref_clean, cur_clean)

        ks_drifted = ks_p < self.config.ks_test_threshold
        psi_drifted = psi > self.config.psi_threshold
        js_drifted = js_div > self.config.js_divergence_threshold

        vote_count = sum([ks_drifted, psi_drifted, js_drifted])
        is_drifted = vote_count >= 2

        severity = self._feature_severity(js_div, psi)

        return FeatureDriftResult(
            feature_name=name,
            is_drifted=is_drifted,
            ks_statistic=float(ks_stat),
            ks_p_value=float(ks_p),
            psi_value=float(psi),
            js_divergence=float(js_div),
            severity=severity,
            details={
                "ks_drifted": ks_drifted,
                "psi_drifted": psi_drifted,
                "js_drifted": js_drifted,
                "vote_count": vote_count,
                "ref_mean": float(np.mean(ref_clean)),
                "cur_mean": float(np.mean(cur_clean)),
                "ref_std": float(np.std(ref_clean)),
                "cur_std": float(np.std(cur_clean)),
            },
        )

    @staticmethod
    def _ks_test(
        reference: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform the two-sample Kolmogorov-Smirnov test.

        Tests the null hypothesis that both samples are drawn from the
        same continuous distribution.

        Args:
            reference: Reference samples.
            current: Current samples.

        Returns:
            Tuple of (KS statistic, p-value).
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)

    def _compute_psi(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """
        Compute Population Stability Index (PSI).

        PSI measures the shift between two distributions by comparing
        the proportion of observations in each bin.

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change

        Args:
            reference: Reference samples.
            current: Current samples.

        Returns:
            PSI value (non-negative).
        """
        combined = np.concatenate([reference, current])
        min_val = np.min(combined)
        max_val = np.max(combined)

        if min_val == max_val:
            return 0.0

        edges = np.linspace(min_val, max_val, self.n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)

        eps = 1e-6
        ref_pct = (ref_counts + eps) / (len(reference) + eps * self.n_bins)
        cur_pct = (cur_counts + eps) / (len(current) + eps * self.n_bins)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(max(psi, 0.0))

    def _compute_js_divergence(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.

        The JS divergence is a symmetric, bounded (0 to ln(2) for base-e)
        measure derived from KL divergence. Uses histogram-based density
        estimation.

        Args:
            reference: Reference samples.
            current: Current samples.

        Returns:
            JS divergence value in [0, 1] range (using base-2 log).
        """
        combined = np.concatenate([reference, current])
        min_val = np.min(combined)
        max_val = np.max(combined)

        if min_val == max_val:
            return 0.0

        edges = np.linspace(min_val, max_val, self.n_bins + 1)

        ref_hist, _ = np.histogram(reference, bins=edges, density=False)
        cur_hist, _ = np.histogram(current, bins=edges, density=False)

        eps = 1e-10
        ref_prob = (ref_hist + eps) / (ref_hist.sum() + eps * self.n_bins)
        cur_prob = (cur_hist + eps) / (cur_hist.sum() + eps * self.n_bins)

        ref_prob = ref_prob / ref_prob.sum()
        cur_prob = cur_prob / cur_prob.sum()

        js_dist = jensenshannon(ref_prob, cur_prob, base=2)
        js_div = float(js_dist ** 2)

        return js_div

    @staticmethod
    def _feature_severity(js_div: float, psi: float) -> DriftSeverity:
        """Determine severity based on JS divergence and PSI."""
        score = (js_div + psi) / 2

        if score < 0.05:
            return DriftSeverity.NONE
        elif score < 0.1:
            return DriftSeverity.LOW
        elif score < 0.2:
            return DriftSeverity.MEDIUM
        elif score < 0.4:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    @staticmethod
    def _compute_severity(
        drift_score: float, drifted_fraction: float
    ) -> DriftSeverity:
        """Compute overall severity from aggregated metrics."""
        combined = 0.6 * drift_score + 0.4 * drifted_fraction

        if combined < 0.05:
            return DriftSeverity.NONE
        elif combined < 0.1:
            return DriftSeverity.LOW
        elif combined < 0.25:
            return DriftSeverity.MEDIUM
        elif combined < 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
