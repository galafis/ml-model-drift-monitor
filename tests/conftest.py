"""
Shared test fixtures for ML Model Drift Monitor tests.
"""

import numpy as np
import pytest

from src.config.settings import AppSettings, DriftConfig, MonitoringConfig


@pytest.fixture
def drift_config() -> DriftConfig:
    """Provide default drift configuration for tests."""
    return DriftConfig()


@pytest.fixture
def monitoring_config() -> MonitoringConfig:
    """Provide default monitoring configuration for tests."""
    return MonitoringConfig()


@pytest.fixture
def app_settings() -> AppSettings:
    """Provide default application settings for tests."""
    return AppSettings()


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def reference_data(rng: np.random.Generator):
    """Generate reference dataset for drift detection tests."""
    n_samples = 500
    n_features = 5
    features = rng.standard_normal((n_samples, n_features))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    predictions = labels.copy()
    noise_idx = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    predictions[noise_idx] = 1 - predictions[noise_idx]
    return features, predictions, labels


@pytest.fixture
def drifted_data(rng: np.random.Generator):
    """Generate drifted dataset for drift detection tests."""
    n_samples = 500
    n_features = 5
    features = rng.standard_normal((n_samples, n_features))
    features[:, 0] += 2.0
    features[:, 1] *= 3.0
    features[:, 2] += 1.5
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    predictions = labels.copy()
    noise_idx = rng.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    predictions[noise_idx] = 1 - predictions[noise_idx]
    return features, predictions, labels


@pytest.fixture
def no_drift_data(rng: np.random.Generator):
    """Generate non-drifted dataset (same distribution as reference)."""
    n_samples = 500
    n_features = 5
    features = rng.standard_normal((n_samples, n_features))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    predictions = labels.copy()
    noise_idx = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    predictions[noise_idx] = 1 - predictions[noise_idx]
    return features, predictions, labels


@pytest.fixture
def feature_names() -> list:
    """Provide feature names for tests."""
    return ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
