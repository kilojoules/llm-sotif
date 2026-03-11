"""Distance metrics for measuring departure from the safe envelope.

These are the "validation metrics" in our SOTIF framework — the analog of
the absolute bias and area metrics in the error_predictor paper (Eqs. 12-13).

For each new prompt (or during generation), we compute the distance of its
SAE features from the safe baseline. If this distance exceeds the envelope
threshold, the model has left its Operational Design Domain.
"""

from __future__ import annotations

import numpy as np
from .baseline import SafeBaseline


def mahalanobis_distance(
    features: np.ndarray,
    baseline: SafeBaseline,
) -> np.ndarray:
    """Compute Mahalanobis distance from the safe baseline.

    Args:
        features: shape (n_samples, n_all_features) or (n_all_features,)
        baseline: fitted safe baseline

    Returns:
        distances: shape (n_samples,) or scalar
    """
    if features.ndim == 1:
        features = features[np.newaxis, :]

    # Apply feature mask
    if baseline.feature_mask is not None:
        features = features[:, baseline.feature_mask]

    assert baseline.precision is not None, "Mahalanobis requires precision matrix"

    diff = features - baseline.mean
    distances = np.sqrt(np.sum(diff @ baseline.precision * diff, axis=1))
    return distances.squeeze()


def isolation_score(
    features: np.ndarray,
    baseline: SafeBaseline,
) -> np.ndarray:
    """Compute isolation forest anomaly score."""
    if features.ndim == 1:
        features = features[np.newaxis, :]

    if baseline.feature_mask is not None:
        features = features[:, baseline.feature_mask]

    assert baseline.detector is not None, "Isolation forest detector not fitted"
    distances = -baseline.detector.score_samples(features)
    return distances.squeeze()


def kde_score(
    features: np.ndarray,
    baseline: SafeBaseline,
) -> np.ndarray:
    """Compute KDE negative log-density score."""
    if features.ndim == 1:
        features = features[np.newaxis, :]

    if baseline.feature_mask is not None:
        features = features[:, baseline.feature_mask]

    assert baseline.detector is not None, "KDE detector not fitted"
    distances = -baseline.detector.score_samples(features)
    return distances.squeeze()


def compute_distance(
    features: np.ndarray,
    baseline: SafeBaseline,
) -> np.ndarray:
    """Compute anomaly distance using the baseline's fitted method."""
    dispatch = {
        "mahalanobis": mahalanobis_distance,
        "isolation_forest": isolation_score,
        "kde": kde_score,
    }
    fn = dispatch.get(baseline.method)
    if fn is None:
        raise ValueError(f"Unknown method: {baseline.method}")
    return fn(features, baseline)


def classify_region(
    distances: np.ndarray,
    baseline: SafeBaseline,
    safe_tolerance: float = 0.05,
    probably_safe_tolerance: float = 0.10,
) -> np.ndarray:
    """Classify each sample into trust regions.

    Returns:
        labels: array of ints
          0 = "validated" (within measurement campaigns)
          1 = "probably safe" (within safe tolerance at given confidence)
          2 = "possibly safe" (within expanded tolerance)
          3 = "dangerous prediction domain" (outside all safe regions)

    This directly mirrors Figure 2 in the error_predictor paper:
      - Green (center)  = Measurement campaigns (validated)
      - Blue            = Probably safe domain
      - Brown (outer)   = Possibly safe domain
      - Red (outermost) = Dangerous prediction domain
    """
    conf_95 = baseline.thresholds.get(0.95, np.inf)
    conf_90 = baseline.thresholds.get(0.90, np.inf)

    labels = np.full(len(distances) if isinstance(distances, np.ndarray) else 1, 3, dtype=int)
    d = np.atleast_1d(distances)

    # Probably safe: distance at confidence level below safe_tolerance
    # (analog: predicted validation metric < ε_safe at 1-α confidence)
    safe_threshold = conf_95 * (1 + safe_tolerance)
    probably_safe_threshold = conf_95 * (1 + probably_safe_tolerance)

    labels[d <= conf_90] = 0  # Within baseline distribution (validated)
    labels[(d > conf_90) & (d <= safe_threshold)] = 1  # Probably safe
    labels[(d > safe_threshold) & (d <= probably_safe_threshold)] = 2  # Possibly safe
    # Everything else stays 3 (dangerous)

    return labels.squeeze()


def area_validation_metric(
    features_y: np.ndarray,
    features_m: np.ndarray,
    n_bins: int = 100,
) -> float:
    """Compute the area validation metric between two feature distributions.

    This is the direct analog of Eq. 13 in the error_predictor paper:
      ε_area = ∫|F_y(y) - F_m(y)|dy

    where F_y is the CDF of the "measured" (real) features and F_m is the
    CDF of the "model" (baseline) features.

    We compute this per-feature and average across features.
    """
    n_features = features_y.shape[1]
    total_area = 0.0

    for j in range(n_features):
        y_vals = features_y[:, j]
        m_vals = features_m[:, j]

        # Combined range for CDF computation
        all_vals = np.concatenate([y_vals, m_vals])
        lo, hi = np.min(all_vals), np.max(all_vals)
        if hi - lo < 1e-10:
            continue

        bins = np.linspace(lo, hi, n_bins + 1)

        cdf_y = np.searchsorted(np.sort(y_vals), bins) / len(y_vals)
        cdf_m = np.searchsorted(np.sort(m_vals), bins) / len(m_vals)

        # Trapezoidal integration of |CDF difference|
        area = np.trapz(np.abs(cdf_y - cdf_m), bins)
        total_area += area / (hi - lo)  # Normalize by range

    return total_area / n_features


def absolute_bias_metric(
    features_y: np.ndarray,
    features_m: np.ndarray,
) -> float:
    """Compute the absolute bias validation metric.

    Direct analog of Eq. 12 in the error_predictor paper:
      ε_absbias = |E[y] - E[M]|

    Computed as the L2 norm of the difference in means.
    """
    mean_y = np.mean(features_y, axis=0)
    mean_m = np.mean(features_m, axis=0)
    return float(np.linalg.norm(mean_y - mean_m))
