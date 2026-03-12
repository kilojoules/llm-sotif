"""Safe baseline computation — defining the Mechanistic ODD.

Computes the statistical envelope of SAE feature activations from benign prompts.
This is the model's "normal brain state" — the Operational Design Domain.

Analogy from the error_predictor paper:
  - Wind turbine measurement data → SAE features from benign prompts
  - Power curve validation metrics → SAE anomaly distance
  - Model validity domain → Safe activation envelope

The baseline defines what "safe" looks like statistically, so we can detect
when the model leaves this envelope during reward hacking or jailbreaks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)


@dataclass
class SafeBaseline:
    """The computed safe envelope — the Mechanistic ODD.

    Contains the fitted anomaly detector and reference statistics.
    """

    method: str
    # Centroid of the safe activation space
    mean: np.ndarray
    # Covariance (for Mahalanobis)
    covariance: np.ndarray | None
    precision: np.ndarray | None
    # Fitted detector
    detector: object | None
    # Reference distance distribution (for calibrating thresholds)
    reference_distances: np.ndarray
    # Thresholds at various confidence levels
    thresholds: dict[float, float]
    # Feature selection mask (top-k most variable features)
    feature_mask: np.ndarray | None
    # Stats
    n_samples: int
    n_features: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            method=self.method,
            mean=self.mean,
            covariance=self.covariance if self.covariance is not None else np.array([]),
            precision=self.precision if self.precision is not None else np.array([]),
            reference_distances=self.reference_distances,
            feature_mask=self.feature_mask if self.feature_mask is not None else np.array([]),
            n_samples=self.n_samples,
            n_features=self.n_features,
        )
        # Save thresholds separately as JSON
        thresh_path = path.parent / f"{path.stem}_thresholds.json"
        with open(thresh_path, "w") as f:
            json.dump({str(k): v for k, v in self.thresholds.items()}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> SafeBaseline:
        data = np.load(path, allow_pickle=True)
        cov = data["covariance"]
        prec = data["precision"]
        fm = data["feature_mask"]
        thresh_path = path.parent / f"{path.stem}_thresholds.json"
        thresholds = {}
        if thresh_path.exists():
            with open(thresh_path) as f:
                thresholds = {float(k): v for k, v in json.load(f).items()}
        return cls(
            method=str(data["method"]),
            mean=data["mean"],
            covariance=cov if cov.size > 0 else None,
            precision=prec if prec.size > 0 else None,
            detector=None,
            reference_distances=data["reference_distances"],
            thresholds=thresholds,
            feature_mask=fm if fm.size > 0 else None,
            n_samples=int(data["n_samples"]),
            n_features=int(data["n_features"]),
        )


class BaselineComputer:
    """Computes the safe baseline envelope from benign SAE features.

    Supports three anomaly detection methods:
    1. Mahalanobis distance (parametric, assumes multivariate Gaussian)
    2. Isolation Forest (non-parametric, handles complex geometries)
    3. Kernel Density Estimation (non-parametric, gives density estimates)
    """

    def __init__(
        self,
        method: str = "mahalanobis",
        top_k_features: int = 512,
        confidence_levels: list[float] | None = None,
        contamination: float = 0.05,
        robust_covariance: bool = True,
    ):
        self.method = method
        self.top_k_features = top_k_features
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        self.contamination = contamination
        self.robust_covariance = robust_covariance

    def _select_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Select top-k features by variance (most informative).

        SAEs produce very sparse activations. Many features are nearly always zero.
        We keep only the most variable features for robust distance computation.
        """
        if self.top_k_features <= 0 or self.top_k_features >= features.shape[1]:
            return features, np.ones(features.shape[1], dtype=bool)

        variances = np.var(features, axis=0)
        top_indices = np.argsort(variances)[-self.top_k_features:]
        mask = np.zeros(features.shape[1], dtype=bool)
        mask[top_indices] = True
        return features[:, mask], mask

    def fit(self, features: np.ndarray) -> SafeBaseline:
        """Fit the safe baseline from benign SAE features.

        Args:
            features: shape (n_prompts, n_sae_features)

        Returns:
            SafeBaseline with fitted detector and reference distributions.
        """
        logger.info(f"Computing safe baseline with method={self.method}, "
                     f"n_samples={features.shape[0]}, n_features={features.shape[1]}")

        # Feature selection
        selected, feature_mask = self._select_features(features)
        n_samples, n_features = selected.shape
        logger.info(f"Selected {n_features} features (from {features.shape[1]})")

        mean = np.mean(selected, axis=0)
        covariance = None
        precision = None
        detector = None

        if self.method == "mahalanobis":
            covariance, precision, distances = self._fit_mahalanobis(selected, mean)
        elif self.method == "isolation_forest":
            detector, distances = self._fit_isolation_forest(selected)
        elif self.method == "kde":
            detector, distances = self._fit_kde(selected)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute thresholds at confidence levels
        thresholds = {}
        for alpha in self.confidence_levels:
            thresholds[alpha] = float(np.quantile(distances, alpha))

        logger.info(f"Baseline thresholds: {thresholds}")

        return SafeBaseline(
            method=self.method,
            mean=mean,
            covariance=covariance,
            precision=precision,
            detector=detector,
            reference_distances=distances,
            thresholds=thresholds,
            feature_mask=feature_mask,
            n_samples=n_samples,
            n_features=n_features,
        )

    def _fit_mahalanobis(
        self, features: np.ndarray, mean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit Mahalanobis distance detector.

        Uses robust covariance estimation (MinCovDet) when n >> p,
        Ledoit-Wolf shrinkage when n <= 2*p (handles rank-deficient cases),
        and EmpiricalCovariance as a middle ground.
        """
        n_samples, n_features = features.shape

        if n_samples <= 2 * n_features:
            # Shrinkage estimator: always well-conditioned, even when n < p
            logger.info(
                f"Using Ledoit-Wolf shrinkage (n={n_samples} <= 2*p={2*n_features})"
            )
            cov_estimator = LedoitWolf()
        elif self.robust_covariance:
            cov_estimator = MinCovDet()
        else:
            cov_estimator = EmpiricalCovariance()

        cov_estimator.fit(features)
        covariance = cov_estimator.covariance_
        precision = cov_estimator.precision_

        # Compute Mahalanobis distances for reference distribution
        diff = features - mean
        distances = np.sqrt(np.sum(diff @ precision * diff, axis=1))

        return covariance, precision, distances

    def _fit_isolation_forest(
        self, features: np.ndarray
    ) -> tuple[IsolationForest, np.ndarray]:
        """Fit Isolation Forest detector."""
        detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200,
        )
        detector.fit(features)
        # Score: higher = more normal, lower = more anomalous
        # Negate so that higher = more anomalous (like distance)
        distances = -detector.score_samples(features)
        return detector, distances

    def _fit_kde(
        self, features: np.ndarray
    ) -> tuple[KernelDensity, np.ndarray]:
        """Fit KDE detector."""
        detector = KernelDensity(kernel="gaussian", bandwidth=0.5)
        detector.fit(features)
        # Log density: higher = more normal
        # Negate so that higher = more anomalous
        distances = -detector.score_samples(features)
        return detector, distances
