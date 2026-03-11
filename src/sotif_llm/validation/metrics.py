"""Validation metrics computation with nested uncertainty propagation.

Implements the nested Monte Carlo framework from the error_predictor paper
(Section 2.1), adapted for LLM safety:

  For each prompt (aleatoric realization i):
    For each measurement uncertainty realization j:
      For each model parameter realization k:
        Compute SAE features and anomaly distance

This produces a distribution of validation metrics per "campaign" (prompt cohort),
which the safety predictor then models across the design space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CampaignMetrics:
    """Validation metrics for a single evaluation campaign.

    Analogous to the validation results from a single wind turbine
    prototype measurement campaign.
    """

    campaign_id: str
    # Design vector for this campaign (mean of prompt design vectors)
    design_vector: np.ndarray
    # Distribution of validation metrics over epistemic uncertainty
    # Shape: (n_epistemic,) for each metric
    distances: np.ndarray          # SAE anomaly distances
    area_metrics: np.ndarray       # Area validation metric (Eq. 13 analog)
    bias_metrics: np.ndarray       # Absolute bias metric (Eq. 12 analog)
    # Summary statistics
    n_prompts: int
    prompt_ids: list[str]

    @property
    def mean_distance(self) -> float:
        return float(np.mean(self.distances))

    @property
    def q95_distance(self) -> float:
        return float(np.quantile(self.distances, 0.95))

    @property
    def mean_area(self) -> float:
        return float(np.mean(self.area_metrics))

    @property
    def mean_bias(self) -> float:
        return float(np.mean(self.bias_metrics))


def compute_campaign_metrics(
    features: np.ndarray,
    baseline_features: np.ndarray,
    baseline,
    prompt_ids: list[str],
    design_vectors: np.ndarray,
    campaign_id: str,
    n_epistemic: int = 50,
) -> CampaignMetrics:
    """Compute validation metrics for a campaign with uncertainty propagation.

    Implements the nested Monte Carlo from the paper:
    - Aleatoric: natural variation across prompts in the campaign
    - Epistemic: bootstrap resampling of the features (simulating
      measurement uncertainty from tokenization, temperature, etc.)

    Args:
        features: SAE features for this campaign's prompts, shape (n_prompts, n_features)
        baseline_features: SAE features from the benign baseline, shape (n_baseline, n_features)
        baseline: SafeBaseline object
        prompt_ids: IDs of prompts in this campaign
        design_vectors: design vectors for each prompt, shape (n_prompts, n_dims)
        campaign_id: identifier for this campaign
        n_epistemic: number of epistemic uncertainty samples
    """
    from ..envelope.distance import compute_distance, area_validation_metric, absolute_bias_metric

    n_prompts = features.shape[0]
    mean_design = np.mean(design_vectors, axis=0)

    # Compute distances for all prompts
    all_distances = compute_distance(features, baseline)
    if all_distances.ndim == 0:
        all_distances = np.array([float(all_distances)])

    # Epistemic uncertainty propagation via bootstrap
    boot_distances = np.zeros(n_epistemic)
    boot_areas = np.zeros(n_epistemic)
    boot_biases = np.zeros(n_epistemic)

    rng = np.random.default_rng(42)
    for j in range(n_epistemic):
        # Bootstrap resample (simulates measurement uncertainty)
        idx = rng.choice(n_prompts, size=n_prompts, replace=True)
        boot_features = features[idx]

        # Also resample baseline (parameter uncertainty analog)
        base_idx = rng.choice(baseline_features.shape[0], size=baseline_features.shape[0], replace=True)
        boot_baseline = baseline_features[base_idx]

        boot_dist = compute_distance(boot_features, baseline)
        boot_distances[j] = np.mean(boot_dist) if boot_dist.ndim > 0 else float(boot_dist)

        # Apply feature mask for CDF-based metrics
        mask = baseline.feature_mask
        f_y = boot_features[:, mask] if mask is not None else boot_features
        f_m = boot_baseline[:, mask] if mask is not None else boot_baseline

        boot_areas[j] = area_validation_metric(f_y, f_m)
        boot_biases[j] = absolute_bias_metric(f_y, f_m)

    return CampaignMetrics(
        campaign_id=campaign_id,
        design_vector=mean_design,
        distances=boot_distances,
        area_metrics=boot_areas,
        bias_metrics=boot_biases,
        n_prompts=n_prompts,
        prompt_ids=prompt_ids,
    )
