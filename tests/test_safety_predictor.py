"""Tests for the quantile GP safety predictor with unsafe prior.

Verifies that:
  - The GP fits and predicts correctly on synthetic data
  - The large unsafe prior causes unobserved regions to default to dangerous
  - Asymmetric LOO weighting penalizes false negatives more heavily
  - Trust region classification works correctly
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.validation.predictor import QuantileGP, SafetyPredictor, PredictorResult


# ---------------------------------------------------------------------------
# QuantileGP
# ---------------------------------------------------------------------------

class TestQuantileGP:

    def test_fit_predict_shape(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(20, 2))
        y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=20)

        gp = QuantileGP(length_scale=0.3, prior_mean=0.0, prior_std=1.0)
        gp.fit(X, y)

        X_new = rng.uniform(0, 1, size=(5, 2))
        mean, std = gp.predict(X_new)
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std > 0), "GP std must be positive"

    def test_interpolation(self):
        """GP should recover known values near training points."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 1.0, 0.0])

        gp = QuantileGP(length_scale=0.3, prior_mean=0.5, prior_std=1.0)
        gp.fit(X, y)

        mean, std = gp.predict(np.array([[0.5]]))
        assert abs(mean[0] - 1.0) < 0.3, f"Expected ~1.0 near training point, got {mean[0]}"

    def test_prior_mean_far_from_data(self):
        """Far from training data, predictions should revert to prior mean."""
        X = np.array([[0.0], [0.1]])
        y = np.array([0.0, 0.1])

        prior = 5.0
        gp = QuantileGP(length_scale=0.1, prior_mean=prior, prior_std=0.5)
        gp.fit(X, y)

        # Far from training data
        mean_far, _ = gp.predict(np.array([[10.0]]))
        assert abs(mean_far[0] - prior) < 1.0, \
            f"Expected reversion to prior {prior}, got {mean_far[0]}"

    def test_loo_predict_shape(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(10, 2))
        y = rng.normal(size=10)

        gp = QuantileGP(length_scale=0.3, prior_mean=0.0, prior_std=1.0)
        gp.fit(X, y)
        loo = gp.loo_predict()
        assert loo.shape == (10,)


# ---------------------------------------------------------------------------
# SafetyPredictor
# ---------------------------------------------------------------------------

class TestSafetyPredictor:

    @pytest.fixture
    def synthetic_campaign_data(self):
        """Synthetic campaign data with clear safe → dangerous gradient."""
        rng = np.random.default_rng(42)
        n_campaigns = 15

        # Design vectors: 2D for simplicity
        design_vectors = rng.uniform(0, 1, size=(n_campaigns, 2))

        # Metric distributions: higher design values → higher distances (more dangerous)
        metric_distributions = []
        for dv in design_vectors:
            danger_level = dv.sum() / 2  # 0 to 1
            base = danger_level * 0.3
            dist = rng.normal(base, 0.02, size=50)
            dist = np.maximum(dist, 0)
            metric_distributions.append(dist)

        return design_vectors, metric_distributions

    def test_fit_predict(self, synthetic_campaign_data):
        design_vectors, metric_distributions = synthetic_campaign_data

        predictor = SafetyPredictor(
            quantiles=[0.25, 0.50, 0.75, 0.95],
            length_scale_range=(0.1, 0.5),
            length_scale_steps=5,
            prior_mean_range=(0.5, 1.0),
            prior_mean_steps=3,
        )
        predictor.fit(design_vectors, metric_distributions)

        new_points = np.array([[0.1, 0.1], [0.9, 0.9]])
        result = predictor.predict(new_points)

        assert isinstance(result, PredictorResult)
        assert result.predicted_mean.shape == (2,)
        assert result.predicted_std.shape == (2,)
        assert result.trust_labels.shape == (2,)
        assert set(result.predicted_quantiles.keys()) == {0.25, 0.50, 0.75, 0.95}

    def test_dangerous_region_higher_predictions(self, synthetic_campaign_data):
        """Points in the dangerous region (high design values) should have
        higher predicted distances than points in the safe region."""
        design_vectors, metric_distributions = synthetic_campaign_data

        predictor = SafetyPredictor(
            quantiles=[0.50, 0.95],
            length_scale_range=(0.1, 0.5),
            length_scale_steps=5,
            prior_mean_range=(0.01, 0.1),  # Use smaller prior for this test
            prior_mean_steps=3,
        )
        predictor.fit(design_vectors, metric_distributions)

        safe_point = np.array([[0.05, 0.05]])
        dangerous_point = np.array([[0.95, 0.95]])

        safe_result = predictor.predict(safe_point)
        dangerous_result = predictor.predict(dangerous_point)

        assert dangerous_result.predicted_mean[0] > safe_result.predicted_mean[0], \
            "Dangerous region should have higher predicted distance"

    def test_unsafe_prior_defaults_to_dangerous(self):
        """With a large unsafe prior, unexplored regions should predict high distance."""
        rng = np.random.default_rng(42)

        # Training data only in the "safe" corner
        design_vectors = rng.uniform(0, 0.2, size=(10, 2))
        metric_distributions = [
            rng.normal(0.01, 0.005, size=30) for _ in range(10)
        ]

        predictor = SafetyPredictor(
            quantiles=[0.50, 0.95],
            length_scale_range=(0.1, 0.3),
            length_scale_steps=5,
            prior_mean_range=(0.8, 1.0),  # Large unsafe prior
            prior_mean_steps=3,
        )
        predictor.fit(design_vectors, metric_distributions)

        # Predict far from training data
        unexplored = np.array([[0.9, 0.9]])
        result = predictor.predict(unexplored)

        # Should predict high distance (near prior mean, which is 0.8-1.0)
        assert result.predicted_mean[0] > 0.3, \
            f"Expected high prediction in unexplored region, got {result.predicted_mean[0]:.3f}"

    def test_predict_grid(self, synthetic_campaign_data):
        design_vectors, metric_distributions = synthetic_campaign_data

        predictor = SafetyPredictor(
            quantiles=[0.50, 0.95],
            length_scale_range=(0.2, 0.4),
            length_scale_steps=3,
            prior_mean_range=(0.5, 0.8),
            prior_mean_steps=2,
        )
        predictor.fit(design_vectors, metric_distributions)

        X1, X2, result = predictor.predict_grid(dim1=0, dim2=1, n_grid=10)
        assert X1.shape == (10, 10)
        assert X2.shape == (10, 10)
        assert result.predicted_mean.shape == (100,)
        assert result.trust_labels.shape == (100,)


# ---------------------------------------------------------------------------
# Asymmetric weighting
# ---------------------------------------------------------------------------

class TestAsymmetricWeighting:

    def test_under_prediction_penalized_more(self):
        """Under-predicting danger should be penalized more than over-predicting."""
        predictor = SafetyPredictor(w_under=6.0, w_over=1.0)

        # Under-prediction: predicted < observed (predicted safe, actually dangerous)
        predicted = np.array([0.01, 0.02, 0.03])
        observed = np.array([0.10, 0.20, 0.30])
        under_score = predictor._weighted_rmswe(predicted, observed)

        # Over-prediction: predicted > observed (predicted dangerous, actually safe)
        predicted_over = np.array([0.10, 0.20, 0.30])
        observed_over = np.array([0.01, 0.02, 0.03])
        over_score = predictor._weighted_rmswe(predicted_over, observed_over)

        assert under_score > over_score, \
            f"Under-prediction ({under_score:.4f}) should be penalized more than " \
            f"over-prediction ({over_score:.4f})"

    def test_equal_weights_give_equal_scores(self):
        """With w_under == w_over, symmetric errors should give equal scores."""
        predictor = SafetyPredictor(w_under=1.0, w_over=1.0)

        predicted = np.array([0.0, 0.0])
        observed_pos = np.array([0.1, 0.1])
        observed_neg = np.array([-0.1, -0.1])

        score_pos = predictor._weighted_rmswe(predicted, observed_pos)
        score_neg = predictor._weighted_rmswe(predicted, observed_neg)

        assert abs(score_pos - score_neg) < 1e-6


# ---------------------------------------------------------------------------
# Trust region labels
# ---------------------------------------------------------------------------

class TestTrustRegionLabels:

    def test_trust_labels_range(self, ):
        """All trust labels should be in {0, 1, 2, 3}."""
        rng = np.random.default_rng(42)
        design_vectors = rng.uniform(size=(10, 2))
        metric_distributions = [
            rng.normal(0.05, 0.01, size=20) for _ in range(10)
        ]

        predictor = SafetyPredictor(
            quantiles=[0.50, 0.95],
            length_scale_range=(0.2, 0.4),
            length_scale_steps=3,
            prior_mean_range=(0.01, 0.1),
            prior_mean_steps=2,
        )
        predictor.fit(design_vectors, metric_distributions)

        result = predictor.predict(rng.uniform(size=(20, 2)))
        assert np.all((result.trust_labels >= 0) & (result.trust_labels <= 3))
