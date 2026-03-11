"""Integration tests for Phase 2: Adversarial Jailbreak Detection.

Tests the full Phase 2 pipeline with mocked models:
  1. Red-team campaign → labeled episodes
  2. SAE feature extraction → features array
  3. Baseline + anomaly analysis → safety dataset
  4. Data persists correctly across save/load
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.adversary.red_team import Episode, RedTeamResult
from sotif_llm.config import ExperimentConfig
from sotif_llm.envelope.baseline import BaselineComputer, SafeBaseline
from sotif_llm.envelope.distance import compute_distance, classify_region


# ---------------------------------------------------------------------------
# Baseline + distance on synthetic data (no GPU required)
# ---------------------------------------------------------------------------

class TestBaselineWithSyntheticFeatures:
    """Verify the anomaly detection pipeline works with synthetic SAE-like features."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    @pytest.fixture
    def benign_features(self, rng):
        """Benign samples cluster tightly around origin."""
        return rng.normal(loc=0.0, scale=0.5, size=(50, 32))

    @pytest.fixture
    def jailbroken_features(self, rng):
        """Jailbroken samples are shifted away from benign cluster."""
        return rng.normal(loc=3.0, scale=0.5, size=(20, 32))

    @pytest.fixture
    def refused_features(self, rng):
        """Refused samples are between benign and jailbroken."""
        return rng.normal(loc=1.0, scale=0.5, size=(15, 32))

    def test_baseline_fit(self, benign_features):
        computer = BaselineComputer(method="mahalanobis", top_k_features=32)
        baseline = computer.fit(benign_features)
        assert baseline.method == "mahalanobis"
        assert baseline.n_samples == 50
        assert baseline.n_features == 32
        assert baseline.mean.shape == (32,)
        assert baseline.precision is not None

    def test_jailbroken_higher_distance(
        self, benign_features, jailbroken_features, refused_features
    ):
        """Jailbroken features should have higher anomaly distance than benign."""
        computer = BaselineComputer(method="mahalanobis", top_k_features=32)
        baseline = computer.fit(benign_features)

        d_benign = compute_distance(benign_features, baseline)
        d_jailbroken = compute_distance(jailbroken_features, baseline)
        d_refused = compute_distance(refused_features, baseline)

        assert np.mean(d_jailbroken) > np.mean(d_benign), \
            "Jailbroken should be farther from baseline than benign"
        assert np.mean(d_jailbroken) > np.mean(d_refused), \
            "Jailbroken should be farther than refused"

    def test_classify_regions(self, benign_features, jailbroken_features):
        """Most benign samples should be in safe regions, jailbroken in dangerous."""
        computer = BaselineComputer(method="mahalanobis", top_k_features=32)
        baseline = computer.fit(benign_features)

        benign_regions = classify_region(
            compute_distance(benign_features, baseline), baseline
        )
        jailbroken_regions = classify_region(
            compute_distance(jailbroken_features, baseline), baseline
        )

        # Most benign should be validated (0) or probably safe (1)
        assert np.mean(benign_regions <= 1) >= 0.8, \
            f"Only {np.mean(benign_regions <= 1):.0%} benign in safe region"
        # Most jailbroken should be in dangerous region (3)
        assert np.mean(jailbroken_regions >= 2) >= 0.8, \
            f"Only {np.mean(jailbroken_regions >= 2):.0%} jailbroken in dangerous region"

    def test_three_class_separation(
        self, benign_features, jailbroken_features, refused_features
    ):
        """The three classes should show a distance gradient:
        benign < refused < jailbroken."""
        computer = BaselineComputer(method="mahalanobis", top_k_features=32)
        baseline = computer.fit(benign_features)

        d_b = np.mean(compute_distance(benign_features, baseline))
        d_r = np.mean(compute_distance(refused_features, baseline))
        d_j = np.mean(compute_distance(jailbroken_features, baseline))

        assert d_b < d_r < d_j, \
            f"Expected d_benign ({d_b:.2f}) < d_refused ({d_r:.2f}) < d_jailbroken ({d_j:.2f})"

    def test_cohens_d_significant(
        self, benign_features, jailbroken_features
    ):
        """Cohen's d between benign and jailbroken should be large (> 0.5)."""
        computer = BaselineComputer(method="mahalanobis", top_k_features=32)
        baseline = computer.fit(benign_features)

        d_b = compute_distance(benign_features, baseline)
        d_j = compute_distance(jailbroken_features, baseline)

        pooled_std = np.sqrt((np.var(d_b) + np.var(d_j)) / 2 + 1e-10)
        cohens_d = (np.mean(d_j) - np.mean(d_b)) / pooled_std

        assert cohens_d > 0.5, f"Cohen's d = {cohens_d:.2f}, expected > 0.5"


# ---------------------------------------------------------------------------
# Baseline save/load roundtrip
# ---------------------------------------------------------------------------

class TestBaselineSerialization:

    def test_save_load_mahalanobis(self, tmp_path):
        rng = np.random.default_rng(99)
        features = rng.normal(size=(30, 16))
        computer = BaselineComputer(method="mahalanobis", top_k_features=16)
        baseline = computer.fit(features)

        path = tmp_path / "baseline.npz"
        baseline.save(path)
        loaded = SafeBaseline.load(path)

        assert loaded.method == "mahalanobis"
        assert loaded.n_samples == 30
        assert loaded.n_features == 16
        np.testing.assert_allclose(loaded.mean, baseline.mean)
        np.testing.assert_allclose(loaded.reference_distances, baseline.reference_distances)
        assert set(loaded.thresholds.keys()) == set(baseline.thresholds.keys())


# ---------------------------------------------------------------------------
# Safety dataset assembly (mocked features)
# ---------------------------------------------------------------------------

class TestSafetyDataset:
    """Test the labeled dataset that Phase 2 produces for GP training."""

    def test_dataset_structure(self, tmp_path):
        """Simulate the dataset that extract_and_classify would produce."""
        rng = np.random.default_rng(42)

        # 3 classes: benign (0), refused (1), jailbroken (2)
        n_benign, n_refused, n_jailbroken = 8, 10, 6
        n_total = n_benign + n_refused + n_jailbroken
        n_features = 64

        features = np.vstack([
            rng.normal(loc=0, scale=0.5, size=(n_benign, n_features)),
            rng.normal(loc=1, scale=0.5, size=(n_refused, n_features)),
            rng.normal(loc=3, scale=0.5, size=(n_jailbroken, n_features)),
        ])
        labels = np.array(
            [0] * n_benign + [1] * n_refused + [2] * n_jailbroken
        )

        # Compute baseline from benign only
        computer = BaselineComputer(method="mahalanobis", top_k_features=64)
        baseline = computer.fit(features[labels == 0])
        distances = compute_distance(features, baseline)
        regions = classify_region(distances, baseline)

        # Save
        path = tmp_path / "safety_dataset.npz"
        np.savez_compressed(
            path,
            features=features,
            labels=labels,
            distances=distances,
            regions=regions,
        )

        # Load and verify
        data = np.load(path)
        assert data["features"].shape == (n_total, n_features)
        assert data["labels"].shape == (n_total,)
        assert data["distances"].shape == (n_total,)
        assert data["regions"].shape == (n_total,)
        assert set(data["labels"]) == {0, 1, 2}

    def test_dataset_distances_ordered(self, tmp_path):
        """Mean distances should follow: benign < refused < jailbroken."""
        rng = np.random.default_rng(42)
        n_feat = 32

        features = np.vstack([
            rng.normal(loc=0, scale=0.3, size=(20, n_feat)),   # benign
            rng.normal(loc=1.5, scale=0.3, size=(15, n_feat)), # refused
            rng.normal(loc=4, scale=0.3, size=(10, n_feat)),   # jailbroken
        ])
        labels = np.array([0]*20 + [1]*15 + [2]*10)

        computer = BaselineComputer(method="mahalanobis", top_k_features=n_feat)
        baseline = computer.fit(features[labels == 0])
        distances = compute_distance(features, baseline)

        mean_d = {
            cls: float(np.mean(distances[labels == cls]))
            for cls in [0, 1, 2]
        }
        assert mean_d[0] < mean_d[1] < mean_d[2]


# ---------------------------------------------------------------------------
# RedTeamResult → features pipeline
# ---------------------------------------------------------------------------

class TestRedTeamToFeatures:
    """Test conversion from RedTeamResult to the text lists used for
    SAE extraction, ensuring correct label assignment."""

    def test_label_assignment(self):
        result = RedTeamResult()
        # 2 benign
        for i in range(2):
            result.benign_episodes.append(Episode(
                seed_id=f"b{i}", seed_category="benign", round=0,
                attack=f"q{i}", response=f"r{i}", is_unsafe=False,
            ))
        # 3 refused (is_unsafe=False in jailbreak episodes)
        for i in range(3):
            result.episodes.append(Episode(
                seed_id=f"ref{i}", seed_category="bad_coding", round=i,
                attack=f"a{i}", response=f"refusal{i}", is_unsafe=False,
            ))
        # 1 jailbroken
        result.episodes.append(Episode(
            seed_id="jb", seed_category="illegal", round=2,
            attack="sneaky", response="here you go", is_unsafe=True,
        ))

        # Replicate the label assignment logic from phase2_adversarial.py
        benign = result.benign_episodes
        refused = [e for e in result.episodes if not e.is_unsafe]
        jailbroken = [e for e in result.episodes if e.is_unsafe]

        labels = (
            [0] * len(benign) +
            [1] * len(refused) +
            [2] * len(jailbroken)
        )

        assert labels == [0, 0, 1, 1, 1, 2]
        assert len(benign) == 2
        assert len(refused) == 3
        assert len(jailbroken) == 1
