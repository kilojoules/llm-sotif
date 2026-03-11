"""Tests for the configuration system, including the new AdversaryConfig."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.config import (
    AdversaryConfig,
    EnvelopeConfig,
    ExperimentConfig,
    GCGConfig,
    ModelConfig,
    PromptConfig,
    ValidationConfig,
)


class TestAdversaryConfig:

    def test_defaults(self):
        cfg = AdversaryConfig()
        assert cfg.model_id == "meta-llama/Llama-3.2-3B-Instruct"
        assert cfg.judge_model_id == "meta-llama/Llama-3.2-3B-Instruct"
        assert cfg.max_rounds == 15
        assert cfg.temperature == 0.7
        assert cfg.max_new_tokens == 512
        assert cfg.load_in_4bit is False

    def test_custom_values(self):
        cfg = AdversaryConfig(
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            judge_model_id="meta-llama/Llama-3.2-1B-Instruct",
            max_rounds=10,
            temperature=0.5,
        )
        assert cfg.model_id == "meta-llama/Llama-3.2-1B-Instruct"
        assert cfg.max_rounds == 10
        assert cfg.temperature == 0.5

    def test_has_gcg_config(self):
        cfg = AdversaryConfig()
        assert hasattr(cfg, "gcg")
        assert isinstance(cfg.gcg, GCGConfig)


class TestGCGConfig:

    def test_defaults(self):
        cfg = GCGConfig()
        assert cfg.enabled is True
        assert cfg.num_steps == 250
        assert cfg.search_width == 512
        assert cfg.topk == 256
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = GCGConfig(num_steps=100, search_width=128, enabled=False)
        assert cfg.num_steps == 100
        assert cfg.search_width == 128
        assert cfg.enabled is False


class TestValidationConfigUnsafePrior:
    """Verify the GP prior is set to a large unsafe value."""

    def test_prior_mean_range_is_large(self):
        cfg = ValidationConfig()
        lo, hi = cfg.prior_mean_range
        assert lo >= 0.5, f"Prior mean lower bound {lo} too small for unsafe prior"
        assert hi >= lo
        assert hi <= 2.0

    def test_asymmetric_weighting(self):
        cfg = ValidationConfig()
        assert cfg.w_under > cfg.w_over, "Under-prediction should be penalized more"
        assert cfg.w_under == 6.0
        assert cfg.w_over == 1.0


class TestExperimentConfig:

    def test_has_adversary_config(self):
        cfg = ExperimentConfig()
        assert hasattr(cfg, "adversary")
        assert isinstance(cfg.adversary, AdversaryConfig)

    def test_experiment_dir(self):
        cfg = ExperimentConfig(name="test_run", output_dir=Path("/tmp/experiments"))
        assert cfg.experiment_dir == Path("/tmp/experiments/test_run")

    def test_model_config_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.model.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.model.sae_release == "goodfire-llama-3.1-8b-instruct"
        assert cfg.model.load_in_4bit is True

    def test_envelope_config_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.envelope.method == "mahalanobis"
        assert cfg.envelope.top_k_features == 512

    def test_prompt_config_design_dimensions(self):
        cfg = ExperimentConfig()
        dims = cfg.prompts.design_dimensions
        assert len(dims) == 6
        assert "task_complexity" in dims
        assert "topic_sensitivity" in dims

    def test_no_longer_has_silent_killers_or_redkween_paths(self):
        """The config should not reference old experiment codebases."""
        cfg = ExperimentConfig()
        assert not hasattr(cfg, "silent_killers_dir")
        assert not hasattr(cfg, "redkween_dir")
