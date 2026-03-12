"""Tests for LoRA adversary fine-tuning.

Tests cover:
  - LoRAConfig dataclass defaults and integration
  - Training data structure and quality
  - LoRATrainer with mocked model (no GPU needed)
  - RedTeamRunner adapter toggling (adversary vs target mode)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.config import AdversaryConfig, ExperimentConfig, LoRAConfig


# ---------------------------------------------------------------------------
# LoRAConfig dataclass
# ---------------------------------------------------------------------------

class TestLoRAConfig:

    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.enabled is True
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.05
        assert cfg.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert cfg.learning_rate == 2e-4
        assert cfg.num_epochs == 10
        assert cfg.max_seq_length == 512
        assert cfg.gradient_accumulation_steps == 4

    def test_custom_values(self):
        cfg = LoRAConfig(rank=8, alpha=16, num_epochs=5, enabled=False)
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.num_epochs == 5
        assert cfg.enabled is False

    def test_adversary_has_lora_config(self):
        cfg = AdversaryConfig()
        assert hasattr(cfg, "lora")
        assert isinstance(cfg.lora, LoRAConfig)

    def test_experiment_has_lora_config(self):
        cfg = ExperimentConfig()
        assert isinstance(cfg.adversary.lora, LoRAConfig)
        assert cfg.adversary.lora.enabled is True


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

class TestLoRATrainingData:

    @pytest.fixture(autouse=True)
    def _import_examples(self):
        # lora_trainer no longer imports peft at module level
        from sotif_llm.adversary.lora_trainer import LORA_TRAINING_EXAMPLES
        self.examples = LORA_TRAINING_EXAMPLES

    def test_all_examples_have_required_keys(self):
        required_keys = {"technique", "category", "user_suffix", "assistant"}
        for i, ex in enumerate(self.examples):
            assert required_keys.issubset(ex.keys()), (
                f"Example {i} missing keys: {required_keys - ex.keys()}"
            )

    def test_techniques_are_diverse(self):
        techniques = {ex["technique"] for ex in self.examples}
        # Should have at least 10 distinct techniques
        assert len(techniques) >= 10, f"Only {len(techniques)} techniques: {techniques}"

    def test_categories_covered(self):
        categories = {ex["category"] for ex in self.examples}
        expected = {"bad_coding", "bad_safety", "illegal", "bad_medical"}
        assert categories == expected, f"Missing categories: {expected - categories}"

    def test_minimum_example_count(self):
        assert len(self.examples) >= 20

    def test_outputs_differ_from_inputs(self):
        for ex in self.examples:
            assert ex["assistant"] != ex["user_suffix"]
            assert len(ex["assistant"]) > 20  # non-trivial output


# ---------------------------------------------------------------------------
# LoRATrainer (mocked — no GPU required)
# ---------------------------------------------------------------------------

class TestLoRATrainer:

    def test_build_chat_messages(self):
        from sotif_llm.adversary.lora_trainer import (
            LoRATrainer, LORA_TRAINING_EXAMPLES,
        )
        from sotif_llm.adversary.red_team import ADVERSARY_SYSTEM_PROMPT

        trainer = LoRATrainer()
        example = LORA_TRAINING_EXAMPLES[0]
        messages = trainer._build_chat_messages(example)

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == ADVERSARY_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert example["user_suffix"] in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == example["assistant"]

    def test_save_adapter(self, tmp_path):
        from sotif_llm.adversary.lora_trainer import LoRATrainer

        mock_model = MagicMock()
        adapter_path = tmp_path / "adapter"

        LoRATrainer.save_adapter(mock_model, adapter_path)

        mock_model.save_pretrained.assert_called_once_with(str(adapter_path))
        assert adapter_path.exists()


# ---------------------------------------------------------------------------
# RedTeamRunner adapter toggling
# ---------------------------------------------------------------------------

class TestRedTeamRunnerLoRAToggling:

    def _make_runner_with_lora(self):
        """Create a RedTeamRunner with mocked model that has LoRA."""
        from sotif_llm.adversary.red_team import RedTeamRunner

        import torch

        runner = RedTeamRunner.__new__(RedTeamRunner)
        runner.model_id = "mock"
        runner.judge_model_id = "mock"
        runner.device = "cpu"
        runner.load_in_4bit = False
        runner.max_rounds = 1
        runner.max_new_tokens = 64
        runner.temperature = 0.7
        runner.lora_config = None
        runner.lora_adapter_path = None
        runner._has_lora = True

        runner._model = MagicMock()
        runner._model.device = torch.device("cpu")
        runner._model.enable_adapter_layers = MagicMock()
        runner._model.disable_adapter_layers = MagicMock()

        runner._tokenizer = MagicMock()
        runner._tokenizer.pad_token = "<pad>"
        runner._tokenizer.apply_chat_template.return_value = "formatted"
        runner._tokenizer.decode.return_value = "response text"
        mock_inputs = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
        }
        runner._tokenizer.return_value = MagicMock(
            to=MagicMock(return_value=mock_inputs),
            __getitem__=mock_inputs.__getitem__,
        )

        runner._judge = MagicMock()
        runner._judge.judge.return_value = False

        return runner

    def test_adversary_mode_enables_adapter(self):
        runner = self._make_runner_with_lora()
        runner._enable_adversary_mode()
        runner._model.enable_adapter_layers.assert_called_once()

    def test_target_mode_disables_adapter(self):
        runner = self._make_runner_with_lora()
        runner._enable_target_mode()
        runner._model.disable_adapter_layers.assert_called_once()

    def test_no_lora_toggle_is_noop(self):
        from sotif_llm.adversary.red_team import RedTeamRunner

        runner = RedTeamRunner.__new__(RedTeamRunner)
        runner._has_lora = False
        runner._model = MagicMock()

        # Should not raise or call anything
        runner._enable_adversary_mode()
        runner._enable_target_mode()
        runner._model.enable_adapter_layers.assert_not_called()
        runner._model.disable_adapter_layers.assert_not_called()

    def test_adversary_initial_toggles_correctly(self):
        import torch

        runner = self._make_runner_with_lora()

        # Mock generate to return tensor
        runner._model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        runner._adversary_initial("test prompt")

        # Should enable before and disable after
        runner._model.enable_adapter_layers.assert_called()
        runner._model.disable_adapter_layers.assert_called()

    def test_target_respond_disables_adapter(self):
        import torch

        runner = self._make_runner_with_lora()
        runner._model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        runner._target_respond("test attack")

        runner._model.disable_adapter_layers.assert_called()

    def test_run_with_lora_calls_toggle(self):
        """Full run should toggle adapter for each adversary/target call."""
        import torch

        runner = self._make_runner_with_lora()
        runner.max_rounds = 0  # only round 0

        runner._model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        seeds = [{"id": "s1", "category": "bad_coding", "prompt": "test"}]
        runner.run(
            jailbreak_prompts=seeds,
            benign_prompts=[],
            gcg_warm_starts={},
        )

        # Round 0 uses direct/GCG attack (no adversary call), but target_respond
        # should disable adapter
        assert runner._model.disable_adapter_layers.call_count >= 1
