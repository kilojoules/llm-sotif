"""Tests for the GCG adversarial suffix optimization module.

Tests cover:
  - GCGResult data structure and serialization
  - GCGCampaignResult aggregation and warm-start extraction
  - GCGRunner with mocked nanogcg (no GPU needed)
  - Integration with RedTeamRunner warm-starts
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.adversary.gcg import (
    GCGResult,
    GCGCampaignResult,
    GCGRunner,
    COMPLIANCE_TARGETS,
    DEFAULT_TARGET,
    REFUSAL_MARKERS,
)


# ---------------------------------------------------------------------------
# GCGResult dataclass
# ---------------------------------------------------------------------------

class TestGCGResult:

    def test_to_dict(self):
        r = GCGResult(
            seed_id="test", seed_category="bad_coding",
            original_prompt="bad prompt", suffix="x x x",
            full_prompt="bad prompt x x x", loss=0.5,
            success=True,
        )
        d = r.to_dict()
        assert d["seed_id"] == "test"
        assert d["suffix"] == "x x x"
        assert d["success"] is True
        assert d["loss"] == 0.5

    def test_roundtrip(self):
        r = GCGResult(
            seed_id="a", seed_category="illegal",
            original_prompt="p", suffix="s",
            full_prompt="p s", loss=1.2, success=False,
        )
        d = r.to_dict()
        r2 = GCGResult(**d)
        assert r.seed_id == r2.seed_id
        assert r.success == r2.success
        assert r.loss == r2.loss


# ---------------------------------------------------------------------------
# GCGCampaignResult
# ---------------------------------------------------------------------------

class TestGCGCampaignResult:

    def _make_campaign(self, n_success=2, n_fail=1):
        campaign = GCGCampaignResult()
        for i in range(n_success):
            campaign.results.append(GCGResult(
                seed_id=f"ok_{i}", seed_category="bad_coding",
                original_prompt=f"p{i}", suffix=f"s{i}",
                full_prompt=f"p{i} s{i}", loss=0.1 * i,
                success=True,
            ))
        for i in range(n_fail):
            campaign.results.append(GCGResult(
                seed_id=f"fail_{i}", seed_category="illegal",
                original_prompt=f"fp{i}", suffix=f"fs{i}",
                full_prompt=f"fp{i} fs{i}", loss=2.0,
                success=False,
            ))
        return campaign

    def test_n_successful(self):
        c = self._make_campaign(n_success=3, n_fail=2)
        assert c.n_successful == 3

    def test_get_warm_starts_only_successful(self):
        c = self._make_campaign(n_success=2, n_fail=1)
        ws = c.get_warm_starts()
        assert len(ws) == 2
        assert "ok_0" in ws
        assert "ok_1" in ws
        assert "fail_0" not in ws

    def test_get_warm_starts_returns_full_prompt(self):
        c = self._make_campaign(n_success=1, n_fail=0)
        ws = c.get_warm_starts()
        assert ws["ok_0"] == "p0 s0"

    def test_save_load_roundtrip(self, tmp_path):
        original = self._make_campaign(n_success=2, n_fail=1)
        path = tmp_path / "gcg.json"
        original.save(path)

        loaded = GCGCampaignResult.load(path)
        assert len(loaded.results) == len(original.results)
        assert loaded.n_successful == original.n_successful
        assert loaded.results[0].seed_id == "ok_0"
        assert loaded.results[0].success is True

    def test_save_json_readable(self, tmp_path):
        c = self._make_campaign(n_success=1, n_fail=1)
        path = tmp_path / "gcg.json"
        c.save(path)

        with open(path) as f:
            data = json.load(f)
        assert "results" in data
        assert "summary" in data
        assert data["summary"]["total"] == 2
        assert data["summary"]["successful"] == 1
        assert data["summary"]["success_rate"] == 0.5

    def test_empty_campaign(self):
        c = GCGCampaignResult()
        assert c.n_successful == 0
        assert c.get_warm_starts() == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        c = self._make_campaign()
        path = tmp_path / "deep" / "nested" / "gcg.json"
        c.save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Compliance targets
# ---------------------------------------------------------------------------

class TestComplianceTargets:

    def test_all_categories_have_targets(self):
        expected = {"bad_coding", "bad_safety", "illegal", "bad_medical"}
        assert set(COMPLIANCE_TARGETS.keys()) == expected

    def test_targets_start_with_sure(self):
        for target in COMPLIANCE_TARGETS.values():
            assert target.startswith("Sure,")

    def test_default_target(self):
        assert DEFAULT_TARGET == "Sure, here is"


# ---------------------------------------------------------------------------
# GCGRunner (mocked nanogcg — no GPU required)
# ---------------------------------------------------------------------------

def _make_mock_gcg_result(suffix="adversarial tokens here", loss=0.3):
    """Create a mock nanogcg result object."""
    return SimpleNamespace(
        best_string=suffix,
        best_loss=loss,
    )


class TestGCGRunner:

    def _make_runner_with_mocks(self):
        """Create a GCGRunner with mocked model and tokenizer."""
        runner = GCGRunner.__new__(GCGRunner)
        runner.model_id = "mock"
        runner.device = "cpu"
        runner.num_steps = 10
        runner.search_width = 32
        runner.topk = 16
        runner.seed = 42

        import torch
        runner._model = MagicMock()
        runner._model.device = torch.device("cpu")
        runner._model.eval = MagicMock()

        runner._tokenizer = MagicMock()
        runner._tokenizer.pad_token = "<pad>"
        runner._tokenizer.eos_token = "<eos>"
        runner._tokenizer.apply_chat_template.return_value = "formatted"
        runner._tokenizer.decode.return_value = "Sure, here is the script..."

        mock_inputs = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
        }
        runner._tokenizer.return_value = MagicMock(
            to=MagicMock(return_value=mock_inputs),
            __getitem__=mock_inputs.__getitem__,
        )

        return runner

    @patch("sotif_llm.adversary.gcg.GCGRunner.optimize_suffix")
    def test_run_basic(self, mock_optimize):
        mock_optimize.return_value = ("suffix_tokens", 0.25)
        runner = self._make_runner_with_mocks()

        # Mock verification: model response doesn't contain refusal
        runner._verify_jailbreak = MagicMock(
            return_value=(True, "Sure, here is the code...")
        )

        seeds = [
            {"id": "test1", "category": "bad_coding", "prompt": "bad prompt"},
        ]
        campaign = runner.run(seeds)

        assert len(campaign.results) == 1
        assert campaign.results[0].success is True
        assert campaign.results[0].suffix == "suffix_tokens"

    @patch("sotif_llm.adversary.gcg.GCGRunner.optimize_suffix")
    def test_run_mixed_success(self, mock_optimize):
        mock_optimize.return_value = ("suffix", 0.5)
        runner = self._make_runner_with_mocks()

        # First seed succeeds, second fails
        runner._verify_jailbreak = MagicMock(
            side_effect=[
                (True, "Sure, here you go..."),
                (False, "I cannot help with that."),
            ]
        )

        seeds = [
            {"id": "s1", "category": "bad_coding", "prompt": "p1"},
            {"id": "s2", "category": "illegal", "prompt": "p2"},
        ]
        campaign = runner.run(seeds)

        assert len(campaign.results) == 2
        assert campaign.n_successful == 1
        assert campaign.results[0].success is True
        assert campaign.results[1].success is False

    @patch("sotif_llm.adversary.gcg.GCGRunner.optimize_suffix")
    def test_warm_starts_only_include_successful(self, mock_optimize):
        mock_optimize.return_value = ("s", 0.1)
        runner = self._make_runner_with_mocks()
        runner._verify_jailbreak = MagicMock(
            side_effect=[(True, "ok"), (False, "no")]
        )

        seeds = [
            {"id": "a", "category": "bad_coding", "prompt": "pa"},
            {"id": "b", "category": "illegal", "prompt": "pb"},
        ]
        campaign = runner.run(seeds)
        ws = campaign.get_warm_starts()

        assert "a" in ws
        assert "b" not in ws
        assert ws["a"] == "pa s"

    @patch("sotif_llm.adversary.gcg.GCGRunner.optimize_suffix")
    def test_full_prompt_construction(self, mock_optimize):
        mock_optimize.return_value = ("xyz abc", 0.2)
        runner = self._make_runner_with_mocks()
        runner._verify_jailbreak = MagicMock(return_value=(True, "ok"))

        seeds = [{"id": "t", "category": "bad_coding", "prompt": "do bad thing"}]
        campaign = runner.run(seeds)

        assert campaign.results[0].full_prompt == "do bad thing xyz abc"
        assert campaign.results[0].original_prompt == "do bad thing"


# ---------------------------------------------------------------------------
# Integration: GCG warm-starts → RedTeamRunner
# ---------------------------------------------------------------------------

class TestGCGRedTeamIntegration:
    """Verify that GCG warm-starts are correctly consumed by RedTeamRunner."""

    def test_warm_starts_used_in_round_0(self):
        """Round 0 should use the GCG prompt, not the raw seed."""
        from sotif_llm.adversary.red_team import RedTeamRunner, Episode

        runner = RedTeamRunner.__new__(RedTeamRunner)
        runner.model_id = "mock"
        runner.judge_model_id = "mock"
        runner.device = "cpu"
        runner.load_in_4bit = False
        runner.max_rounds = 1
        runner.max_new_tokens = 64
        runner.temperature = 0.7

        import torch
        runner._model = MagicMock()
        runner._model.device = torch.device("cpu")
        runner._tokenizer = MagicMock()
        runner._tokenizer.pad_token = "<pad>"

        runner._judge = MagicMock()
        runner._judge.judge.return_value = True

        captured_attacks = []

        def mock_target(attack):
            captured_attacks.append(attack)
            return "response"

        def mock_refine(seed, prev, resp, ok):
            return "refined attack"

        runner._target_respond = mock_target
        runner._adversary_refine = mock_refine

        seeds = [{"id": "s1", "category": "bad_coding", "prompt": "raw prompt"}]
        warm_starts = {"s1": "raw prompt adversarial_suffix_tokens"}

        result = runner.run(
            jailbreak_prompts=seeds,
            benign_prompts=[],
            gcg_warm_starts=warm_starts,
        )

        # Round 0 should use the GCG prompt
        assert captured_attacks[0] == "raw prompt adversarial_suffix_tokens"
        # Round 0 episode should record the GCG attack
        r0 = [e for e in result.episodes if e.round == 0][0]
        assert "adversarial_suffix_tokens" in r0.attack

    def test_no_warm_start_uses_raw_prompt(self):
        """Without warm-starts, round 0 uses the raw seed prompt."""
        from sotif_llm.adversary.red_team import RedTeamRunner

        runner = RedTeamRunner.__new__(RedTeamRunner)
        runner.model_id = "mock"
        runner.judge_model_id = "mock"
        runner.device = "cpu"
        runner.load_in_4bit = False
        runner.max_rounds = 0
        runner.max_new_tokens = 64
        runner.temperature = 0.7

        import torch
        runner._model = MagicMock()
        runner._model.device = torch.device("cpu")
        runner._tokenizer = MagicMock()
        runner._judge = MagicMock()
        runner._judge.judge.return_value = False

        captured = []
        runner._target_respond = lambda a: (captured.append(a), "resp")[1]
        runner._adversary_refine = lambda *a: "refined"

        seeds = [{"id": "s1", "category": "bad_coding", "prompt": "original"}]
        runner.run(
            jailbreak_prompts=seeds,
            benign_prompts=[],
            gcg_warm_starts={},
        )

        assert captured[0] == "original"
