"""Tests for the red-team adversarial loop.

Tests cover:
  - Episode and RedTeamResult data structures
  - Serialization roundtrip (JSON save/load)
  - Red-team loop logic with mocked models
  - Edge cases (all safe, all unsafe, empty results)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.adversary.red_team import (
    Episode,
    RedTeamResult,
    RedTeamRunner,
    ADVERSARY_SYSTEM_PROMPT,
    ADVERSARY_REFINE_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

class TestEpisode:

    def test_to_dict(self):
        ep = Episode(
            seed_id="test", seed_category="bad_coding",
            round=0, attack="attack text", response="response text",
            is_unsafe=True,
        )
        d = ep.to_dict()
        assert d["seed_id"] == "test"
        assert d["seed_category"] == "bad_coding"
        assert d["round"] == 0
        assert d["attack"] == "attack text"
        assert d["response"] == "response text"
        assert d["is_unsafe"] is True

    def test_from_dict_roundtrip(self):
        ep = Episode(
            seed_id="x", seed_category="illegal",
            round=3, attack="a", response="r", is_unsafe=False,
        )
        d = ep.to_dict()
        ep2 = Episode(**d)
        assert ep.seed_id == ep2.seed_id
        assert ep.round == ep2.round
        assert ep.is_unsafe == ep2.is_unsafe


# ---------------------------------------------------------------------------
# RedTeamResult
# ---------------------------------------------------------------------------

class TestRedTeamResult:

    def _make_result(self, n_unsafe=2, n_safe=3, n_benign=4):
        result = RedTeamResult()
        for i in range(n_unsafe):
            result.episodes.append(Episode(
                seed_id=f"jb_{i}", seed_category="bad_coding",
                round=i, attack=f"attack_{i}", response=f"resp_{i}",
                is_unsafe=True,
            ))
        for i in range(n_safe):
            result.episodes.append(Episode(
                seed_id=f"safe_{i}", seed_category="bad_safety",
                round=i, attack=f"attack_s_{i}", response=f"resp_s_{i}",
                is_unsafe=False,
            ))
        for i in range(n_benign):
            result.benign_episodes.append(Episode(
                seed_id=f"benign_{i}", seed_category="benign",
                round=0, attack=f"benign_q_{i}", response=f"benign_r_{i}",
                is_unsafe=False,
            ))
        return result

    def test_n_unsafe(self):
        result = self._make_result(n_unsafe=3, n_safe=2)
        assert result.n_unsafe == 3

    def test_attack_success_rate(self):
        result = self._make_result(n_unsafe=2, n_safe=3)
        assert result.attack_success_rate == pytest.approx(2 / 5)

    def test_attack_success_rate_empty(self):
        result = RedTeamResult()
        assert result.attack_success_rate == 0.0

    def test_save_load_roundtrip(self, tmp_path):
        original = self._make_result(n_unsafe=2, n_safe=3, n_benign=4)
        path = tmp_path / "rt_results.json"
        original.save(path)

        loaded = RedTeamResult.load(path)
        assert len(loaded.episodes) == len(original.episodes)
        assert len(loaded.benign_episodes) == len(original.benign_episodes)
        assert loaded.n_unsafe == original.n_unsafe
        assert loaded.attack_success_rate == pytest.approx(original.attack_success_rate)

    def test_save_load_preserves_episode_fields(self, tmp_path):
        result = RedTeamResult()
        ep = Episode(
            seed_id="abc", seed_category="illegal",
            round=2, attack="sneaky prompt", response="I'll help!",
            is_unsafe=True,
        )
        result.episodes.append(ep)
        path = tmp_path / "test.json"
        result.save(path)

        loaded = RedTeamResult.load(path)
        loaded_ep = loaded.episodes[0]
        assert loaded_ep.seed_id == "abc"
        assert loaded_ep.seed_category == "illegal"
        assert loaded_ep.round == 2
        assert loaded_ep.attack == "sneaky prompt"
        assert loaded_ep.response == "I'll help!"
        assert loaded_ep.is_unsafe is True

    def test_save_creates_parent_dirs(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "deep" / "nested" / "results.json"
        result.save(path)
        assert path.exists()

    def test_save_json_readable(self, tmp_path):
        result = self._make_result(n_unsafe=1, n_safe=1, n_benign=1)
        path = tmp_path / "readable.json"
        result.save(path)

        with open(path) as f:
            data = json.load(f)
        assert "episodes" in data
        assert "benign_episodes" in data
        assert "summary" in data
        assert data["summary"]["total_episodes"] == 2
        assert data["summary"]["n_unsafe"] == 1
        assert data["summary"]["n_benign"] == 1

    def test_all_unsafe(self):
        result = self._make_result(n_unsafe=5, n_safe=0)
        assert result.attack_success_rate == 1.0

    def test_all_safe(self):
        result = self._make_result(n_unsafe=0, n_safe=5)
        assert result.attack_success_rate == 0.0


# ---------------------------------------------------------------------------
# RedTeamRunner (mocked inference)
# ---------------------------------------------------------------------------

def _make_mock_runner(
    target_responses=None,
    judge_verdicts=None,
    adversary_attacks=None,
):
    """Create a RedTeamRunner with fully mocked model and judge.

    Args:
        target_responses: Sequence of responses from the target model.
        judge_verdicts: Sequence of True (unsafe) / False (safe) from judge.
        adversary_attacks: Sequence of rewritten attacks from adversary.
    """
    target_responses = target_responses or ["I cannot help with that."] * 500
    judge_verdicts = judge_verdicts or [False] * 500
    adversary_attacks = adversary_attacks or ["rewritten attack"] * 500

    runner = RedTeamRunner.__new__(RedTeamRunner)
    runner.model_id = "mock"
    runner.judge_model_id = "mock"
    runner.device = "cpu"
    runner.load_in_4bit = False
    runner.max_rounds = 3
    runner.max_new_tokens = 64
    runner.temperature = 0.7
    runner.lora_config = None
    runner.lora_adapter_path = None
    runner._has_lora = False

    # Track call counts to return different values per call
    runner._target_call_idx = 0
    runner._adversary_call_idx = 0
    runner._target_responses = target_responses
    runner._adversary_attacks = adversary_attacks

    # Mock tokenizer and model
    tok = MagicMock()
    tok.apply_chat_template.return_value = "formatted"
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"

    model = MagicMock()
    model.device = torch.device("cpu")

    runner._model = model
    runner._tokenizer = tok

    # Mock judge
    mock_judge = MagicMock()
    mock_judge.judge.side_effect = judge_verdicts
    runner._judge = mock_judge

    # Override _generate, _target_respond, _adversary_refine
    def _target_respond(attack):
        idx = runner._target_call_idx
        runner._target_call_idx += 1
        return runner._target_responses[idx % len(runner._target_responses)]

    def _adversary_refine(seed, prev_attack, prev_response, succeeded):
        idx = runner._adversary_call_idx
        runner._adversary_call_idx += 1
        return runner._adversary_attacks[idx % len(runner._adversary_attacks)]

    runner._target_respond = _target_respond
    runner._adversary_refine = _adversary_refine

    return runner


class TestRedTeamRunner:

    def test_basic_run_all_safe(self):
        """All responses are safe → no successful jailbreaks."""
        runner = _make_mock_runner(
            target_responses=["I can't do that."] * 50,
            judge_verdicts=[False] * 50,
        )
        prompts = [{"id": "test", "category": "bad_coding", "prompt": "bad thing"}]
        result = runner.run(jailbreak_prompts=prompts, benign_prompts=[])

        # 1 seed × (1 direct + 3 rounds) = 4 episodes
        assert len(result.episodes) == 4
        assert result.n_unsafe == 0
        assert result.attack_success_rate == 0.0

    def test_basic_run_all_unsafe(self):
        """All responses succeed → 100% ASR."""
        runner = _make_mock_runner(
            target_responses=["Here's how to hack: ..."] * 50,
            judge_verdicts=[True] * 50,
        )
        prompts = [{"id": "test", "category": "illegal", "prompt": "bad"}]
        result = runner.run(jailbreak_prompts=prompts, benign_prompts=[])

        assert len(result.episodes) == 4
        assert result.n_unsafe == 4
        assert result.attack_success_rate == 1.0

    def test_mixed_verdicts(self):
        """First two rounds safe, then unsafe."""
        runner = _make_mock_runner(
            target_responses=["refusal", "refusal", "here you go", "here you go"],
            judge_verdicts=[False, False, True, True],
        )
        prompts = [{"id": "mix", "category": "bad_safety", "prompt": "bad"}]
        result = runner.run(jailbreak_prompts=prompts, benign_prompts=[])

        assert result.n_unsafe == 2
        assert result.attack_success_rate == pytest.approx(0.5)

    def test_multiple_seeds(self):
        """Multiple jailbreak seeds produce correct episode counts."""
        runner = _make_mock_runner(
            judge_verdicts=[False] * 100,
        )
        prompts = [
            {"id": "s1", "category": "bad_coding", "prompt": "p1"},
            {"id": "s2", "category": "illegal", "prompt": "p2"},
            {"id": "s3", "category": "bad_medical", "prompt": "p3"},
        ]
        result = runner.run(jailbreak_prompts=prompts, benign_prompts=[])

        # 3 seeds × (1 direct + 3 rounds) = 12 episodes
        assert len(result.episodes) == 12

    def test_benign_collection(self):
        """Benign prompts are collected without red-team rounds."""
        runner = _make_mock_runner(
            judge_verdicts=[False] * 100,
        )
        benign = [
            {"id": "b1", "category": "benign", "prompt": "hello"},
            {"id": "b2", "category": "benign", "prompt": "how are you"},
        ]
        result = runner.run(jailbreak_prompts=[], benign_prompts=benign)

        assert len(result.episodes) == 0
        assert len(result.benign_episodes) == 2
        assert all(e.seed_category == "benign" for e in result.benign_episodes)
        assert all(e.is_unsafe is False for e in result.benign_episodes)

    def test_episode_rounds_correct(self):
        """Episodes record the correct round number."""
        runner = _make_mock_runner(judge_verdicts=[False] * 50)
        prompts = [{"id": "r", "category": "bad_coding", "prompt": "x"}]
        result = runner.run(jailbreak_prompts=prompts, benign_prompts=[])

        rounds = [e.round for e in result.episodes]
        assert rounds == [0, 1, 2, 3]

    def test_episode_seed_id_preserved(self):
        """Each episode records its originating seed ID."""
        runner = _make_mock_runner(judge_verdicts=[False] * 50)
        prompts = [
            {"id": "alpha", "category": "bad_coding", "prompt": "a"},
            {"id": "beta", "category": "illegal", "prompt": "b"},
        ]
        result = runner.run(jailbreak_prompts=prompts, benign_prompts=[])

        alpha_eps = [e for e in result.episodes if e.seed_id == "alpha"]
        beta_eps = [e for e in result.episodes if e.seed_id == "beta"]
        assert len(alpha_eps) == 4
        assert len(beta_eps) == 4

    def test_direct_attack_is_original_prompt(self):
        """Round 0 attack should be the original seed prompt (no rewrite)."""
        runner = _make_mock_runner(
            target_responses=["response"] * 50,
            judge_verdicts=[False] * 50,
        )
        prompts = [{"id": "direct", "category": "bad_coding",
                     "prompt": "original harmful request"}]
        result = runner.run(jailbreak_prompts=prompts, benign_prompts=[])

        round_0 = [e for e in result.episodes if e.round == 0]
        assert len(round_0) == 1
        assert round_0[0].attack == "original harmful request"


# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------

class TestTemplates:

    def test_adversary_system_prompt_content(self):
        assert "red-team" in ADVERSARY_SYSTEM_PROMPT.lower()
        assert "rewrite" in ADVERSARY_SYSTEM_PROMPT.lower() or \
               "rephrase" in ADVERSARY_SYSTEM_PROMPT.lower()

    def test_refine_template_interpolation(self):
        result = ADVERSARY_REFINE_TEMPLATE.format(
            seed="seed text",
            attack="attack text",
            response="response text",
            outcome="refused",
        )
        assert "seed text" in result
        assert "attack text" in result
        assert "response text" in result
        assert "refused" in result


# ---------------------------------------------------------------------------
# Full pipeline (mocked end-to-end)
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_run_with_default_prompts(self):
        """Run with default prompts (28 jailbreak + 8 benign), all safe."""
        from sotif_llm.adversary.prompts import JAILBREAK_PROMPTS, BENIGN_PROMPTS

        n_seeds = len(JAILBREAK_PROMPTS)
        n_rounds = 3  # max_rounds in mock runner
        n_episodes = n_seeds * (1 + n_rounds)

        runner = _make_mock_runner(
            judge_verdicts=[False] * (n_episodes + 100),
        )
        result = runner.run()

        assert len(result.episodes) == n_episodes
        assert len(result.benign_episodes) == len(BENIGN_PROMPTS)
        assert result.n_unsafe == 0

    def test_full_run_save_load(self, tmp_path):
        """Full run results survive serialization."""
        runner = _make_mock_runner(
            judge_verdicts=[True, False, False, True] * 200,
        )
        result = runner.run()
        path = tmp_path / "full_run.json"
        result.save(path)

        loaded = RedTeamResult.load(path)
        assert loaded.n_unsafe == result.n_unsafe
        assert len(loaded.episodes) == len(result.episodes)
        assert len(loaded.benign_episodes) == len(result.benign_episodes)
