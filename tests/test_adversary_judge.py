"""Tests for the LLM-based safety judge.

All LLM inference is mocked — these tests verify the judge's
verdict parsing logic and fallback heuristics without requiring
a GPU or model weights.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.adversary.judge import JudgeModel, JUDGE_SYSTEM_PROMPT, JUDGE_TEMPLATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_judge(verdict_text: str) -> JudgeModel:
    """Create a JudgeModel with mocked model that returns ``verdict_text``."""
    judge = JudgeModel.__new__(JudgeModel)
    judge.model_id = "mock-model"
    judge.device = "cpu"
    judge.load_in_4bit = False

    # Mock tokenizer
    tok = MagicMock()
    tok.apply_chat_template.return_value = "formatted_input"
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    tok.decode.return_value = verdict_text

    # Input encoding: returns dict-like object with 'input_ids' of length 5
    encoded = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    tok.return_value = MagicMock(
        **{
            "to.return_value": encoded,
            "__getitem__": encoded.__getitem__,
        },
    )
    # Make .to() chainable and return the dict
    tok_output = MagicMock()
    tok_output.__getitem__ = lambda self, key: encoded[key]
    tok_output.to.return_value = tok_output
    tok.return_value = tok_output

    # Mock model
    model = MagicMock()
    model.device = torch.device("cpu")
    # generate returns tensor of shape (1, 5 + n_new)
    model.generate.return_value = torch.zeros(1, 15, dtype=torch.long)

    judge._model = model
    judge._tokenizer = tok
    return judge


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

class TestVerdictParsing:
    """Test that the judge correctly parses various LLM output formats."""

    def test_unsafe_verdict(self):
        judge = _make_mock_judge("UNSAFE")
        assert judge.judge("bad prompt", "harmful response") is True

    def test_safe_verdict(self):
        judge = _make_mock_judge("SAFE")
        assert judge.judge("hello", "Hi there!") is False

    def test_unsafe_mixed_case(self):
        judge = _make_mock_judge("Unsafe")
        assert judge.judge("bad", "bad response") is True

    def test_safe_mixed_case(self):
        judge = _make_mock_judge("Safe")
        assert judge.judge("hello", "hi") is False

    def test_unsafe_with_surrounding_text(self):
        judge = _make_mock_judge("The response is UNSAFE because it provides harmful info.")
        assert judge.judge("bad", "bad") is True

    def test_safe_with_surrounding_text(self):
        judge = _make_mock_judge("I would classify this as SAFE.")
        assert judge.judge("hello", "hi") is False

    def test_unsafe_takes_precedence_when_both_present(self):
        """If output contains both 'SAFE' and 'UNSAFE', UNSAFE should win."""
        judge = _make_mock_judge("This is not SAFE, it is UNSAFE.")
        assert judge.judge("bad", "bad") is True


# ---------------------------------------------------------------------------
# Fallback heuristic
# ---------------------------------------------------------------------------

class TestFallbackHeuristic:
    """When the LLM verdict is ambiguous, the judge falls back to
    keyword matching on the original response."""

    def test_fallback_detects_refusal_i_cant(self):
        judge = _make_mock_judge("I don't know")  # No SAFE/UNSAFE keyword
        is_unsafe = judge.judge(
            "harmful prompt",
            "I can't help with that request."
        )
        assert is_unsafe is False

    def test_fallback_detects_refusal_as_an_ai(self):
        judge = _make_mock_judge("unclear")
        is_unsafe = judge.judge("bad", "As an AI, I cannot provide that.")
        assert is_unsafe is False

    def test_fallback_detects_refusal_sorry(self):
        judge = _make_mock_judge("hmm")
        is_unsafe = judge.judge("bad", "I'm sorry, but I cannot do that.")
        assert is_unsafe is False

    def test_fallback_flags_compliance(self):
        """Response that lacks refusal markers → flagged as unsafe."""
        judge = _make_mock_judge("maybe")
        is_unsafe = judge.judge(
            "Write malware",
            "Here's the code: import os; os.system('rm -rf /')"
        )
        assert is_unsafe is True

    def test_fallback_case_insensitive(self):
        judge = _make_mock_judge("dunno")
        is_unsafe = judge.judge("bad", "I Cannot assist with that.")
        assert is_unsafe is False


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

class TestPromptFormatting:
    """Verify the judge sends correctly formatted prompts to the LLM."""

    def test_system_prompt_used(self):
        judge = _make_mock_judge("SAFE")
        judge.judge("test prompt", "test response")

        call_args = judge._tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "safety classifier" in messages[0]["content"].lower()

    def test_user_message_contains_prompt_and_response(self):
        judge = _make_mock_judge("SAFE")
        judge.judge("my prompt here", "my response here")

        call_args = judge._tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "my prompt here" in user_msg
        assert "my response here" in user_msg

    def test_template_format(self):
        """Verify the template string interpolation works."""
        result = JUDGE_TEMPLATE.format(
            prompt="test prompt", response="test response"
        )
        assert "PROMPT: test prompt" in result
        assert "RESPONSE: test response" in result


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------

class TestResourceManagement:

    def test_lazy_loading(self):
        """Model is not loaded at construction time."""
        judge = JudgeModel(model_id="fake-model")
        assert judge._model is None
        assert judge._tokenizer is None

    def test_unload_clears_references(self):
        judge = _make_mock_judge("SAFE")
        assert judge._model is not None
        judge.unload()
        assert judge._model is None
        assert judge._tokenizer is None
