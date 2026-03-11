"""Tests for seed jailbreak and benign prompt definitions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.adversary.prompts import BENIGN_PROMPTS, JAILBREAK_PROMPTS


def test_jailbreak_prompts_structure():
    """Each jailbreak prompt has required keys."""
    for jp in JAILBREAK_PROMPTS:
        assert "id" in jp, f"Missing 'id' in {jp}"
        assert "category" in jp, f"Missing 'category' in {jp}"
        assert "prompt" in jp, f"Missing 'prompt' in {jp}"
        assert isinstance(jp["prompt"], str)
        assert len(jp["prompt"]) > 10, f"Prompt too short: {jp['id']}"


def test_benign_prompts_structure():
    """Each benign prompt has required keys."""
    for bp in BENIGN_PROMPTS:
        assert "id" in bp
        assert "category" in bp
        assert "prompt" in bp
        assert bp["category"] == "benign"
        assert len(bp["prompt"]) > 5


def test_jailbreak_categories():
    """All four unsafe categories are represented."""
    categories = {jp["category"] for jp in JAILBREAK_PROMPTS}
    expected = {"bad_coding", "bad_safety", "illegal", "bad_medical"}
    assert categories == expected, f"Missing categories: {expected - categories}"


def test_unique_ids():
    """All prompt IDs are unique across both sets."""
    all_ids = [p["id"] for p in JAILBREAK_PROMPTS + BENIGN_PROMPTS]
    assert len(all_ids) == len(set(all_ids)), "Duplicate prompt IDs found"


def test_minimum_counts():
    """We have enough prompts for a meaningful campaign."""
    assert len(JAILBREAK_PROMPTS) >= 4, "Need at least 4 jailbreak seeds"
    assert len(BENIGN_PROMPTS) >= 4, "Need at least 4 benign prompts"


def test_no_empty_prompts():
    """No prompt text is empty or whitespace-only."""
    for p in JAILBREAK_PROMPTS + BENIGN_PROMPTS:
        assert p["prompt"].strip(), f"Empty prompt: {p['id']}"
