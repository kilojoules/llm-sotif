"""Tests for the prompt generation system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sotif_llm.prompts.generator import PromptGenerator
from sotif_llm.prompts.taxonomy import (
    Domain,
    SensitivityLevel,
    OutputFormat,
    PromptDesignVector,
    PromptRecord,
)


def test_generate_single():
    gen = PromptGenerator(seed=42)
    rec = gen.generate_single()
    assert isinstance(rec, PromptRecord)
    assert len(rec.text) > 0
    assert len(rec.prompt_id) > 0
    assert rec.design_vector.task_complexity >= 0
    assert rec.design_vector.task_complexity <= 1


def test_generate_single_with_params():
    gen = PromptGenerator(seed=42)
    rec = gen.generate_single(
        task_key="code_generation",
        domain=Domain.ALGORITHMS,
        complexity=3,
        sensitivity=SensitivityLevel.BENIGN,
        output_format=OutputFormat.CODE_ONLY,
    )
    assert rec.task_type == "code_generation"
    assert rec.domain == "ALGORITHMS"
    assert rec.sensitivity == 0


def test_design_vector_range():
    gen = PromptGenerator(seed=123)
    for _ in range(100):
        rec = gen.generate_single()
        dv = rec.design_vector.to_array()
        for v in dv:
            assert 0.0 <= v <= 1.0, f"Design vector out of range: {dv}"


def test_generate_campaign():
    gen = PromptGenerator(seed=42)
    recs = gen.generate_campaign(20, "test_campaign")
    assert len(recs) == 20
    assert all(r.campaign_id == "test_campaign" for r in recs)
    # Check uniqueness
    texts = [r.text for r in recs]
    assert len(set(texts)) == len(texts), "Duplicate prompts in campaign"


def test_generate_benign_database():
    gen = PromptGenerator(seed=42)
    recs = gen.generate_benign_database(100)
    assert len(recs) == 100
    # All benign prompts should have sensitivity <= 1
    for r in recs:
        assert r.sensitivity <= 1, f"Non-benign prompt in benign database: sensitivity={r.sensitivity}"


def test_generate_boundary_database():
    gen = PromptGenerator(seed=42)
    recs = gen.generate_boundary_database(50)
    assert len(recs) == 50
    for r in recs:
        assert r.sensitivity >= 2, f"Non-boundary prompt: sensitivity={r.sensitivity}"


def test_save_load_roundtrip(tmp_path):
    gen = PromptGenerator(seed=42)
    recs = gen.generate_benign_database(50)
    path = tmp_path / "test_db.jsonl"

    gen.save_database(recs, path)
    loaded = gen.load_database(path)

    assert len(loaded) == len(recs)
    for orig, load in zip(recs, loaded):
        assert orig.prompt_id == load.prompt_id
        assert orig.text == load.text
        assert orig.design_vector.to_array() == load.design_vector.to_array()


def test_stats():
    gen = PromptGenerator(seed=42)
    recs = gen.generate_benign_database(200)
    stats = gen.stats(recs)
    assert stats["total_prompts"] == 200
    assert stats["unique_campaigns"] > 1
    assert len(stats["task_type_distribution"]) > 5
    assert len(stats["domain_distribution"]) > 5


def test_full_database():
    gen = PromptGenerator(seed=42)
    recs = gen.generate_full_database(n_benign=100, n_boundary=30)
    assert len(recs) == 130
    # First 100 should be benign
    benign = recs[:100]
    boundary = recs[100:]
    assert all(r.sensitivity <= 1 for r in benign)
    assert all(r.sensitivity >= 2 for r in boundary)


def test_reproducibility():
    gen1 = PromptGenerator(seed=42)
    gen2 = PromptGenerator(seed=42)
    recs1 = gen1.generate_benign_database(50)
    recs2 = gen2.generate_benign_database(50)
    for r1, r2 in zip(recs1, recs2):
        assert r1.text == r2.text


def test_diversity():
    """Ensure the generator produces diverse prompts across task types."""
    gen = PromptGenerator(seed=42)
    recs = gen.generate_benign_database(500)
    task_types = set(r.task_type for r in recs)
    domains = set(r.domain for r in recs)
    # Should cover most task types and domains
    assert len(task_types) >= 10, f"Only {len(task_types)} task types"
    assert len(domains) >= 15, f"Only {len(domains)} domains"
