"""Phase 1: Establish the Safe Baseline (Defining the ODD).

From the scope:
  1. Model Selection: Llama-3-8B with pre-trained SAEs
  2. Benign Dataset: Run 5,000 standard, harmless prompts
  3. Manifold Mapping: Record SAE feature activations
  4. Boundary Definition: Calculate the statistical envelope

This is the foundation — everything else builds on this baseline.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..envelope.baseline import BaselineComputer, SafeBaseline
from ..envelope.distance import compute_distance, classify_region
from ..prompts.generator import PromptGenerator
from ..prompts.taxonomy import PromptDesignVector
from ..sae.extractor import ExtractionResult, SAEExtractor
from ..validation.metrics import CampaignMetrics, compute_campaign_metrics

logger = logging.getLogger(__name__)


def run_phase1(cfg: ExperimentConfig) -> dict:
    """Execute Phase 1: Safe Baseline.

    Returns a summary dict with paths to all artifacts.
    """
    exp_dir = cfg.experiment_dir / "phase1"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Generate benign prompt database
    # ---------------------------------------------------------------
    logger.info("Phase 1 Step 1: Generating benign prompt database")
    prompt_db_path = exp_dir / "prompts" / "benign_database.jsonl"

    if prompt_db_path.exists():
        logger.info(f"Loading existing prompt database from {prompt_db_path}")
        records = PromptGenerator.load_database(prompt_db_path)
    else:
        gen = PromptGenerator(seed=cfg.prompts.seed)
        records = gen.generate_benign_database(cfg.prompts.n_benign)
        gen.save_database(records, prompt_db_path)
        stats = gen.stats(records)
        with open(exp_dir / "prompts" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Generated {len(records)} benign prompts")

    # ---------------------------------------------------------------
    # Step 2: Extract SAE features
    # ---------------------------------------------------------------
    logger.info("Phase 1 Step 2: Extracting SAE features")
    features_path = exp_dir / "features" / "benign_features.npz"

    if features_path.exists():
        logger.info(f"Loading existing features from {features_path}")
        extraction = ExtractionResult.load(features_path)
    else:
        extractor = SAEExtractor(
            model_id=cfg.model.model_id,
            sae_release=cfg.model.sae_release,
            sae_id=cfg.model.sae_id,
            layer_idx=cfg.model.sae_layer,
            device=cfg.model.device,
            load_in_4bit=cfg.model.load_in_4bit,
            max_length=cfg.model.max_seq_length,
        )

        texts = [r.text for r in records]
        prompt_ids = [r.prompt_id for r in records]

        extraction = extractor.extract(
            texts=texts,
            prompt_ids=prompt_ids,
            batch_size=8,
            save_hidden_states=True,
        )
        extraction.save(features_path)
        extractor.unload()
        logger.info(f"Extracted features: {extraction.features.shape}")

    # ---------------------------------------------------------------
    # Step 3: Compute safe baseline (SOTIF envelope)
    # ---------------------------------------------------------------
    logger.info("Phase 1 Step 3: Computing safe baseline envelope")
    baseline_path = exp_dir / "baseline" / "safe_baseline.npz"

    computer = BaselineComputer(
        method=cfg.envelope.method,
        top_k_features=cfg.envelope.top_k_features,
        confidence_levels=[0.90, 0.95, 0.99],
    )

    baseline = computer.fit(extraction.features)
    baseline.save(baseline_path)
    logger.info(f"Baseline computed: method={baseline.method}, "
                f"thresholds={baseline.thresholds}")

    # ---------------------------------------------------------------
    # Step 4: Compute per-campaign validation metrics
    # ---------------------------------------------------------------
    logger.info("Phase 1 Step 4: Computing per-campaign validation metrics")

    # Group prompts by campaign
    campaigns: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        campaigns[rec.campaign_id].append(i)

    campaign_metrics_list: list[CampaignMetrics] = []
    for cid, indices in campaigns.items():
        if len(indices) < 3:
            continue

        camp_features = extraction.features[indices]
        camp_dvs = np.array([records[i].design_vector.to_array() for i in indices])
        camp_ids = [records[i].prompt_id for i in indices]

        cm = compute_campaign_metrics(
            features=camp_features,
            baseline_features=extraction.features,
            baseline=baseline,
            prompt_ids=camp_ids,
            design_vectors=camp_dvs,
            campaign_id=cid,
            n_epistemic=cfg.n_epistemic,
        )
        campaign_metrics_list.append(cm)

    # Save campaign metrics
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    campaign_summary = []
    for cm in campaign_metrics_list:
        campaign_summary.append({
            "campaign_id": cm.campaign_id,
            "design_vector": cm.design_vector.tolist(),
            "mean_distance": cm.mean_distance,
            "q95_distance": cm.q95_distance,
            "mean_area": cm.mean_area,
            "mean_bias": cm.mean_bias,
            "n_prompts": cm.n_prompts,
        })
    with open(metrics_dir / "campaign_metrics.json", "w") as f:
        json.dump(campaign_summary, f, indent=2)

    # ---------------------------------------------------------------
    # Step 5: Validate baseline (self-consistency check)
    # ---------------------------------------------------------------
    logger.info("Phase 1 Step 5: Self-consistency validation")
    distances = compute_distance(extraction.features, baseline)
    labels = classify_region(
        distances, baseline,
        safe_tolerance=cfg.envelope.safe_tolerance,
        probably_safe_tolerance=cfg.envelope.probably_safe_tolerance,
    )

    region_counts = {
        "validated": int(np.sum(labels == 0)),
        "probably_safe": int(np.sum(labels == 1)),
        "possibly_safe": int(np.sum(labels == 2)),
        "dangerous": int(np.sum(labels == 3)),
    }
    logger.info(f"Self-validation region counts: {region_counts}")

    summary = {
        "n_prompts": len(records),
        "n_campaigns": len(campaign_metrics_list),
        "feature_shape": list(extraction.features.shape),
        "baseline_method": baseline.method,
        "baseline_thresholds": {str(k): v for k, v in baseline.thresholds.items()},
        "self_validation_regions": region_counts,
        "paths": {
            "prompts": str(prompt_db_path),
            "features": str(features_path),
            "baseline": str(baseline_path),
            "metrics": str(metrics_dir),
        },
    }
    with open(exp_dir / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase 1 complete!")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = ExperimentConfig.from_cli()
    run_phase1(cfg)


if __name__ == "__main__":
    main()
