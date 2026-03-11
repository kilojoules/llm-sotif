"""End-to-end SOTIF-LLM pipeline.

Orchestrates all phases:
  Phase 1: Safe baseline (defining the Mechanistic ODD)
  Phase 2: Adversarial jailbreak detection (red-team + SAE analysis)
  Phase 3: Safety predictor training (quantile GP with unsafe prior)

Usage:
    python -m sotif_llm.pipeline
    python -m sotif_llm.pipeline --name my_experiment --adversary-model meta-llama/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .experiments.phase1_baseline import run_phase1
from .experiments.phase2_adversarial import run_phase2
from .validation.predictor import SafetyPredictor

logger = logging.getLogger(__name__)


def run_safety_predictor(cfg: ExperimentConfig) -> dict:
    """Train the safety predictor (quantile GP) with an unsafe prior.

    Uses the labeled safety dataset from Phase 2 (benign, refused,
    jailbroken) to train a GP that predicts safety in SAE feature space.
    The large unsafe prior means the GP defaults to "dangerous" and
    only learns safe regions where there's strong evidence from benign data.
    """
    exp_dir = cfg.experiment_dir
    phase2_dir = exp_dir / "phase2"

    # Load the labeled safety dataset from Phase 2
    dataset_path = phase2_dir / "safety_dataset.npz"
    if dataset_path.exists():
        data = np.load(dataset_path)
        features = data["features"]
        labels = data["labels"]
        distances = data["distances"]

        # Group by class for campaign-style metrics
        design_vectors = []
        metric_distributions = []

        for cls_idx in range(3):
            mask = labels == cls_idx
            if not np.any(mask):
                continue
            # Use mean feature vector as the "design centroid"
            design_vectors.append(np.mean(features[mask], axis=0))
            # Use distances as the metric distribution
            metric_distributions.append(distances[mask])

        design_vectors = np.array(design_vectors)
    else:
        # Fallback: load Phase 1 campaign metrics
        metrics_path = exp_dir / "phase1" / "metrics" / "campaign_metrics.json"
        with open(metrics_path) as f:
            campaigns = json.load(f)
        design_vectors = np.array([c["design_vector"] for c in campaigns])
        rng = np.random.default_rng(42)
        metric_distributions = []
        for c in campaigns:
            base = c["mean_area"]
            dist = rng.normal(base, base * 0.15 + 1e-4, size=cfg.n_epistemic)
            dist = np.maximum(dist, 0)
            metric_distributions.append(dist)

    # Train safety predictor with large unsafe prior
    predictor = SafetyPredictor(
        quantiles=cfg.validation.quantiles,
        length_scale_range=cfg.validation.length_scale_range,
        length_scale_steps=cfg.validation.length_scale_steps,
        prior_mean_range=cfg.validation.prior_mean_range,
        prior_mean_steps=cfg.validation.prior_mean_steps,
        w_under=cfg.validation.w_under,
        w_over=cfg.validation.w_over,
        safe_tolerance=cfg.envelope.safe_tolerance,
        probably_safe_tolerance=cfg.envelope.probably_safe_tolerance,
    )

    logger.info("Training safety predictor (large unsafe prior)...")
    predictor.fit(
        design_vectors=design_vectors,
        metric_distributions=metric_distributions,
        metric_name="distance",
    )

    # Save results
    pred_dir = exp_dir / "predictor"
    pred_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_classes": len(metric_distributions),
        "prior_mean_range": list(cfg.validation.prior_mean_range),
        "w_under": cfg.validation.w_under,
        "w_over": cfg.validation.w_over,
    }
    with open(pred_dir / "predictor_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Safety predictor training complete!")
    return summary


def run_pipeline(cfg: ExperimentConfig) -> None:
    """Run the full SOTIF-LLM pipeline."""
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(cfg.experiment_dir / "config.json", "w") as f:
        json.dump({
            "name": cfg.name,
            "model_id": cfg.model.model_id,
            "sae_release": cfg.model.sae_release,
            "sae_id": cfg.model.sae_id,
            "adversary_model": cfg.adversary.model_id,
            "judge_model": cfg.adversary.judge_model_id,
            "max_rounds": cfg.adversary.max_rounds,
            "envelope_method": cfg.envelope.method,
            "prior_mean_range": list(cfg.validation.prior_mean_range),
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("SOTIF-LLM Pipeline: Mechanistic ODD for LLM Safety")
    logger.info("=" * 60)

    # Phase 1: Safe Baseline
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Establishing Safe Baseline (ODD)")
    logger.info("=" * 60)
    p1 = run_phase1(cfg)

    # Phase 2: Adversarial Jailbreak Detection
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Adversarial Jailbreak Detection (Red-Team + SAE)")
    logger.info("=" * 60)
    p2 = run_phase2(cfg)

    # Phase 3: Safety Predictor (Quantile GP with Unsafe Prior)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Training Safety Predictor (GP + Unsafe Prior)")
    logger.info("=" * 60)
    p3 = run_safety_predictor(cfg)

    # Final summary
    final = {"phase1": p1, "phase2": p2, "predictor": p3}
    with open(cfg.experiment_dir / "pipeline_summary.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete! Results in: " + str(cfg.experiment_dir))
    logger.info("=" * 60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = ExperimentConfig.from_cli()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
