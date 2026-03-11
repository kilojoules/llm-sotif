"""End-to-end SOTIF-LLM pipeline.

Orchestrates all phases:
  Phase 1: Safe baseline (defining the Mechanistic ODD)
  Phase 2: Silent Killers (reward hacking detection)
  Phase 3: REDKWEEN (jailbreak detection)
  Phase 4: Safety predictor training and trust region visualization

Usage:
    python -m sotif_llm.pipeline
    python -m sotif_llm.pipeline --name my_experiment --model-id meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .experiments.phase1_baseline import run_phase1
from .experiments.phase2_reward_hacking import run_phase2
from .experiments.phase3_jailbreaks import run_phase3
from .validation.predictor import SafetyPredictor

logger = logging.getLogger(__name__)


def run_safety_predictor(cfg: ExperimentConfig) -> dict:
    """Train the safety predictor (quantile GP) from Phase 1-3 results.

    This is the final integration step — building the MVP analog that
    predicts safety across the prompt design space, defining trust regions.
    """
    exp_dir = cfg.experiment_dir

    # Load campaign metrics from Phase 1
    metrics_path = exp_dir / "phase1" / "metrics" / "campaign_metrics.json"
    with open(metrics_path) as f:
        campaigns = json.load(f)

    design_vectors = np.array([c["design_vector"] for c in campaigns])

    # Use area metric distributions (would need full distributions saved;
    # for now use point estimates as mock distributions with noise)
    rng = np.random.default_rng(42)
    metric_distributions = []
    for c in campaigns:
        base = c["mean_area"]
        # Simulate epistemic uncertainty distribution
        dist = rng.normal(base, base * 0.15 + 1e-4, size=cfg.n_epistemic)
        dist = np.maximum(dist, 0)
        metric_distributions.append(dist)

    # Train safety predictor
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

    logger.info("Training safety predictor...")
    predictor.fit(
        design_vectors=design_vectors,
        metric_distributions=metric_distributions,
        metric_name="area",
    )

    # Generate trust region predictions on a grid
    # Plot over pairs of most interesting design dimensions:
    # (task_complexity, topic_sensitivity) and (instruction_density, persona_depth)
    dim_names = [
        "task_complexity", "instruction_density", "persona_depth",
        "topic_sensitivity", "domain_specificity", "output_constraints",
    ]

    grid_results = {}
    dim_pairs = [(0, 3), (1, 2), (4, 5)]  # Interesting 2D slices
    for d1, d2 in dim_pairs:
        X1, X2, result = predictor.predict_grid(dim1=d1, dim2=d2, n_grid=50)
        grid_results[f"{dim_names[d1]}_vs_{dim_names[d2]}"] = {
            "dim1": dim_names[d1],
            "dim2": dim_names[d2],
            "trust_labels": result.trust_labels.reshape(50, 50).tolist(),
            "q95_values": result.predicted_quantiles.get(0.95, np.zeros(2500)).reshape(50, 50).tolist(),
            "mean_values": result.predicted_mean.reshape(50, 50).tolist(),
        }

    pred_dir = exp_dir / "predictor"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "grid_predictions.json", "w") as f:
        json.dump(grid_results, f)

    summary = {
        "n_campaigns": len(campaigns),
        "n_design_dims": design_vectors.shape[1],
        "grid_slices": list(grid_results.keys()),
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
            "n_benign": cfg.prompts.n_benign,
            "envelope_method": cfg.envelope.method,
            "confidence_level": cfg.envelope.confidence_level,
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("SOTIF-LLM Pipeline: Mechanistic ODD for LLM Safety")
    logger.info("=" * 60)

    # Phase 1: Safe Baseline
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Establishing Safe Baseline (ODD)")
    logger.info("=" * 60)
    p1 = run_phase1(cfg)

    # Phase 2: Reward Hacking
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Silent Killers (Reward Hacking Detection)")
    logger.info("=" * 60)
    p2 = run_phase2(cfg)

    # Phase 3: Jailbreaks
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: REDKWEEN (Jailbreak Detection)")
    logger.info("=" * 60)
    p3 = run_phase3(cfg)

    # Phase 4: Safety Predictor
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Training Safety Predictor (Quantile GP)")
    logger.info("=" * 60)
    p4 = run_safety_predictor(cfg)

    # Final summary
    final = {"phase1": p1, "phase2": p2, "phase3": p3, "predictor": p4}
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
