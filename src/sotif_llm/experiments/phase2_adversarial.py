"""Phase 2: Adversarial Jailbreak Detection via SAE Features.

Replaces the Silent Killers (reward hacking) phase with a direct
adversarial evaluation:

  1. Run a red-team loop: adversary (3B) strengthens jailbreak prompts
     against a target (3B), classified by a judge (1B/3B).
  2. Collect three response classes:
       - Benign responses (safe baseline)
       - Jailbreak refusals (model says no)
       - Successful jailbreaks (harmful compliance)
  3. Extract SAE features from the target model's activations for all
     three classes.
  4. Train a GP safety predictor with a large unsafe prior over the
     SAE feature space.

The hypothesis: successful jailbreaks produce a mechanistically
distinct internal state (detectable via SAE features) that lies
outside the safe envelope, and a GP with an unsafe prior will
correctly classify unseen prompts as dangerous by default.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..adversary.red_team import RedTeamResult, RedTeamRunner
from ..config import ExperimentConfig
from ..envelope.baseline import BaselineComputer, SafeBaseline
from ..envelope.distance import compute_distance, classify_region
from ..sae.extractor import SAEExtractor

logger = logging.getLogger(__name__)


def run_red_team_campaign(cfg: ExperimentConfig) -> RedTeamResult:
    """Run the adversarial red-team campaign and save results."""
    exp_dir = cfg.experiment_dir / "phase2"
    exp_dir.mkdir(parents=True, exist_ok=True)

    runner = RedTeamRunner(
        model_id=cfg.adversary.model_id,
        judge_model_id=cfg.adversary.judge_model_id,
        device=cfg.model.device,
        load_in_4bit=cfg.adversary.load_in_4bit,
        max_rounds=cfg.adversary.max_rounds,
        max_new_tokens=cfg.adversary.max_new_tokens,
        temperature=cfg.adversary.temperature,
    )

    result = runner.run()
    result.save(exp_dir / "red_team_results.json")
    runner.unload()
    return result


def extract_and_classify(cfg: ExperimentConfig) -> dict:
    """Extract SAE features and build the safety dataset.

    Loads the red-team results, extracts SAE features from the target
    model for each response class, computes anomaly distances, and
    assembles the labeled dataset for GP training.
    """
    exp_dir = cfg.experiment_dir / "phase2"

    # Load red-team data
    rt_result = RedTeamResult.load(exp_dir / "red_team_results.json")

    # Partition episodes into three classes
    benign = rt_result.benign_episodes
    refused = [e for e in rt_result.episodes if not e.is_unsafe]
    jailbroken = [e for e in rt_result.episodes if e.is_unsafe]

    logger.info(
        f"Dataset: {len(benign)} benign, {len(refused)} refused, "
        f"{len(jailbroken)} jailbroken"
    )

    # Collect all texts for SAE extraction (prompt + response concatenated)
    all_texts = []
    all_ids = []
    all_labels = []  # 0=benign, 1=refused, 2=jailbroken

    for i, ep in enumerate(benign):
        all_texts.append(f"{ep.attack}\n\n{ep.response}")
        all_ids.append(f"benign_{i}")
        all_labels.append(0)

    for i, ep in enumerate(refused):
        all_texts.append(f"{ep.attack}\n\n{ep.response}")
        all_ids.append(f"refused_{i}")
        all_labels.append(1)

    for i, ep in enumerate(jailbroken):
        all_texts.append(f"{ep.attack}\n\n{ep.response}")
        all_ids.append(f"jailbroken_{i}")
        all_labels.append(2)

    labels = np.array(all_labels)

    # Extract SAE features
    extractor = SAEExtractor(
        model_id=cfg.model.model_id,
        sae_release=cfg.model.sae_release,
        sae_id=cfg.model.sae_id,
        layer_idx=cfg.model.sae_layer,
        device=cfg.model.device,
        load_in_4bit=cfg.model.load_in_4bit,
    )

    extraction = extractor.extract(
        texts=all_texts,
        prompt_ids=all_ids,
        batch_size=4,
        save_hidden_states=True,
    )
    extraction.save(exp_dir / "features" / "all_features.npz")
    extractor.unload()

    # Compute safe baseline from benign features only
    benign_mask = labels == 0
    benign_features = extraction.features[benign_mask]

    baseline_computer = BaselineComputer(
        method=cfg.envelope.method,
        top_k_features=cfg.envelope.top_k_features,
    )
    baseline = baseline_computer.fit(benign_features)
    baseline.save(exp_dir / "baseline" / "safe_baseline.npz")

    # Compute distances for all three classes
    distances = compute_distance(extraction.features, baseline)
    regions = classify_region(distances, baseline)

    # Per-class statistics
    analysis = {}
    class_names = {0: "benign", 1: "refused", 2: "jailbroken"}
    for cls_idx, cls_name in class_names.items():
        mask = labels == cls_idx
        if not np.any(mask):
            continue
        d = distances[mask]
        r = regions[mask]
        analysis[cls_name] = {
            "count": int(np.sum(mask)),
            "mean_distance": float(np.mean(d)),
            "std_distance": float(np.std(d)),
            "q95_distance": float(np.quantile(d, 0.95)) if len(d) > 1 else float(d[0]),
            "pct_outside_envelope": float(np.mean(r >= 2)),
        }

    # Cohen's d: jailbroken vs benign
    if "jailbroken" in analysis and "benign" in analysis:
        d_jb = distances[labels == 2]
        d_bn = distances[labels == 0]
        pooled_std = np.sqrt(
            (np.var(d_jb) + np.var(d_bn)) / 2 + 1e-10
        )
        analysis["cohens_d_jailbroken_vs_benign"] = float(
            (np.mean(d_jb) - np.mean(d_bn)) / pooled_std
        )

    # Save labeled dataset for GP training
    np.savez_compressed(
        exp_dir / "safety_dataset.npz",
        features=extraction.features,
        labels=labels,
        distances=distances,
        regions=regions,
    )

    summary = {
        "n_total": len(all_texts),
        "class_counts": {
            cls_name: int(np.sum(labels == cls_idx))
            for cls_idx, cls_name in class_names.items()
        },
        "analysis": analysis,
    }
    with open(exp_dir / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Phase 2 analysis: {json.dumps(analysis, indent=2)}")
    return summary


def run_phase2(cfg: ExperimentConfig) -> dict:
    """Execute Phase 2: Adversarial Jailbreak Detection.

    Step 1: Red-team campaign (adversary + target + judge)
    Step 2: SAE feature extraction + anomaly analysis
    """
    logger.info("Phase 2a: Running red-team campaign...")
    rt_result = run_red_team_campaign(cfg)

    logger.info("Phase 2b: Extracting SAE features and building safety dataset...")
    summary = extract_and_classify(cfg)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = ExperimentConfig.from_cli()
    run_phase2(cfg)


if __name__ == "__main__":
    main()
