"""Phase 2: Adversarial Jailbreak Detection via SAE Features.

Replaces the Silent Killers (reward hacking) phase with a direct
adversarial evaluation:

  1. (Optional) Run GCG suffix optimization to find verified
     jailbreak prompts for each seed.  These gradient-optimised
     adversarial suffixes bypass safety alignment reliably.
  2. Run a red-team loop: adversary (3B) strengthens jailbreak prompts
     against a target (3B), classified by a judge (1B/3B).  When GCG
     warm-starts are available, round 0 begins from a known-working
     attack instead of the raw seed prompt.
  3. Collect three response classes:
       - Benign responses (safe baseline)
       - Jailbreak refusals (model says no)
       - Successful jailbreaks (harmful compliance)
  4. Extract SAE features from the target model's activations for all
     three classes.
  5. Train a GP safety predictor with a large unsafe prior over the
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

from ..adversary.gcg import GCGRunner
from ..adversary.red_team import RedTeamResult, RedTeamRunner
from ..config import ExperimentConfig
from ..envelope.baseline import BaselineComputer, SafeBaseline
from ..envelope.distance import compute_distance, classify_region
from ..sae.extractor import SAEExtractor

logger = logging.getLogger(__name__)


def run_gcg_campaign(cfg: ExperimentConfig) -> dict[str, str]:
    """Run GCG suffix optimization and return warm-start prompts.

    Returns a mapping ``{seed_id: gcg_full_prompt}`` for seeds where
    GCG found a verified jailbreak.  Results are saved to
    ``<experiment_dir>/phase2/gcg_results.json``.
    """
    from ..adversary.prompts import JAILBREAK_PROMPTS

    exp_dir = cfg.experiment_dir / "phase2"
    exp_dir.mkdir(parents=True, exist_ok=True)

    gcg_cfg = cfg.adversary.gcg
    runner = GCGRunner(
        model_id=cfg.adversary.model_id,
        device=cfg.model.device,
        num_steps=gcg_cfg.num_steps,
        search_width=gcg_cfg.search_width,
        topk=gcg_cfg.topk,
        seed=gcg_cfg.seed,
    )

    campaign = runner.run(JAILBREAK_PROMPTS)
    campaign.save(exp_dir / "gcg_results.json")
    runner.unload()

    warm_starts = campaign.get_warm_starts()
    logger.info(
        f"GCG produced {len(warm_starts)}/{len(JAILBREAK_PROMPTS)} "
        f"verified warm-starts"
    )
    return warm_starts


def run_red_team_campaign(
    cfg: ExperimentConfig,
    gcg_warm_starts: dict[str, str] | None = None,
) -> RedTeamResult:
    """Run the adversarial red-team campaign and save results.

    If *gcg_warm_starts* is provided, the adversary loop uses
    GCG-optimized prompts for round 0 instead of the raw seeds.
    """
    exp_dir = cfg.experiment_dir / "phase2"
    exp_dir.mkdir(parents=True, exist_ok=True)

    lora_config = cfg.adversary.lora if cfg.adversary.lora.enabled else None
    lora_adapter_path = (
        cfg.experiment_dir / "phase2" / "lora_adapter"
        if lora_config is not None else None
    )

    runner = RedTeamRunner(
        model_id=cfg.adversary.model_id,
        judge_model_id=cfg.adversary.judge_model_id,
        device=cfg.model.device,
        load_in_4bit=cfg.adversary.load_in_4bit,
        max_rounds=cfg.adversary.max_rounds,
        max_new_tokens=cfg.adversary.max_new_tokens,
        temperature=cfg.adversary.temperature,
        lora_config=lora_config,
        lora_adapter_path=lora_adapter_path,
    )

    result = runner.run(gcg_warm_starts=gcg_warm_starts)
    result.save(exp_dir / "red_team_results.json")
    runner.unload()
    return result


def extract_and_classify(cfg: ExperimentConfig) -> dict:
    """Extract SAE features and build the safety dataset.

    Loads the red-team results, extracts SAE features from the target
    model for each response class, computes anomaly distances, and
    assembles the labeled dataset for GP training.

    Distance computation uses the Phase 1 baseline (fitted on 500+
    benign prompts) rather than a separate baseline from the small
    Phase 2 benign set.  This avoids rank-deficient covariance when
    only a handful of benign episodes exist.
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

    # Collect attack prompts for SAE extraction (prompt only, matching Phase 1)
    # The SAE baseline is fitted on prompt-only activations in Phase 1, so
    # Phase 2 must also extract from prompts to stay in-distribution.
    all_texts = []
    all_ids = []
    all_labels = []  # 0=benign, 1=refused, 2=jailbroken

    for i, ep in enumerate(benign):
        all_texts.append(ep.attack)
        all_ids.append(f"benign_{i}")
        all_labels.append(0)

    for i, ep in enumerate(refused):
        all_texts.append(ep.attack)
        all_ids.append(f"refused_{i}")
        all_labels.append(1)

    for i, ep in enumerate(jailbroken):
        all_texts.append(ep.attack)
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

    # --- Baseline selection ---
    # Prefer the Phase 1 baseline (fitted on 500+ benign prompts) for
    # well-conditioned Mahalanobis.  Fall back to fitting from Phase 2
    # benign data (with Ledoit-Wolf shrinkage) if Phase 1 is unavailable.
    phase1_baseline_path = cfg.experiment_dir / "phase1" / "baseline" / "safe_baseline.npz"
    if phase1_baseline_path.exists():
        logger.info("Using Phase 1 baseline for distance computation")
        baseline = SafeBaseline.load(phase1_baseline_path)
    else:
        logger.info("Phase 1 baseline not found — fitting from Phase 2 benign data")
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

    Step 1 (optional): GCG suffix optimization for warm-starts
    Step 2: Red-team campaign (adversary + target + judge)
    Step 3: SAE feature extraction + anomaly analysis
    """
    gcg_warm_starts = None

    if cfg.adversary.gcg.enabled:
        logger.info("Phase 2a-gcg: Running GCG suffix optimization...")
        gcg_warm_starts = run_gcg_campaign(cfg)
    else:
        logger.info("GCG warm-start disabled, using raw seed prompts.")

    logger.info("Phase 2a: Running red-team campaign...")
    rt_result = run_red_team_campaign(cfg, gcg_warm_starts=gcg_warm_starts)

    logger.info("Phase 2b: Extracting SAE features and building safety dataset...")
    summary = extract_and_classify(cfg)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = ExperimentConfig.from_cli()
    run_phase2(cfg)


if __name__ == "__main__":
    main()
