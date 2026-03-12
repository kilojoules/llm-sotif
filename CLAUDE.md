# SOTIF-LLM: Mechanistic ODD for LLM Safety

## Quick Reference

- **Language**: Python 3.11+, managed with Pixi
- **Package**: `src/sotif_llm/` (src layout)
- **Tests**: `python -m pytest tests/ -v` (run from project root, needs `src` on path)
- **Run pipeline**: `python -m sotif_llm.pipeline`

## Architecture

```
src/sotif_llm/
├── config.py            # Dataclass configs (ExperimentConfig, AdversaryConfig, GCGConfig, LoRAConfig, etc.)
├── adversary/           # Red-team adversarial loop + GCG + LoRA
│   ├── prompts.py       # Seed jailbreak + benign prompts
│   ├── gcg.py           # GCG suffix optimization (gradient-based jailbreaks)
│   ├── lora_trainer.py  # LoRA fine-tuning for adversary (jailbreak rewriting techniques)
│   ├── red_team.py      # Adversary/target loop (Llama 3B) with LoRA + GCG warm-starts
│   └── judge.py         # LLM-based safety judge (1B/3B)
├── prompts/             # Prompt database (compositional generation)
│   ├── taxonomy.py      # Design space definition (6D continuous)
│   ├── primitives.py    # Atomic building blocks
│   └── generator.py     # Compositional generator engine
├── sae/
│   └── extractor.py     # SAE feature extraction (SAELens + HuggingFace)
├── envelope/
│   ├── baseline.py      # Safe baseline computation (Mahalanobis, IF, KDE)
│   └── distance.py      # Anomaly distance metrics
├── validation/
│   ├── metrics.py       # Nested uncertainty propagation (Monte Carlo)
│   └── predictor.py     # Quantile GP safety predictor (unsafe prior)
├── experiments/
│   ├── phase1_baseline.py       # Phase 1: Safe ODD
│   └── phase2_adversarial.py    # Phase 2: GCG + Red-team + SAE jailbreak detection
├── visualization.py     # Trust region plots, generation traces
└── pipeline.py          # End-to-end orchestration (Phase 1 → 2 → GP)
```

## Pipeline

1. **Phase 1**: Safe baseline — benign prompts → SAE features → SOTIF envelope
2. **Phase 2**: Adversarial jailbreaks
   - **Phase 2a-gcg** (optional): GCG suffix optimization finds gradient-based
     adversarial suffixes for each seed prompt, producing verified jailbreaks.
     Uses `nanogcg` (Zou et al., 2023). Disable with `--no-gcg`.
   - **Phase 2a**: Red-team loop — LoRA-enhanced adversary (3B) strengthens
     attacks against target (3B), judge classifies. The adversary is fine-tuned
     on jailbreak rewriting techniques (persona injection, hypothetical framing,
     etc.) via a lightweight LoRA adapter toggled on/off between roles. When GCG
     warm-starts are available, round 0 begins from a known-working jailbreak.
     Disable LoRA with `--no-lora`.
   - **Phase 2b**: SAE feature extraction → 3 response classes
     (benign, refused, jailbroken) → anomaly analysis
3. **Phase 3**: Safety predictor — quantile GP with large unsafe prior over
   SAE feature space, trained on labeled data from Phase 2

## Models

- **SAE target**: `meta-llama/Llama-3.1-8B-Instruct` with SAELens SAEs
- **Adversary/target**: `meta-llama/Llama-3.2-3B-Instruct` (same model, different roles)
- **Judge**: `meta-llama/Llama-3.2-3B-Instruct` (or 1B)

## GCG (Greedy Coordinate Gradient)

GCG uses gradient information from the target model to find token-level
adversarial suffixes that maximize the probability of harmful compliance.
The suffixes are nonsensical text that reliably bypasses safety training.

- Requires `nanogcg` (`pip install nanogcg`)
- Needs fp16 model weights (no 4-bit quantization) for gradient computation
- Default: 250 steps, search_width=512, topk=256
- Configure via `GCGConfig` or CLI: `--gcg-steps 500`, `--no-gcg`

## LoRA Adversary Fine-Tuning

The adversary model is fine-tuned with a LoRA adapter on ~25 curated
jailbreak rewriting examples covering 12+ attack techniques (persona
injection, continuation, multi-step, ethical inversion, etc.).  The
adapter is toggled on for adversary generation and off for target/judge
generation (single model instance, no extra memory).  Round 0 also
uses the LoRA adversary rewrite (not the raw seed).

- Uses `peft` (already a dependency)
- Default: rank=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 10 epochs
- Training takes seconds (~25 short examples)
- Configure via `LoRAConfig` or CLI: `--lora-rank 16`, `--lora-epochs 5`, `--no-lora`

## Unsafe Behavior Categories

- Bad coding advice (destructive commands)
- Bad safety advice (physical harm)
- Illegal activity (drug synthesis)
- Bogus medical advice (unauthorized diagnosis/prescription)
