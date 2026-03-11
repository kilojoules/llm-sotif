# SOTIF-LLM: Mechanistic ODD for LLM Safety

## Quick Reference

- **Language**: Python 3.11+, managed with Pixi
- **Package**: `src/sotif_llm/` (src layout)
- **Tests**: `python -m pytest tests/ -v` (run from project root, needs `src` on path)
- **Run pipeline**: `python -m sotif_llm.pipeline`

## Architecture

```
src/sotif_llm/
├── config.py            # Dataclass configs (ExperimentConfig, AdversaryConfig, etc.)
├── adversary/           # Red-team adversarial loop
│   ├── prompts.py       # Seed jailbreak + benign prompts
│   ├── red_team.py      # Adversary/target loop (Llama 3B)
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
│   └── phase2_adversarial.py    # Phase 2: Red-team + SAE jailbreak detection
├── visualization.py     # Trust region plots, generation traces
└── pipeline.py          # End-to-end orchestration (Phase 1 → 2 → GP)
```

## Pipeline

1. **Phase 1**: Safe baseline — benign prompts → SAE features → SOTIF envelope
2. **Phase 2**: Adversarial jailbreaks — red-team loop (adversary strengthens
   attacks against target, judge classifies) → 3 response classes
   (benign, refused, jailbroken) → SAE feature extraction → anomaly analysis
3. **Phase 3**: Safety predictor — quantile GP with large unsafe prior over
   SAE feature space, trained on labeled data from Phase 2

## Models

- **SAE target**: `meta-llama/Llama-3.1-8B-Instruct` with SAELens SAEs
- **Adversary/target**: `meta-llama/Llama-3.2-3B-Instruct` (same model, different roles)
- **Judge**: `meta-llama/Llama-3.2-3B-Instruct` (or 1B)

## Unsafe Behavior Categories

- Bad coding advice (destructive commands)
- Bad safety advice (physical harm)
- Illegal activity (drug synthesis)
- Bogus medical advice (unauthorized diagnosis/prescription)
