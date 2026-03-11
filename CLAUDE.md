# SOTIF-LLM: Mechanistic ODD for LLM Safety

## Quick Reference

- **Language**: Python 3.11+, managed with Pixi
- **Package**: `src/sotif_llm/` (src layout)
- **Tests**: `python -m pytest tests/ -v` (run from project root, needs `src` on path)
- **Generate prompts**: `python -m sotif_llm.prompts.generator` (from `src/`)

## Architecture

```
src/sotif_llm/
├── config.py            # Dataclass configs (ExperimentConfig, ModelConfig, etc.)
├── prompts/             # Prompt database (THE key creative piece)
│   ├── taxonomy.py      # Design space definition (6D continuous)
│   ├── primitives.py    # Atomic building blocks (templates, topics, code snippets)
│   └── generator.py     # Compositional generator engine
├── sae/
│   └── extractor.py     # SAE feature extraction (SAELens + HuggingFace)
├── envelope/
│   ├── baseline.py      # Safe baseline computation (Mahalanobis, IsolationForest, KDE)
│   └── distance.py      # Anomaly distance metrics
├── validation/
│   ├── metrics.py       # Nested uncertainty propagation (Monte Carlo)
│   └── predictor.py     # Quantile GP safety predictor (the MVP analog)
├── experiments/
│   ├── phase1_baseline.py       # Phase 1: Safe ODD
│   ├── phase2_reward_hacking.py # Phase 2: Silent Killers
│   └── phase3_jailbreaks.py     # Phase 3: REDKWEEN
├── visualization.py     # Trust region plots, generation traces
└── pipeline.py          # End-to-end orchestration
```

## Key Concepts (mapping from error_predictor paper)

| Wind Turbine Paper | This Project |
|---|---|
| Turbine design params z | Prompt design vector z (6D: complexity, sensitivity, etc.) |
| Measurement campaign | Evaluation campaign (cohort of ~20 related prompts) |
| Validation metric ε | SAE anomaly distance from safe baseline |
| MVP (quantile GP) | Safety Predictor (quantile GP over prompt space) |
| Trust regions | Safe / probably safe / possibly safe / dangerous |

## Model

Default target: `meta-llama/Llama-3.1-8B-Instruct` with SAELens SAEs.
