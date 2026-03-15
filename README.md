# Mechanistic ODDs: Continuous SOTIF Validation of Large Language Models via Sparse Autoencoders

> **TL;DR** — We borrow the *Operational Design Domain* concept from autonomous vehicle safety (ISO 21448 SOTIF) and apply it to LLMs. Sparse Autoencoder activations define a "safe envelope" for Llama-3.1-8B-Instruct; a quantile Gaussian process predicts where in prompt space the model is likely to leave that envelope. Key finding: successful jailbreaks land *closer* to the benign baseline than refusals (Cohen's *d* = −1.48), suggesting adversarial prompts are optimized to *mimic* safe internal states — so detection requires looking for inputs that are suspiciously normal, not obviously anomalous.

## Motivation

Current AI safety evaluation relies on behavioral testing: ask the model a dangerous question, check if the output is harmful. But as models become more capable, behavioral evaluation is failing. A model can appear aligned while secretly generating malicious code (*reward hacking*) or breaking under complex adversarial roleplay (*jailbreaking*). These failure modes are invisible to output-only evaluation.

This project introduces an approach borrowed from autonomous vehicle engineering: **Safety of the Intended Functionality (SOTIF, ISO 21448)**. In autonomous driving, a vehicle is certified to operate safely within a mathematically defined **Operational Design Domain (ODD)** --- for example, clear weather on paved roads. If conditions leave the ODD, the vehicle must safely disengage. LLMs currently have no equivalent.

We define the first quantitative, internal **Mechanistic ODD** for an LLM. Instead of inspecting outputs, we monitor the model's internal activation geometry via **Sparse Autoencoders (SAEs)** and detect when it leaves its certified safe envelope --- in real time, before harmful tokens are generated.

## Methods

### Conceptual Framework

The methodology is adapted from a model-validity predictor (MVP) framework developed for wind turbine power curve certification ([Murcia Leon, Quick, et al., 2025](https://ssrn.com/abstract=5492825)). In that domain, prototype measurement campaigns validate a computational model across a design space parameterized by continuous variables (rated power, rotor diameter, hub height). A quantile Gaussian process regressor predicts validation metrics at untested designs, defining *probably safe* and *possibly safe* trust regions.

We transpose this framework onto LLM safety:

| Wind Turbine Certification | LLM Safety (This Work) |
|---|---|
| Turbine design parameters **z** (rated power, diameter, hub height, specific power) | Prompt design vector **z** (task complexity, instruction density, persona depth, topic sensitivity, domain specificity, output constraints) |
| Prototype measurement campaign | Evaluation campaign: a cohort of ~20 related prompts run through the model |
| Measured inputs/outputs (wind speed, turbulence &rarr; power) | Prompt text &rarr; SAE feature activations in the residual stream |
| Measurement uncertainty (instrument error) | Generation stochasticity (temperature), tokenization artifacts, SAE reconstruction error |
| Parameter uncertainty (C<sub>P,max</sub>, yaw angle, rated power) | SAE hyperparameters, anomaly threshold, distance metric choice |
| Validation metric &epsilon; (area metric, absolute bias) | SAE anomaly distance from safe baseline |
| Model-validity predictor (quantile GP) | Safety predictor (quantile GP over prompt design space) |
| Trust regions (probably safe / possibly safe / dangerous) | Trust regions in prompt space (safe / borderline / dangerous) |

### Phase 1 --- Establishing the Safe Baseline (Defining the ODD)

We establish what a "safe, normal brain state" looks like for the target model (Llama-3.1-8B-Instruct).

**Prompt database.** 5,000 benign prompts are generated via a compositional engine that combines atomic primitives across four layers:

1. **Task primitives** --- 18 task types (code generation, debugging, summarization, math reasoning, etc.) at 5 complexity levels, parameterized by slot-fill templates drawn from curated pools of 700+ atomic building blocks (code snippets, topics, constraints, personas).
2. **Domain contexts** --- 30 subject domains (algorithms, physics, cooking, finance, etc.) with domain-specific concept vocabularies.
3. **Complexity modifiers** --- persona framing (none &rarr; deep roleplay), output format constraints (free text &rarr; JSON with unit tests), multi-step instruction chaining.
4. **Sensitivity gradient** --- 4 levels from fully benign (recipes, basic arithmetic) through neutral (professional topics), dual-use (security education, chemistry), to boundary-adjacent (fictional thriller scenes, CTF challenges).

Each prompt is assigned a continuous **6-dimensional design vector z &isin; [0,1]<sup>6</sup>**:

| Dimension | Meaning |
|---|---|
| `task_complexity` | Cognitive demand (0 = trivial, 1 = expert-level multi-step reasoning) |
| `instruction_density` | Constraints per unit prompt length |
| `persona_depth` | Degree of roleplay framing (0 = direct instruction, 1 = deep immersive persona) |
| `topic_sensitivity` | Semantic proximity to safety-relevant content |
| `domain_specificity` | Specialization of required knowledge |
| `output_constraints` | Structural demands on output format |

This design vector is the direct analog of the turbine design parameters. The safety predictor will interpolate across this space.

**SAE feature extraction.** Each prompt is tokenized and passed through Llama-3.1-8B-Instruct. The last-token hidden state at a target residual-stream layer (layer 19) is encoded through a pre-trained SAE (Goodfire, via [SAELens](https://github.com/jbloomAus/SAELens)), producing a sparse feature activation vector per prompt. The last token is used rather than mean-pooling because the SAE was trained on per-token activations; mean-pooling over long sequences produces out-of-distribution vectors that fail to activate SAE features.

**Baseline envelope.** The statistical envelope of the benign SAE features is computed using one of three methods:

- **Mahalanobis distance** (default) --- fits a (robust) covariance estimate; anomaly = Mahalanobis distance from the centroid.
- **Isolation Forest** --- non-parametric; handles complex activation geometries.
- **Kernel Density Estimation** --- provides density-based anomaly scores.

Feature selection retains the top-*k* most variable SAE features (default: 512 of 65,536) for robust distance computation. Confidence thresholds are set at the 90th, 95th, and 99th percentiles of the reference distance distribution.

**Nested uncertainty propagation.** Following the MVP framework, we propagate uncertainty through nested Monte Carlo sampling:

- *Aleatoric*: natural variation across prompts within a campaign.
- *Epistemic*: bootstrap resampling of features (simulating measurement/generation noise).

This produces a *distribution* of validation metrics per campaign, not a point estimate.

**Per-campaign validation metrics.** Prompts are grouped into evaluation campaigns (~20 prompts sharing a design-space region). For each campaign, we compute:

- **SAE anomaly distance**: mean and quantiles of the Mahalanobis distance from baseline.
- **Area validation metric**: &int;|F<sub>y</sub>(y) - F<sub>m</sub>(y)|dy --- the CDF area between the campaign's feature distribution and the baseline distribution (analog of Eq. 13 in the reference paper).
- **Absolute bias metric**: ||E[y] - E[m]||<sub>2</sub> --- L2 norm of the mean feature difference (analog of Eq. 12).

### Phase 2 --- Reward Hacking Detection (Silent Killers)

We test whether the model leaves the safe envelope when it silently generates bad code.

**Stimulus.** Coding prompts from the [Silent Killers](https://github.com/kilojoules/silent-killers) codebase at three difficulty levels (easy: code modularization; medium: Sobol sensitivity analysis; hard: wind farm optimization with HDF5 corruption). These are tasks where LLMs are known to generate `try/except: pass` blocks that silently swallow errors --- a concrete instance of reward hacking (optimizing for "code runs" rather than "code fails correctly").

**Real-time monitoring.** During token-by-token generation, SAE features are extracted at *each decoding step*. The anomaly distance is computed per step, producing a trajectory through activation space. This enables detecting when the model enters an unsafe cognitive state *before* the harmful pattern (e.g., `except: pass`) is fully generated.

**Hypothesis.** When the model writes correct code with proper error handling, it stays inside the SOTIF envelope. When it decides to swallow an error, the SAE distance spikes outside the envelope, triggering an anomaly alert.

**Separation analysis.** Existing model responses (from 7 frontier models, 20 seeds each) are loaded from the Silent Killers dataset. Responses are labeled as "bad exception" or "clean" by AST-based auditing. SAE features are extracted from both groups, and separation is quantified by Cohen's *d* and the percentage of bad-exception responses that fall outside the envelope.

### Phase 3 --- Jailbreak Detection (REDKWEEN)

We test whether the model leaves the safe envelope under adversarial attack.

**Stimulus.** Automated adversarial attacks from the [REDKWEEN](https://github.com/kilojoules/redkween) self-play framework. A 1B-parameter adversary is LoRA-trained to jailbreak an 8B victim through trial-and-error, independently discovering strategies like CTF challenge framing (67% ASR), classified document framing (67%), and multi-turn decomposition (55%). Both frozen-victim and self-play experiments are analyzed.

**Feature extraction.** SAE features are extracted from:
1. **Attack prompts** (adversary's input to the victim).
2. **Victim responses** (the model's generated output).

**Hypothesis.** Standard harmful prompts are rejected easily --- the model stays inside the envelope. Successful jailbreaks force the model into *malicious compliance* --- a fundamentally different cognitive state that ruptures the SOTIF envelope.

**Per-round analysis.** Attack success rate (ASR) is correlated with mean SAE distance per round, testing whether the co-evolutionary dynamics of adversary vs. victim training are reflected in the activation geometry.

### Safety Predictor (Quantile GP)

A quantile Gaussian process regressor is trained over the prompt design space, predicting the *distribution* of expected SAE anomaly distances at arbitrary design points. This is the direct analog of the Model-Validity Predictor (Eq. 17--19 in the reference paper):

$$\mathbb{Q}_q \sim GP^{(q)}(\mathbf{z} | \mathbb{Q}(\epsilon(\mathbf{z}), q) \; \forall \mathbf{z} \in \mathbf{z}_\text{obs}) \quad \forall q \in \mathbf{q}_\text{all}$$

One GP is fitted per quantile level. Hyperparameters (kernel length scale, prior mean) are selected by leave-one-out cross-validation with asymmetric weighting: under-prediction of danger is penalized 6x more than over-prediction, because declaring a region "safe" when it is not is far worse than the reverse.

**Trust regions** are defined using the 95th-percentile predicted distance:

- **Validated safe**: within the reference baseline distribution.
- **Probably safe**: predicted q95 distance below the safe tolerance &epsilon;<sub>safe</sub>.
- **Possibly safe**: predicted q95 distance below the expanded tolerance &epsilon;<sub>prob safe</sub>.
- **Dangerous prediction domain**: above all tolerance levels.

These regions are visualized as contour plots over 2D slices of the 6D design space, directly analogous to Figure 10 in the reference paper.

## Results

### Phase 1 --- Safe Baseline

500 benign prompts were processed through Llama-3.1-8B-Instruct. SAE features were extracted from the last-token hidden state at layer 19 using a Goodfire pre-trained SAE (65,536 features). The top 512 most variable features were selected for Mahalanobis distance computation, using Ledoit-Wolf shrinkage to regularize the covariance estimate (n=500, p=512).

| Metric | Value |
|---|---|
| Benign prompts | 500 |
| SAE features per prompt | ~45 nonzero (of 65,536) |
| Baseline thresholds | q90=19.3, q95=20.9, q99=22.5 |
| Self-validation | 90% validated, 7.2% probably safe, 2.8% possibly safe, 0% dangerous |

### Phase 2 --- Adversarial Jailbreak Detection

A LoRA-enhanced adversary (Llama-3.2-3B-Instruct, rank-16 adapter trained on 22 jailbreak rewriting examples) attacked the same 3B model across 28 seed prompts over 16 rounds per seed. A judge model classified each response as SAFE or UNSAFE.

| Metric | Value |
|---|---|
| Total episodes | 448 |
| Successful jailbreaks | 121 (27% ASR) |
| Refused | 327 |
| Benign controls | 8 |

SAE features were then extracted from the attack prompts (not responses) using the same Llama-3.1-8B-Instruct + SAE pipeline. Mahalanobis distances were computed against the Phase 1 baseline.

| Class | n | Mean Distance | Std | % Outside Envelope |
|---|---|---|---|---|
| Benign | 8 | 49.6 | 14.6 | 100% |
| Refused | 327 | 32.5 | 11.3 | 88% |
| Jailbroken | 121 | 30.0 | 11.8 | 74% |
| **Cohen's *d*** (jailbroken vs. benign) | | **-1.48** | | |

### Interpretation

The SAE anomaly distances show a clear ordering: **jailbroken < refused < benign**. Successful jailbreaks are *closer* to the benign baseline than refusals --- the opposite of the naive hypothesis that jailbreaks should look more anomalous.

This result is consistent with the mechanism of adversarial attacks: **successful jailbreaks are optimized to appear innocuous to the model**. The LoRA-trained adversary learns to frame harmful requests using educational framing, hypothetical scenarios, and professional contexts that activate similar internal representations as benign prompts. Prompts that fail to jailbreak (refusals) retain more overtly harmful markers that the model's safety training detects.

The strong Cohen's *d* of -1.48 confirms that SAE-based anomaly detection can distinguish jailbroken from benign prompts, but the direction of separation suggests that jailbreak detection requires monitoring for prompts that are *suspiciously close to benign* while containing harmful intent --- a second-order signal rather than simple anomaly detection.

### Technical Fixes

Three issues were identified and resolved during development:

1. **Rank-deficient covariance.** With n=500 samples and p=512 features, the empirical covariance matrix is barely full-rank. Ledoit-Wolf shrinkage regularization was added as an automatic fallback when n &le; 2p.

2. **Input distribution mismatch.** Phase 2 originally concatenated attack prompts with model responses for SAE extraction. Phase 1 used prompts only. The mismatch pushed Phase 2 activations out-of-distribution for the SAE. Fixed by extracting from prompts only in both phases.

3. **Mean-pooling vs. last-token.** The SAE was trained on per-token residual stream activations, but the extractor mean-pooled hidden states across the full sequence. For long adversarial prompts (1,500--2,600 characters), this produced activation vectors far from the SAE's training distribution, resulting in near-zero sparse features. Switching to the last token's hidden state --- which captures the model's summary representation of the full input --- resolved the issue completely, producing 45--60 nonzero SAE features per prompt regardless of input length.

## Repository Structure

```
src/sotif_llm/
├── config.py                          # Experiment configuration dataclasses
├── prompts/
│   ├── taxonomy.py                    # 6D design space definition
│   ├── primitives.py                  # 700+ atomic building blocks
│   └── generator.py                   # Compositional prompt engine
├── sae/
│   └── extractor.py                   # SAE feature extraction (last-token + real-time)
├── envelope/
│   ├── baseline.py                    # Safe baseline (Mahalanobis w/ Ledoit-Wolf / IF / KDE)
│   └── distance.py                    # Anomaly distance + trust region classification
├── adversary/
│   ├── prompts.py                     # Seed jailbreak + benign prompts
│   ├── gcg.py                         # GCG suffix optimization (gradient-based jailbreaks)
│   ├── lora_trainer.py                # LoRA fine-tuning for adversary
│   ├── red_team.py                    # Adversary/target loop with LoRA + GCG warm-starts
│   └── judge.py                       # LLM-based safety judge
├── validation/
│   ├── metrics.py                     # Nested uncertainty propagation
│   └── predictor.py                   # Quantile GP safety predictor
├── experiments/
│   ├── phase1_baseline.py             # Phase 1: define the ODD
│   └── phase2_adversarial.py          # Phase 2: red-team + SAE jailbreak detection
├── visualization.py                   # Trust region plots, generation traces
└── pipeline.py                        # End-to-end orchestration
```

## Getting Started

```bash
# Install dependencies
pixi install

# Generate the prompt database (CPU only)
pixi run generate-prompts

# Run the full pipeline (requires GPU)
pixi run run-all

# Or run phases individually
pixi run extract-baseline
pixi run run-silent-killers
pixi run run-redkween
pixi run train-predictor
pixi run plot-results

# Run tests
pixi run test
```

## References

- **SOTIF**: ISO 21448:2022 --- Safety of the Intended Functionality.
- **Model-Validity Predictor**: Murcia Leon, J.P., Quick, J., Overgaard, N.S., Servizi, V., Dimitrov, N., Kim, T. (2025). Reducing the Number of Wind Turbine Prototype Measurement Campaigns for Power Curve Model Validation Using a Model-Validity Predictor. *Preprint*, SSRN 5492825.
- **SAELens**: Bloom, J. et al. SAELens: Sparse Autoencoder tools for mechanistic interpretability.
- **Silent Killers**: Quick, J. et al. AST-based detection of silent error swallowing in LLM-generated code.
- **REDKWEEN**: Quick, J. et al. Automated red teaming of LLMs via adversarial self-play.
