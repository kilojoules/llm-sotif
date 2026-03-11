# Mechanistic ODDs: Continuous SOTIF Validation of Large Language Models via Sparse Autoencoders

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

**SAE feature extraction.** Each prompt is tokenized and passed through Llama-3.1-8B-Instruct. Hidden states at a target residual-stream layer (default: layer 16) are mean-pooled across the sequence and encoded through a pre-trained SAE (via [SAELens](https://github.com/jbloomAus/SAELens)), producing a sparse feature activation vector per prompt.

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

## Expected Results

The following results are hypothesized and will be validated experimentally:

1. **The safe baseline is well-defined.** Benign prompts cluster tightly in SAE feature space. The Mahalanobis distance distribution is approximately chi-squared, and the 95th percentile provides a clean separation threshold. Self-validation should show >90% of benign prompts classified as "validated safe."

2. **Reward hacking is mechanistically detectable.** Code responses with bad exception handling (`try/except: pass`) will show significantly higher SAE anomaly distances than clean code responses (expected Cohen's *d* > 0.5). Real-time monitoring will show distance spikes *during* the generation of error-swallowing patterns, not just after.

3. **Jailbreaks rupture the envelope.** Successful jailbreak responses will fall outside the SOTIF envelope at significantly higher rates than refusals (expected >70% outside vs. <10%). The response-side separation (malicious compliance vs. refusal) will be stronger than the attack-side separation, because the victim's *cognitive state during compliance* is the mechanistically distinct phenomenon.

4. **Reward hacking and jailbreaking are mechanistically unified.** Both failure modes represent the model leaving its safe ODD to satisfy a localized proxy incentive. The safety predictor will assign both types of stimuli to overlapping dangerous regions of the design space.

5. **The safety predictor generalizes.** The quantile GP will predict elevated risk in high-sensitivity, high-persona-depth regions of design space even without direct observations, interpolating from nearby campaigns --- just as the wind turbine MVP predicts validation metrics at untested turbine designs.

## Repository Structure

```
src/sotif_llm/
├── config.py                          # Experiment configuration dataclasses
├── prompts/
│   ├── taxonomy.py                    # 6D design space definition
│   ├── primitives.py                  # 700+ atomic building blocks
│   └── generator.py                   # Compositional prompt engine
├── sae/
│   └── extractor.py                   # SAE feature extraction (batch + real-time)
├── envelope/
│   ├── baseline.py                    # Safe baseline (Mahalanobis / IF / KDE)
│   └── distance.py                    # Anomaly distance + trust region classification
├── validation/
│   ├── metrics.py                     # Nested uncertainty propagation
│   └── predictor.py                   # Quantile GP safety predictor
├── experiments/
│   ├── phase1_baseline.py             # Phase 1: define the ODD
│   ├── phase2_reward_hacking.py       # Phase 2: Silent Killers
│   └── phase3_jailbreaks.py           # Phase 3: REDKWEEN
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
