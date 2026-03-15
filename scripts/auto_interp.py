"""Auto-interpretability for top SAE features.

Loads Llama-3.1-8B-Instruct + Goodfire SAE (layer 19), runs a diverse
corpus through it, finds top-activating token contexts per feature,
then uses the model to generate one-line feature descriptions.

Output: auto_interp_results.json with feature descriptions + top examples.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Top 20 most discriminative SAE features (from Mahalanobis analysis)
TOP_FEATURES = [
    46011, 58699, 14767, 7962, 48703, 53314, 33380, 23820,
    14128, 59508, 19435, 65237, 32636, 55536, 50220, 12518,
    28598, 20518, 41237, 26440,
]

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "goodfire-llama-3.1-8b-instruct"
SAE_ID = "layer_19"
TARGET_LAYER = 19
N_CORPUS = 3000        # Number of diverse text samples to process
TOP_K_EXAMPLES = 30    # Top activating contexts per feature
CONTEXT_WINDOW = 80    # Characters of context around top-activating token
BATCH_SIZE = 4
MAX_SEQ_LEN = 256


def load_diverse_corpus(n: int = N_CORPUS) -> list[str]:
    """Load diverse text samples from C4."""
    logger.info(f"Loading {n} samples from C4...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        text = row["text"][:1200]  # Cap length
        if len(text) > 100:  # Skip very short
            texts.append(text)
    logger.info(f"Loaded {len(texts)} text samples")
    return texts


def extract_per_token_activations(
    model, tokenizer, sae, texts: list[str], feature_indices: list[int],
) -> dict[int, list[dict]]:
    """Run texts through model+SAE, collect per-token activations for target features.

    Returns: {feature_idx: [{'text': str, 'token': str, 'activation': float, 'context': str}, ...]}
    """
    device = model.device
    feat_set = set(feature_indices)

    # Per-feature top-k heap (activation, text_idx, token_idx)
    from heapq import heappush, heappushpop
    heaps: dict[int, list] = {f: [] for f in feature_indices}

    all_texts_tokens = []  # Store tokenized texts for context retrieval

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get hidden states at target layer
        hs = outputs.hidden_states[TARGET_LAYER + 1]  # +1 because index 0 is embeddings

        # Run through SAE
        for bi in range(hs.shape[0]):
            text_idx = batch_start + bi
            seq_len = inputs.attention_mask[bi].sum().item()
            h = hs[bi, :seq_len, :]  # (seq_len, hidden_dim)

            # Encode through SAE
            sae_acts = sae.encode(h)  # (seq_len, n_features)

            # Get token strings
            token_ids = inputs.input_ids[bi, :seq_len]
            tokens = [tokenizer.decode(tid) for tid in token_ids]
            all_texts_tokens.append(tokens)

            # Collect activations for target features
            for fi in feature_indices:
                acts = sae_acts[:, fi].detach().cpu().numpy()
                for ti in range(len(acts)):
                    val = float(acts[ti])
                    if val <= 0:
                        continue
                    entry = (val, text_idx, ti)
                    if len(heaps[fi]) < TOP_K_EXAMPLES:
                        heappush(heaps[fi], entry)
                    elif val > heaps[fi][0][0]:
                        heappushpop(heaps[fi], entry)

        if (batch_start // BATCH_SIZE) % 50 == 0:
            logger.info(f"  Processed {batch_start + len(batch_texts)}/{len(texts)} texts")

    # Convert heaps to sorted results with context
    results = {}
    for fi in feature_indices:
        entries = sorted(heaps[fi], reverse=True)
        examples = []
        for act_val, text_idx, token_idx in entries:
            tokens = all_texts_tokens[text_idx]
            # Build context: surrounding tokens
            start = max(0, token_idx - 15)
            end = min(len(tokens), token_idx + 15)
            context = "".join(tokens[start:end])
            target_token = tokens[token_idx]
            examples.append({
                "activation": round(act_val, 4),
                "token": target_token,
                "context": context[:CONTEXT_WINDOW * 2],
                "text_idx": text_idx,
            })
        results[fi] = examples
        logger.info(f"Feature {fi}: {len(examples)} examples, "
                     f"max_act={examples[0]['activation']:.3f}" if examples else "no examples")

    return results


def generate_descriptions(
    model, tokenizer, feature_examples: dict[int, list[dict]],
) -> dict[int, str]:
    """Use the model to generate one-line descriptions for each feature."""
    descriptions = {}

    for fi, examples in feature_examples.items():
        if not examples:
            descriptions[fi] = "Unknown"
            continue

        # Build prompt with top activating contexts
        top_contexts = examples[:15]
        examples_text = "\n".join(
            f"  - \"{ex['context'].strip()}\" (token: \"{ex['token'].strip()}\")"
            for ex in top_contexts
        )

        prompt = f"""Below are text excerpts where a specific neuron in a language model activates strongly. The activated token is shown in quotes.

{examples_text}

Based on these examples, write a concise label (3-8 words) describing what concept or pattern this neuron detects. Be specific — avoid generic descriptions like "text" or "language". Focus on the common theme.

Label:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=30, temperature=0.3,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        label = response.strip().split("\n")[0].strip().strip('"').strip("'")
        descriptions[fi] = label
        logger.info(f"Feature {fi}: {label}")

    return descriptions


def main():
    output_path = Path("auto_interp_results.json")

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        hf_path = Path.home() / ".hf_token"
        if hf_path.exists():
            hf_token = hf_path.read_text().strip()

    logger.info(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map="auto", token=hf_token,
    )
    model.eval()

    logger.info(f"Loading SAE: {SAE_RELEASE} / {SAE_ID}")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE, sae_id=SAE_ID,
    )[0]
    sae = sae.to(model.device)
    sae.eval()

    # Load corpus
    texts = load_diverse_corpus(N_CORPUS)

    # Extract per-token activations
    logger.info(f"Extracting activations for {len(TOP_FEATURES)} features...")
    feature_examples = extract_per_token_activations(
        model, tokenizer, sae, texts, TOP_FEATURES,
    )

    # Generate descriptions
    logger.info("Generating feature descriptions...")
    descriptions = generate_descriptions(model, tokenizer, feature_examples)

    # Save results
    results = {
        "model": MODEL_ID,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "target_layer": TARGET_LAYER,
        "n_corpus": len(texts),
        "features": {},
    }

    for fi in TOP_FEATURES:
        results["features"][str(fi)] = {
            "sae_index": fi,
            "description": descriptions.get(fi, "Unknown"),
            "top_examples": feature_examples.get(fi, [])[:TOP_K_EXAMPLES],
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Print summary
    print("\n" + "=" * 60)
    print("AUTO-INTERP RESULTS")
    print("=" * 60)
    for fi in TOP_FEATURES:
        desc = descriptions.get(fi, "?")
        n_ex = len(feature_examples.get(fi, []))
        print(f"  Feature {fi:>6d}: {desc} ({n_ex} examples)")


if __name__ == "__main__":
    main()
