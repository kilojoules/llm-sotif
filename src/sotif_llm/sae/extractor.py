"""SAE feature extraction from LLM hidden states.

Extracts Sparse Autoencoder feature activations from an LLM's residual stream.
Supports both:
  1. Pre-trained SAEs via SAELens (preferred for reproducibility)
  2. Custom SAEs trained on the fly (via REDKWEEN's sae_analysis.py)

The extracted features form the "measurements" in our SOTIF framework —
analogous to the wind speed, turbulence, and power measurements in the
error_predictor paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of SAE feature extraction for a batch of prompts."""

    prompt_ids: list[str]
    # SAE feature activations: shape (n_prompts, n_features)
    features: np.ndarray
    # Raw hidden states (optional, for diagnostics): shape (n_prompts, hidden_dim)
    hidden_states: np.ndarray | None = None
    # Per-prompt metadata
    layer_idx: int = 0
    model_id: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            prompt_ids=np.array(self.prompt_ids, dtype=object),
            features=self.features,
            hidden_states=self.hidden_states if self.hidden_states is not None else np.array([]),
            layer_idx=self.layer_idx,
            model_id=self.model_id,
        )

    @classmethod
    def load(cls, path: Path) -> ExtractionResult:
        data = np.load(path, allow_pickle=True)
        hs = data["hidden_states"]
        return cls(
            prompt_ids=data["prompt_ids"].tolist(),
            features=data["features"],
            hidden_states=hs if hs.size > 0 else None,
            layer_idx=int(data["layer_idx"]),
            model_id=str(data["model_id"]),
        )


class SAEExtractor:
    """Extract SAE features from an LLM.

    Architecture:
      prompt text → tokenizer → LLM forward pass → hidden states at layer L
      → SAE encoder → sparse feature activations

    The feature activations are the "measurements" we use for:
      - Defining the safe baseline (Phase 1)
      - Monitoring anomaly distance (Phases 2 & 3)
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        sae_release: str = "sae_bench_llama_3.1_8b_instruct",
        sae_id: str = "layers.16/65536",
        layer_idx: int = 16,
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.layer_idx = layer_idx
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length

        self._model = None
        self._tokenizer = None
        self._sae = None

    def load_model(self) -> None:
        """Load the target LLM."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading model: {self.model_id}")

        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def load_sae(self) -> None:
        """Load a pre-trained SAE from SAELens."""
        from sae_lens import SAE

        logger.info(f"Loading SAE: {self.sae_release} / {self.sae_id}")
        self._sae = SAE.from_pretrained(
            release=self.sae_release,
            sae_id=self.sae_id,
            device=self.device,
        )

    def _extract_hidden_states(self, texts: list[str]) -> torch.Tensor:
        """Extract residual-stream hidden states at the target layer.

        Returns: tensor of shape (len(texts), hidden_dim)
        """
        assert self._model is not None and self._tokenizer is not None

        all_states = []
        for text in texts:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False,
            ).to(self._model.device)

            with torch.inference_mode():
                outputs = self._model(
                    **inputs,
                    output_hidden_states=True,
                )

            # Get hidden state at target layer — use last token (not mean pool).
            # The SAE was trained on per-token activations, so mean-pooling
            # produces OOD vectors that cause near-zero sparse features,
            # especially for long sequences. The last token's hidden state
            # captures the model's summary of the full input.
            hs = outputs.hidden_states[self.layer_idx]  # (1, seq_len, hidden_dim)
            last_token_hs = hs[:, -1, :]  # (1, hidden_dim)
            all_states.append(last_token_hs.squeeze(0).float().cpu())

        return torch.stack(all_states)  # (n_texts, hidden_dim)

    def _encode_with_sae(self, hidden_states: torch.Tensor) -> np.ndarray:
        """Pass hidden states through the SAE encoder to get sparse features.

        Returns: numpy array of shape (n_texts, n_sae_features)
        """
        assert self._sae is not None

        hs_device = hidden_states.to(self.device)
        with torch.inference_mode():
            feature_acts = self._sae.encode(hs_device)
        return feature_acts.float().cpu().numpy()

    def extract(
        self,
        texts: list[str],
        prompt_ids: list[str],
        batch_size: int = 8,
        save_hidden_states: bool = False,
    ) -> ExtractionResult:
        """Extract SAE features for a list of prompts.

        This is the main entry point. Processes in batches.
        """
        if self._model is None:
            self.load_model()
        if self._sae is None:
            self.load_sae()

        all_features = []
        all_hidden = [] if save_hidden_states else None

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting SAE features"):
            batch_texts = texts[i : i + batch_size]
            hidden_states = self._extract_hidden_states(batch_texts)

            if save_hidden_states:
                all_hidden.append(hidden_states.numpy())

            features = self._encode_with_sae(hidden_states)
            all_features.append(features)

        features_array = np.concatenate(all_features, axis=0)
        hidden_array = np.concatenate(all_hidden, axis=0) if all_hidden else None

        return ExtractionResult(
            prompt_ids=prompt_ids,
            features=features_array,
            hidden_states=hidden_array,
            layer_idx=self.layer_idx,
            model_id=self.model_id,
        )

    def extract_during_generation(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> tuple[str, list[np.ndarray]]:
        """Extract SAE features at each generation step (real-time monitoring).

        Returns (generated_text, list_of_feature_vectors_per_step).
        This enables detecting when the model leaves the safe envelope
        *during* generation — before the harmful tokens are emitted.
        """
        if self._model is None:
            self.load_model()
        if self._sae is None:
            self.load_sae()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._model.device)

        input_ids = inputs["input_ids"]
        step_features = []

        for _ in range(max_new_tokens):
            with torch.inference_mode():
                outputs = self._model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )

            # Extract hidden state at target layer for the LAST token
            hs = outputs.hidden_states[self.layer_idx][:, -1, :]  # (1, hidden_dim)

            # SAE encode
            features = self._encode_with_sae(hs.float().cpu())
            step_features.append(features.squeeze(0))

            # Sample next token
            logits = outputs.logits[:, -1, :]
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop at EOS
            if next_token.item() == self._tokenizer.eos_token_id:
                break

        generated_ids = input_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, step_features

    def unload(self) -> None:
        """Free GPU memory."""
        import gc
        del self._model
        del self._tokenizer
        del self._sae
        self._model = None
        self._tokenizer = None
        self._sae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
