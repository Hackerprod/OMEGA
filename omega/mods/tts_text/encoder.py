from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from omega.mods.base import BaseEncoder

# Transliteration map keeps the implementation ASCII-only.
_ACCENT_MAP = {
    ord("\u00E1"): "a",  # á
    ord("\u00E9"): "e",  # é
    ord("\u00ED"): "i",  # í
    ord("\u00F3"): "o",  # ó
    ord("\u00FA"): "u",  # ú
    ord("\u00FC"): "u",  # ü
    ord("\u00F1"): "ny",  # ñ
    ord("\u00E4"): "a",  # ä
    ord("\u00EB"): "e",  # ë
    ord("\u00EF"): "i",  # ï
    ord("\u00F6"): "o",  # ö
}

_LANGUAGE_RULES: Dict[str, Dict[str, object]] = {
    "en": {
        "letters": {
            "a": "aa",
            "b": "b",
            "c": "k",
            "d": "d",
            "e": "eh",
            "f": "f",
            "g": "g",
            "h": "hh",
            "i": "iy",
            "j": "jh",
            "k": "k",
            "l": "l",
            "m": "m",
            "n": "n",
            "o": "ow",
            "p": "p",
            "q": "k",
            "r": "r",
            "s": "s",
            "t": "t",
            "u": "uw",
            "v": "v",
            "w": "w",
            "x": "ks",
            "y": "iy",
            "z": "z",
        },
        "digraphs": {
            "th": "th",
            "sh": "sh",
            "ch": "ch",
            "ph": "f",
            "gh": "g",
            "ng": "ng",
            "wh": "w",
            "qu": "k",
            "ee": "iy",
            "oo": "uw",
            "ea": "iy",
            "ai": "ey",
            "ou": "aw",
        },
        "vowel_letters": set("aeiouy"),
        "pause_tokens": {"pau", "sil"},
    },
    "es": {
        "letters": {
            "a": "a",
            "b": "b",
            "c": "k",
            "d": "d",
            "e": "e",
            "f": "f",
            "g": "g",
            "h": "h",
            "i": "i",
            "j": "x",
            "k": "k",
            "l": "l",
            "m": "m",
            "n": "n",
            "o": "o",
            "p": "p",
            "q": "k",
            "r": "r",
            "s": "s",
            "t": "t",
            "u": "u",
            "v": "b",
            "w": "w",
            "x": "ks",
            "y": "i",
            "z": "s",
        },
        "digraphs": {
            "ch": "ch",
            "ll": "ly",
            "rr": "rr",
            "ny": "ny",
            "qu": "k",
            "gu": "g",
            "ue": "ue",
            "ia": "ia",
        },
        "vowel_letters": set("aeiou"),
        "pause_tokens": {"pau", "sil"},
    },
}

_DEFAULT_PAUSE_TOKENS = {"pau", "sil"}
_SPECIAL_TOKENS = {"pau", "sil", "unk", "num"}


class PhonemeTokenizer:
    """Light rule-based phoneme tokenizer with language-aware digraphs."""

    def __init__(
        self,
        language: str = "en",
        custom_map: Optional[Dict[str, str]] = None,
        extra_tokens: Optional[Sequence[str]] = None,
        pause_tokens: Optional[Sequence[str]] = None,
    ):
        if language not in _LANGUAGE_RULES:
            raise ValueError(f"Unsupported language '{language}'")
        rules = _LANGUAGE_RULES[language]
        self.language = language
        self.map: Dict[str, str] = dict(rules["letters"])  # type: ignore[arg-type]
        if custom_map:
            self.map.update({k.lower(): v for k, v in custom_map.items()})
        digraphs = dict(rules["digraphs"])  # type: ignore[arg-type]
        self.digraphs = {k.lower(): v for k, v in digraphs.items()}
        self._digraph_sizes = sorted({len(k) for k in self.digraphs}, reverse=True)
        self.vowel_letters = set(rules["vowel_letters"])  # type: ignore[arg-type]
        self.inventory = set(self.map.values()) | set(self.digraphs.values()) | _SPECIAL_TOKENS
        if extra_tokens:
            self.inventory.update(extra_tokens)
        self.inventory.update(_DEFAULT_PAUSE_TOKENS)
        base_pauses = _DEFAULT_PAUSE_TOKENS | set(rules["pause_tokens"])  # type: ignore[arg-type]
        if pause_tokens:
            base_pauses |= set(pause_tokens)
        self.pause_tokens = base_pauses

    def tokenize(self, text: str) -> List[str]:
        text = text.lower().translate(_ACCENT_MAP)
        tokens: List[str] = ["sil"]
        i = 0
        length = len(text)
        while i < length:
            char = text[i]
            if char.isspace():
                i += 1
                continue
            if char in ",.;:!?":
                if tokens[-1] != "pau":
                    tokens.append("pau")
                i += 1
                continue
            if char.isdigit():
                tokens.append("num")
                i += 1
                continue

            matched = False
            for size in self._digraph_sizes:
                if i + size > length:
                    continue
                segment = text[i : i + size]
                if segment in self.digraphs:
                    tokens.append(self.digraphs[segment])
                    i += size
                    matched = True
                    break
            if matched:
                continue

            phoneme = self.map.get(char)
            tokens.append(phoneme if phoneme is not None else "unk")
            i += 1

        if tokens[-1] != "pau":
            tokens.append("pau")
        tokens.append("sil")

        cleaned: List[str] = []
        for token in tokens:
            if not cleaned:
                cleaned.append(token)
                continue
            if token == "pau" and cleaned[-1] == "pau":
                continue
            cleaned.append(token)
        return cleaned

    def is_vowel(self, token: str) -> bool:
        # Determine vowels by reverse lookup on letters and common vowel phonemes.
        return token in {
            "aa",
            "ah",
            "ao",
            "aw",
            "ay",
            "eh",
            "ey",
            "iy",
            "ow",
            "oy",
            "uw",
            "a",
            "e",
            "i",
            "o",
            "u",
            "ia",
        }


class TextTrajectoryEncoder(BaseEncoder):
    """
    Omega-native text encoder that produces smooth trajectories for ACP.

    Pipeline
    --------
    1. Phoneme tokenization with rule-based durations (pseudo phonemizer).
    2. One-hot embeddings initialised via SVD to ensure orthogonality.
    3. Local convolution + gated units (GLU-like) without global attention.
    4. Exponential smoothing to make the output suitable for Arnoldi updates.
    """

    def __init__(
        self,
        d_model: int,
        smoothing: float = 0.2,
        seed: Optional[int] = 1312,
        language: str = "en",
        custom_phonemes: Optional[Dict[str, str]] = None,
        embed_dim: Optional[int] = None,
        hidden_channels: Optional[Sequence[int]] = None,
        kernel_size: int = 5,
        vowel_duration: int = 6,
        consonant_duration: int = 3,
        pause_duration: int = 8,
        duration_overrides: Optional[Dict[str, int]] = None,
        pause_tokens: Optional[Sequence[str]] = None,
        dtype: np.dtype = np.float32,
    ):
        super().__init__(d_model)
        self.dtype = np.dtype(dtype)
        self.latent_smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.rng = np.random.default_rng(seed)

        self.tokenizer = PhonemeTokenizer(language=language, custom_map=custom_phonemes, pause_tokens=pause_tokens)
        self.inventory = sorted(self.tokenizer.inventory)
        self.phoneme_to_id = {symbol: idx for idx, symbol in enumerate(self.inventory)}
        self.vowel_tokens = {tok for tok in self.inventory if self.tokenizer.is_vowel(tok)}
        self.pause_tokens = set(self.tokenizer.pause_tokens)
        self.unk_id = self.phoneme_to_id["unk"]

        self.vowel_duration = max(1, int(vowel_duration))
        self.consonant_duration = max(1, int(consonant_duration))
        self.pause_duration = max(1, int(pause_duration))
        self.duration_overrides: Dict[str, int] = {
            "rr": 5,
            "ch": 5,
            "ly": 4,
            "th": 4,
        }
        if duration_overrides:
            for key, value in duration_overrides.items():
                self.duration_overrides[key] = max(1, int(value))

        self.embed_dim = embed_dim or max(self.d_model, 48)
        self.embedding = self._initialise_embeddings()

        aux_features = 4  # phase, vowel flag, pause flag, relative position
        input_dim = self.embed_dim + aux_features

        if hidden_channels is None:
            base = max(64, min(128, self.d_model * 2))
            hidden_channels = (base, base)
        self.hidden_channels = list(hidden_channels)

        self.layers: List[Dict[str, np.ndarray]] = []
        in_channels = input_dim
        for width in self.hidden_channels:
            conv_weight = self._init_conv_weight(in_channels, width)
            conv_bias = np.zeros(width, dtype=self.dtype)
            gate_weight = self._init_conv_weight(in_channels, width)
            gate_bias = np.zeros(width, dtype=self.dtype)
            self.layers.append(
                {
                    "weight": conv_weight,
                    "bias": conv_bias,
                    "gate_weight": gate_weight,
                    "gate_bias": gate_bias,
                }
            )
            in_channels = width

        self.output_projection = self._init_projection(in_channels, self.d_model)
        self.output_bias = np.zeros(self.d_model, dtype=self.dtype)

    # ------------------------------------------------------------------ #
    def encode(self, source: str | Iterable[str]) -> np.ndarray:
        if isinstance(source, (list, tuple)):
            text = "".join(source)
        else:
            text = str(source)
        if not text:
            return np.zeros((0, self.d_model), dtype=self.dtype)

        phonemes = self.tokenizer.tokenize(text)
        frame_ids, aux = self._expand_phonemes(phonemes)
        if frame_ids.size == 0:
            return np.zeros((0, self.d_model), dtype=self.dtype)

        embedded = self.embedding[frame_ids]
        features = np.concatenate([embedded, aux], axis=1).astype(self.dtype, copy=False)

        x = features
        for layer in self.layers:
            conv_out = self._conv1d(x, layer["weight"], layer["bias"])
            gate_out = self._conv1d(x, layer["gate_weight"], layer["gate_bias"])
            x = np.tanh(conv_out) * self._sigmoid(gate_out)
            x = self._local_normalize(x)

        latents = x @ self.output_projection.T + self.output_bias
        latents = latents.astype(self.dtype, copy=False)
        if 0.0 < self.latent_smoothing < 1.0 and latents.shape[0] > 1:
            latents = self._smooth(latents, self.latent_smoothing)
        return latents

    # Convenience helpers -------------------------------------------------
    def expand_text(
        self, source: str | Iterable[str]
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[int]]:
        if isinstance(source, (list, tuple)):
            text = "".join(source)
        else:
            text = str(source)
        phonemes = self.tokenizer.tokenize(text)
        frame_ids, aux, durations = self._expand_phonemes(phonemes, with_metadata=True)
        return phonemes, frame_ids, aux, durations

    def encode_lines(self, lines: Iterable[str], separator: str = "\n") -> np.ndarray:
        text = separator.join(lines)
        return self.encode(text)

    def vocab(self) -> Iterable[str]:
        return list(self.inventory)

    def get_state(self) -> Dict[str, np.ndarray | Dict[str, float] | List[int] | str | int | float]:
        return {
            "embedding": self.embedding.copy(),
            "layers": [
                {
                    "weight": layer["weight"].copy(),
                    "bias": layer["bias"].copy(),
                    "gate_weight": layer["gate_weight"].copy(),
                    "gate_bias": layer["gate_bias"].copy(),
                }
                for layer in self.layers
            ],
            "output_projection": self.output_projection.copy(),
            "output_bias": self.output_bias.copy(),
            "vowel_duration": int(self.vowel_duration),
            "consonant_duration": int(self.consonant_duration),
            "pause_duration": int(self.pause_duration),
            "duration_overrides": dict(self.duration_overrides),
        }

    def set_state(self, state: Dict[str, object]) -> None:
        self.embedding = np.asarray(state["embedding"], dtype=self.dtype).copy()
        layer_states = state["layers"]
        if not isinstance(layer_states, list) or len(layer_states) != len(self.layers):
            raise ValueError("Layer state mismatch.")
        for layer, layer_state in zip(self.layers, layer_states):
            layer["weight"] = np.asarray(layer_state["weight"], dtype=self.dtype).copy()
            layer["bias"] = np.asarray(layer_state["bias"], dtype=self.dtype).copy()
            layer["gate_weight"] = np.asarray(layer_state["gate_weight"], dtype=self.dtype).copy()
            layer["gate_bias"] = np.asarray(layer_state["gate_bias"], dtype=self.dtype).copy()
        self.output_projection = np.asarray(state["output_projection"], dtype=self.dtype).copy()
        self.output_bias = np.asarray(state["output_bias"], dtype=self.dtype).copy()
        self.vowel_duration = int(state.get("vowel_duration", self.vowel_duration))
        self.consonant_duration = int(state.get("consonant_duration", self.consonant_duration))
        self.pause_duration = int(state.get("pause_duration", self.pause_duration))
        overrides = state.get("duration_overrides", {})
        if isinstance(overrides, dict):
            self.duration_overrides = {k: int(v) for k, v in overrides.items()}

    # Internal helpers ----------------------------------------------------
    def _initialise_embeddings(self) -> np.ndarray:
        vocab = len(self.inventory)
        random_matrix = self.rng.standard_normal((vocab, self.embed_dim))
        try:
            u, _, vh = np.linalg.svd(random_matrix, full_matrices=False)
            base = (u @ vh).astype(self.dtype, copy=False)
        except np.linalg.LinAlgError:
            base = random_matrix.astype(self.dtype, copy=False)
        base *= self.dtype.type(0.5)
        return base

    def _init_conv_weight(self, in_channels: int, out_channels: int) -> np.ndarray:
        weight = self.rng.standard_normal((out_channels, self.kernel_size, in_channels)).astype(self.dtype)
        scale = 1.0 / np.sqrt(in_channels * self.kernel_size)
        weight *= self.dtype.type(scale)
        return weight

    def _init_projection(self, in_channels: int, out_channels: int) -> np.ndarray:
        proj = self.rng.standard_normal((out_channels, in_channels)).astype(self.dtype)
        proj /= self.dtype.type(np.sqrt(in_channels))
        return proj

    def _expand_phonemes(
        self, phonemes: Sequence[str], with_metadata: bool = False
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, List[int]]:
        frame_ids: List[int] = []
        phases: List[float] = []
        vowel_flags: List[float] = []
        pause_flags: List[float] = []
        rel_positions: List[float] = []
        token_durations: List[int] = []

        total = len(phonemes)
        denom = max(total - 1, 1)

        for idx, token in enumerate(phonemes):
            duration = self._duration(token, idx, total)
            token_id = self.phoneme_to_id.get(token, self.unk_id)
            is_vowel = 1.0 if token in self.vowel_tokens else 0.0
            is_pause = 1.0 if token in self.pause_tokens else 0.0
            rel_pos = idx / denom if total > 1 else 0.0
            if duration <= 0:
                continue
            for step in range(duration):
                frame_ids.append(token_id)
                phase = step / max(duration - 1, 1)
                phases.append(phase)
                vowel_flags.append(is_vowel)
                pause_flags.append(is_pause)
                rel_positions.append(rel_pos)
            if with_metadata:
                token_durations.append(duration)

        if not frame_ids:
            empty_frames = np.array([], dtype=np.int32)
            empty_aux = np.zeros((0, 4), dtype=self.dtype)
            if with_metadata:
                return empty_frames, empty_aux, []
            return empty_frames, empty_aux

        aux = np.column_stack([phases, vowel_flags, pause_flags, rel_positions]).astype(self.dtype, copy=False)
        frames = np.asarray(frame_ids, dtype=np.int32)
        if with_metadata:
            return frames, aux, token_durations
        return frames, aux

    def _duration(self, token: str, index: int, total: int) -> int:
        if token in self.duration_overrides:
            return self.duration_overrides[token]
        if token in self.pause_tokens:
            return self.pause_duration
        base = self.vowel_duration if token in self.vowel_tokens else self.consonant_duration
        if index == 0 or index == total - 1:
            base = int(base * 1.2)
        return max(1, int(base))

    def _conv1d(self, inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        time_steps, channels = inputs.shape
        out_channels, kernel, in_channels = weight.shape
        if in_channels != channels:
            raise ValueError("Input channel mismatch in conv layer.")
        pad = kernel // 2
        padded = np.pad(inputs, ((pad, pad), (0, 0)), mode="edge")
        windows = sliding_window_view(padded, (kernel, channels))[:, 0]
        outputs = np.tensordot(windows, weight, axes=([1, 2], [1, 2])) + bias
        return outputs.astype(self.dtype, copy=False)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _local_normalize(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _smooth(self, latents: np.ndarray, alpha: float) -> np.ndarray:
        smoothed = np.empty_like(latents)
        smoothed[0] = latents[0]
        for t in range(1, latents.shape[0]):
            smoothed[t] = self.dtype.type(alpha) * smoothed[t - 1] + self.dtype.type(1.0 - alpha) * latents[t]
        return smoothed
