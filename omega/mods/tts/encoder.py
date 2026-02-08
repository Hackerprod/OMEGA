from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from omega.mods.base import BaseEncoder


class ContinuousAudioEncoder(BaseEncoder):
    """
    Omega-native audio encoder.

    1. Applies optional pre-emphasis and framing.
    2. Computes STFT magnitudes and filters them with a learnable FIR bank.
    3. Runs local conv / gating blocks with light pooling (no global attention).
    4. Projects to `d_model` with SVD-initialised weights and exponential smoothing.

    `get_state` and `set_state` allow offline optimisation (e.g., via PyTorch) and
    then reloading the trained parameters into the NumPy pipeline.
    """

    def __init__(
        self,
        d_model: int,
        frame_size: int = 1024,
        hop_size: int = 256,
        fft_size: Optional[int] = None,
        filter_bands: int = 96,
        conv_channels: Optional[Sequence[int]] = None,
        kernel_size: int = 5,
        pool_size: int = 2,
        smoothing: float = 0.1,
        pre_emphasis: float = 0.0,
        log_eps: float = 1e-4,
        dtype: np.dtype = np.float32,
        seed: Optional[int] = 1729,
    ):
        super().__init__(d_model)
        if frame_size <= 0 or hop_size <= 0:
            raise ValueError("frame_size and hop_size must be positive integers.")
        if filter_bands <= 0:
            raise ValueError("filter_bands must be positive.")
        self.dtype = np.dtype(dtype)
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.fft_size = self._next_pow_two(fft_size or frame_size)
        self.freq_bins = self.fft_size // 2 + 1
        self.filter_bands = int(filter_bands)
        self.kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.pool_size = max(1, int(pool_size))
        self.latent_smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.pre_emphasis = float(np.clip(pre_emphasis, 0.0, 0.99))
        self.log_eps = float(max(log_eps, 1e-8))

        rng = np.random.default_rng(seed)
        self.rng = rng
        self.window = np.hanning(self.frame_size).astype(self.dtype)
        self.filter_bank = self._init_filter_bank().astype(self.dtype)

        if conv_channels is None:
            base = max(64, min(192, d_model * 2))
            conv_channels = (base, base)
        self.conv_channels = list(conv_channels)

        input_channels = self.filter_bands + 3  # + energy, delta, acceleration
        self.layers: List[Dict[str, np.ndarray]] = []
        in_ch = input_channels
        for width in self.conv_channels:
            conv_weight = self._init_conv_weight(in_ch, width)
            conv_bias = np.zeros(width, dtype=self.dtype)
            gate_weight = self._init_conv_weight(in_ch, width)
            gate_bias = np.zeros(width, dtype=self.dtype)
            self.layers.append(
                {
                    "weight": conv_weight,
                    "bias": conv_bias,
                    "gate_weight": gate_weight,
                    "gate_bias": gate_bias,
                }
            )
            in_ch = width

        self.output_projection = self._init_projection(in_ch, self.d_model)
        self.output_bias = np.zeros(self.d_model, dtype=self.dtype)

    # ------------------------------------------------------------------ #
    def encode(self, source: Iterable[float] | np.ndarray) -> np.ndarray:
        waveform = np.asarray(source, dtype=self.dtype).flatten()
        if waveform.size < self.frame_size:
            return np.zeros((0, self.d_model), dtype=self.dtype)

        if self.pre_emphasis > 0.0:
            waveform = self._pre_emphasis(waveform, self.pre_emphasis)

        frames = sliding_window_view(waveform, self.frame_size)[:: self.hop_size]
        frames = frames.astype(self.dtype, copy=False)

        if self.window is not None:
            frames = frames * self.window

        stft = np.fft.rfft(frames, n=self.fft_size, axis=1)
        magnitude = np.abs(stft).astype(self.dtype, copy=False)

        log_mag = np.log(magnitude + self.log_eps)
        filter_feats = log_mag @ self.filter_bank.T

        energy = np.sqrt(np.mean(frames ** 2, axis=1, keepdims=True, dtype=np.float64)).astype(self.dtype)
        delta = self._temporal_delta(filter_feats)
        accel = self._temporal_delta(delta)
        features = np.concatenate([filter_feats, energy, delta, accel], axis=1).astype(self.dtype, copy=False)

        x = features
        for layer in self.layers:
            conv = self._conv1d_time(x, layer["weight"], layer["bias"])
            gate = self._conv1d_time(x, layer["gate_weight"], layer["gate_bias"])
            x = np.tanh(conv) * self._sigmoid(gate)
            if self.pool_size > 1 and x.shape[0] > self.pool_size:
                x = self._avg_pool(x, self.pool_size)
            x = self._local_normalize(x)

        latents = x @ self.output_projection.T + self.output_bias
        latents = latents.astype(self.dtype, copy=False)
        if 0.0 < self.latent_smoothing < 1.0 and latents.shape[0] > 1:
            latents = self._smooth(latents, self.latent_smoothing)
        return latents

    def expand_audio(self, source: Iterable[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        waveform = np.asarray(source, dtype=self.dtype).flatten()
        if waveform.size < self.frame_size:
            return waveform, np.zeros((0, self.frame_size), dtype=self.dtype), np.zeros((0, self.freq_bins), dtype=self.dtype)
        if self.pre_emphasis > 0.0:
            waveform = self._pre_emphasis(waveform, self.pre_emphasis)
        frames = sliding_window_view(waveform, self.frame_size)[:: self.hop_size]
        frames = frames.astype(self.dtype, copy=False)
        if self.window is not None:
            frames = frames * self.window
        stft = np.fft.rfft(frames, n=self.fft_size, axis=1)
        magnitude = np.abs(stft).astype(self.dtype, copy=False)
        return waveform, frames, magnitude

    def get_state(self) -> Dict[str, object]:
        return {
            "filter_bank": self.filter_bank.copy(),
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
        }

    def set_state(self, state: Dict[str, object]) -> None:
        self.filter_bank = np.asarray(state["filter_bank"], dtype=self.dtype).copy()
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

    # ------------------------------------------------------------------ #
    def _init_filter_bank(self) -> np.ndarray:
        # Triangular mel-like filter bank covering [50 Hz, nyquist]
        nyquist = 0.5 * self.fft_size
        mel_min = self._hz_to_mel(50.0)
        mel_max = self._hz_to_mel(nyquist)
        mel_points = np.linspace(mel_min, mel_max, self.filter_bands + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((self.fft_size + 1) * hz_points / self.fft_size).astype(int)
        bank = np.zeros((self.filter_bands, self.freq_bins), dtype=np.float64)
        for i in range(1, self.filter_bands + 1):
            left, center, right = bin_points[i - 1 : i + 2]
            left = max(left, 0)
            right = min(right, self.freq_bins - 1)
            for k in range(left, center):
                if center != left:
                    bank[i - 1, k] = (k - left) / (center - left)
            for k in range(center, right):
                if right != center:
                    bank[i - 1, k] = (right - k) / (right - center)
        bank /= np.maximum(bank.sum(axis=1, keepdims=True), 1e-8)
        return bank

    def _init_conv_weight(self, in_channels: int, out_channels: int) -> np.ndarray:
        weight = self.rng.standard_normal((out_channels, self.kernel_size, in_channels)).astype(self.dtype)
        scale = 1.0 / np.sqrt(in_channels * self.kernel_size)
        weight *= self.dtype.type(scale)
        return weight

    def _init_projection(self, in_channels: int, out_channels: int) -> np.ndarray:
        matrix = self.rng.standard_normal((out_channels, in_channels))
        try:
            u, _, vh = np.linalg.svd(matrix, full_matrices=False)
            base = (u @ vh).astype(self.dtype, copy=False)
        except np.linalg.LinAlgError:
            base = matrix.astype(self.dtype, copy=False)
        base /= self.dtype.type(np.sqrt(in_channels))
        return base

    def _conv1d_time(self, inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        time_steps, channels = inputs.shape
        out_channels, kernel, in_channels = weight.shape
        if in_channels != channels:
            raise ValueError("Input channel mismatch in conv layer.")
        pad = kernel // 2
        padded = np.pad(inputs, ((pad, pad), (0, 0)), mode="edge")
        windows = sliding_window_view(padded, (kernel, channels))[:, 0]
        outputs = np.tensordot(windows, weight, axes=([1, 2], [1, 2])) + bias
        return outputs.astype(self.dtype, copy=False)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _avg_pool(self, x: np.ndarray, size: int) -> np.ndarray:
        length = x.shape[0]
        trimmed = length - (length % size)
        if trimmed == 0:
            return x
        pooled = x[:trimmed].reshape(trimmed // size, size, x.shape[1]).mean(axis=1)
        if trimmed < length:
            tail = x[trimmed:].mean(axis=0, keepdims=True)
            pooled = np.vstack([pooled, tail])
        return pooled

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

    def _pre_emphasis(self, waveform: np.ndarray, coeff: float) -> np.ndarray:
        emphasized = waveform.copy()
        emphasized[1:] = waveform[1:] - coeff * waveform[:-1]
        return emphasized

    def _temporal_delta(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return features
        diff = np.diff(features, axis=0, prepend=features[:1])
        return diff

    @staticmethod
    def _next_pow_two(value: int) -> int:
        return 1 << (int(value - 1).bit_length())

    @staticmethod
    def _hz_to_mel(freq: float) -> float:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)
