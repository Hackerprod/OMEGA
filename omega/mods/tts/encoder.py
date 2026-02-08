from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from omega.mods.base import BaseEncoder


class ContinuousAudioEncoder(BaseEncoder):
    """
    Token-free encoder for 1-D audio waveforms.

    The encoder segments the waveform into overlapping frames, optionally applies
    a Hann window, and projects each frame onto a dense latent space using a
    random Gaussian matrix. A light exponential smoothing step encourages
    temporal continuity.
    """

    def __init__(
        self,
        d_model: int,
        frame_size: int = 1024,
        hop_size: int = 256,
        smoothing: float = 0.1,
        seed: Optional[int] = 1729,
        apply_window: bool = True,
    ):
        super().__init__(d_model)
        if frame_size <= 0 or hop_size <= 0:
            raise ValueError("frame_size and hop_size must be positive integers.")
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.apply_window = bool(apply_window)
        rng = np.random.default_rng(seed)
        self.projection = rng.standard_normal((self.d_model, self.frame_size)).astype(np.float32)
        self.projection /= np.sqrt(self.frame_size)
        self._window = np.hanning(self.frame_size).astype(np.float32) if self.apply_window else None

    def encode(self, source: Iterable[float] | np.ndarray) -> np.ndarray:
        waveform = np.asarray(source, dtype=np.float32).flatten()
        if waveform.size < self.frame_size + 1:
            return np.zeros((0, self.d_model), dtype=np.float32)

        # Frame extraction
        frames = sliding_window_view(waveform, self.frame_size)[:: self.hop_size]
        frames = frames.astype(np.float32, copy=False)

        if self.apply_window and self._window is not None:
            frames = frames * self._window

        # Random projection
        projected = frames @ self.projection.T

        # Light smoothing for temporal continuity
        if self.smoothing > 0.0 and projected.shape[0] > 1:
            alpha = self.smoothing
            for i in range(1, projected.shape[0]):
                projected[i] = (1.0 - alpha) * projected[i] + alpha * projected[i - 1]

        return projected.astype(np.float32, copy=False)
