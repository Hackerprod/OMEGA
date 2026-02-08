from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from omega.mods.base import BaseEncoder


class TextTrajectoryEncoder(BaseEncoder):
    """
    Lightweight encoder that maps text into continuous trajectories suitable for OMEGA.

    - Character-level embeddings projected to `d_model`.
    - Optional exponential smoothing to encourage temporal continuity.
    - Deterministic projection given a RNG seed.
    """

    def __init__(
        self,
        d_model: int,
        smoothing: float = 0.15,
        seed: Optional[int] = 1312,
        charset: Optional[Iterable[str]] = None,
        dtype: np.dtype = np.float32,
    ):
        super().__init__(d_model)
        self.dtype = np.dtype(dtype)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        rng = np.random.default_rng(seed)
        if charset is None:
            charset = [chr(i) for i in range(32, 127)]
        self.charset = list(charset)
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.embed = rng.standard_normal((len(self.charset), self.d_model)).astype(self.dtype)
        self.embed /= self.dtype.type(np.sqrt(self.d_model))

    def encode(self, source: str | Iterable[str]) -> np.ndarray:
        if isinstance(source, (list, tuple)):
            text = "".join(source)
        else:
            text = str(source)
        if not text:
            return np.zeros((0, self.d_model), dtype=self.dtype)

        vectors = np.empty((len(text), self.d_model), dtype=self.dtype)
        prev = np.zeros(self.d_model, dtype=self.dtype)
        for idx, ch in enumerate(text):
            emb = self.embed[self.char_to_idx.get(ch, 0)]
            vectors[idx] = (1.0 - self.smoothing) * emb + self.smoothing * prev
            prev = vectors[idx]
        return vectors

    # Convenience helpers -------------------------------------------------
    def encode_lines(self, lines: Iterable[str], separator: str = "\n") -> np.ndarray:
        text = separator.join(lines)
        return self.encode(text)

    def vocab(self) -> Iterable[str]:
        return self.charset
