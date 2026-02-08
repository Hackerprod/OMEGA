from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from omega.mods.base import BaseEncoder


class ContinuousTextEncoder(BaseEncoder):
    """
    Token-free encoder that maps raw text into continuous trajectories.
    Each character is projected onto a dense subspace and smoothed in time.
    """

    def __init__(
        self,
        d_model: int,
        smoothing: float = 0.2,
        seed: Optional[int] = 1729,
        charset: Optional[Iterable[str]] = None,
        dtype: np.dtype = np.float32,
    ):
        super().__init__(d_model)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.rng = np.random.default_rng(seed)
        if charset is None:
            charset = [chr(i) for i in range(32, 127)]
        self.charset = list(charset)
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.output_dtype = np.dtype(dtype)
        self.matrix = self._build_projection(len(self.charset))

    @property
    def d(self) -> int:
        """Backward compatibility alias."""
        return self.d_model

    def _build_projection(self, vocab_size: int) -> np.ndarray:
        scale = 1.0 / np.sqrt(vocab_size)
        matrix = self.rng.standard_normal((self.d_model, vocab_size), dtype=self.output_dtype)
        matrix *= self.output_dtype.type(scale)
        return matrix.astype(self.output_dtype, copy=False)

    def encode(self, source: str | Iterable[str]) -> np.ndarray:
        if isinstance(source, (list, tuple)):
            text = "".join(source)
        else:
            text = str(source)
        return self.encode_text(text)

    def encode_text(self, text: str) -> np.ndarray:
        vectors: List[np.ndarray] = []
        prev = np.zeros(self.d_model, dtype=self.output_dtype)
        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx.get(" ", 0))
            column = self.matrix[:, idx]
            current = (1.0 - self.smoothing) * column + self.smoothing * prev
            prev = current
            vectors.append(current)
        if not vectors:
            return np.zeros((0, self.d_model), dtype=self.output_dtype)
        return np.stack(vectors, axis=0).astype(self.output_dtype, copy=False)

    def encode_lines(self, lines: Iterable[str], separator: str = "\n") -> np.ndarray:
        text = separator.join(lines)
        return self.encode_text(text)

    def encode_file_to_memmap(
        self,
        text_path: Path,
        output_path: Path,
        encoding: str = "utf-8",
        chunk_chars: int = 65536,
        dtype: Optional[np.dtype] = None,
        max_chars: Optional[int] = None,
    ) -> np.memmap:
        """Stream encode a text file into a memmap."""
        text_path = Path(text_path)
        output_path = Path(output_path)
        mem_dtype = self.output_dtype if dtype is None else np.dtype(dtype)
        total_chars = 0
        remaining = max_chars
        with text_path.open("r", encoding=encoding) as fh:
            while True:
                read_size = chunk_chars if remaining is None else min(chunk_chars, remaining)
                chunk = fh.read(read_size)
                if not chunk:
                    break
                total_chars += len(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break

        mmap = np.memmap(output_path, mode="w+", dtype=mem_dtype, shape=(total_chars, self.d_model))
        prev = np.zeros(self.d_model, dtype=self.output_dtype)
        offset = 0
        remaining = max_chars
        with text_path.open("r", encoding=encoding) as fh:
            while True:
                read_size = chunk_chars if remaining is None else min(chunk_chars, remaining)
                chunk = fh.read(read_size)
                if not chunk:
                    break
                encoded = self.encode_text(chunk)
                if encoded.size == 0:
                    continue
                if offset > 0 and encoded.shape[0] > 0:
                    encoded[0] = (1.0 - self.smoothing) * encoded[0] + self.smoothing * prev
                mmap[offset : offset + encoded.shape[0]] = encoded.astype(mem_dtype, copy=False)
                prev = encoded[-1]
                offset += encoded.shape[0]
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
        mmap.flush()
        return mmap
