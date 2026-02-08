import numpy as np
from pathlib import Path
from typing import Iterable, List, Optional


class ContinuousTextEncoder:
    """
    Token-free encoder that maps raw text into continuous trajectories.
    Each character is projected onto a dense subspace and smoothed in time,
    yielding a vector of dimensión d por paso lingüístico.
    """

    def __init__(
        self,
        d_model: int,
        smoothing: float = 0.2,
        seed: Optional[int] = 1729,
        charset: Optional[Iterable[str]] = None,
    ):
        self.d = d_model
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.rng = np.random.default_rng(seed)
        if charset is None:
            charset = [chr(i) for i in range(32, 127)]
        self.charset = list(charset)
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.matrix = self._build_projection(len(self.charset))

    def _build_projection(self, vocab_size: int) -> np.ndarray:
        scale = 1.0 / np.sqrt(vocab_size)
        matrix = self.rng.standard_normal((self.d, vocab_size)) * scale
        return matrix.astype(np.float64)

    def encode_text(self, text: str) -> np.ndarray:
        vectors: List[np.ndarray] = []
        prev = np.zeros(self.d, dtype=np.float64)
        for char in text:
            idx = self.char_to_idx.get(char)
            if idx is None:
                idx = self.char_to_idx.get(" ", 0)
            column = self.matrix[:, idx]
            current = (1.0 - self.smoothing) * column + self.smoothing * prev
            prev = current
            vectors.append(current)
        if not vectors:
            return np.zeros((0, self.d), dtype=np.float64)
        return np.stack(vectors, axis=0)

    def encode_lines(self, lines: Iterable[str], separator: str = "\n") -> np.ndarray:
        text = separator.join(lines)
        return self.encode_text(text)

    def encode_file_to_memmap(
        self,
        text_path: Path,
        output_path: Path,
        encoding: str = "utf-8",
        chunk_chars: int = 65536,
        dtype: np.dtype = np.float32,
        max_chars: Optional[int] = None,
    ) -> np.memmap:
        """
        Encode a text file directly to a memory-mapped matrix to avoid keeping the
        full corpus in RAM.
        """
        text_path = Path(text_path)
        output_path = Path(output_path)
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

        mmap = np.memmap(output_path, mode="w+", dtype=dtype, shape=(total_chars, self.d))
        prev = np.zeros(self.d, dtype=np.float64)
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
                # Preserve smoothing continuity across chunks
                if offset > 0 and encoded.shape[0] > 0:
                    encoded[0] = (1.0 - self.smoothing) * encoded[0] + self.smoothing * prev
                mmap[offset : offset + encoded.shape[0]] = encoded.astype(dtype, copy=False)
                prev = encoded[-1]
                offset += encoded.shape[0]
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
        mmap.flush()
        return mmap


class ContinuousTextDecoder:
    """
    Placeholder para decodificador continuo->texto.
    Requiere un modelo externo (e.g. VQ-VAE) para reconstrucción explícita.
    """

    def __init__(self):
        raise NotImplementedError(
            "ContinuousTextDecoder es un placeholder; integra tu VQ-VAE o vocoder textual."
        )
