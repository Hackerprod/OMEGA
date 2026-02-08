import csv
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional


class TimeSeriesDataLoader:
    """
    Streaming loader for multivariate time-series.
    Supports CSV/NumPy sources, sliding windows and mini-batching on CPU.
    """

    def __init__(
        self,
        data: np.ndarray,
        window: int = 1,
        batch_size: int = 1,
        stride: int = 1,
        shuffle: bool = False,
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError("time-series data must be 2D: (steps, features)")

        self.data = data.astype(np.float64)
        self.window = max(1, int(window))
        self.batch_size = max(1, int(batch_size))
        self.stride = max(1, int(stride))
        self.shuffle = shuffle
        self.normalize = normalize
        self.mean = None
        self.std = None
        if normalize:
            self.mean = np.mean(self.data, axis=0, keepdims=True)
            self.std = np.std(self.data, axis=0, keepdims=True)
            self.std = np.where(self.std < eps, 1.0, self.std)
            self.data = (self.data - self.mean) / self.std
        self.indices = self._compute_indices()

    @classmethod
    def from_path(
        cls,
        path: str,
        window: int = 1,
        batch_size: int = 1,
        stride: int = 1,
        shuffle: bool = False,
        delimiter: str = ",",
        normalize: bool = False,
    ):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(path)

        if path_obj.suffix in (".npy", ".npz"):
            data = np.load(path_obj)
            if isinstance(data, np.lib.npyio.NpzFile):
                if "data" in data:
                    data = data["data"]
                else:
                    raise ValueError("npz file must contain array under key 'data'")
        else:
            data = cls._load_numeric_csv(path_obj, delimiter=delimiter)

        return cls(
            data=data,
            window=window,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            normalize=normalize,
        )

    @staticmethod
    def _load_numeric_csv(path: Path, delimiter: str) -> np.ndarray:
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            for row in reader:
                numeric = []
                for value in row:
                    try:
                        numeric.append(float(value))
                    except ValueError:
                        continue
                if numeric:
                    rows.append(numeric)
        if not rows:
            raise ValueError(f"no numeric data found in {path}")
        widths = {len(r) for r in rows}
        max_width = max(widths)
        matrix = np.zeros((len(rows), max_width), dtype=np.float64)
        for i, r in enumerate(rows):
            r = r[:max_width]
            matrix[i, : len(r)] = r
        return matrix

    def _compute_indices(self) -> np.ndarray:
        max_start = self.data.shape[0] - self.window - 1
        if max_start < 0:
            raise ValueError("time-series shorter than required window+1")
        indices = np.arange(0, max_start + 1, self.stride)
        if self.shuffle:
            np.random.shuffle(indices)
        return indices

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self.iter_batches()

    def iter_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        total = len(self.indices)
        for start in range(0, total, self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            windows, targets = [], []
            for idx in batch_idx:
                window_slice = self.data[idx : idx + self.window]
                target = self.data[idx + self.window]
                windows.append(window_slice)
                targets.append(target)
            x_batch = np.stack(windows, axis=0)
            y_batch = np.stack(targets, axis=0)
            yield x_batch, y_batch

    def epoch(self, shuffle: Optional[bool] = None):
        if shuffle is not None:
            self.shuffle = shuffle
        self.indices = self._compute_indices()
