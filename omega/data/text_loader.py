import numpy as np
from pathlib import Path
from typing import Optional

from omega.data.loader import TimeSeriesDataLoader
from omega.nlp.continuous import ContinuousTextEncoder


class TextWindowDataLoader(TimeSeriesDataLoader):
    """
    Loader especializado para corpora textuales continuos.
    Convierte el texto en trayectorias densas mediante ContinuousTextEncoder.
    """

    def __init__(
        self,
        encoded: np.ndarray,
        window: int,
        batch_size: int,
        stride: int,
        shuffle: bool,
        normalize: bool = False,
    ):
        super().__init__(
            data=encoded,
            window=window,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            normalize=normalize,
        )

    @classmethod
    def from_path(
        cls,
        path: str,
        encoder: ContinuousTextEncoder,
        window: int = 16,
        batch_size: int = 1,
        stride: int = 1,
        shuffle: bool = False,
        encoding: str = "utf-8",
        max_chars: Optional[int] = None,
        normalize: bool = False,
    ):
        text = Path(path).read_text(encoding=encoding)
        if max_chars is not None:
            text = text[:max_chars]
        encoded = encoder.encode_text(text)
        return cls(
            encoded=encoded,
            window=window,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            normalize=normalize,
        )
