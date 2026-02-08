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
        dtype: np.dtype = np.float64,
    ):
        super().__init__(
            data=encoded,
            window=window,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            normalize=normalize,
            dtype=dtype,
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
        dtype: np.dtype = np.float64,
        memmap_path: Optional[str] = None,
        chunk_chars: int = 65536,
    ):
        path_obj = Path(path)
        if memmap_path is not None:
            memmap_file = Path(memmap_path)
            mmap = encoder.encode_file_to_memmap(
                text_path=path_obj,
                output_path=memmap_file,
                encoding=encoding,
                chunk_chars=chunk_chars,
                dtype=dtype,
                max_chars=max_chars,
            )
            encoded = np.memmap(memmap_file, mode="r", dtype=dtype, shape=mmap.shape)
        else:
            text = path_obj.read_text(encoding=encoding)
            if max_chars is not None:
                text = text[:max_chars]
            encoded = encoder.encode_text(text).astype(dtype, copy=False)
        return cls(
            encoded=encoded,
            window=window,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            normalize=normalize,
            dtype=dtype,
        )
