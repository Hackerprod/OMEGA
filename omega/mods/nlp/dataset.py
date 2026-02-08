from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterator, Tuple, Optional

import numpy as np

from omega.data.text_loader import TextWindowDataLoader
from omega.mods.base import BaseDataset
from omega.mods.nlp.encoder import ContinuousTextEncoder

Batch = Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]


class TextWindowDataset(BaseDataset):
    """Dataset wrapper around TextWindowDataLoader using a ContinuousTextEncoder."""

    def __init__(self, loader: TextWindowDataLoader):
        self.loader = loader
        self.dtype = loader.data.dtype

    @classmethod
    def from_config(
        cls,
        encoder: ContinuousTextEncoder,
        config: Dict[str, Any],
    ) -> "TextWindowDataset":
        path = config["text_path"]
        dtype = np.float32 if config.get("dtype", "float64") == "float32" else np.float64
        loader = TextWindowDataLoader.from_path(
            path=path,
            encoder=encoder,
            window=int(config.get("window", 16)),
            batch_size=int(config.get("batch_size", 4)),
            stride=int(config.get("stride", 1)),
            shuffle=bool(config.get("shuffle", False)),
            encoding=config.get("text_encoding", "utf-8"),
            max_chars=config.get("max_chars"),
            normalize=bool(config.get("normalize", False)),
            dtype=dtype,
            memmap_path=config.get("memmap_path"),
            chunk_chars=int(config.get("chunk_chars", 65536)),
        )
        return cls(loader)

    def __iter__(self) -> Iterator[Batch]:
        for windows, targets in self.loader:
            meta = {"preprojected": True}
            yield windows, targets, meta

    def epoch(self, shuffle: bool = False) -> None:
        self.loader.epoch(shuffle=shuffle)

    def steps_per_epoch(self) -> Optional[int]:
        return len(self.loader.indices)
