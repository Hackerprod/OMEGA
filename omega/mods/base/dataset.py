from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional, Dict, Any

import numpy as np


Batch = Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]


class BaseDataset(ABC):
    """Interface for datasets exposing windowed batches compatible with OMEGAAgent."""

    @abstractmethod
    def __iter__(self) -> Iterator[Batch]:
        ...

    @abstractmethod
    def epoch(self, shuffle: bool = False) -> None:
        ...

    def steps_per_epoch(self) -> Optional[int]:
        return None

    @classmethod
    def from_config(cls, *args, **kwargs) -> "BaseDataset":
        raise NotImplementedError(f"{cls.__name__} must implement from_config.")
