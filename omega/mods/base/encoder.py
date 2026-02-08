from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEncoder(ABC):
    """Interface for modality-specific encoders producing float trajectories."""

    def __init__(self, d_model: int):
        self.d_model = int(d_model)

    @abstractmethod
    def encode(self, source: Any) -> np.ndarray:
        """Encode raw input into a (steps, d_model) float matrix."""

    def __call__(self, source: Any) -> np.ndarray:
        return self.encode(source)
