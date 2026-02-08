"""Base interfaces for modality-specific components."""

from .encoder import BaseEncoder
from .dataset import BaseDataset

__all__ = ["BaseEncoder", "BaseDataset"]
