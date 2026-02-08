from __future__ import annotations

from typing import Dict, Any

from omega.mods.nlp import ContinuousTextEncoder, TextWindowDataset
from omega.mods.tts import ContinuousAudioEncoder, AudioWindowDataset


MOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nlp": {
        "encoder": ContinuousTextEncoder,
        "dataset": TextWindowDataset,
        "default_config": "configs/nlp.json",
    },
    "tts": {
        "encoder": ContinuousAudioEncoder,
        "dataset": AudioWindowDataset,
        "default_config": "configs/tts.json",
    },
}


def get_module(name: str) -> Dict[str, Any]:
    if name not in MOD_REGISTRY:
        raise KeyError(f"Module '{name}' is not registered.")
    return MOD_REGISTRY[name]
