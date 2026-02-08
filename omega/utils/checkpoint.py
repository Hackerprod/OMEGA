import pickle
from pathlib import Path
from typing import Any, Dict, Optional


class CheckpointManager:
    """
    Serializa el estado completo del agente para reanudar entrenamiento.
    Usa pickle por simplicidad (solo en entornos controlados).
    """

    def __init__(self, directory: str):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.pkl"

    def save(self, name: str, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        payload = {"state": state, "meta": metadata or {}}
        with self._path(name).open("wb") as fh:
            pickle.dump(payload, fh)

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._path(name)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return payload
