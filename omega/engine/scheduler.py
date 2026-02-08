import numpy as np
from typing import Dict, Any


class AdaptiveScheduler:
    """
    Ajusta hiperparámetros de ACP/local units en función de métricas agregadas.
    Simple heurística para CPU: reduce alpha cuando la estabilidad es baja
    y afloja lambda si el progreso se estanca.
    """

    def __init__(
        self,
        target_gain: float = 2.0,
        patience: int = 3,
        min_alpha: float = 1e-4,
        max_alpha: float = 5e-3,
        min_lambda: float = 0.90,
        max_lambda: float = 0.999,
    ):
        self.target_gain = target_gain
        self.patience = patience
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        self._no_improve = 0
        self._best_gain = -np.inf

    def step(self, metrics: Dict[str, Any], agent):
        gain = metrics.get("gain", 0.0)
        spectral_radius = metrics.get("spectral_radius", 0.0)
        orth_error = metrics.get("orth_error", 0.0)

        if gain > self._best_gain:
            self._best_gain = gain
            self._no_improve = 0
        else:
            self._no_improve += 1

        # Ajuste de alpha (regularización RLS) basado en radio espectral
        if spectral_radius > 0.95:
            agent.acp.alpha = min(self.max_alpha, agent.acp.alpha * 1.2)
        elif spectral_radius < 0.7:
            agent.acp.alpha = max(self.min_alpha, agent.acp.alpha * 0.8)

        # Ajuste de lambda (decay en Arnoldi) para mejorar ortogonalidad
        if orth_error > agent.acp.orthogonality_tol * 5:
            agent.acp.l = max(self.min_lambda, agent.acp.l * 0.98)
        elif gain > self.target_gain and orth_error < agent.acp.orthogonality_tol:
            agent.acp.l = min(self.max_lambda, agent.acp.l * 1.01)

        # Early stopping sugerido
        stop = self._no_improve >= self.patience
        return {
            "alpha": agent.acp.alpha,
            "lambda": agent.acp.l,
            "stop": stop,
            "patience_left": max(0, self.patience - self._no_improve),
        }
