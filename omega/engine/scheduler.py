import numpy as np
from typing import Dict, Any


class AdaptiveScheduler:
    """
    Ajusta hiperparámetros de ACP y unidades locales usando métricas suavizadas
    para evitar oscilaciones. Responde a anomalías SCSI de forma gradual y
    devuelve trazas diagnósticas para análisis posterior.
    """

    def __init__(
        self,
        target_gain: float = 2.0,
        patience: int = 3,
        min_alpha: float = 1e-4,
        max_alpha: float = 5e-3,
        min_lambda: float = 0.90,
        max_lambda: float = 0.999,
        smooth_factor: float = 0.2,
        scsi_angle_deg: float = 75.0,
        scsi_eig_threshold: float = 0.7,
        alpha_step: float = 0.1,
        lambda_step: float = 0.01,
    ):
        self.target_gain = target_gain
        self.patience = patience
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.smooth_factor = float(np.clip(smooth_factor, 0.05, 0.9))
        self.scsi_angle_threshold = np.deg2rad(scsi_angle_deg)
        self.scsi_eig_threshold = scsi_eig_threshold
        self.alpha_step = alpha_step
        self.lambda_step = lambda_step

        self._no_improve = 0
        self._best_gain = -np.inf
        self._ema_gain = None
        self._ema_rho = None
        self._ema_orth = None

    def _update_ema(self, value: float, ema: float | None) -> float:
        if ema is None:
            return value
        return (1.0 - self.smooth_factor) * ema + self.smooth_factor * value

    def step(self, metrics: Dict[str, Any], agent):
        gain = metrics.get("gain", 0.0)
        spectral_radius = metrics.get("spectral_radius", 0.0)
        orth_error = metrics.get("orth_error", 0.0)
        scsi_angle = metrics.get("max_scsi_angle", 0.0)
        scsi_eig = metrics.get("max_scsi_eig", 0.0)

        if gain > self._best_gain:
            self._best_gain = gain
            self._no_improve = 0
        else:
            self._no_improve += 1

        self._ema_gain = self._update_ema(gain, self._ema_gain)
        self._ema_rho = self._update_ema(spectral_radius, self._ema_rho)
        self._ema_orth = self._update_ema(orth_error, self._ema_orth)

        alpha_adjust = 0.0
        lambda_adjust = 0.0

        if self._ema_rho is not None:
            if self._ema_rho > 0.95:
                agent.acp.alpha = min(self.max_alpha, agent.acp.alpha * (1.0 + self.alpha_step))
                alpha_adjust = self.alpha_step
            elif self._ema_rho < 0.7:
                agent.acp.alpha = max(self.min_alpha, agent.acp.alpha * (1.0 - self.alpha_step))
                alpha_adjust = -self.alpha_step

        tol = agent.acp.orthogonality_tol
        if self._ema_orth is not None and tol > 0:
            if self._ema_orth > tol * 4:
                agent.acp.l = max(self.min_lambda, agent.acp.l * (1.0 - self.lambda_step))
                lambda_adjust = -self.lambda_step
            elif self._ema_gain is not None and self._ema_gain > self.target_gain and self._ema_orth < tol:
                agent.acp.l = min(self.max_lambda, agent.acp.l * (1.0 + self.lambda_step))
                lambda_adjust = self.lambda_step

        scsi_event = False
        if scsi_angle > self.scsi_angle_threshold and scsi_eig > self.scsi_eig_threshold:
            agent.acp.l = max(self.min_lambda, agent.acp.l * (1.0 - 2 * self.lambda_step))
            agent.acp.alpha = min(self.max_alpha, agent.acp.alpha * (1.0 + 2 * self.alpha_step))
            scsi_event = True

        stop = self._no_improve >= self.patience
        return {
            "alpha": agent.acp.alpha,
            "lambda": agent.acp.l,
            "stop": stop,
            "patience_left": max(0, self.patience - self._no_improve),
            "ema_gain": self._ema_gain,
            "ema_rho": self._ema_rho,
            "ema_orth": self._ema_orth,
            "alpha_adjust": alpha_adjust,
            "lambda_adjust": lambda_adjust,
            "scsi_event": scsi_event,
        }
