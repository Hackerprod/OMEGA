import numpy as np

from omega.engine.scheduler import AdaptiveScheduler
from omega.core.acp import ACPModule


def test_scheduler_smoothing_and_adjustments():
    scheduler = AdaptiveScheduler(smooth_factor=0.3, alpha_step=0.05, lambda_step=0.02)
    acp = ACPModule(d_model=8)

    metrics = {
        "gain": 1.0,
        "spectral_radius": 0.96,
        "orth_error": 0.001,
        "max_scsi_angle": 0.1,
        "max_scsi_eig": 0.1,
    }
    adjust1 = scheduler.step(metrics, type("Agent", (), {"acp": acp})())
    assert "ema_gain" in adjust1
    assert adjust1["alpha_adjust"] > 0

    metrics["spectral_radius"] = 0.5
    metrics["gain"] = 3.0
    metrics["orth_error"] = 1.0
    metrics["max_scsi_angle"] = np.deg2rad(80)
    metrics["max_scsi_eig"] = 1.0
    adjust2 = scheduler.step(metrics, type("Agent", (), {"acp": acp})())

    assert adjust2["lambda_adjust"] <= 0
    assert adjust2["scsi_event"] is True
