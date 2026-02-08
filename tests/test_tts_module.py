import numpy as np

from omega.mods.tts.encoder import ContinuousAudioEncoder
from omega.mods.tts.dataset import AudioWindowDataset


def test_audio_dataset_from_waveforms():
    encoder = ContinuousAudioEncoder(d_model=8, frame_size=16, hop_size=8, smoothing=0.0, seed=0)
    waveforms = [
        np.sin(np.linspace(0, 4 * np.pi, 200, dtype=np.float32)),
        np.cos(np.linspace(0, 6 * np.pi, 256, dtype=np.float32)),
    ]
    dataset = AudioWindowDataset.from_config(
        encoder=encoder,
        config={
            "waveforms": waveforms,
            "window": 4,
            "batch_size": 2,
            "shuffle": False,
            "dtype": "float32",
        },
    )

    batch_windows, batch_targets, meta = next(iter(dataset))
    assert batch_windows.shape[-1] == encoder.d_model
    assert batch_targets.shape[-1] == encoder.d_model
    assert meta is not None and meta.get("preprojected") is True
