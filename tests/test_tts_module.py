import numpy as np

from omega.mods.tts.encoder import ContinuousAudioEncoder
from omega.mods.tts.dataset import AudioWindowDataset


def _sine_wave(length: int, freq: float = 3.0) -> np.ndarray:
    t = np.linspace(0, 1, length, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t)


def test_audio_dataset_from_waveforms():
    encoder = ContinuousAudioEncoder(
        d_model=16,
        frame_size=32,
        hop_size=16,
        fft_size=64,
        filter_bands=12,
        conv_channels=(24,),
        smoothing=0.0,
        pre_emphasis=0.05,
        seed=0,
    )
    waveforms = [
        _sine_wave(320, freq=2.0),
        _sine_wave(400, freq=3.5),
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
    assert batch_windows.shape[0] == 2
    assert meta is not None and meta.get("preprojected") is True


def test_audio_encoder_expand_and_state_roundtrip():
    encoder = ContinuousAudioEncoder(
        d_model=8,
        frame_size=32,
        hop_size=16,
        fft_size=64,
        filter_bands=10,
        conv_channels=(18,),
        smoothing=0.2,
        pre_emphasis=0.1,
        seed=42,
    )
    waveform = _sine_wave(256, freq=5.0)
    latents = encoder.encode(waveform)
    assert latents.ndim == 2
    assert latents.shape[1] == 8
    assert not np.isnan(latents).any()

    restored = ContinuousAudioEncoder(
        d_model=8,
        frame_size=32,
        hop_size=16,
        fft_size=64,
        filter_bands=10,
        conv_channels=(18,),
        smoothing=0.2,
        pre_emphasis=0.1,
        seed=0,
    )
    restored.set_state(encoder.get_state())
    np.testing.assert_allclose(latents, restored.encode(waveform), atol=1e-6)

    waveform_out, frames, spec = encoder.expand_audio(waveform)
    assert waveform_out.shape[0] == waveform.shape[0]
    assert frames.shape[1] == encoder.frame_size
    assert spec.shape[1] == encoder.freq_bins
