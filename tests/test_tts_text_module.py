import numpy as np
import pytest

from omega.mods.tts_text.dataset import TextSpeechDataset
from omega.mods.tts_text.decoder import OmegaAudioDecoder
from omega.mods.tts_text.encoder import TextTrajectoryEncoder


def test_text_speech_dataset_missing_manifest(tmp_path):
    encoder = TextTrajectoryEncoder(d_model=16)
    config = {
        "manifest_path": tmp_path / "missing.jsonl",
        "window": 4,
        "batch_size": 1,
    }
    with pytest.raises(FileNotFoundError):
        TextSpeechDataset.from_config(encoder, config)


def test_text_encoder_tokenizer_inserts_pauses():
    encoder = TextTrajectoryEncoder(d_model=16, language="en")
    tokens = encoder.tokenizer.tokenize("Hello, world!")
    assert tokens[0] == "sil"
    assert tokens[-1] == "sil"
    assert "pau" in tokens


def test_text_encoder_generates_smooth_latents():
    encoder = TextTrajectoryEncoder(d_model=32, seed=42, language="es")
    latents = encoder.encode("Hola mundo")

    assert latents.ndim == 2
    assert latents.shape[1] == 32
    assert latents.dtype == np.float32
    assert latents.shape[0] >= encoder.vowel_duration
    assert not np.isnan(latents).any()

    diffs = np.abs(np.diff(latents, axis=0))
    assert float(np.max(diffs)) < 5.0


def test_text_encoder_duration_expansion():
    encoder = TextTrajectoryEncoder(d_model=16, language="en", vowel_duration=8, consonant_duration=2)
    _, frame_ids, aux, durations = encoder.expand_text("aaa")
    assert frame_ids.size >= 8 * 3
    assert aux.shape[0] == frame_ids.size
    assert len(durations) >= 3


def test_text_encoder_duration_override():
    encoder = TextTrajectoryEncoder(
        d_model=8,
        language="en",
        duration_overrides={"pau": 12, "rr": 9},
        pause_duration=4,
    )
    tokens, frame_ids, _, durations = encoder.expand_text("Arr!")
    assert "pau" in tokens
    assert frame_ids.size >= 9 + 12  # override applied
    assert 9 in durations or any(d >= 9 for d in durations)


def test_text_encoder_state_roundtrip():
    encoder = TextTrajectoryEncoder(d_model=12, language="es", seed=0)
    original = encoder.encode("Hola mundo polarizado")
    state = encoder.get_state()
    new_encoder = TextTrajectoryEncoder(d_model=12, language="es", seed=123)
    new_encoder.set_state(state)
    reconstructed = new_encoder.encode("Hola mundo polarizado")
    np.testing.assert_allclose(original, reconstructed, atol=1e-6)


def test_text_tokenizer_custom_pause_tokens():
    encoder = TextTrajectoryEncoder(d_model=8, language="en", pause_tokens=["breath"])
    assert "breath" in encoder.pause_tokens
    assert "breath" in encoder.tokenizer.pause_tokens


def test_omega_decoder_rls_adaptation():
    rng = np.random.default_rng(7)
    latents = rng.standard_normal((12, 8)).astype(np.float32)

    teacher = OmegaAudioDecoder(d_model=8, frame_size=64, hop_size=32, fft_size=128, smoothing=0.0, seed=3)
    target_wave = teacher.decode(latents)

    student = OmegaAudioDecoder(d_model=8, frame_size=64, hop_size=32, fft_size=128, smoothing=0.0, seed=123)
    pred_before = student.decode(latents)
    mse_before = float(np.mean((pred_before - target_wave) ** 2))

    for _ in range(5):
        student.train_step(latents, target_wave)

    pred_after = student.decode(latents)
    mse_after = float(np.mean((pred_after - target_wave) ** 2))
    assert mse_after < mse_before * 0.4

    state = student.get_state()
    clone = OmegaAudioDecoder(d_model=8, frame_size=64, hop_size=32, fft_size=128, smoothing=0.0, seed=0)
    clone.set_state(state)
    np.testing.assert_allclose(pred_after, clone.decode(latents), atol=1e-6)
