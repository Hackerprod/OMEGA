import pytest

from omega.mods.tts_text.encoder import TextTrajectoryEncoder
from omega.mods.tts_text.dataset import TextSpeechDataset


def test_text_speech_dataset_missing_manifest(tmp_path):
    encoder = TextTrajectoryEncoder(d_model=16)
    config = {
        "manifest_path": tmp_path / "missing.jsonl",
        "window": 4,
        "batch_size": 1,
    }
    with pytest.raises(FileNotFoundError):
        TextSpeechDataset.from_config(encoder, config)
