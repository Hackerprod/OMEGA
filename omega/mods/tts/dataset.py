from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Tuple

import numpy as np

from omega.mods.base import BaseDataset
from omega.mods.tts.encoder import ContinuousAudioEncoder

Batch = Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]


class AudioWindowDataset(BaseDataset):
    """
    Dataset that loads audio waveforms (either synthetic or from Hugging Face)
    encodes them with ContinuousAudioEncoder, and yields sliding windows suitable
    for OMEGAAgent.
    """

    def __init__(
        self,
        encoded_sequences: List[np.ndarray],
        window: int,
        batch_size: int,
        shuffle: bool,
        dtype: np.dtype,
    ):
        self.encoded_sequences = encoded_sequences
        self.window = max(1, int(window))
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.dtype = np.dtype(dtype)
        self._windows: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._prepare_samples()

    @classmethod
    def from_config(
        cls,
        encoder: ContinuousAudioEncoder,
        config: Dict[str, Any],
    ) -> "AudioWindowDataset":
        dtype = np.float32 if config.get("dtype", "float32") == "float32" else np.float64
        window = int(config.get("window", 16))
        batch_size = int(config.get("batch_size", 4))
        shuffle = bool(config.get("shuffle", False))
        encoded_sequences: List[np.ndarray] = []

        if "waveforms" in config:
            raw_waveforms = [np.asarray(w, dtype=np.float32) for w in config["waveforms"]]
            for waveform in raw_waveforms:
                encoded_sequences.append(encoder.encode(waveform))
        else:
            encoded_sequences = cls._load_from_huggingface(encoder, config)

        if not encoded_sequences:
            raise ValueError("AudioWindowDataset received no encoded sequences.")

        return cls(
            encoded_sequences=encoded_sequences,
            window=window,
            batch_size=batch_size,
            shuffle=shuffle,
            dtype=dtype,
        )

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(len(self._windows))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            windows = np.stack([self._windows[i] for i in batch_idx]).astype(self.dtype, copy=False)
            targets = np.stack([self._targets[i] for i in batch_idx]).astype(self.dtype, copy=False)
            metas = [self._metadata[i] for i in batch_idx]
            yield windows, targets, {"clips": metas, "preprojected": True}

    def epoch(self, shuffle: bool = False) -> None:
        if shuffle != self.shuffle:
            self.shuffle = shuffle

    def steps_per_epoch(self) -> Optional[int]:
        return len(self._windows)

    def _prepare_samples(self) -> None:
        self._windows.clear()
        self._targets.clear()
        self._metadata.clear()

        for clip_idx, features in enumerate(self.encoded_sequences):
            if features.shape[0] <= self.window:
                continue
            total_steps = features.shape[0] - self.window
            for step in range(total_steps):
                window = features[step : step + self.window].astype(self.dtype, copy=False)
                target = features[step + self.window].astype(self.dtype, copy=False)
                self._windows.append(window)
                self._targets.append(target)
                self._metadata.append({"clip": clip_idx, "step": step})

    @staticmethod
    def _load_from_huggingface(
        encoder: ContinuousAudioEncoder,
        config: Dict[str, Any],
    ) -> List[np.ndarray]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install 'datasets' to use the TTS module.") from exc

        dataset_name = config.get("dataset_name", "ylacombe/google-colombian-spanish")
        subset = config.get("dataset_subset")
        split = config.get("dataset_split", "train")
        sample_rate = int(config.get("sample_rate", 16000))
        max_clips = int(config.get("max_clips", 64))
        streaming = bool(config.get("streaming", False))

        ds = load_dataset(dataset_name, subset, split=split, streaming=streaming)

        encoded_sequences: List[np.ndarray] = []
        count = 0
        for sample in ds:
            audio = sample.get("audio")
            if audio is None:
                continue
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            waveform = AudioWindowDataset._resample_if_needed(waveform, sr, sample_rate)
            encoded = encoder.encode(waveform)
            if encoded.shape[0] > encoder.d_model:
                encoded_sequences.append(encoded)
                count += 1
            if max_clips and count >= max_clips:
                break
        return encoded_sequences

    @staticmethod
    def _resample_if_needed(waveform: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
        if sr == target_sr:
            return waveform
        from scipy.signal import resample  # local import to avoid imposing dependency at import time

        duration = waveform.shape[0] / sr
        new_length = int(duration * target_sr)
        if new_length <= 0:
            return waveform
        return resample(waveform, new_length).astype(np.float32, copy=False)
