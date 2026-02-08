from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf

from omega.mods.base import BaseDataset
from omega.mods.tts.encoder import ContinuousAudioEncoder
from omega.mods.tts_text.encoder import TextTrajectoryEncoder


Batch = Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]


class TextSpeechDataset(BaseDataset):
    """
    Loads text/audio pairs, encodes both modalities, and yields pre-projected windows.

    Expected manifest format: JSON lines with fields
        {
          "text": "...",
          "audio_path": "relative/or/absolute/path.wav"
        }
    Config fields:
        manifest_path (str)        : required path to jsonl manifest
        text_field (str)           : default "text"
        audio_field (str)          : default "audio_path"
        base_dir (str)             : optional base directory for audio paths
        max_samples (int)          : optional limit
        sample_rate (int)          : resample target for audio
        window (int)               : window size for OMEGA
        batch_size (int)           : batch size for iteration
        stride (int)               : stride between windows
        dtype (str)                : "float32" (default) or "float64"
        shuffle (bool)             : shuffle windows per epoch
        audio_encoder (dict)       : overrides for ContinuousAudioEncoder
    """

    def __init__(
        self,
        audio_windows: List[np.ndarray],
        audio_targets: List[np.ndarray],
        text_windows: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        window: int,
        batch_size: int,
        shuffle: bool,
        dtype: np.dtype,
    ):
        self.audio_windows = audio_windows
        self.audio_targets = audio_targets
        self.text_windows = text_windows
        self.metadata = metadata
        self.window = window
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = np.dtype(dtype)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, encoder: TextTrajectoryEncoder, config: Dict[str, Any]) -> "TextSpeechDataset":
        manifest_path = Path(config["manifest_path"])
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        base_dir = Path(config.get("base_dir", manifest_path.parent))
        text_field = config.get("text_field", "text")
        audio_field = config.get("audio_field", "audio_path")
        max_samples = config.get("max_samples")
        sample_rate = int(config.get("sample_rate", 16000))
        window = int(config.get("window", 16))
        stride = int(config.get("stride", 1))
        batch_size = int(config.get("batch_size", 4))
        shuffle = bool(config.get("shuffle", False))
        dtype = np.float32 if config.get("dtype", "float32") == "float32" else np.float64

        audio_encoder_cfg = config.get("audio_encoder", {})
        audio_encoder = ContinuousAudioEncoder(
            d_model=encoder.d_model,
            frame_size=int(audio_encoder_cfg.get("frame_size", 1024)),
            hop_size=int(audio_encoder_cfg.get("hop_size", 256)),
            smoothing=float(audio_encoder_cfg.get("smoothing", 0.1)),
            seed=audio_encoder_cfg.get("seed", 1234),
        )

        entries: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                entries.append(sample)
                if max_samples and len(entries) >= max_samples:
                    break

        audio_windows: List[np.ndarray] = []
        audio_targets: List[np.ndarray] = []
        text_windows: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []

        for idx, sample in enumerate(entries):
            text = sample.get(text_field, "")
            audio_path = sample.get(audio_field)
            if not audio_path:
                continue
            wav_path = (base_dir / audio_path).resolve()
            if not wav_path.exists():
                continue

            waveform, sr = sf.read(wav_path, always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = waveform.astype(np.float32)
            if sr != sample_rate:
                waveform = cls._resample_waveform(waveform, sr, sample_rate)

            audio_latents = audio_encoder.encode(waveform)
            text_latents = encoder.encode(text)
            if audio_latents.shape[0] <= window or text_latents.shape[0] == 0:
                continue

            aligned_text = cls._match_length(text_latents, audio_latents.shape[0])

            for start in range(0, audio_latents.shape[0] - window, stride):
                a_window = audio_latents[start : start + window]
                a_target = audio_latents[start + window]
                t_window = aligned_text[start : start + window]

                audio_windows.append(a_window.astype(dtype, copy=False))
                audio_targets.append(a_target.astype(dtype, copy=False))
                text_windows.append(t_window.astype(dtype, copy=False))
                meta.append({"sample_index": idx, "offset": start})

        if not audio_windows:
            raise ValueError("No usable pairs found in manifest. Check window/stride settings or dataset integrity.")

        return cls(
            audio_windows=audio_windows,
            audio_targets=audio_targets,
            text_windows=text_windows,
            metadata=meta,
            window=window,
            batch_size=batch_size,
            shuffle=shuffle,
            dtype=dtype,
        )

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(len(self.audio_windows))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            a_windows = np.stack([self.audio_windows[i] for i in batch_idx]).astype(self.dtype, copy=False)
            a_targets = np.stack([self.audio_targets[i] for i in batch_idx]).astype(self.dtype, copy=False)
            t_windows = np.stack([self.text_windows[i] for i in batch_idx]).astype(self.dtype, copy=False)
            metas = [self.metadata[i] for i in batch_idx]
            yield a_windows, a_targets, {
                "context": t_windows,
                "preprojected": True,
                "context_preprojected": True,
                "metadata": metas,
            }

    def epoch(self, shuffle: bool = False) -> None:
        if shuffle != self.shuffle:
            self.shuffle = shuffle

    def steps_per_epoch(self) -> Optional[int]:
        return len(self.audio_windows)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resample_waveform(waveform: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
        if src_sr == tgt_sr:
            return waveform
        import librosa

        return librosa.resample(waveform, orig_sr=src_sr, target_sr=tgt_sr).astype(np.float32)

    @staticmethod
    def _match_length(latents: np.ndarray, target_len: int) -> np.ndarray:
        if latents.shape[0] == target_len:
            return latents
        x_old = np.linspace(0.0, 1.0, latents.shape[0])
        x_new = np.linspace(0.0, 1.0, target_len)
        stretched = np.empty((target_len, latents.shape[1]), dtype=latents.dtype)
        for dim in range(latents.shape[1]):
            stretched[:, dim] = np.interp(x_new, x_old, latents[:, dim])
        return stretched
