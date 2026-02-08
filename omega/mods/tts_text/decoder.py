from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import librosa
from numpy.lib.stride_tricks import sliding_window_view


class GriffinLimDecoder:
    """
    Simple continuous-to-waveform decoder using Griffin-Lim.

    The latent sequence is interpreted as a log-magnitude spectrogram. This is a
    placeholder decoder intended for experimentation; production-quality TTS would
    replace it with a neural vocoder.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        n_iter: int = 32,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_iter = n_iter

    def decode(self, latent_sequence: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        latent_sequence : np.ndarray, shape (steps, features)
            Continuous latent trajectory produced by OMEGA.

        Returns
        -------
        waveform : np.ndarray
            Time-domain signal reconstructed via Griffin-Lim.
        """
        if latent_sequence.ndim != 2:
            raise ValueError("Expected latent_sequence with shape (frames, features).")

        mag = np.exp(latent_sequence.T).astype(np.float32)
        waveform = librosa.griffinlim(
            mag,
            n_iter=self.n_iter,
            hop_length=self.hop_length,
            win_length=self.n_fft,
        )
        return waveform.astype(np.float32)


class OmegaAudioDecoder:
    """
    Omega-native decoder that maps latent trajectories back to waveforms
    using a learnable local ISTFT. Parameters are updated with Recursive
    Least Squares so the decoder can adapt on CPU alongside ACP.
    """

    def __init__(
        self,
        d_model: int,
        frame_size: int = 1024,
        hop_size: int = 256,
        fft_size: Optional[int] = None,
        smoothing: float = 0.1,
        rls_lambda: float = 0.995,
        alpha: float = 1e-3,
        seed: Optional[int] = 1729,
        dtype: np.dtype = np.float32,
    ):
        self.d_model = int(d_model)
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.fft_size = self._next_pow_two(fft_size or frame_size)
        self.freq_bins = self.fft_size // 2 + 1
        self.latent_smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.rls_lambda = float(np.clip(rls_lambda, 0.8, 0.9999))
        self.alpha = float(max(alpha, 1e-6))
        self.dtype = np.dtype(dtype)

        rng = np.random.default_rng(seed)
        self.aug_dim = self.d_model + 1
        scale = 1.0 / np.sqrt(self.aug_dim)
        self.W_real = (rng.standard_normal((self.freq_bins, self.aug_dim)) * scale).astype(self.dtype)
        self.W_imag = (rng.standard_normal((self.freq_bins, self.aug_dim)) * scale).astype(self.dtype)
        self.P_real = np.eye(self.aug_dim, dtype=self.dtype) / self.alpha
        self.P_imag = np.eye(self.aug_dim, dtype=self.dtype) / self.alpha
        self._eye = np.eye(self.aug_dim, dtype=self.dtype)

        self.window = np.hanning(self.frame_size).astype(self.dtype)
        self._window_sq = self.window**2

    # ------------------------------------------------------------------ #
    def decode(self, latents: np.ndarray) -> np.ndarray:
        lat_aug = self._prepare_latents(latents)
        if lat_aug.size == 0:
            return np.zeros(0, dtype=self.dtype)
        spectrum = self._latents_to_complex(lat_aug)
        frames = np.fft.irfft(spectrum, n=self.fft_size, axis=1)[:, : self.frame_size]
        frames = frames.astype(self.dtype, copy=False) * self.window
        waveform = self._overlap_add(frames)
        return np.clip(waveform, -1.0, 1.0)

    def train_step(self, latents: np.ndarray, target_waveform: np.ndarray) -> float:
        lat_aug = self._prepare_latents(latents)
        waveform = np.asarray(target_waveform, dtype=self.dtype).flatten()
        frames = self._frame_waveform(waveform)
        if frames.shape[0] == 0 or lat_aug.shape[0] == 0:
            return 0.0

        if frames.shape[0] != lat_aug.shape[0]:
            count = min(frames.shape[0], lat_aug.shape[0])
            frames = frames[:count]
            lat_aug = lat_aug[:count]

        windowed = frames * self.window
        target_spec = np.fft.rfft(windowed, n=self.fft_size, axis=1)
        preds = self._latents_to_complex(lat_aug)
        error = target_spec - preds
        mse = float(np.mean(np.abs(error) ** 2))

        for x_vec, spec in zip(lat_aug, target_spec):
            self.P_real, self.W_real = self._rls_update(self.P_real, self.W_real, x_vec, spec.real)
            self.P_imag, self.W_imag = self._rls_update(self.P_imag, self.W_imag, x_vec, spec.imag)
        return mse

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "W_real": self.W_real.copy(),
            "W_imag": self.W_imag.copy(),
            "P_real": self.P_real.copy(),
            "P_imag": self.P_imag.copy(),
        }

    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        self.W_real = np.asarray(state["W_real"], dtype=self.dtype).copy()
        self.W_imag = np.asarray(state["W_imag"], dtype=self.dtype).copy()
        self.P_real = np.asarray(state["P_real"], dtype=self.dtype).copy()
        self.P_imag = np.asarray(state["P_imag"], dtype=self.dtype).copy()

    # Internal helpers --------------------------------------------------
    def _prepare_latents(self, latents: np.ndarray) -> np.ndarray:
        if latents.ndim == 1:
            latents = latents[None, :]
        if latents.ndim != 2:
            raise ValueError("Latents must be 2D (frames, d_model).")
        lat = np.asarray(latents, dtype=self.dtype)
        if 0.0 < self.latent_smoothing < 1.0 and lat.shape[0] > 1:
            lat = self._smooth(lat, self.latent_smoothing)
        ones = np.ones((lat.shape[0], 1), dtype=self.dtype)
        return np.concatenate([lat, ones], axis=1)

    def _latents_to_complex(self, lat_aug: np.ndarray) -> np.ndarray:
        real = lat_aug @ self.W_real.T
        imag = lat_aug @ self.W_imag.T
        real = np.tanh(real)
        imag = np.tanh(imag)
        return real + 1j * imag

    def _frame_waveform(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.size < self.frame_size:
            return np.zeros((0, self.frame_size), dtype=self.dtype)
        frames = sliding_window_view(waveform, self.frame_size)[:: self.hop_size]
        return frames.astype(self.dtype, copy=False)

    def _overlap_add(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] == 0:
            return np.zeros(0, dtype=self.dtype)
        total = self.hop_size * (frames.shape[0] - 1) + self.frame_size
        signal = np.zeros(total, dtype=self.dtype)
        norm = np.zeros(total, dtype=self.dtype)
        for idx, frame in enumerate(frames):
            start = idx * self.hop_size
            signal[start : start + self.frame_size] += frame[: self.frame_size]
            norm[start : start + self.frame_size] += self._window_sq
        mask = norm > 1e-6
        signal[mask] /= norm[mask]
        return signal

    def _rls_update(self, P: np.ndarray, W: np.ndarray, x_vec: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = x_vec.reshape(-1, 1)
        Px = P @ x
        denom = self.rls_lambda + float(x.T @ Px)
        gain = Px / denom
        pred = W @ x
        err = target.reshape(-1, 1) - pred
        W = W + err @ gain.T
        P = P - gain @ x.T @ P
        P = P / self.rls_lambda
        P = 0.5 * (P + P.T)
        P = P + self.alpha * self._eye
        return P, W

    @staticmethod
    def _smooth(latents: np.ndarray, alpha: float) -> np.ndarray:
        smoothed = np.empty_like(latents)
        smoothed[0] = latents[0]
        for idx in range(1, latents.shape[0]):
            smoothed[idx] = alpha * smoothed[idx - 1] + (1.0 - alpha) * latents[idx]
        return smoothed

    @staticmethod
    def _next_pow_two(value: int) -> int:
        return 1 << (int(value - 1).bit_length())
