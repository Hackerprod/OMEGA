from __future__ import annotations

from typing import Optional

import numpy as np
import librosa


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
