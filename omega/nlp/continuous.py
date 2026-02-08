"""Backwards-compatible import for ContinuousTextEncoder/Decoder."""

from omega.mods.nlp.encoder import ContinuousTextEncoder


class ContinuousTextDecoder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ContinuousTextDecoder is a placeholder; integrate your own VQ-VAE o vocoder textual."
        )


__all__ = ["ContinuousTextEncoder", "ContinuousTextDecoder"]
