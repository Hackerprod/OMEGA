#!/usr/bin/env python
"""
Generate synthetic benchmark datasets (audio/time-series and continuous text).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "benchmarks" / "datasets" / "config.json"
GENERATED_DIR = ROOT / "benchmarks" / "generated"


def load_config() -> Dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def generate_audio(profile: Dict[str, Any], seed: int = 1729) -> Path:
    rng = np.random.default_rng(seed)
    sample_rate = int(profile["sample_rate"])
    duration = float(profile["duration_seconds"])
    channels = int(profile["channels"])
    steps = int(sample_rate * duration)
    t = np.arange(steps) / sample_rate

    freqs = np.array([60, 250, 440, 880], dtype=np.float32)
    signal = np.zeros((steps, channels), dtype=np.float32)
    for ch in range(channels):
        phase = rng.uniform(0, math.pi, size=len(freqs))
        components = np.sin(2 * math.pi * freqs[:, None] * t + phase[:, None])
        weighted = components.T @ rng.uniform(0.1, 0.9, size=len(freqs))
        signal[:, ch] = weighted + 0.02 * rng.standard_normal(steps)

    path = GENERATED_DIR / f"audio_{profile['name']}.npy"
    np.save(path, signal.astype(np.float32))
    return path


def generate_text(profile: Dict[str, Any], seed: int = 2718) -> Path:
    rng = np.random.default_rng(seed)
    vocab = list(" abcdefghijklmnopqrstuvwxyz,.!?;:-")
    length = int(profile["characters"])
    chars = rng.choice(vocab, size=length)
    text = "".join(chars)

    path = GENERATED_DIR / f"text_{profile['name']}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark data.")
    parser.add_argument("--profile", type=str, required=True, help="Benchmark profile name.")
    parser.add_argument("--seed", type=int, default=1729, help="RNG seed.")
    args = parser.parse_args()

    config = load_config()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for group in ("audio", "text"):
        for profile in config[group]:
            if profile["name"] == args.profile:
                if group == "audio":
                    path = generate_audio(profile, args.seed)
                    print(f"Generated {path}")
                else:
                    path = generate_text(profile, args.seed)
                    print(f"Generated {path}")
                return

    raise SystemExit(f"Profile '{args.profile}' not found in {CONFIG_PATH}")


if __name__ == "__main__":
    main()
