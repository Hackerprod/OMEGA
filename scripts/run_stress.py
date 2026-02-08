#!/usr/bin/env python
"""
Stress harness for OMEGA ACP.

Genera datasets (si faltan), ejecuta entrenamiento de una época y mide throughput (pasos/seg).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "benchmarks" / "datasets" / "config.json"
GENERATED_DIR = ROOT / "benchmarks" / "generated"
RESULTS_PATH = ROOT / "benchmarks" / "stress_latest.json"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_benchmark_data import (  # type: ignore
    load_config,
    generate_audio,
    generate_text,
)
from omega.engine.pipeline import OMEGAAgent, train_agent
from omega.data.loader import TimeSeriesDataLoader
from omega.mods.nlp.encoder import ContinuousTextEncoder
from omega.mods.nlp.dataset import TextWindowDataset
from omega.engine.scheduler import AdaptiveScheduler
from omega.utils.checkpoint import CheckpointManager


def ensure_audio(profile: Dict[str, Any]) -> Path:
    path = GENERATED_DIR / f"audio_{profile['name']}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_audio(profile)
    return path


def ensure_text(profile: Dict[str, Any]) -> Path:
    path = GENERATED_DIR / f"text_{profile['name']}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_text(profile)
    return path


def run_audio(profile: Dict[str, Any], dtype: np.dtype) -> Dict[str, Any]:
    data_path = ensure_audio(profile)
    series = np.load(data_path).astype(dtype, copy=False)
    loader = TimeSeriesDataLoader(
        data=series,
        window=int(profile["window"]),
        batch_size=int(profile["batch_size"]),
        stride=int(profile["stride"]),
        shuffle=False,
        normalize=True,
        dtype=dtype,
    )
    agent = OMEGAAgent(d_model=series.shape[1])
    scheduler = AdaptiveScheduler()
    checkpoint_manager = CheckpointManager(ROOT / "benchmarks" / "tmp_checkpoints")
    start = time.perf_counter()
    history = train_agent(
        agent,
        loader,
        epochs=1,
        shuffle=False,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        checkpoint_every=10_000,
    )
    elapsed = time.perf_counter() - start
    samples = sum(item["samples"] for item in history)
    throughput = samples / elapsed if elapsed > 0 else 0.0
    return {
        "profile": profile["name"],
        "type": "audio",
        "samples": samples,
        "elapsed": elapsed,
        "throughput": throughput,
    }


def run_text(profile: Dict[str, Any], dtype: np.dtype) -> Dict[str, Any]:
    text_path = ensure_text(profile)
    encoder = ContinuousTextEncoder(d_model=int(profile["d_model"]), smoothing=0.2)
    memmap_path = GENERATED_DIR / f"text_{profile['name']}.dat"
    dataset_cfg = {
        "text_path": str(text_path),
        "window": int(profile["window"]),
        "batch_size": int(profile["batch_size"]),
        "stride": int(profile["stride"]),
        "shuffle": False,
        "dtype": "float32" if dtype == np.float32 else "float64",
        "memmap_path": str(memmap_path),
        "normalize": False,
    }
    dataset = TextWindowDataset.from_config(encoder, dataset_cfg)
    loader = dataset.loader
    agent = OMEGAAgent(d_model=encoder.d_model)
    scheduler = AdaptiveScheduler()
    checkpoint_manager = CheckpointManager(ROOT / "benchmarks" / "tmp_checkpoints")
    start = time.perf_counter()
    history = train_agent(
        agent,
        loader,
        epochs=1,
        shuffle=False,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        checkpoint_every=10_000,
    )
    elapsed = time.perf_counter() - start
    samples = sum(item["samples"] for item in history)
    throughput = samples / elapsed if elapsed > 0 else 0.0
    return {
        "profile": profile["name"],
        "type": "text",
        "samples": samples,
        "elapsed": elapsed,
        "throughput": throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="Run stress benchmarks.")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float32")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--profiles", nargs="*", default=None, help="Subset of profiles to run.")
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    cfg = load_config()

    records: List[Dict[str, Any]] = []
    for profile in cfg["audio"]:
        if args.profiles and profile["name"] not in args.profiles:
            continue
        print(f"[AUDIO] {profile['name']} ...")
        records.append(run_audio(profile, dtype))

    for profile in cfg["text"]:
        if args.profiles and profile["name"] not in args.profiles:
            continue
        print(f"[TEXT] {profile['name']} ...")
        records.append(run_text(profile, dtype))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
