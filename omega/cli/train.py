from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from omega.engine.pipeline import train_agent, OMEGAAgent
from omega.engine.scheduler import AdaptiveScheduler
from omega.utils.checkpoint import CheckpointManager
from omega.mods import get_module
from omega.mods.base import BaseDataset, BaseEncoder
from omega.mods.tts_text.decoder import OmegaAudioDecoder


def load_config(module_name: str, config_path: str | None) -> Dict[str, Any]:
    if config_path is not None:
        path = Path(config_path)
    else:
        registry_entry = get_module(module_name)
        default_cfg = registry_entry.get("default_config")
        if default_cfg is None:
            raise FileNotFoundError(f"No default config for module '{module_name}'")
        path = Path(default_cfg)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def instantiate_module(module_name: str, config: Dict[str, Any]) -> tuple[BaseEncoder, BaseDataset, np.dtype]:
    entry = get_module(module_name)
    encoder_cls = entry["encoder"]
    dataset_cls = entry["dataset"]

    encoder_cfg = config.get("encoder", {})
    encoder = encoder_cls(**encoder_cfg)

    dataset_cfg = config.get("dataset", {})
    dataset = dataset_cls.from_config(encoder=encoder, config=dataset_cfg)  # type: ignore[arg-type]
    dataset_dtype = getattr(dataset, "dtype", None)
    if dataset_dtype is None:
        dtype_name = dataset_cfg.get("dtype", "float32")
        dtype_np = np.float32 if dtype_name == "float32" else np.float64
    else:
        dtype_np = np.dtype(dataset_dtype)
    return encoder, dataset, dtype_np


def build_agent(config: Dict[str, Any], dataset: BaseDataset, d_model: int, dtype: np.dtype) -> OMEGAAgent:
    agent_cfg = config.get("agent", {})
    agent_cfg.setdefault("d_model", d_model)
    agent_cfg.setdefault("dtype", dtype)
    train_cfg = config.get("training", {})
    agent_cfg.setdefault("scsi_log_interval", int(train_cfg.get("scsi_log_interval", 250)))
    return OMEGAAgent(**agent_cfg)


def _load_decoder_state(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "W_real": data["W_real"],
        "W_imag": data["W_imag"],
        "P_real": data["P_real"],
        "P_imag": data["P_imag"],
    }


def _decode_predictions(
    agent: OMEGAAgent,
    dataset: BaseDataset,
    decoder: OmegaAudioDecoder,
    sample_rate: int,
    output_dir: Path,
) -> None:
    try:
        import soundfile as sf  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("soundfile is required to export decoded waveforms.") from exc

    if hasattr(dataset, "epoch"):
        dataset.epoch(shuffle=False)

    predicted: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    targets: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)

    for windows, targets_batch, meta in dataset:
        meta = meta or {}
        preprojected = bool(meta.get("preprojected"))
        context = meta.get("context")
        context_preprojected = bool(meta.get("context_preprojected", False))
        meta_list = meta.get("metadata", [{}] * len(windows))

        if preprojected:
            projected_windows = np.asarray(windows, dtype=agent.dtype, copy=False)
            projected_targets = np.asarray(targets_batch, dtype=agent.dtype, copy=False)
        else:
            projected_windows = agent.project_windows(windows)
            projected_targets = agent.project_batch(targets_batch)

        if context is not None:
            if context_preprojected:
                projected_context = np.asarray(context, dtype=agent.dtype, copy=False)
            else:
                projected_context = agent.project_windows(context)
        else:
            projected_context = projected_windows
            context_preprojected = preprojected

        for idx in range(projected_targets.shape[0]):
            window_vec = projected_windows[idx]
            x_t = window_vec[-1]
            ctx = (
                projected_context[idx]
                if projected_context is not None and projected_context.ndim == 3
                else projected_context
            )
            pred = agent.predict_latent(
                x_t,
                context=ctx,
                preprojected=preprojected,
                context_preprojected=context_preprojected,
            )
            meta_item = meta_list[idx] if idx < len(meta_list) else {}
            sample_idx = int(meta_item.get("sample_index", 0))
            offset = int(meta_item.get("offset", 0))
            predicted[sample_idx].append((offset, np.asarray(pred, dtype=agent.dtype)))
            targets[sample_idx].append((offset, projected_targets[idx].astype(agent.dtype, copy=False)))

    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx, pred_list in predicted.items():
        if not pred_list:
            continue
        pred_ordered = np.stack([vec for _, vec in sorted(pred_list, key=lambda item: item[0])])
        pred_audio = decoder.decode(pred_ordered)
        sf.write(output_dir / f"sample_{sample_idx:04d}_pred.wav", pred_audio, sample_rate)

        tgt_list = targets.get(sample_idx, [])
        if tgt_list:
            tgt_ordered = np.stack([vec for _, vec in sorted(tgt_list, key=lambda item: item[0])])
            tgt_audio = decoder.decode(tgt_ordered)
            sf.write(output_dir / f"sample_{sample_idx:04d}_target.wav", tgt_audio, sample_rate)
    print(f"Decoded waveforms written to {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OMEGA modular trainer CLI")
    parser.add_argument("--module", required=True, help="Registered module name (e.g., nlp)")
    parser.add_argument("--config", help="Path to module config JSON")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--checkpoint-dir", help="Override checkpoint directory")
    parser.add_argument("--dtype", choices=["float64", "float32"], help="Override dataset dtype")
    parser.add_argument("--shuffle", action="store_true", help="Force shuffling per epoch")
    parser.add_argument("--decode-dir", help="Directory to save decoded audio (tts_text module)")
    parser.add_argument("--decoder-state", help="Optional path to decoder state (.npz) for decoding")
    args = parser.parse_args(argv)

    config = load_config(args.module, args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.checkpoint_dir is not None:
        config.setdefault("training", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.dtype is not None:
        config.setdefault("dataset", {})["dtype"] = args.dtype
    if args.shuffle:
        config.setdefault("training", {})["shuffle"] = True

    encoder, dataset, dtype_np = instantiate_module(args.module, config)
    agent = build_agent(config, dataset=dataset, d_model=encoder.d_model, dtype=dtype_np)

    train_cfg = config.setdefault("training", {})
    epochs = int(train_cfg.get("epochs", 1))
    shuffle = bool(train_cfg.get("shuffle", False))
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1))

    checkpoint_manager = CheckpointManager(checkpoint_dir)
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler = AdaptiveScheduler(**scheduler_cfg)

    history = train_agent(
        agent,
        dataset,
        epochs=epochs,
        shuffle=shuffle,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        checkpoint_every=checkpoint_every,
    )

    stats_dir = Path(checkpoint_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    history_path = stats_dir / f"training_history_{args.module}.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training history saved to {history_path}")

    if args.decode_dir and args.module == "tts_text":
        dataset_cfg = config.get("dataset", {})
        sample_rate = int(dataset_cfg.get("sample_rate", 16000))
        decoder_cfg = config.get("decoder", {})
        audio_encoder = getattr(dataset, "audio_encoder", None)
        if audio_encoder is None:
            raise RuntimeError("Audio encoder reference not available for decoding.")
        decoder = OmegaAudioDecoder(
            d_model=encoder.d_model,
            frame_size=audio_encoder.frame_size,
            hop_size=audio_encoder.hop_size,
            fft_size=decoder_cfg.get("fft_size"),
            smoothing=decoder_cfg.get("smoothing", 0.1),
            rls_lambda=decoder_cfg.get("rls_lambda", 0.995),
            alpha=decoder_cfg.get("alpha", 1e-3),
            seed=decoder_cfg.get("seed"),
            dtype=dtype_np,
        )
        if args.decoder_state:
            decoder.set_state(_load_decoder_state(Path(args.decoder_state)))
            print(f"Loaded decoder state from {args.decoder_state}")
        _decode_predictions(agent, dataset, decoder, sample_rate, Path(args.decode_dir))


if __name__ == "__main__":
    main()
