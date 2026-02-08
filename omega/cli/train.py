from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from omega.engine.pipeline import train_agent, OMEGAAgent
from omega.engine.scheduler import AdaptiveScheduler
from omega.utils.checkpoint import CheckpointManager
from omega.mods import get_module
from omega.mods.base import BaseDataset, BaseEncoder


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


def instantiate_module(module_name: str, config: Dict[str, Any]) -> tuple[BaseEncoder, BaseDataset]:
    entry = get_module(module_name)
    encoder_cls = entry["encoder"]
    dataset_cls = entry["dataset"]

    encoder_cfg = config.get("encoder", {})
    encoder = encoder_cls(**encoder_cfg)

    dataset_cfg = config.get("dataset", {})
    dataset = dataset_cls.from_config(encoder=encoder, config=dataset_cfg)  # type: ignore[arg-type]
    return encoder, dataset


def build_agent(config: Dict[str, Any], d_model: int) -> OMEGAAgent:
    agent_cfg = config.get("agent", {})
    agent_cfg.setdefault("d_model", d_model)
    return OMEGAAgent(**agent_cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OMEGA modular trainer CLI")
    parser.add_argument("--module", required=True, help="Registered module name (e.g., nlp)")
    parser.add_argument("--config", help="Path to module config JSON")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--checkpoint-dir", help="Override checkpoint directory")
    parser.add_argument("--dtype", choices=["float64", "float32"], help="Override dataset dtype")
    parser.add_argument("--shuffle", action="store_true", help="Force shuffling per epoch")
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

    encoder, dataset = instantiate_module(args.module, config)
    agent = build_agent(config, d_model=encoder.d_model)

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


if __name__ == "__main__":
    main()
