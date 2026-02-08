from omega.engine.pipeline import (
    OMEGAAgent,
    build_synthetic_loader,
    generate_dynamic_signal,
    train_agent,
)
from omega.cli.train import main as cli_main

__all__ = [
    "OMEGAAgent",
    "build_synthetic_loader",
    "generate_dynamic_signal",
    "train_agent",
]


if __name__ == "__main__":
    cli_main()
