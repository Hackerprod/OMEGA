import numpy as np

from omega.engine.pipeline import OMEGAAgent, build_synthetic_loader, train_agent
from omega.engine.scheduler import AdaptiveScheduler
from omega.utils.checkpoint import CheckpointManager


def test_float32_pipeline(tmp_path):
    loader = build_synthetic_loader(
        steps=20,
        d_model=4,
        batch=2,
        window=2,
        normalize=False,
        dtype=np.float32,
    )
    agent = OMEGAAgent(d_model=4, dtype=np.float32)
    scheduler = AdaptiveScheduler()
    checkpoint_dir = tmp_path / "ckpts"
    manager = CheckpointManager(checkpoint_dir)

    history = train_agent(
        agent,
        loader,
        epochs=1,
        shuffle=False,
        scheduler=scheduler,
        checkpoint_manager=manager,
        checkpoint_every=5,
    )

    assert history, "Expected at least one epoch summary"
    assert agent.acp.A.dtype == np.float32
    assert agent.layers[0].W.dtype == np.float32
    assert agent.memory.memory.dtype == np.float32
