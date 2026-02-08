# OMEGA v3 – Arnoldi-Causal Projection Prototype

OMEGA v3 is a research prototype that explores **Arnoldi-Causal Projection (ACP)** and **local recursive updates** as an alternative to transformer-style sequence models. The system keeps a compact Krylov basis, adjusts its operators via Recursive Least Squares (RLS), and enforces structural stability through Spectral Checksum Structural Identity (SCSI). Everything runs on CPU using NumPy/SciPy; optional accelerators (Numba or native extensions) can be enabled for additional speed.

> **Status:** experimental. Expect incomplete features, aggressive logging, and un‑optimised code paths.

---
## Repository Layout
```
omega/
  core/        # ACP/DTP core (modality-agnostic)
  engine/      # Training pipeline, scheduler
  brain/       # Regime detector + symbolic bridge
  memory/      # Persistent memory matrix
  data/        # Generic loaders (time-series, text windows)
  mods/        # Vertical modules (e.g., NLP, TTS)
    base/      # BaseEncoder / BaseDataset interfaces
    nlp/       # Continuous text encoder/dataset
    tts/       # Speech encoder/dataset
  cli/         # CLI entry points (`python -m omega.cli.train`)
  utils/       # Shared helpers (checkpoint manager, etc.)
configs/        # JSON configs per module
experiments/    # Optional benchmarks / stress tests
main.py         # Thin wrapper delegating to the CLI
```

---
## Core Capabilities
- **Local learning without backpropagation:** Difference Target Propagation (DTP) and RLS updates per layer (`omega/core`).  
- **Spectral stability:** ACP projects operators to keep `ρ(A) < 1` and monitors basis drift with SCSI (`omega/core/acp.py`, `omega/brain/regime.py`).  
- **Symbolic hooks:** Competitive prototypes with Lipschitz-constrained updates (`omega/brain/symbolic.py`).  
- **Persistent memory:** NTK-inspired matrix with write uniqueness, soft attention reads, and decay (`omega/memory/persistent.py`).  
- **Modular front-ends:** Encoders/datasets registered per modality (NLP, TTS, …) under `omega/mods/`.  
- **Pipeline & CLI:** Shared training loop (`omega/engine/pipeline.py`) and CLI (`omega/cli/train.py`) that load module configs and orchestrate checkpoints.  

---
## Requirements

| Component | Purpose | Version |
| --- | --- | --- |
| Python | Runtime | 3.11+ |
| NumPy | Linear algebra | ≥ 1.24 |
| SciPy | Signal ops / SVD | ≥ 1.10 |
| Datasets | Hugging Face loader (TTS) | ≥ 2.18 |
| SoundFile | Audio decoding (TTS) | ≥ 0.12 |
| Numba *(optional)* | JIT kernels | ≥ 0.59 |

Install via:
```bash
python -m venv .venv
source .venv/Scripts/activate      # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
`requirements.txt` lists the exact packages; remove entries you do not need (e.g., skip `datasets` if you will not run the TTS module).

---
## Quick Start
### Modular CLI
The CLI is the recommended entry point. Every module ships with a JSON config:
```bash
python -m omega.cli.train --module <name> --config configs/<name>.json
```
The CLI loads the config, instantiates the module’s encoder/dataset, and launches the common training loop.

### Example – Continuous Text (NLP)
```bash
python -m omega.cli.train \
  --module nlp \
  --config configs/nlp.json \
  --epochs 1 \
  --shuffle
```
This encodes the configured corpus into continuous trajectories and trains for one epoch. The default config expects you to provide a local text file; adjust the path or max characters as needed.

### Example – Speech (TTS, Google Colombian Spanish)
```bash
python -m omega.cli.train \
  --module tts \
  --config configs/tts.json \
  --epochs 1
```
Notes:
- Uses the Hugging Face dataset `ylacombe/google-colombian-spanish` (downloaded on first use).  
- The config limits the run to the first clips via `max_clips`. Increase this value for deeper experiments.  
- Requires the optional dependencies `datasets` and `soundfile` listed above.

### Synthetic Benchmark (scripted)
You can still exercise the raw pipeline without the CLI:
```bash
python - <<'PY'
from omega.engine.pipeline import OMEGAAgent, build_synthetic_loader, train_agent
from omega.engine.scheduler import AdaptiveScheduler
from omega.utils.checkpoint import CheckpointManager

agent = OMEGAAgent(d_model=32)
loader = build_synthetic_loader(steps=120, d_model=32, batch=4, window=8)
scheduler = AdaptiveScheduler()
history = train_agent(
    agent,
    loader,
    epochs=1,
    shuffle=False,
    scheduler=scheduler,
    checkpoint_manager=CheckpointManager("checkpoints"),
    checkpoint_every=1,
)
print(history)
PY
```
This uses the shared pipeline functions directly for a quick sanity check.

---
## Modular Architecture
- **Core (`omega/core`, `omega/engine`, `omega/brain`, `omega/memory`)** – numerical backbone; modality-agnostic.  
- **Mods (`omega/mods/<name>`)** – each modality provides encoders/datasets (and optional decoders) built on `BaseEncoder` / `BaseDataset`. Current examples: `nlp` and `tts`.  
- **Configs (`configs/<name>.json`)** – describe encoder/dataset/training defaults, including dtypes and checkpoint directories.  
- **CLI (`omega/cli/train.py`)** – resolves the module via the registry, loads the config, and executes the pipeline.  

### Adding a New Module
1. Create `omega/mods/<module>/` with classes inheriting `BaseEncoder` / `BaseDataset`.  
2. Register the module in `omega/mods/registry.py` (optionally point to a default config).  
3. Provide `configs/<module>.json`.  
4. (Optional) Add notebooks / scripts in `experiments/<module>/`.  
5. Launch training: `python -m omega.cli.train --module <module> --config configs/<module>.json`.  

---
## Training Loop & Checkpoints
- **Scheduler** (`omega/engine/scheduler.py`): keeps exponential moving averages of gain, spectral radius, and orthogonality error; adjusts `alpha`/`lambda` smoothly; reacts to SCSI anomalies before forcing rollbacks.  
- **Checkpoints**: full agent state serialized to `checkpoints/epoch_XXXX.pkl`. Metadata includes per-epoch metrics so runs are reproducible.  
- **Resume**: `python -m omega.cli.train --module nlp --config ... --resume epoch_0003`.  
- **Metrics log**: Each run produces `training_history_<module>.json` with metrics such as `error_pre/post`, `gain`, `ρ(A)`, SCSI angles, memory hit rate, α, λ, etc.  

---
## Performance & Tooling
- `experiments/profile_acp.py` – microbenchmark for ACP/DTP kernels (supports cProfile).  
- `scripts/bench_quick.py` – quick throughput/regression check (compare against `benchmarks/baseline.json`).  
- `scripts/run_stress.py` – long-form stress test (text/audio). Dumps reports to `benchmarks/stress_latest.json`.  
- Optional accelerators: set `ACPModule(..., compression_backend="randomized")` or install the native extension under `omega/core/native`.  
- Use `--dtype float32` and dataset memmaps for large corpora to decrease RAM pressure.  

**Heads-up:** The default logging prints every SCSI anomaly, which can produce a lot of output. Throttle or aggregate the messages before large-scale runs.

---
## Roadmap
1. Continuous decoder (VQ-VAE or other vocoder) to reconstruct audio/text from latent trajectories.  
2. Streaming encoders to avoid materialising very large corpora in RAM.  
3. Smarter handling of SCSI alerts (aggregation, telemetry).  
4. Optimised kernels (Numba/C++/SIMD) for ACP/DTP operations.  
5. Benchmarks on public datasets (Forex, speech, curated text corpora) with standard metrics (MAE, perceptual scores, semantic coherence).  
6. Multimodal plug-ins beyond NLP/TTS, plus richer use of the symbolic bridge.  
7. Cleaner decoding/evaluation pipelines for downstream tasks (topic tracking, intention forecasting, etc.).  

---
## FAQ
**Is this a transformer replacement?** No. OMEGA focuses on causal identification and stability. Transformers remain superior for large-scale NLP benchmarks.  
**GPU support?** Not yet. Everything runs on CPU; porting to GPU would require bespoke kernels.  
**Does it handle large corpora well?** The prototype is CPU-first but still Python-heavy. Expect long runtimes unless you enable the optimised kernels, reduce `k_max`, or use memmapped datasets.  
**Decoder provided?** No. Generation back to discrete tokens/audio requires an external decoder (e.g., VQ-VAE, diffusion vocoder).  

---
## Contributing
1. Fork the repo and create a feature branch.  
2. Document your changes and add unit tests or scripts where relevant.  
3. Keep checkpoints / history files compatible with the existing format.  
4. Open a pull request summarising the motivation and results.  

---
## License
No explicit licence is included. Treat the code as research material; if you reuse substantial portions, credit the “OMEGA v3 – Arnoldi-Causal Projection Prototype” project. 
