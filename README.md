# OMEGA 

## Overview

OMEGA is a research prototype that explores **Arnoldi-Causal Projection (ACP)** and **local recursive learning** as an alternative to transformer-based language and sequence models. The system maintains a compact Krylov subspace, updates operators through Recursive Least Squares (RLS), and enforces structural stability with Spectral Checksum Structural Identity (SCSI). The implementation is CPU-first and relies exclusively on NumPy / SciPy.

> **Status:** Early-stage prototype. Expect missing features, limited benchmarks, and unoptimised numerical routines.

## Core Capabilities

- **Local learning without backpropagation:** Difference Target Propagation (DTP) and RLS updates per layer (`omega/core`).
- **Spectral stability:** Projection of operators to keep `ρ(A) < 1` and monitoring via SCSI (`omega/core/acp.py`, `omega/brain/regime.py`).
- **Symbolic hooks:** Competitive prototypes with Lipschitz constraints for neuro-symbolic reasoning (`omega/brain/symbolic.py`).
- **Persistent memory:** NTK-inspired matrix with uniqueness checks, soft attention reads, and decay (`omega/memory/persistent.py`).
- **Flexible data ingestion:** Numeric time-series loader with batching/normalisation and token-free continuous NLP encoding (`omega/data`).
- **Checkpointable training loop:** Adaptive scheduler that tunes `α` and `λ`, plus full-state snapshot/restore (`main.py`).

## Repository Layout

```
omega/
 ├── core/            # ACP module and local predictive units
 ├── brain/           # Regime detector, symbolic bridge
 ├── memory/          # Persistent memory matrix
 ├── data/            # Time-series and continuous-text loaders
 ├── engine/          # Adaptive hyper-parameter scheduler
 ├── nlp/             # Continuous (token-free) text encoder
 └── utils/           # Checkpoint manager
experiments/          # Example benchmark scripts
configs/              # (Reserved for future configuration files)
main.py               # Training / evaluation entry point
``` 

## Requirements

| Component | Version |
| --- | --- |
| Python | 3.11 or newer |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.10 |

Install dependencies via:

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The provided `requirements.txt` lists the minimum packages needed by the codebase.

## Quick Start

### Synthetic Benchmark

```bash
python main.py --epochs 1 --steps 120
```

This generates a sinusoidal signal, trains the agent for one epoch, prints summary metrics, and stores checkpoints under `checkpoints/`.

### Numeric Time Series (e.g., OHLCV)

```bash
python main.py \
  --data-path data/btc_live_simulation_30days.csv \
  --window 16 \
  --batch-size 8 \
  --epochs 5 \
  --shuffle \
  --normalize
```

Key behaviours:

- Numeric CSV files are parsed column-wise; non-numeric columns such as timestamps are ignored automatically.
- Z-score normalisation is optional but recommended for raw financial data.
- Sliding windows feed the ACP module; checkpoints are generated per epoch.

### Continuous Text (Token-Free NLP)

```bash
python main.py \
  --text-path Corpus/small_corpus.txt \
  --window 16 \
  --batch-size 4 \
  --epochs 2 \
  --text-max-chars 250000 \
  --encoder-smoothing 0.2 \
  --d-model 32
```

Continuous text mode uses `ContinuousTextEncoder` to create smoothed vector trajectories from characters. When dealing with large corpora, limit the number of characters (`--text-max-chars`) or reduce `d_model` to control memory consumption. A decoder back to text is **not** included; integrate your own VQ-VAE or vocoder if generation is required.

## Training Loop and Checkpoints

- **Scheduler:** `omega/engine/scheduler.py` inspects per-epoch statistics (error, gain, spectral radius, orthogonality drift, SCSI metrics) and adjusts RLS regularisation (`alpha`) and decay (`lambda`). Early-stop suggestions are triggered when improvements stagnate.
- **Checkpoints:** Full model state (ACP, layers, regime detector, symbolic bridge, memory, input projection) is saved to `checkpoints/epoch_XXXX.pkl`. Metadata includes the metrics recorded above.
- **Resume Training:** `python main.py --resume epoch_0007 ...` reloads state and continues from the next epoch.
- **Metrics Log:** `checkpoints/training_history.json` collects per-epoch summaries (`error_pre`, `error_post`, `gain`, `rho(A)`, SCSI angles/eigenvalues, memory hit rate, alpha, lambda).

## Roadmap

1. **Continuous decoder:** Integrate a VQ-VAE or similar module to reconstruct text/audio from latent trajectories.
2. **Streaming encoders:** Process corpora in chunks to reduce peak memory usage during continuous encoding.
3. **SCSI logging:** Aggregate or debounce anomaly notifications to avoid console saturation on long jobs.
4. **Performance:** Port heavy routines (RLS, SVD/QR) to Numba or C++ to improve throughput on multi-core CPUs.
5. **Benchmarks:** Evaluate on public datasets (e.g., Forex, speech, curated corpora) with well-defined metrics (MAE, semantic stability).
6. **Multimodal encoders:** Provide plug-ins for audio/vision embeddings and extend persistent memory to cross-modal cues.
7. **Decoding pipeline:** Design evaluation protocols for tasks such as intention forecasting, topic shifts, or textual generation using the symbolic bridge.

## FAQ

**Is this a drop-in replacement for transformers?**  
No. OMEGA demonstrates an alternative learning strategy. It prioritises causal identification and stability, not large-scale token modelling. Transformers remain superior for most state-of-the-art NLP benchmarks.

**Does it support GPU acceleration?**  
Not currently. All kernels are written in NumPy / SciPy. Porting to GPU would require custom kernels or an accelerated linear algebra backend.

**Can it handle multimodal inputs?**  
The architecture accepts any continuous vector stream. To operate multimodally, provide encoders for each modality and feed their trajectories through the same training loop. Further integration work is required for production use.

**How does it perform on large corpora?**  
Processing multi-hundred-megabyte corpora in Python/NumPy is computationally expensive. Expect long runtimes unless encoders and ACP updates are optimised (compiled kernels, reduced `k_max`, increased `svd_interval`, etc.).

**Is there a text/audio decoder?**  
No. Generation requires an external model to map the learned trajectories back to discrete tokens or waveforms.

## Contributing

1. Fork the repository and create a feature branch.
2. Document your changes and add tests or scripts where relevant.
3. Ensure checkpoints and historical logs remain compatible with your modifications.
4. Submit a pull request describing the rationale and results.

## License

No explicit license is included. Treat the code as research material. If you reuse significant portions, credit the “OMEGA – Arnoldi-Causal Projection Prototype” project.
