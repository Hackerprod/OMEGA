## `./main.py`

```py
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from omega.core.acp import ACPModule
from omega.core.lpc import LocalPredictiveUnit
from omega.brain.regime import RegimeDetector
from omega.brain.symbolic import SymbolicInterface, LogicEngine
from omega.memory.persistent import PersistentMemory
from omega.data.loader import TimeSeriesDataLoader
from omega.engine.scheduler import AdaptiveScheduler
from omega.utils.checkpoint import CheckpointManager
from omega.data.text_loader import TextWindowDataLoader
from omega.nlp.continuous import ContinuousTextEncoder

class OMEGAAgent:
    """
    OMEGA V3: Integrated Architecture.
    - Uses DTP (Difference Target Propagation) for non-linear learning.
    - Integrates ACP for causal prediction refinement.
    - Activates Memory and Regime Monitoring.
    """
    def __init__(self, d_model=32):
        print("Initializing OMEGA v3 (Integrated & Non-Linear)...")
        self.d = d_model
        self.acp = ACPModule(d_model)
        self.layers = [LocalPredictiveUnit(d_model, d_model) for _ in range(2)]
        self.regime_detector = RegimeDetector(d_model, window_size=15, threshold=1.8)
        self.symbolic_bridge = SymbolicInterface(d_model, n_predicates=20)
        self.memory = PersistentMemory(d_model)
        self.logic_engine = LogicEngine()
        self.input_proj = None
        self.memory_recall_threshold = np.sqrt(self.d) * 0.5
        self.memory_decay = 0.995
        self._commit_stable_state()

    def run_step(self, x_t, x_next, context=None):
        x_t = self._project_input(x_t)
        x_next = self._project_input(x_next)
        context_model = None
        if context is not None:
            context_model = np.stack([self._project_input(vec) for vec in context], axis=0)

        # 1. Forward Pass (Pre-update)
        z = x_t
        if context_model is not None:
            contextual_bias = np.mean(context_model, axis=0)
            z = 0.7 * z + 0.3 * contextual_bias
        for layer in self.layers:
            z = layer.forward(z)
        raw_pred = z
        
        # 2. Refinement via ACP (Integration)
        refined_pred = self.acp.refine_prediction(raw_pred)

        # 3. Learning Phase (CTP/DTP)
        # Global Goal: Predict x_next (modulado por memoria persistente)
        error_signal = np.linalg.norm(x_next - refined_pred)
        memory_used = False
        memory_hint = None
        global_target = x_next
        if self.memory.cursor > 0 and error_signal > self.memory_recall_threshold:
            memory_hint = self.memory.read(x_t)
            if np.linalg.norm(memory_hint) > 0:
                global_target = 0.7 * x_next + 0.3 * memory_hint
                memory_used = True

        current_target = global_target
        basis_vector = self.acp.Q[:, self.acp.k - 1] if self.acp.k > 0 else None
        for i in reversed(range(len(self.layers))):
            self.layers[i].local_update(current_target, basis_vector)
            basis_vector = self.layers[i].project_basis(basis_vector)
            current_target = self.layers[i].propagate_target(current_target)
        
        # 4. Update ACP Subspace
        self.acp.update_operator(x_t, x_next)
        self.acp.step(seed_vector=x_t)

        # 5. Post-Update Forward Pass (Measuring real convergence)
        z_post = x_t
        for layer in self.layers:
            z_post = layer.forward(z_post)
        refined_post = self.acp.refine_prediction(z_post)
        
        # 6. Brain Integration (Memory & Regime)
        symbol_id, conf = self.symbolic_bridge.map_and_learn(refined_post)
        regime_shift = self.regime_detector.update(refined_post)
        basis_matrix = self.acp.Q[:, :self.acp.k] if self.acp.k > 0 else None
        scsi_anomaly = self.regime_detector.check_scsi(self.acp.scsi_signature, basis_matrix, regime_shift)

        if regime_shift:
            print(f"\n[!] OMEGA: Regime Shift detected at output. Adapting basis...")
            self.regime_detector.reset_baseline()
            # Write key experience to Persistent Memory
            self.memory.write(refined_post, importance=conf)
            self._commit_stable_state()
        elif scsi_anomaly:
            metrics = self.regime_detector.last_scsi_metrics
            angle_deg = np.degrees(metrics.get("angle", 0.0))
            eig_drift = metrics.get("eig_drift", 0.0)
            print(f"\n[!] OMEGA: SCSI anomaly detected (angle={angle_deg:.1f}deg, eig-drift={eig_drift:.3f}). Reverting to last stable state.")
            self._restore_state(self.stable_state)
            self._commit_stable_state()
        else:
            self._commit_stable_state()

        if memory_used and memory_hint is not None:
            self.memory.write(memory_hint, importance=error_signal)
        self.memory.decay(self.memory_decay)

        return {
            "error_pre": np.linalg.norm(x_next - refined_pred),
            "error_post": np.linalg.norm(x_next - refined_post),
            "symbol": symbol_id,
            "memory_used": memory_used
        }
    
    def _project_input(self, vector):
        vector = np.asarray(vector, dtype=np.float64).flatten()
        if vector.shape[0] == self.d and self.input_proj is None:
            return vector
        if self.input_proj is None:
            in_dim = vector.shape[0]
            if in_dim == self.d:
                self.input_proj = np.eye(self.d)
            else:
                rng = np.random.default_rng()
                self.input_proj = rng.standard_normal((self.d, in_dim)) / np.sqrt(in_dim)
        if self.input_proj.shape[1] != vector.shape[0]:
            raise ValueError("Input dimension changed; cannot reuse projection.")
        return self.input_proj @ vector

    def _snapshot_state(self):
        """Captures a full system snapshot for structural rollbacks."""
        return self.get_state()

    def _restore_state(self, snapshot):
        """Restores the system to a previously captured snapshot."""
        self.set_state(snapshot)

    def _commit_stable_state(self):
        """Marks the current state as the new stable configuration."""
        self.stable_state = self._snapshot_state()
        basis_matrix = self.acp.Q[:, :self.acp.k] if self.acp.k > 0 else None
        self.regime_detector.mark_scsi_baseline(self.acp.scsi_signature, basis_matrix)

    def get_state(self):
        return {
            "d": self.d,
            "acp": self.acp.get_state(),
            "layers": [layer.get_state() for layer in self.layers],
            "regime": self.regime_detector.get_state(),
            "symbolic": self.symbolic_bridge.get_state(),
            "memory": self.memory.get_state(),
            "logic_rules": dict(self.logic_engine.rules),
            "input_proj": None if self.input_proj is None else self.input_proj.copy(),
            "memory_threshold": float(self.memory_recall_threshold),
            "memory_decay": float(self.memory_decay),
        }

    def set_state(self, state: Dict[str, Any]):
        self.d = state.get("d", self.d)
        self.acp.set_state(state["acp"])
        for layer, layer_state in zip(self.layers, state["layers"]):
            layer.set_state(layer_state)
        self.regime_detector.set_state(state["regime"])
        self.symbolic_bridge.set_state(state["symbolic"])
        self.memory.set_state(state["memory"])
        self.logic_engine.rules = dict(state.get("logic_rules", {}))
        self.input_proj = None if state.get("input_proj") is None else state["input_proj"].copy()
        self.memory_recall_threshold = float(state.get("memory_threshold", self.memory_recall_threshold))
        self.memory_decay = float(state.get("memory_decay", self.memory_decay))

def generate_dynamic_signal(t):
    """3D Signal projected to 32D with phase shifts"""
    proj = np.sin(np.arange(32) * 0.1)
    if t < 60:
        val = np.sin(0.1 * t)
    else:
        # High frequency phase shift
        val = 0.5 * np.cos(0.4 * t)
    return proj * val + np.random.randn(32) * 0.01

def build_synthetic_loader(
    steps: int,
    d_model: int,
    batch: int,
    window: int,
    normalize: bool = False,
) -> TimeSeriesDataLoader:
    series = np.stack([generate_dynamic_signal(t, d_model) for t in range(steps)], axis=0)
    return TimeSeriesDataLoader(
        series,
        window=window,
        batch_size=batch,
        stride=1,
        shuffle=False,
        normalize=normalize,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OMEGA v3 Trainer")
    parser.add_argument("--d-model", type=int, default=32, help="Dimensión del modelo OMEGA")
    parser.add_argument("--data-path", type=str, default=None, help="Ruta a CSV/NPY con serie temporal")
    parser.add_argument("--delimiter", type=str, default=",", help="Delimitador para CSV (por defecto ',')")
    parser.add_argument("--window", type=int, default=1, help="Tamaño de ventana/contexto para loader")
    parser.add_argument("--batch-size", type=int, default=1, help="Tamaño de mini-batch en CPU")
    parser.add_argument("--stride", type=int, default=1, help="Stride de ventana")
    parser.add_argument("--epochs", type=int, default=1, help="Cantidad de épocas sobre el dataset")
    parser.add_argument("--steps", type=int, default=200, help="Pasos sintéticos si no hay dataset")
    parser.add_argument("--shuffle", action="store_true", help="Barajar ventanas en cada época")
    parser.add_argument("--normalize", action="store_true", help="Normalizar series numéricas (z-score)")
    parser.add_argument("--text-path", type=str, default=None, help="Ruta a corpus de texto continuo")
    parser.add_argument("--text-encoding", type=str, default="utf-8", help="Codificación del archivo de texto")
    parser.add_argument("--text-max-chars", type=int, default=None, help="Limitar caracteres cargados del corpus")
    parser.add_argument("--encoder-smoothing", type=float, default=0.2, help="Factor de suavizado temporal del encoder continuo")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directorio para guardar checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Frecuencia de guardado (épocas)")
    parser.add_argument("--resume", type=str, default=None, help="Nombre de checkpoint a reanudar (sin extensión)")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def train_agent(
    agent: OMEGAAgent,
    loader: TimeSeriesDataLoader,
    epochs: int,
    shuffle: bool,
    scheduler: AdaptiveScheduler,
    checkpoint_manager: CheckpointManager,
    checkpoint_every: int,
    start_epoch: int = 0,
) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    prev_post = None
    for epoch in range(start_epoch, start_epoch + epochs):
        loader.epoch(shuffle=shuffle)
        epoch_metrics: List[Dict[str, Any]] = []
        for window_batch, y_batch in loader:
            for window_t, x_next in zip(window_batch, y_batch):
                x_t = window_t[-1]
                metrics = agent.run_step(x_t, x_next, context=window_t)
                epoch_metrics.append(
                    {
                        "error_pre": float(metrics["error_pre"]),
                        "error_post": float(metrics["error_post"]),
                        "symbol": int(metrics["symbol"]),
                        "memory_used": bool(metrics.get("memory_used", False)),
                    }
                )
        if not epoch_metrics:
            continue

        avg_pre = float(np.mean([m["error_pre"] for m in epoch_metrics]))
        avg_post = float(np.mean([m["error_post"] for m in epoch_metrics]))
        improvement = (avg_pre - avg_post) / avg_pre * 100 if avg_pre > 0 else 0.0
        spectral_radius = agent.acp.spectral_radius
        orth_error = float(agent.acp.last_orth_error)
        monotonic_grad = float(max(agent.acp.last_monotonic_gradient) if agent.acp.last_monotonic_gradient else 0.0)
        scsi_angles = agent.acp.scsi_signature.get("principal_angles") if agent.acp.scsi_signature else None
        max_scsi_angle = float(np.max(scsi_angles)) if scsi_angles is not None and len(scsi_angles) > 0 else 0.0
        scsi_eigs = agent.acp.scsi_signature.get("eigenvalues") if agent.acp.scsi_signature else None
        max_scsi_eig = float(np.max(np.abs(scsi_eigs))) if scsi_eigs is not None and len(scsi_eigs) > 0 else 0.0

        memory_hits = sum(1 for m in epoch_metrics if m["memory_used"])
        epoch_summary = {
            "epoch": epoch,
            "samples": len(epoch_metrics),
            "error_pre": avg_pre,
            "error_post": avg_post,
            "gain": improvement,
            "spectral_radius": spectral_radius,
            "orth_error": orth_error,
            "monotonic_grad": monotonic_grad,
            "max_scsi_angle": max_scsi_angle,
            "max_scsi_eig": max_scsi_eig,
            "memory_hit_rate": memory_hits / len(epoch_metrics),
            "alpha": float(agent.acp.alpha),
            "lambda": float(agent.acp.l),
        }

        adjust = scheduler.step(epoch_summary, agent)
        epoch_summary.update(adjust)
        history.append(epoch_summary)

        print(
            f"Epoch {epoch:03d} | Pre-Err: {avg_pre:.4f} | Post-Err: {avg_post:.4f} | "
            f"Gain: {improvement:.1f}% | rho(A): {spectral_radius:.3f} | "
            f"SCSIdeg: {np.degrees(max_scsi_angle):.1f} | MemHit: {epoch_summary['memory_hit_rate']*100:.1f}%"
        )

        if (epoch - start_epoch + 1) % checkpoint_every == 0 or adjust.get("stop"):
            name = f"epoch_{epoch:04d}"
            checkpoint_manager.save(name, agent.get_state(), metadata=epoch_summary)
            print(f"Checkpoint guardado en {checkpoint_manager._path(name)}")

        if prev_post is not None and avg_post > prev_post and adjust.get("stop"):
            print("Scheduler sugiere detención temprana por falta de mejora.")
            break
        prev_post = avg_post
    return history


def generate_dynamic_signal(t, d_model=32):
    """3D Signal projected to d_model with phase shifts"""
    proj = np.sin(np.arange(d_model) * 0.1)
    if t < 60:
        val = np.sin(0.1 * t)
    else:
        # High frequency phase shift
        val = 0.5 * np.cos(0.4 * t)
    return proj * val + np.random.randn(d_model) * 0.01


def main():
    args = parse_args()
    agent = OMEGAAgent(d_model=args.d_model)
    print()

    if args.data_path and args.text_path:
        raise ValueError("Especifique solo --data-path o --text-path, no ambos.")

    if args.text_path:
        encoder = ContinuousTextEncoder(
            d_model=args.d_model,
            smoothing=args.encoder_smoothing,
        )
        loader = TextWindowDataLoader.from_path(
            path=args.text_path,
            encoder=encoder,
            window=args.window,
            batch_size=args.batch_size,
            stride=args.stride,
            shuffle=args.shuffle,
            encoding=args.text_encoding,
            max_chars=args.text_max_chars,
        )
        print(f"Cargando corpus continuo desde {args.text_path} | pasos disponibles: {loader.data.shape[0]}")
    elif args.data_path:
        loader = TimeSeriesDataLoader.from_path(
            path=args.data_path,
            window=args.window,
            batch_size=args.batch_size,
            stride=args.stride,
            shuffle=args.shuffle,
            delimiter=args.delimiter,
            normalize=args.normalize,
        )
        print(f"Cargando datos desde {args.data_path} | pasos disponibles: {loader.data.shape[0]}")
    else:
        loader = build_synthetic_loader(
            steps=args.steps,
            d_model=agent.d,
            batch=args.batch_size,
            window=args.window,
            normalize=args.normalize,
        )
        print(f"Generando serie sintética ({args.steps} pasos) para entrenamiento.")

    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    start_epoch = 0
    if args.resume:
        payload = checkpoint_manager.load(args.resume)
        if payload is None:
            print(f"[WARN] No se encontró checkpoint {args.resume}, se entrenará desde cero.")
        else:
            agent.set_state(payload["state"])
            meta = payload.get("meta", {})
            start_epoch = int(meta.get("epoch", 0) + 1)
            print(f"Reanudando desde checkpoint '{args.resume}' (época {start_epoch})")

    start = time.time()
    scheduler = AdaptiveScheduler()
    history = train_agent(
        agent,
        loader,
        epochs=args.epochs,
        shuffle=args.shuffle,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        checkpoint_every=max(1, args.checkpoint_every),
        start_epoch=start_epoch,
    )
    elapsed = time.time() - start
    total_samples = sum(entry["samples"] for entry in history)
    print(f"\nEntrenamiento finalizado en {elapsed:.2f}s con {total_samples} pasos efectivos.")

    stats_path = Path(args.checkpoint_dir) / "training_history.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"Histórico de métricas guardado en {stats_path}")


if __name__ == "__main__":
    main()

```

## `./experiments/synthetic_benchmark.py`

```py
import numpy as np
from scipy.linalg import svd, norm

class OMEGA_ACP:
    def __init__(self, d_model, k_max=20, decay=0.99):
        self.d = d_model
        self.k_max = k_max
        self.l = decay
        self.Q = np.random.randn(d_model, 1)
        self.Q /= (norm(self.Q) + 1e-8)
        self.k = 1
        self.A = np.eye(d_model) * 0.1
        self.P = np.eye(d_model) * 0.1 

    def predict(self, x):
        return (self.A @ x.reshape(-1, 1)).flatten()

    def update(self, x_t, x_next):
        x_t = x_t.reshape(-1, 1); x_next = x_next.reshape(-1, 1)
        pred = self.A @ x_t; err = x_next - pred
        # RLS with clipping
        gain = (self.P @ x_t) / (1.0 + x_t.T @ self.P @ x_t + 1e-6)
        self.A += 0.5 * err @ gain.T
        self.P = (self.P - gain @ x_t.T @ self.P) / 0.999
        # Spectral projection (Crucial for stability)
        u, s, vh = svd(self.A); s = np.clip(s, 0, 0.95); self.A = (u * s) @ vh
        # Arnoldi update
        w = self.A @ self.Q[:, -1]
        for i in range(self.k):
            h = (self.l ** (self.k - i)) * np.dot(w, self.Q[:, i])
            w -= h * self.Q[:, i]
        h_next = norm(w)
        if h_next > 1e-4 and self.k < self.k_max:
            self.Q = np.column_stack([self.Q, w / h_next]); self.k += 1
        elif self.k >= self.k_max:
            u, _, _ = svd(self.Q); self.Q = u[:, :self.k_max // 2]; self.k = self.Q.shape[1]

class RobustLSTM:
    def __init__(self, d_model):
        self.d = d_model; self.h = np.zeros(d_model)
        self.W = np.random.randn(d_model, d_model * 2) * 0.01
    def predict(self, x):
        concat = np.concatenate([x.flatten(), self.h.flatten()])
        return (self.W @ concat)[:self.d]
    def update(self, x_t, x_next):
        concat = np.concatenate([x_t.flatten(), self.h.flatten()])
        pred = self.W @ concat; err = x_next.flatten() - pred[:self.d]
        grad = np.outer(np.clip(err, -1, 1), concat)
        self.W += 0.01 * grad # Backprop-like update
        self.h = np.tanh(pred[:self.d])

def lorenz_step(state, sigma, rho, beta, dt=0.01):
    x, y, z = state
    dx = sigma * (y - x); dy = x * (rho - z) - y; dz = x * y - beta * z
    return state + np.array([dx, dy, dz]) * dt

def generate_lorenz_data(steps, d_model):
    data = []; projection = np.random.randn(d_model, 3) * 0.5
    state = np.array([1.0, 1.0, 1.0])
    for t in range(steps):
        p = (10.0, 28.0, 8/3) if (t < steps//3 or t > 2*steps//3) else (10.0, 15.0, 2.0)
        state = lorenz_step(state, *p)
        obs = projection @ state + np.random.randn(d_model) * 0.01
        data.append(obs)
    return np.array(data)

def run_benchmark():
    D = 16; STEPS = 600; data = generate_lorenz_data(STEPS, D)
    # Normalize data
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
    models = {"LSTM-Lite": RobustLSTM(D), "OMEGA-ACP": OMEGA_ACP(D)}
    results = {name: [] for name in models}
    print(f"{'Step':<6} | {'Regime':<8} | {'LSTM Err':<10} | {'OMEGA Err':<10}")
    print("-" * 50)
    for t in range(STEPS - 1):
        x_t, x_next = data[t], data[t+1]
        regime = "A1" if t < STEPS//3 else ("B" if t < 2*STEPS//3 else "A2")
        for name, model in models.items():
            pred = model.predict(x_t); err = norm(x_next - pred); results[name].append(err); model.update(x_t, x_next)
        if t % 100 == 0:
            print(f"{t:<6} | {regime:<8} | {results['LSTM-Lite'][-1]:.4f}   | {results['OMEGA-ACP'][-1]:.4f}")
    print("-" * 50 + "\nRESULTS")
    for n in results:
        s3 = STEPS // 3; a1, b, a2 = np.mean(results[n][:s3]), np.mean(results[n][s3:2*s3]), np.mean(results[n][2*s3:])
        forget = (a2 - a1) / a1
        print(f"-> {n:12}: A1: {a1:.4f} | B (Shift): {b:.4f} | A2 (Return): {a2:.4f} | Forget: {forget:+.2%}")

if __name__ == "__main__":
    run_benchmark()

```

## `./omega/brain/regime.py`

```py
import numpy as np


class RegimeDetector:
    """
    Bayesian Entropy Monitor for Out-of-Distribution (OOD) detection.
    Extends detection with Spectral Checksum Structural Identity (SCSI) monitoring.
    """

    def __init__(
        self,
        d_latents,
        window_size=50,
        threshold=2.5,
        scsi_angle_threshold=np.deg2rad(80.0),
        scsi_eigen_threshold=0.8
    ):
        self.d = d_latents
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
        self.baseline_mu = None
        self.baseline_sigma = None

        # SCSI parameters
        self.scsi_angle_threshold = scsi_angle_threshold
        self.scsi_eigen_threshold = scsi_eigen_threshold
        self.scsi_reference = None
        self.last_scsi_metrics = {"angle": 0.0, "eig_drift": 0.0}

    def update(self, latent_vector):
        """Monitors statistics of latent activations."""
        self.history.append(latent_vector)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < self.window_size:
            return False  # Still warming up

        current_mu = np.mean(self.history, axis=0)
        current_sigma = np.std(self.history, axis=0)

        if self.baseline_mu is None:
            self.baseline_mu = current_mu
            self.baseline_sigma = current_sigma
            return False

        # Calculate Mahalanobis-like distance / KL Divergence heuristic
        z_score = np.abs(current_mu - self.baseline_mu) / (self.baseline_sigma + 1e-6)
        max_deviation = np.max(z_score)

        if max_deviation > self.threshold:
            # Regime shift detected
            return True

        return False

    def reset_baseline(self):
        """Update baseline after a confirmed regime shift."""
        self.baseline_mu = np.mean(self.history, axis=0)
        self.baseline_sigma = np.std(self.history, axis=0)

    def mark_scsi_baseline(self, scsi_signature, basis_matrix):
        """Stores the current structural signature as the stable reference."""
        if scsi_signature is None and basis_matrix is None:
            return

        angles = self._extract_angles(None if scsi_signature is None else scsi_signature.get("principal_angles"))
        eigenvalues = self._extract_eigenvalues(None if scsi_signature is None else scsi_signature.get("eigenvalues"))
        basis = None
        if basis_matrix is not None:
            basis = np.array(basis_matrix, copy=True)
        self.scsi_reference = {"angles": angles, "eigenvalues": eigenvalues, "basis": basis}

    def check_scsi(self, scsi_signature, basis_matrix, regime_shift=False):
        """
        Evaluates structural identity drift. Returns True when a structural anomaly
        is detected without an accompanying regime shift.
        """
        if scsi_signature is None and basis_matrix is None:
            return False

        current_angles = self._extract_angles(None if scsi_signature is None else scsi_signature.get("principal_angles"))
        current_eigs = self._extract_eigenvalues(None if scsi_signature is None else scsi_signature.get("eigenvalues"))
        current_basis = None if basis_matrix is None else np.array(basis_matrix, copy=False)

        if current_angles.size == 0 and current_eigs.size == 0 and (current_basis is None or current_basis.size == 0):
            return False

        if self.scsi_reference is None or regime_shift:
            baseline_basis = None if current_basis is None else np.array(current_basis, copy=True)
            self.scsi_reference = {"angles": current_angles, "eigenvalues": current_eigs, "basis": baseline_basis}
            return False

        baseline_basis = self.scsi_reference.get("basis")
        if current_basis is not None and baseline_basis is not None and current_basis.size and baseline_basis.size:
            min_dim = min(baseline_basis.shape[1], current_basis.shape[1])
            if min_dim > 0:
                cross = baseline_basis[:, :min_dim].T @ current_basis[:, :min_dim]
                _, sing_vals, _ = np.linalg.svd(cross, full_matrices=False)
                sing_vals = np.clip(sing_vals, -1.0, 1.0)
                principal_angles = np.arccos(sing_vals)
                max_angle = float(np.max(principal_angles)) if principal_angles.size else 0.0
            else:
                max_angle = 0.0
        else:
            max_angle = float(np.max(current_angles)) if current_angles.size else 0.0

        ref_eigs = self.scsi_reference["eigenvalues"]
        if ref_eigs.size and current_eigs.size:
            cur_sorted = np.sort_complex(current_eigs)
            ref_sorted = np.sort_complex(ref_eigs)
            max_len = max(cur_sorted.size, ref_sorted.size)
            cur_pad = np.pad(cur_sorted, (0, max_len - cur_sorted.size), constant_values=0)
            ref_pad = np.pad(ref_sorted, (0, max_len - ref_sorted.size), constant_values=0)
            diff = np.linalg.norm(cur_pad - ref_pad)
            base = max(np.linalg.norm(ref_pad), 1.0)
            eig_drift = diff / base
        else:
            eig_drift = 0.0

        anomaly = (max_angle > self.scsi_angle_threshold) and (eig_drift > self.scsi_eigen_threshold)
        self.last_scsi_metrics = {"angle": max_angle, "eig_drift": eig_drift}
        return anomaly

    @staticmethod
    def _extract_angles(values):
        if values is None:
            return np.array([])
        arr = np.array(values, dtype=float)
        return arr

    @staticmethod
    def _extract_eigenvalues(values):
        if values is None:
            return np.array([], dtype=complex)
        return np.array(values, dtype=complex)

    def get_state(self):
        return {
            "history": np.array(self.history, copy=True),
            "baseline_mu": None if self.baseline_mu is None else self.baseline_mu.copy(),
            "baseline_sigma": None if self.baseline_sigma is None else self.baseline_sigma.copy(),
            "scsi_reference": None
            if self.scsi_reference is None
            else {
                "angles": None
                if self.scsi_reference.get("angles") is None
                else np.array(self.scsi_reference["angles"], copy=True),
                "eigenvalues": None
                if self.scsi_reference.get("eigenvalues") is None
                else np.array(self.scsi_reference["eigenvalues"], copy=True),
                "basis": None
                if self.scsi_reference.get("basis") is None
                else np.array(self.scsi_reference["basis"], copy=True),
            },
            "last_scsi_metrics": dict(self.last_scsi_metrics),
        }

    def set_state(self, state):
        history = state.get("history")
        self.history = [] if history is None else [h.copy() for h in history]
        self.baseline_mu = None
        self.baseline_sigma = None
        if state.get("baseline_mu") is not None:
            self.baseline_mu = state["baseline_mu"].copy()
        if state.get("baseline_sigma") is not None:
            self.baseline_sigma = state["baseline_sigma"].copy()

        ref = state.get("scsi_reference")
        if ref is None:
            self.scsi_reference = None
        else:
            self.scsi_reference = {
                "angles": None if ref.get("angles") is None else np.array(ref["angles"], copy=True),
                "eigenvalues": None if ref.get("eigenvalues") is None else np.array(ref["eigenvalues"], copy=True),
                "basis": None if ref.get("basis") is None else np.array(ref["basis"], copy=True),
            }
        self.last_scsi_metrics = dict(state.get("last_scsi_metrics", {"angle": 0.0, "eig_drift": 0.0}))

```

## `./omega/brain/symbolic.py`

```py
import numpy as np

class SymbolicInterface:
    """
    Adaptive Neuro-Symbolic Bridge.
    Learns symbolic anchors from the data distribution via online competitive learning.
    Enforces a Lipschitz constraint on the anchor updates to keep the mapping contractive.
    """

    def __init__(self, d_model, n_predicates, anchor_radius=5.0, max_update_norm=0.1):
        self.d = d_model
        self.n = n_predicates
        self.anchor_radius = anchor_radius
        self.max_update_norm = max_update_norm
        # Learnable anchors (Prototypes)
        self.anchors = np.random.randn(n_predicates, d_model)
        self._project_anchors()
        self.usage_counts = np.ones(n_predicates)
        self.last_update_norm = 0.0

    def map_and_learn(self, z, lr=0.05):
        """
        Maps vector to symbol and updates the anchor to track the data cluster.
        The update is clipped to satisfy a Lipschitz bound.
        """
        # Distances to prototypes
        diffs = self.anchors - z
        dist_sq = np.sum(diffs**2, axis=1)
        symbol_id = np.argmin(dist_sq)

        # Competitive Learning with Lipschitz-bounded update
        delta = z - self.anchors[symbol_id]
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 0.0:
            max_step = min(self.max_update_norm, lr * delta_norm)
            scaled_delta = delta * (max_step / (delta_norm + 1e-8))
        else:
            scaled_delta = np.zeros_like(delta)

        self.anchors[symbol_id] += scaled_delta
        self._project_anchor(symbol_id)
        self.usage_counts[symbol_id] += 1
        self.last_update_norm = np.linalg.norm(scaled_delta)

        confidence = 1.0 / (1.0 + np.sqrt(dist_sq[symbol_id]))
        return symbol_id, confidence

    def _project_anchor(self, idx):
        norm_anchor = np.linalg.norm(self.anchors[idx])
        if norm_anchor > self.anchor_radius:
            self.anchors[idx] *= (self.anchor_radius / (norm_anchor + 1e-8))

    def _project_anchors(self):
        for i in range(self.n):
            self._project_anchor(i)

    def get_state(self):
        return {
            "anchors": self.anchors.copy(),
            "usage_counts": self.usage_counts.copy(),
            "last_update_norm": float(self.last_update_norm),
            "anchor_radius": float(self.anchor_radius),
            "max_update_norm": float(self.max_update_norm),
        }

    def set_state(self, state):
        self.anchors = state["anchors"].copy()
        self.usage_counts = state["usage_counts"].copy()
        self.anchor_radius = float(state.get("anchor_radius", self.anchor_radius))
        self.max_update_norm = float(state.get("max_update_norm", self.max_update_norm))
        self.last_update_norm = float(state.get("last_update_norm", 0.0))

class LogicEngine:
    def __init__(self):
        self.rules = {}

    def add_rule(self, premise, conclusion):
        self.rules[premise] = conclusion

    def infer(self, symbol_id):
        return self.rules.get(symbol_id, None)

```

## `./omega/core/acp.py`

```py
import numpy as np
from scipy.linalg import svd, norm, qr

class ACPModule:
    """
    Arnoldi-Causal Projection (ACP) Module.
    Refines predictions by projecting them onto the causal subspace.
    """
    def __init__(self, d_model, k_max=16, decay_lambda=0.99, tau=1e-4,
                 rls_lambda=0.99, alpha=1e-3, frobenius_radius=1.0,
                 orthogonality_tol=1e-4, svd_interval=5):
        self.d = d_model
        self.k_max = k_max
        self.l = decay_lambda
        self.tau = tau
        self.rls_lambda = rls_lambda
        self.alpha = alpha
        self.frobenius_radius = frobenius_radius
        self.orthogonality_tol = orthogonality_tol
        self.svd_interval = max(1, int(svd_interval))
        self.Q = np.random.randn(d_model, 1); self.Q /= (norm(self.Q) + 1e-8)
        self.k = 1
        self.A = np.eye(d_model) * 0.1
        self.P = np.eye(d_model) / max(self.alpha, 1e-6)
        self.H = np.zeros((self.k_max + 1, self.k_max))
        self.prev_basis = None
        self.scsi_signature = {"eigenvalues": None, "principal_angles": None}
        self.last_orth_error = 0.0
        self.last_monotonic_gradient = []
        self.step_counter = 0
        self.power_vector = np.random.randn(d_model, 1)
        self.power_vector /= (norm(self.power_vector) + 1e-8)
        self._spectral_radius = 0.0

    def refine_prediction(self, raw_pred):
        """
        Projects the raw neural prediction onto the Causal Subspace.
        This filters out components that don't match the learned system dynamics.
        """
        raw_pred = raw_pred.reshape(-1, 1)
        # Project onto Q basis: Q Q^T z
        refined = self.Q @ (self.Q.T @ raw_pred)
        return refined.flatten()

    def update_operator(self, x_t, x_next):
        x_t, x_next = x_t.reshape(-1, 1), x_next.reshape(-1, 1)

        if self.k > 0:
            q_t = self.Q[:, self.k - 1].reshape(-1, 1)
        else:
            norm_x = norm(x_t)
            q_t = x_t / (norm_x + 1e-8)

        phi = q_t
        denom = float(self.rls_lambda + phi.T @ self.P @ phi)
        gain = (self.P @ phi) / (denom + 1e-8)
        error = x_next - self.A @ phi
        self.A += error @ gain.T

        self.P = (self.P - gain @ phi.T @ self.P) / self.rls_lambda
        self.P += self.alpha * np.eye(self.d)
        self.P = 0.5 * (self.P + self.P.T)

        self.A *= (1.0 - self.alpha)
        frob = norm(self.A, ord='fro')
        if frob > self.frobenius_radius:
            self.A *= self.frobenius_radius / (frob + 1e-8)

        v_new = self.A @ self.power_vector
        radius_est = norm(v_new)
        if radius_est > 0:
            self.power_vector = v_new / (radius_est + 1e-8)
        self._spectral_radius = float(radius_est)

        if self._spectral_radius >= 1.0:
            self.A *= (0.99 / (self._spectral_radius + 1e-8))
            self._spectral_radius = 0.99
            self.power_vector = self.A @ self.power_vector
            pv_norm = norm(self.power_vector)
            if pv_norm > 0:
                self.power_vector /= pv_norm

    def step(self, seed_vector=None):
        """
        Updates the causal Krylov basis using the latest transition operator.
        Optionally seeds the base vector with the provided observation.
        """
        self.step_counter += 1
        prev_basis = self.Q[:, :self.k].copy() if self.k > 0 else None
        spectral_signature = None

        if seed_vector is not None:
            v = seed_vector.reshape(-1, 1)
            v_norm = norm(v)
            if v_norm > 1e-8:
                self.Q[:, 0:1] = v / v_norm
                self.k = max(self.k, 1)

        q_last = self.Q[:, self.k - 1].reshape(-1, 1)
        w = self.A @ q_last
        monotonic_gradients = []
        for i in range(self.k):
            q_i = self.Q[:, i].reshape(-1, 1)
            dot_val = float(q_i.T @ w)
            weight = self.l ** (self.k - 1 - i)
            h_mag = weight * abs(dot_val)
            coeff = np.sign(dot_val) * h_mag
            self.H[i, self.k - 1] = coeff
            w -= coeff * q_i
            exponent = self.k - 1 - i
            if exponent > 0:
                grad_mag = exponent * (self.l ** (exponent - 1)) * abs(dot_val)
            else:
                grad_mag = 0.0
            monotonic_gradients.append(grad_mag)

        h_next = float(norm(w))
        self.H[self.k, self.k - 1] = h_next

        if h_next > self.tau and self.k < self.k_max:
            q_new = (w / h_next).flatten()
            self.Q = np.column_stack([self.Q, q_new])
            self.k += 1
            self.H[:, self.k - 1] = 0.0
        elif self.k >= self.k_max:
            if self.step_counter % self.svd_interval == 0:
                spectral_signature = self._compress_basis()
            else:
                monotonic_gradients = [0.0] * self.k
        else:
            self.H[self.k, self.k - 1] = 0.0

        self._enforce_orthogonality()
        self.last_monotonic_gradient = monotonic_gradients
        self._update_scsi(prev_basis, spectral_signature)
        return self.Q

    def _compress_basis(self):
        """Spectral compression via truncated SVD of the Hessenberg matrix."""
        active_cols = min(self.k, self.H.shape[1])
        if active_cols == 0:
            return None

        H_k = self.H[:active_cols + 1, :active_cols]
        u, s, vt = svd(H_k, full_matrices=False)
        target_dim = max(1, min(self.k_max // 2, vt.shape[0]))
        V_r = vt[:target_dim, :].T  # shape (active_cols, target_dim)
        compressed_basis = self.Q[:, :active_cols] @ V_r
        q_new, _ = qr(compressed_basis, mode='economic')
        self.Q = q_new[:, :target_dim]
        self.k = self.Q.shape[1]
        spectral_signature = np.linalg.eigvals(H_k[:target_dim, :target_dim])
        self.H = np.zeros((self.k_max + 1, self.k_max))
        return spectral_signature

    def _update_scsi(self, prev_basis, spectral_signature=None):
        """Updates the spectral checksum structural identity (SCSI) metrics."""
        current_basis = self.Q[:, :self.k]

        if prev_basis is not None and prev_basis.size > 0 and current_basis.size > 0:
            shared_dim = min(prev_basis.shape[1], current_basis.shape[1])
            if shared_dim > 0:
                cross = prev_basis[:, :shared_dim].T @ current_basis[:, :shared_dim]
                _, sing_vals, _ = svd(cross, full_matrices=False)
                sing_vals = np.clip(sing_vals, -1.0, 1.0)
                principal_angles = np.arccos(sing_vals)
            else:
                principal_angles = np.array([])
        else:
            principal_angles = np.array([])

        if spectral_signature is None and self.k > 0:
            H_active = self.H[:self.k, :self.k]
            spectral_signature = np.linalg.eigvals(H_active) if H_active.size > 0 else np.array([])
        elif spectral_signature is None:
            spectral_signature = np.array([])

        self.scsi_signature = {
            "eigenvalues": spectral_signature,
            "principal_angles": principal_angles
        }
        self.prev_basis = current_basis.copy() if current_basis.size > 0 else None

    def get_state(self):
        """Returns a deep copy of the internal ACP state."""
        return {
            "A": self.A.copy(),
            "Q": self.Q.copy(),
            "H": self.H.copy(),
            "P": self.P.copy(),
            "k": int(self.k),
            "last_orth_error": float(self.last_orth_error),
            "last_monotonic_gradient": np.array(self.last_monotonic_gradient, copy=True),
            "step_counter": int(self.step_counter),
            "power_vector": self.power_vector.copy(),
            "spectral_radius": float(self._spectral_radius),
            "scsi": {
                "eigenvalues": None if self.scsi_signature["eigenvalues"] is None
                else np.array(self.scsi_signature["eigenvalues"], copy=True),
                "principal_angles": None if self.scsi_signature["principal_angles"] is None
                else np.array(self.scsi_signature["principal_angles"], copy=True)
            },
            "prev_basis": None if self.prev_basis is None else self.prev_basis.copy()
        }

    def set_state(self, state):
        """Restores the ACP state from a snapshot produced by get_state."""
        self.A = state["A"].copy()
        self.Q = state["Q"].copy()
        self.H = state["H"].copy()
        self.P = state["P"].copy()
        self.k = int(state["k"])
        self.last_orth_error = float(state.get("last_orth_error", 0.0))
        grad = state.get("last_monotonic_gradient")
        self.last_monotonic_gradient = [] if grad is None else list(np.array(grad, copy=True))
        self.step_counter = int(state.get("step_counter", self.step_counter))
        self.power_vector = state.get("power_vector", self.power_vector).copy()
        self._spectral_radius = float(state.get("spectral_radius", self._spectral_radius))

        scsi_state = state.get("scsi", {})
        eigs = scsi_state.get("eigenvalues")
        angles = scsi_state.get("principal_angles")
        self.scsi_signature = {
            "eigenvalues": None if eigs is None else np.array(eigs, copy=True),
            "principal_angles": None if angles is None else np.array(angles, copy=True)
        }
        prev_basis = state.get("prev_basis")
        self.prev_basis = None if prev_basis is None else prev_basis.copy()

    def _enforce_orthogonality(self):
        """Re-orthogonalises Q when numerical drift exceeds the tolerance."""
        if self.Q.size == 0:
            return
        gram = self.Q.T @ self.Q
        deviation = gram - np.eye(self.Q.shape[1])
        deviation_norm = norm(deviation, ord='fro')
        self.last_orth_error = deviation_norm
        if deviation_norm > self.orthogonality_tol:
            q_new, _ = qr(self.Q, mode='economic')
            self.Q = q_new[:, :self.Q.shape[1]]

    @property
    def spectral_radius(self):
        return float(self._spectral_radius)

```

## `./omega/core/lpc.py`

```py
import numpy as np


def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2


class LocalPredictiveUnit:
    """
    OMEGA v3: Difference Target Propagation (DTP).
    Each unit also maintains a local causal operator learned via RLS against the global ACP basis.
    """

    def __init__(self, d_in, d_out, rls_lambda=0.99, alpha=1e-3, frobenius_radius=1.0):
        self.d_in = d_in
        self.d_out = d_out
        self.rls_lambda = rls_lambda
        self.alpha = alpha
        self.frobenius_radius = frobenius_radius

        # Forward weights
        self.W = np.random.randn(d_out, d_in) * (1.0 / np.sqrt(d_in))
        # Inverse/Backward weights (Feedback)
        self.V = np.random.randn(d_in, d_out) * (1.0 / np.sqrt(d_out))
        # RLS state for Forward
        self.P = np.eye(d_in)

        # Local causal operator (A_L) and covariance
        self.A_local = np.eye(d_out, d_in)
        self.P_local = np.eye(d_in) / max(self.alpha, 1e-6)

        self.last_x = None
        self.last_z_pre = None  # Pre-activation
        self.last_z = None  # Post-activation

    def forward(self, x):
        self.last_x = x.reshape(-1, 1)
        self.last_z_pre = self.W @ self.last_x
        self.last_z = np.tanh(self.last_z_pre)
        return self.last_z.flatten()

    def propagate_target(self, target_z):
        """
        Implements Difference Target Propagation (DTP).
        Target_x = last_x + V(target_z) - V(last_z)
        This ensures the target respects the local non-linear mapping.
        """
        target_z = target_z.reshape(-1, 1)
        # Compute the target for the layer below
        delta_x = self.V @ target_z - self.V @ self.last_z
        target_x = self.last_x + delta_x
        return target_x.flatten()

    def local_update(self, target_z, basis_vector=None):
        """
        Updates Forward weights (W), Backward weights (V), and the local causal operator A_L.
        """
        target_z = target_z.reshape(-1, 1)

        # 1. Update Forward W (using RLS on the linear part to reach atanh(target_z))
        target_z_clipped = np.clip(target_z, -0.99, 0.99)
        z_pre_target = np.arctanh(target_z_clipped)

        error_f = z_pre_target - self.last_z_pre
        gain = (self.P @ self.last_x) / (1.0 + self.last_x.T @ self.P @ self.last_x + 1e-8)
        self.W += error_f @ gain.T
        self.P = (self.P - gain @ self.last_x.T @ self.P) / self.rls_lambda
        self.P = 0.5 * (self.P + self.P.T)

        # 2. Update Inverse V (The layer learns to invert its own forward pass)
        error_v = self.last_x - self.V @ self.last_z
        self.V += 0.1 * error_v @ self.last_z.T

        # 3. Update Local Causal Operator using the ACP basis
        if basis_vector is not None:
            phi = basis_vector.reshape(-1, 1)
            denom = float(self.rls_lambda + phi.T @ self.P_local @ phi)
            k_gain = (self.P_local @ phi) / (denom + 1e-8)
            err_local = target_z - self.A_local @ phi
            self.A_local += err_local @ k_gain.T

            self.P_local = (self.P_local - k_gain @ phi.T @ self.P_local) / self.rls_lambda
            self.P_local += self.alpha * np.eye(self.d_in)
            self.P_local = 0.5 * (self.P_local + self.P_local.T)

            # Regularize A_L
            self.A_local *= (1.0 - self.alpha)
            frob = np.linalg.norm(self.A_local, ord="fro")
            if frob > self.frobenius_radius:
                self.A_local *= self.frobenius_radius / (frob + 1e-8)

            if self.d_in == self.d_out:
                eigvals = np.linalg.eigvals(self.A_local)
                rho = np.max(np.abs(eigvals)) if eigvals.size > 0 else 0.0
                if rho >= 1.0:
                    self.A_local *= (0.99 / (rho + 1e-8))

        # Recalculate local state for metrics
        new_z = np.tanh(self.W @ self.last_x)
        return np.linalg.norm(target_z - new_z)

    def project_basis(self, basis_vector):
        """
        Projects the global ACP basis through the local operator to produce a basis estimate for lower layers.
        """
        if basis_vector is None:
            return None
        phi = basis_vector.reshape(-1, 1)
        projected = self.A_local @ phi
        norm_proj = np.linalg.norm(projected)
        if norm_proj < 1e-8:
            return basis_vector
        return (projected / norm_proj).flatten()

    def get_state(self):
        """Create a snapshot of the local predictive unit."""
        return {
            "W": self.W.copy(),
            "V": self.V.copy(),
            "P": self.P.copy(),
            "A_local": self.A_local.copy(),
            "P_local": self.P_local.copy()
        }

    def set_state(self, state):
        """Restore parameters from a snapshot created by get_state."""
        self.W = state["W"].copy()
        self.V = state["V"].copy()
        self.P = state["P"].copy()
        self.A_local = state["A_local"].copy()
        self.P_local = state["P_local"].copy()

```

## `./omega/data/loader.py`

```py
import csv
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional


class TimeSeriesDataLoader:
    """
    Streaming loader for multivariate time-series.
    Supports CSV/NumPy sources, sliding windows and mini-batching on CPU.
    """

    def __init__(
        self,
        data: np.ndarray,
        window: int = 1,
        batch_size: int = 1,
        stride: int = 1,
        shuffle: bool = False,
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError("time-series data must be 2D: (steps, features)")

        self.data = data.astype(np.float64)
        self.window = max(1, int(window))
        self.batch_size = max(1, int(batch_size))
        self.stride = max(1, int(stride))
        self.shuffle = shuffle
        self.normalize = normalize
        self.mean = None
        self.std = None
        if normalize:
            self.mean = np.mean(self.data, axis=0, keepdims=True)
            self.std = np.std(self.data, axis=0, keepdims=True)
            self.std = np.where(self.std < eps, 1.0, self.std)
            self.data = (self.data - self.mean) / self.std
        self.indices = self._compute_indices()

    @classmethod
    def from_path(
        cls,
        path: str,
        window: int = 1,
        batch_size: int = 1,
        stride: int = 1,
        shuffle: bool = False,
        delimiter: str = ",",
        normalize: bool = False,
    ):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(path)

        if path_obj.suffix in (".npy", ".npz"):
            data = np.load(path_obj)
            if isinstance(data, np.lib.npyio.NpzFile):
                if "data" in data:
                    data = data["data"]
                else:
                    raise ValueError("npz file must contain array under key 'data'")
        else:
            data = cls._load_numeric_csv(path_obj, delimiter=delimiter)

        return cls(
            data=data,
            window=window,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            normalize=normalize,
        )

    @staticmethod
    def _load_numeric_csv(path: Path, delimiter: str) -> np.ndarray:
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            for row in reader:
                numeric = []
                for value in row:
                    try:
                        numeric.append(float(value))
                    except ValueError:
                        continue
                if numeric:
                    rows.append(numeric)
        if not rows:
            raise ValueError(f"no numeric data found in {path}")
        widths = {len(r) for r in rows}
        max_width = max(widths)
        matrix = np.zeros((len(rows), max_width), dtype=np.float64)
        for i, r in enumerate(rows):
            r = r[:max_width]
            matrix[i, : len(r)] = r
        return matrix

    def _compute_indices(self) -> np.ndarray:
        max_start = self.data.shape[0] - self.window - 1
        if max_start < 0:
            raise ValueError("time-series shorter than required window+1")
        indices = np.arange(0, max_start + 1, self.stride)
        if self.shuffle:
            np.random.shuffle(indices)
        return indices

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self.iter_batches()

    def iter_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        total = len(self.indices)
        for start in range(0, total, self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            windows, targets = [], []
            for idx in batch_idx:
                window_slice = self.data[idx : idx + self.window]
                target = self.data[idx + self.window]
                windows.append(window_slice)
                targets.append(target)
            x_batch = np.stack(windows, axis=0)
            y_batch = np.stack(targets, axis=0)
            yield x_batch, y_batch

    def epoch(self, shuffle: Optional[bool] = None):
        if shuffle is not None:
            self.shuffle = shuffle
        self.indices = self._compute_indices()

```

## `./omega/data/text_loader.py`

```py
import numpy as np
from pathlib import Path
from typing import Optional

from omega.data.loader import TimeSeriesDataLoader
from omega.nlp.continuous import ContinuousTextEncoder


class TextWindowDataLoader(TimeSeriesDataLoader):
    """
    Loader especializado para corpora textuales continuos.
    Convierte el texto en trayectorias densas mediante ContinuousTextEncoder.
    """

    def __init__(
        self,
        encoded: np.ndarray,
        window: int,
        batch_size: int,
        stride: int,
        shuffle: bool,
        normalize: bool = False,
    ):
        super().__init__(
            data=encoded,
            window=window,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            normalize=normalize,
        )

    @classmethod
    def from_path(
        cls,
        path: str,
        encoder: ContinuousTextEncoder,
        window: int = 16,
        batch_size: int = 1,
        stride: int = 1,
        shuffle: bool = False,
        encoding: str = "utf-8",
        max_chars: Optional[int] = None,
        normalize: bool = False,
    ):
        text = Path(path).read_text(encoding=encoding)
        if max_chars is not None:
            text = text[:max_chars]
        encoded = encoder.encode_text(text)
        return cls(
            encoded=encoded,
            window=window,
            batch_size=batch_size,
            stride=stride,
            shuffle=shuffle,
            normalize=normalize,
        )

```

## `./omega/engine/scheduler.py`

```py
import numpy as np
from typing import Dict, Any


class AdaptiveScheduler:
    """
    Ajusta hiperparámetros de ACP/local units en función de métricas agregadas.
    Simple heurística para CPU: reduce alpha cuando la estabilidad es baja
    y afloja lambda si el progreso se estanca.
    """

    def __init__(
        self,
        target_gain: float = 2.0,
        patience: int = 3,
        min_alpha: float = 1e-4,
        max_alpha: float = 5e-3,
        min_lambda: float = 0.90,
        max_lambda: float = 0.999,
    ):
        self.target_gain = target_gain
        self.patience = patience
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        self._no_improve = 0
        self._best_gain = -np.inf

    def step(self, metrics: Dict[str, Any], agent):
        gain = metrics.get("gain", 0.0)
        spectral_radius = metrics.get("spectral_radius", 0.0)
        orth_error = metrics.get("orth_error", 0.0)

        if gain > self._best_gain:
            self._best_gain = gain
            self._no_improve = 0
        else:
            self._no_improve += 1

        # Ajuste de alpha (regularización RLS) basado en radio espectral
        if spectral_radius > 0.95:
            agent.acp.alpha = min(self.max_alpha, agent.acp.alpha * 1.2)
        elif spectral_radius < 0.7:
            agent.acp.alpha = max(self.min_alpha, agent.acp.alpha * 0.8)

        # Ajuste de lambda (decay en Arnoldi) para mejorar ortogonalidad
        if orth_error > agent.acp.orthogonality_tol * 5:
            agent.acp.l = max(self.min_lambda, agent.acp.l * 0.98)
        elif gain > self.target_gain and orth_error < agent.acp.orthogonality_tol:
            agent.acp.l = min(self.max_lambda, agent.acp.l * 1.01)

        # Early stopping sugerido
        stop = self._no_improve >= self.patience
        return {
            "alpha": agent.acp.alpha,
            "lambda": agent.acp.l,
            "stop": stop,
            "patience_left": max(0, self.patience - self._no_improve),
        }

```

## `./omega/memory/persistent.py`

```py
import numpy as np

class PersistentMemory:
    """
    NTK-Stabilized Persistent Memory Matrix.
    Stores episodic experiences with protection against catastrophic forgetting.
    """
    def __init__(self, d_model, capacity=1000):
        self.d = d_model
        self.capacity = capacity
        self.memory = np.zeros((capacity, d_model))
        self.usage = np.zeros(capacity)
        self.cursor = 0

    def write(self, experience_vector, importance=1.0):
        """
        Selective write using GUM-inspired uniqueness detection.
        """
        # Uniqueness check: check if already exists
        if self.cursor > 0:
            similarities = self.memory[:self.cursor] @ experience_vector
            if np.max(similarities) > 0.98:
                return # Redundant info
                
        # NTK-inspired update: only update if the importance is high
        if self.cursor < self.capacity:
            self.memory[self.cursor] = experience_vector
            self.usage[self.cursor] = importance
            self.cursor += 1
        else:
            # Overwrite least important
            idx = np.argmin(self.usage)
            self.memory[idx] = experience_vector
            self.usage[idx] = importance

    def read(self, query_vector):
        """Differentiable read (Attention-like) over the matrix."""
        if self.cursor == 0:
            return np.zeros(self.d)
        
        # Dot-product similarity
        weights = self.memory[:self.cursor] @ query_vector
        weights -= np.max(weights)
        weights = np.exp(weights)
        denom = np.sum(weights) + 1e-8
        weights /= denom
        return weights @ self.memory[:self.cursor]

    def get_state(self):
        return {
            "memory": self.memory.copy(),
            "usage": self.usage.copy(),
            "cursor": int(self.cursor),
        }

    def set_state(self, state):
        self.memory = state["memory"].copy()
        self.usage = state["usage"].copy()
        self.cursor = int(state["cursor"])

    def decay(self, rate: float = 0.995):
        if self.cursor == 0:
            return
        self.usage[: self.cursor] *= rate

```

## `./omega/nlp/continuous.py`

```py
import numpy as np
from pathlib import Path
from typing import Iterable, List, Optional


class ContinuousTextEncoder:
    """
    Token-free encoder that maps raw text into continuous trajectories.
    Each character is projected onto a dense subspace and smoothed in time,
    yielding a vector of dimensión d por paso lingüístico.
    """

    def __init__(
        self,
        d_model: int,
        smoothing: float = 0.2,
        seed: Optional[int] = 1729,
        charset: Optional[Iterable[str]] = None,
    ):
        self.d = d_model
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.rng = np.random.default_rng(seed)
        if charset is None:
            charset = [chr(i) for i in range(32, 127)]
        self.charset = list(charset)
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.matrix = self._build_projection(len(self.charset))

    def _build_projection(self, vocab_size: int) -> np.ndarray:
        scale = 1.0 / np.sqrt(vocab_size)
        matrix = self.rng.standard_normal((self.d, vocab_size)) * scale
        return matrix.astype(np.float64)

    def encode_text(self, text: str) -> np.ndarray:
        vectors: List[np.ndarray] = []
        prev = np.zeros(self.d, dtype=np.float64)
        for char in text:
            idx = self.char_to_idx.get(char)
            if idx is None:
                idx = self.char_to_idx.get(" ", 0)
            column = self.matrix[:, idx]
            current = (1.0 - self.smoothing) * column + self.smoothing * prev
            prev = current
            vectors.append(current)
        if not vectors:
            return np.zeros((0, self.d), dtype=np.float64)
        return np.stack(vectors, axis=0)

    def encode_lines(self, lines: Iterable[str], separator: str = "\n") -> np.ndarray:
        text = separator.join(lines)
        return self.encode_text(text)


class ContinuousTextDecoder:
    """
    Placeholder para decodificador continuo->texto.
    Requiere un modelo externo (e.g. VQ-VAE) para reconstrucción explícita.
    """

    def __init__(self):
        raise NotImplementedError(
            "ContinuousTextDecoder es un placeholder; integra tu VQ-VAE o vocoder textual."
        )

```

## `./omega/utils/checkpoint.py`

```py
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


class CheckpointManager:
    """
    Serializa el estado completo del agente para reanudar entrenamiento.
    Usa pickle por simplicidad (solo en entornos controlados).
    """

    def __init__(self, directory: str):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.pkl"

    def save(self, name: str, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        payload = {"state": state, "meta": metadata or {}}
        with self._path(name).open("wb") as fh:
            pickle.dump(payload, fh)

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._path(name)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return payload

```

