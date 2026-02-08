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

    def run_step(self, x_t, x_next, context=None, preprojected=False):
        if preprojected:
            x_t = np.asarray(x_t, dtype=np.float64).flatten()
            x_next = np.asarray(x_next, dtype=np.float64).flatten()
            context_model = None if context is None else np.asarray(context, dtype=np.float64)
        else:
            x_t = self._project_input(x_t)
            x_next = self._project_input(x_next)
            context_model = None if context is None else self._project_inputs(context)
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
        projected = self._project_inputs(vector)
        return projected if projected.ndim == 1 else projected.flatten()

    def _project_inputs(self, array):
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim == 1:
            vector = arr.flatten()
            if vector.shape[0] == self.d and self.input_proj is None:
                return vector
            self._ensure_projection(vector.shape[0])
            if self.input_proj.shape[1] != vector.shape[0]:
                raise ValueError("Input dimension changed; cannot reuse projection.")
            return (self.input_proj @ vector).astype(np.float64, copy=False)
        elif arr.ndim == 2:
            if arr.shape[1] == self.d and self.input_proj is None:
                return arr
            self._ensure_projection(arr.shape[1])
            if self.input_proj.shape[1] != arr.shape[1]:
                raise ValueError("Input dimension changed; cannot reuse projection.")
            return (arr @ self.input_proj.T).astype(np.float64, copy=False)
        else:
            raise ValueError("Unsupported input dimensions for projection.")

    def _ensure_projection(self, in_dim):
        if self.input_proj is not None:
            return
        if in_dim == self.d:
            self.input_proj = np.eye(self.d, dtype=np.float64)
        else:
            rng = np.random.default_rng()
            self.input_proj = rng.standard_normal((self.d, in_dim)) / np.sqrt(in_dim)

    def project_windows(self, windows):
        windows = np.asarray(windows)
        if windows.ndim != 3:
            raise ValueError("Expected windows with shape (batch, window, features).")
        batch, window, feat = windows.shape
        flat = windows.reshape(batch * window, feat)
        projected = self._project_inputs(flat)
        return projected.reshape(batch, window, self.d)

    def project_batch(self, batch):
        batch = np.asarray(batch)
        if batch.ndim == 1:
            return self._project_input(batch)
        if batch.ndim != 2:
            raise ValueError("Batch must be 1D or 2D.")
        return self._project_inputs(batch)

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

def generate_dynamic_signal(t, d_model=32):
    """3D signal projected to d_model dimensions with phase shifts."""
    proj = np.sin(np.arange(d_model) * 0.1)
    if t < 60:
        val = np.sin(0.1 * t)
    else:
        # High frequency phase shift
        val = 0.5 * np.cos(0.4 * t)
    return proj * val + np.random.randn(d_model) * 0.01

def build_synthetic_loader(
    steps: int,
    d_model: int,
    batch: int,
    window: int,
    normalize: bool = False,
    dtype: np.dtype = np.float64,
) -> TimeSeriesDataLoader:
    series = np.stack([generate_dynamic_signal(t, d_model) for t in range(steps)], axis=0).astype(dtype, copy=False)
    return TimeSeriesDataLoader(
        series,
        window=window,
        batch_size=batch,
        stride=1,
        shuffle=False,
        normalize=normalize,
        dtype=dtype,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OMEGA v3 Trainer")
    parser.add_argument("--d-model", type=int, default=32, help="Dimension del modelo OMEGA")
    parser.add_argument("--data-path", type=str, default=None, help="Ruta a CSV/NPY con serie temporal")
    parser.add_argument("--delimiter", type=str, default=",", help="Delimitador para CSV (por defecto ',')")
    parser.add_argument("--window", type=int, default=1, help="Tamano de ventana/contexto para loader")
    parser.add_argument("--batch-size", type=int, default=1, help="Tamano de mini-batch en CPU")
    parser.add_argument("--stride", type=int, default=1, help="Stride de ventana")
    parser.add_argument("--epochs", type=int, default=1, help="Cantidad de epocas sobre el dataset")
    parser.add_argument("--steps", type=int, default=200, help="Pasos sinteticos si no hay dataset")
    parser.add_argument("--shuffle", action="store_true", help="Barajar ventanas en cada epoca")
    parser.add_argument("--normalize", action="store_true", help="Normalizar series numericas (z-score)")
    parser.add_argument("--text-path", type=str, default=None, help="Ruta a corpus de texto continuo")
    parser.add_argument("--text-encoding", type=str, default="utf-8", help="Codificacion del archivo de texto")
    parser.add_argument("--text-max-chars", type=int, default=None, help="Limitar caracteres cargados del corpus")
    parser.add_argument("--encoder-smoothing", type=float, default=0.2, help="Factor de suavizado temporal del encoder continuo")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directorio para guardar checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Frecuencia de guardado (epocas)")
    parser.add_argument("--resume", type=str, default=None, help="Nombre de checkpoint a reanudar (sin extension)")
    parser.add_argument("--dtype", type=str, choices=["float64", "float32"], default="float64", help="Precision del loader (float64|float32)")
    parser.add_argument("--text-memmap", type=str, default=None, help="Ruta de memmap para corpus continuo")
    parser.add_argument("--text-chunk-size", type=int, default=65536, help="Tamano de bloque para el encoder continuo")
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
            projected_windows = agent.project_windows(window_batch)
            projected_targets = agent.project_batch(y_batch)
            for window_t, x_next in zip(projected_windows, projected_targets):
                x_t = window_t[-1]
                metrics = agent.run_step(x_t, x_next, context=window_t, preprojected=True)
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


def main():
    args = parse_args()
    agent = OMEGAAgent(d_model=args.d_model)
    print()

    dtype = np.float32 if args.dtype == "float32" else np.float64

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
            dtype=dtype,
            memmap_path=args.text_memmap,
            chunk_chars=args.text_chunk_size,
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
            dtype=dtype,
        )
        print(f"Cargando datos desde {args.data_path} | pasos disponibles: {loader.data.shape[0]}")
    else:
        loader = build_synthetic_loader(
            steps=args.steps,
            d_model=agent.d,
            batch=args.batch_size,
            window=args.window,
            normalize=args.normalize,
            dtype=dtype,
        )
        print(f"Generando serie sintetica ({args.steps} pasos) para entrenamiento.")

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
