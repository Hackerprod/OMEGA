import argparse
import cProfile
import pstats
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.core.acp import ACPModule
from omega.core.lpc import LocalPredictiveUnit


def run_microbench(d_model: int, steps: int, context_window: int, seed: int | None) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    acp = ACPModule(d_model=d_model)
    layers = [LocalPredictiveUnit(d_model, d_model) for _ in range(2)]

    series = rng.standard_normal((steps + 1, d_model))
    stats: dict[str, float] = defaultdict(float)
    for t in range(steps):
        x_t = series[t]
        x_next = series[t + 1]
        context = rng.standard_normal((context_window, d_model))

        start = perf_counter()
        for layer in layers:
            layer.forward(x_t)
        stats["lpu.forward"] += perf_counter() - start

        start = perf_counter()
        acp.update_operator(x_t, x_next)
        stats["acp.update_operator"] += perf_counter() - start

        start = perf_counter()
        acp.step(seed_vector=x_t)
        stats["acp.step"] += perf_counter() - start

        basis_vector = acp.Q[:, acp.k - 1] if acp.k > 0 else None
        for layer in reversed(layers):
            start = perf_counter()
            layer.local_update(x_next, basis_vector)
            stats["lpu.local_update"] += perf_counter() - start
            basis_vector = layer.project_basis(basis_vector)

    return stats


def profile_agent(d_model: int, steps: int, context_window: int, seed: int | None) -> pstats.Stats:
    acp = ACPModule(d_model=d_model)
    layers = [LocalPredictiveUnit(d_model, d_model) for _ in range(2)]
    rng = np.random.default_rng(seed)
    series = rng.standard_normal((steps + 1, d_model))

    def _loop():
        for t in range(steps):
            x_t = series[t]
            x_next = series[t + 1]
            for layer in layers:
                layer.forward(x_t)
            acp.update_operator(x_t, x_next)
            acp.step(seed_vector=x_t)
            basis_vector = acp.Q[:, acp.k - 1] if acp.k > 0 else None
            for layer in reversed(layers):
                layer.local_update(x_next, basis_vector)
                basis_vector = layer.project_basis(basis_vector)

    profiler = cProfile.Profile()
    profiler.enable()
    _loop()
    profiler.disable()
    return pstats.Stats(profiler)


def main():
    parser = argparse.ArgumentParser(description="Profile OMEGA ACP/DTP core routines.")
    parser.add_argument("--d-model", type=int, default=64, help="Dimensionality to profile.")
    parser.add_argument("--steps", type=int, default=500, help="Number of synthetic steps to simulate.")
    parser.add_argument("--context-window", type=int, default=8, help="Context size for synthetic batches.")
    parser.add_argument("--seed", type=int, default=1729, help="Random seed for reproducibility.")
    parser.add_argument("--profile", action="store_true", help="Emit cProfile stats in addition to wall timings.")
    args = parser.parse_args()

    stats = run_microbench(args.d_model, args.steps, args.context_window, args.seed)
    total = sum(stats.values())
    print("Wall-clock micro-benchmark (seconds):")
    for key, value in sorted(stats.items(), key=lambda item: item[1], reverse=True):
        pct = (value / total * 100.0) if total else 0.0
        print(f"  {key:<22} {value:8.4f}s ({pct:5.1f}%)")
    print(f"  {'total':<22} {total:8.4f}s (100.0%)")

    if args.profile:
        stats_obj = profile_agent(args.d_model, args.steps, args.context_window, args.seed)
        buffer = StringIO()
        stats_obj.sort_stats("cumulative").print_stats(20, stream=buffer)
        print("\ncProfile cumulative top 20:")
        print(buffer.getvalue())


if __name__ == "__main__":
    main()
