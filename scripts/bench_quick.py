#!/usr/bin/env python
"""
Quick performance probe for the OMEGA ACP core.

Collects micro-benchmark timings using experiments/profile_acp and optionally
compares the results against a JSON baseline with tolerances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.profile_acp import run_microbench


def compare_to_baseline(current: Dict[str, float], baseline_path: Path, tolerance: float) -> bool:
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    ok = True
    for key, value in current.items():
        if key not in data:
            continue
        baseline = data[key]
        if baseline <= 0:
            continue
        ratio = value / baseline
        if ratio > 1.0 + tolerance:
            print(f"[REGRESSION] {key}: {value:.4f}s vs baseline {baseline:.4f}s (+{(ratio-1)*100:.1f}%)")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast ACP micro-benchmarks.")
    parser.add_argument("--d-model", type=int, default=48, help="Model dimensionality.")
    parser.add_argument("--steps", type=int, default=200, help="Synthetic steps to simulate.")
    parser.add_argument("--context-window", type=int, default=8, help="Context window size.")
    parser.add_argument("--seed", type=int, default=1729, help="Random seed.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write JSON timings.")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline JSON for comparison.")
    parser.add_argument("--tolerance", type=float, default=0.25, help="Allowed slowdown ratio.")
    args = parser.parse_args()

    stats = run_microbench(args.d_model, args.steps, args.context_window, args.seed)
    for key, value in sorted(stats.items()):
        print(f"{key:<24}: {value:.4f}s")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nTimings saved to {args.output}")

    if args.baseline:
        if not args.baseline.exists():
            print(f"[WARN] Baseline {args.baseline} missing.")
        else:
            ok = compare_to_baseline(stats, args.baseline, args.tolerance)
            if not ok:
                return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
