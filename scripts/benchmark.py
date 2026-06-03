#!/usr/bin/env python3
"""Unified RelBench benchmark: all models x all tasks -> one results matrix.

Example (rel-f1, three task types, all three models):
    python scripts/benchmark.py \
        --dataset rel-f1 \
        --tasks driver-dnf driver-top3 driver-position \
        --models flat graphsage relgt-lite \
        --train-seeds 512 --val-seeds 256 --epochs 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--tasks", nargs="+", default=["driver-dnf", "driver-top3", "driver-position"])
    parser.add_argument("--models", nargs="+", default=["flat", "graphsage", "relgt-lite"])
    parser.add_argument("--train-seeds", type=int, default=512)
    parser.add_argument("--val-seeds", type=int, default=256)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs" / "benchmark_matrix.json")
    args = parser.parse_args()

    from src.relbench_pipeline.benchmark import benchmark

    payload = benchmark(
        dataset_name=args.dataset,
        task_names=args.tasks,
        models=args.models,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        K=args.k,
        channels=args.channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    matrix = payload["matrix"]
    tasks = payload["tasks"]
    models = payload["models"]

    # Column header: task (metric)
    metric_by_task = {}
    for t in tasks:
        for m in models:
            cell = matrix[m].get(t, {})
            if cell.get("metric", "n/a") != "n/a":
                metric_by_task[t] = cell["metric"]
                break
        metric_by_task.setdefault(t, "n/a")

    col_w = 22
    print("\n" + "=" * (20 + col_w * len(tasks)))
    print(f"BENCHMARK  {payload['dataset']}  (CPU, first-N-seed protocol)")
    print(f"train_seeds={args.train_seeds} val_seeds={args.val_seeds} "
          f"K={args.k} channels={args.channels} epochs={args.epochs}")
    print("=" * (20 + col_w * len(tasks)))
    header = f"{'model':<20}" + "".join(f"{t + ' (' + metric_by_task[t] + ')':<{col_w}}" for t in tasks)
    print(header)
    print("-" * (20 + col_w * len(tasks)))
    for m in models:
        row = f"{m:<20}"
        for t in tasks:
            v = matrix[m].get(t, {}).get("value", float("nan"))
            row += f"{v:<{col_w}.4f}"
        print(row)
    print("=" * (20 + col_w * len(tasks)))
    print("higher is better except MAE (regression), where lower is better.")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.results}")


if __name__ == "__main__":
    main()
