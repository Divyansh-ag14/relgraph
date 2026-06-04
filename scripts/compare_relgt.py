#!/usr/bin/env python3
"""Apples-to-apples comparison on a RelBench binary task (CPU, identical subset protocol).

Runs three configurations on the *same* dataset/task/seeds/K so the numbers are directly
comparable, and prints a results table:

  1. RelGT-lite (local + global)   - our from-scratch reimplementation
  2. RelGT-lite (local only)        - ablation: removes the global centroid attention
  3. Official RelGT (snap-stanford) - reference architecture, single-process CPU port

This demonstrates (a) the from-scratch model reproduces the official architecture's
behavior, and (b) the global centroid attention contributes.

Example:
    python scripts/compare_relgt.py --train-seeds 1024 --val-seeds 512 --epochs 8
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
    parser.add_argument("--task", default="driver-dnf")
    parser.add_argument("--train-seeds", type=int, default=1024)
    parser.add_argument("--val-seeds", type=int, default=512)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-official", action="store_true", help="skip the official RelGT run")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs" / "relgt_comparison.json")
    args = parser.parse_args()

    from src.relbench_pipeline.relgt_lite import run_relgt_lite_experiment

    rows: list[dict] = []

    print("\n### [1/3] RelGT-lite (local + global) ###")
    lite_full = run_relgt_lite_experiment(
        dataset_name=args.dataset,
        task_name=args.task,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        K=args.k,
        channels=args.channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_global=True,
        seed=args.seed,
    )
    rows.append({"model": "RelGT-lite (local+global)", **lite_full})

    print("\n### [2/3] RelGT-lite (local only, ablation) ###")
    lite_local = run_relgt_lite_experiment(
        dataset_name=args.dataset,
        task_name=args.task,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        K=args.k,
        channels=args.channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_global=False,
        seed=args.seed,
    )
    rows.append({"model": "RelGT-lite (local only)", **lite_local})

    if not args.skip_official:
        print("\n### [3/3] Official RelGT (snap-stanford) ###")
        from scripts.run_official_relgt import run_official_relgt

        official = run_official_relgt(
            dataset=args.dataset,
            task=args.task,
            train_seeds=args.train_seeds,
            val_seeds=args.val_seeds,
            k=args.k,
            channels=args.channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            conv_type="full",
            seed=args.seed,
        )
        rows.append({"model": "Official RelGT (full)", **official})

    print("\n" + "=" * 64)
    print(f"RESULTS  {args.dataset} / {args.task}  (CPU, subset protocol)")
    print(f"train_seeds={args.train_seeds} val_seeds={args.val_seeds} K={args.k} "
          f"channels={args.channels} epochs={args.epochs}")
    metric_name = rows[0].get("metric", "roc_auc")
    print("=" * 64)
    print(f"{'model':<32}{('best val ' + metric_name):>18}{'params':>12}")
    print("-" * 64)
    for r in rows:
        print(f"{r['model']:<32}{r['best_val_metric']:>18.4f}{int(r['params']):>12,}")
    print("=" * 64)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "task": args.task,
        "protocol": {
            "train_seeds": args.train_seeds,
            "val_seeds": args.val_seeds,
            "K": args.k,
            "channels": args.channels,
            "epochs": args.epochs,
            "device": "cpu",
        },
        "results": rows,
    }
    with open(args.results, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.results}")


if __name__ == "__main__":
    main()
