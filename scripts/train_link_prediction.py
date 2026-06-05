#!/usr/bin/env python3
"""Transductive link prediction (recommendation-style) on a RelBench graph.

Reconstructs a held-out foreign-key edge type with a HeteroEncoder + GraphSAGE encoder
and a dot-product decoder, evaluated by ROC-AUC against negative samples.

Example:
    python scripts/train_link_prediction.py --dataset rel-f1
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
    parser.add_argument(
        "--target-edge",
        default="results->constructors",
        help="relation to predict as 'src->dst' or relation name (default: results->constructors)",
    )
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs" / "link_prediction.json")
    args = parser.parse_args()

    from src.relbench_pipeline.link_prediction import run_link_prediction

    result = run_link_prediction(
        dataset_name=args.dataset,
        target_edge=args.target_edge,
        channels=args.channels,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"\n{result['task']} | best test ROC-AUC: {result['best_test_metric']:.4f}")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.results}")


if __name__ == "__main__":
    main()
