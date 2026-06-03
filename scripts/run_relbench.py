#!/usr/bin/env python3
"""List RelBench datasets/tasks. For training, use scripts/benchmark.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-dnf")
    args = parser.parse_args()

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task, get_task_names

    print(f"Tasks for {args.dataset}: {get_task_names(args.dataset)}")
    dataset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)
    print(f"Loaded {args.dataset} / {args.task}")
    print(f"Tables: {list(dataset.get_db().table_dict.keys())}")
    print(f"Train rows: {len(task.get_table('train'))}")
    print("Train with: python scripts/benchmark.py")


if __name__ == "__main__":
    main()
