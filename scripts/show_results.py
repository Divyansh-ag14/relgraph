#!/usr/bin/env python3
"""Print every saved result in outputs/ as formatted tables, and save a timestamped report.

Reads the JSON artifacts written by the other scripts and renders them:
  * benchmark_matrix.json   -> models x tasks matrix (all models, all tasks)
  * relgt_comparison.json   -> RelGT-lite vs official vs ablation
  * link_prediction.json    -> link-prediction ROC-AUC

Prints to the console AND writes the same content to a report file (default: outputs/report.txt),
prefixed with a generation timestamp.

Usage:
    python scripts/show_results.py                 # print + write report.txt
    python scripts/show_results.py --output run.txt
    python scripts/show_results.py --no-save       # print only
    python scripts/show_results.py --timestamped   # write report_YYYYMMDD_HHMMSS.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

_LINES: list[str] = []


def emit(line: str = "") -> None:
    """Print a line and capture it for the report file."""
    print(line)
    _LINES.append(line)


def _load(name: str):
    path = OUT / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _protocol_str(p: dict | None) -> str:
    if not p:
        return ""
    keys = ["train_seeds", "val_seeds", "K", "channels", "epochs", "device"]
    return "  ".join(f"{k}={p[k]}" for k in keys if k in p)


def _header(title: str) -> None:
    emit("\n" + "=" * 72)
    emit(title)
    emit("=" * 72)


def show_benchmark_matrix() -> None:
    data = _load("benchmark_matrix.json")
    if not data:
        return
    _header(f"UNIFIED BENCHMARK  —  {data.get('dataset', '?')}")
    emit(_protocol_str(data.get("protocol")))
    tasks = data["tasks"]
    models = data["models"]
    matrix = data["matrix"]

    metric_of = {}
    for t in tasks:
        for m in models:
            cell = matrix.get(m, {}).get(t)
            if cell and cell.get("metric") not in (None, "n/a"):
                metric_of[t] = cell["metric"]
                break
    arrow = {"mae": "↓", "rmse": "↓"}

    def col_label(t: str) -> str:
        met = metric_of.get(t, "?")
        return f"{t} ({met}{arrow.get(met, '↑')})"

    cols = [col_label(t) for t in tasks]
    widths = [max(len(c), 9) for c in cols]
    emit("")
    emit(f"{'model':<16}" + "".join(f"{c:>{w + 3}}" for c, w in zip(cols, widths)))
    emit("-" * (16 + sum(w + 3 for w in widths)))
    for m in models:
        row = f"{m:<16}"
        for t, w in zip(tasks, widths):
            cell = matrix.get(m, {}).get(t, {})
            v = cell.get("value")
            txt = "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.3f}"
            row += f"{txt:>{w + 3}}"
        emit(row)


def show_relgt_comparison() -> None:
    data = _load("relgt_comparison.json")
    if not data:
        return
    _header(f"RELGT COMPARISON  —  {data.get('dataset', '?')} / {data.get('task', '?')}")
    emit(_protocol_str(data.get("protocol")))
    emit("")
    emit(f"{'model':<42}{'best val':>12}{'params':>12}")
    emit("-" * 66)
    for r in data["results"]:
        metric = r.get("best_val_metric", r.get("best_val_roc_auc", float("nan")))
        params = int(r.get("params", 0))
        emit(f"{r['model']:<42}{metric:>12.4f}{params:>12,}")


def show_link_prediction() -> None:
    data = _load("link_prediction.json")
    if not data:
        return
    _header("LINK PREDICTION")
    metric = data.get("metric", "roc_auc")
    emit(f"{data.get('task', '?'):<42}{metric} = {data.get('best_test_metric', float('nan')):.4f}")
    emit(f"params: {int(data.get('params', 0)):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/report.txt",
        help="report file path (default: outputs/report.txt)",
    )
    parser.add_argument("--no-save", action="store_true", help="print only, do not write a report file")
    parser.add_argument(
        "--timestamped",
        action="store_true",
        help="append a timestamp to the report filename (outputs/report_YYYYMMDD_HHMMSS.txt)",
    )
    args = parser.parse_args()

    if not OUT.exists() or not any(OUT.glob("*.json")):
        print(f"No result JSONs found in {OUT}. Run a script first, e.g.:")
        print("  python scripts/benchmark.py --tasks driver-dnf driver-top3 driver-position \\")
        print("      --models flat graphsage relgt-lite")
        sys.exit(0)

    now = datetime.now()
    _header("RELGRAPH — RESULTS REPORT")
    emit(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.astimezone().tzname()})")
    emit(f"Source JSONs: {OUT}")

    show_benchmark_matrix()
    show_relgt_comparison()
    show_link_prediction()

    emit("\n" + "=" * 72)
    emit("Note: these are saved results; re-run the individual scripts to refresh them.")

    if not args.no_save:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        if args.timestamped:
            out_path = out_path.with_name(f"{out_path.stem}_{now.strftime('%Y%m%d_%H%M%S')}{out_path.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
        print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
