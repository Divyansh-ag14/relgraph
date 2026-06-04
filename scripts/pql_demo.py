#!/usr/bin/env python3
"""Demo + validation of the PQL compiler on a RelBench database.

Parses PQL queries, compiles them into RelBench-format label tables via temporal
foreign-key aggregation, and validates each against an independent pandas recomputation.

Example:
    python scripts/pql_demo.py --dataset rel-f1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A few illustrative queries spanning classification + regression.
DEFAULT_QUERIES = [
    "PREDICT COUNT(results, 365) = 0 FOR EACH drivers",     # driver inactivity / churn (binary)
    "PREDICT COUNT(results, 30) > 0 FOR EACH drivers",       # will race in next 30 days (binary)
    "PREDICT SUM(results.points, 90) FOR EACH drivers",      # points in next 90 days (regression)
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--cutoff", default="2008-06-01", help="prediction timestamp (anchor; rel-f1 data spans 1950-2009)")
    args = parser.parse_args()

    import pandas as pd

    from relbench.datasets import get_dataset
    from src.relbench_pipeline.pql import compile_labels, parse_pql, validate_against_pandas

    db = get_dataset(args.dataset, download=True).get_db()
    t0 = pd.Timestamp(args.cutoff)

    print(f"\nPQL compiler on {args.dataset}  (anchor timestamp = {t0.date()})")
    print("=" * 78)
    all_ok = True
    for q in args.queries:
        query = parse_pql(q)
        labels = compile_labels(db, query, [t0])
        report = validate_against_pandas(db, query, t0)
        all_ok = all_ok and report["ok"]
        print(f"\n{query.describe()}")
        print(f"  query string : {q}")
        print(f"  label table  : {len(labels)} rows, columns = {list(labels.columns)}")
        if query.is_classification:
            print(f"  positive rate: {report['positive_rate']:.3f}")
        else:
            print(f"  mean target  : {report['positive_rate']:.3f}")
        status = "OK" if report["ok"] else f"MISMATCH ({report['mismatches']})"
        print(f"  validation   : {status} (checked {report['checked']} entities vs pandas)")

    print("\n" + "=" * 78)
    print("All queries validated against pandas recomputation." if all_ok else "VALIDATION FAILED.")
    print(
        "These label tables share RelBench's [entity, timestamp, target] schema, so they "
        "drop straight into the training pipeline."
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
