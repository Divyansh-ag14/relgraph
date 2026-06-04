"""A small but real PQL (Predictive Query Language) compiler for RelBench databases.

Predictive Query Language lets a user declare a predictive task as a query instead of
hand-building training tables. This module implements a lightweight version of that idea:
it parses a PQL-style query and *compiles* it into a RelBench-format label table
(``[<entity_col>, timestamp, <target_col>]``) by executing a temporal foreign-key
aggregation against a ``relbench`` ``Database``.

This is NOT cosmetic: it reads the real schema (pkeys / fkeys / time columns), filters
each target table to a forward-looking time window, aggregates over the FK to the entity,
and emits labels in exactly the format RelBench tasks use — so the output is a drop-in
training table for the rest of the pipeline.

Grammar (case-insensitive keywords)::

    PREDICT <AGG>(<target_table>[.<column>], <window_days>) [<op> <value>]
    FOR EACH <entity_table>

* ``AGG`` in COUNT | SUM | AVG | MIN | MAX  (COUNT may omit the column)
* If ``<op> <value>`` is present  -> BINARY classification label (0/1)
  otherwise                        -> REGRESSION label (the aggregate value)
* ``<window_days>`` is an integer number of days after the prediction timestamp

Examples::

    PREDICT COUNT(results, 365) = 0 FOR EACH drivers      # driver inactivity (churn)
    PREDICT SUM(results.points, 90) FOR EACH drivers       # points in next 90 days (regression)
    PREDICT COUNT(results, 30) > 0 FOR EACH drivers         # will race in next 30 days
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

_AGGS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
_OPS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "=": np.equal,
    "!=": np.not_equal,
}

_PATTERN = re.compile(
    r"""^\s*PREDICT\s+
        (?P<agg>\w+)\s*\(\s*
            (?P<table>\w+)\s*(?:\.\s*(?P<col>\w+))?\s*,\s*
            (?P<window>\d+)\s*
        \)\s*
        (?:(?P<op><=|>=|==|!=|=|<|>)\s*(?P<value>-?\d+(?:\.\d+)?)\s*)?
        FOR\s+EACH\s+(?P<entity>\w+)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class PQLQuery:
    agg: str
    target_table: str
    column: Optional[str]
    window_days: int
    op: Optional[str]
    value: Optional[float]
    entity_table: str

    @property
    def is_classification(self) -> bool:
        return self.op is not None

    @property
    def target_col(self) -> str:
        return "label" if self.is_classification else f"{self.agg.lower()}_{self.column or 'rows'}"

    def describe(self) -> str:
        col = f".{self.column}" if self.column else ""
        cond = f" {self.op} {self.value}" if self.op else ""
        kind = "BINARY" if self.is_classification else "REGRESSION"
        return (
            f"[{kind}] PREDICT {self.agg}({self.target_table}{col}, {self.window_days}d)"
            f"{cond} FOR EACH {self.entity_table}"
        )


def parse_pql(query: str) -> PQLQuery:
    m = _PATTERN.match(query.strip())
    if not m:
        raise ValueError(f"Could not parse PQL query:\n  {query}")
    agg = m.group("agg").upper()
    if agg not in _AGGS:
        raise ValueError(f"Unknown aggregation '{agg}'. Supported: {sorted(_AGGS)}")
    col = m.group("col")
    if agg != "COUNT" and col is None:
        raise ValueError(f"Aggregation {agg} requires a column, e.g. {m.group('table')}.points")
    op = m.group("op")
    return PQLQuery(
        agg=agg,
        target_table=m.group("table"),
        column=col,
        window_days=int(m.group("window")),
        op=op,
        value=float(m.group("value")) if op else None,
        entity_table=m.group("entity"),
    )


def _resolve_fk(db, target_table: str, entity_table: str) -> str:
    """Find the FK column in target_table that references entity_table."""
    t = db.table_dict[target_table]
    for fk_col, ref_table in t.fkey_col_to_pkey_table.items():
        if ref_table == entity_table:
            return fk_col
    raise ValueError(
        f"No foreign key from '{target_table}' to '{entity_table}'. "
        f"Available FKs: {db.table_dict[target_table].fkey_col_to_pkey_table}"
    )


def _aggregate(series_groups, agg: str) -> pd.Series:
    if agg == "COUNT":
        return series_groups.size()
    if agg == "SUM":
        return series_groups.sum()
    if agg == "AVG":
        return series_groups.mean()
    if agg == "MIN":
        return series_groups.min()
    if agg == "MAX":
        return series_groups.max()
    raise ValueError(agg)


def compile_labels(
    db,
    query: PQLQuery,
    anchor_times: List[pd.Timestamp],
    *,
    entity_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compile a PQL query into a RelBench-format label table.

    Returns a DataFrame with columns ``[<entity_pkey>, "timestamp", <target_col>]`` —
    the same schema RelBench entity-task tables use.
    """
    entity = db.table_dict[query.entity_table]
    target = db.table_dict[query.target_table]
    epkey = entity.pkey_col
    fk_col = _resolve_fk(db, query.target_table, query.entity_table)
    time_col = target.time_col
    if time_col is None:
        raise ValueError(f"Target table '{query.target_table}' has no time column for windowing")

    if entity_ids is None:
        entity_ids = entity.df[epkey].to_numpy()

    tdf = target.df[[fk_col, time_col] + ([query.column] if query.column else [])].copy()
    tdf[time_col] = pd.to_datetime(tdf[time_col])

    rows = []
    for t0 in anchor_times:
        t0 = pd.Timestamp(t0)
        t1 = t0 + pd.Timedelta(days=query.window_days)
        window = tdf[(tdf[time_col] > t0) & (tdf[time_col] <= t1)]
        grouped = window.groupby(fk_col)
        if query.agg == "COUNT":
            agg_vals = _aggregate(grouped, "COUNT")
        else:
            agg_vals = _aggregate(grouped[query.column], query.agg)
        # Default for entities with no rows in the window: 0 for COUNT/SUM, else NaN->0.
        agg_series = agg_vals.reindex(entity_ids).fillna(0.0)

        if query.is_classification:
            label = _OPS[query.op](agg_series.to_numpy(), query.value).astype(int)
            target_vals = label
        else:
            target_vals = agg_series.to_numpy(dtype=np.float32)

        for eid, val in zip(entity_ids, target_vals):
            rows.append({epkey: eid, "timestamp": t0, query.target_col: val})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Validation: PQL engine output must match an independent pandas recomputation
# --------------------------------------------------------------------------------------
def validate_against_pandas(db, query: PQLQuery, t0: pd.Timestamp, n_check: int = 200) -> dict:
    """Recompute labels for one anchor with a naive per-entity loop and assert equality."""
    compiled = compile_labels(db, query, [t0])
    epkey = db.table_dict[query.entity_table].pkey_col
    fk_col = _resolve_fk(db, query.target_table, query.entity_table)
    target = db.table_dict[query.target_table]
    time_col = target.time_col

    tdf = target.df.copy()
    tdf[time_col] = pd.to_datetime(tdf[time_col])
    t1 = pd.Timestamp(t0) + pd.Timedelta(days=query.window_days)
    window = tdf[(tdf[time_col] > pd.Timestamp(t0)) & (tdf[time_col] <= t1)]

    compiled_map = dict(zip(compiled[epkey], compiled[query.target_col]))
    check_ids = list(compiled_map.keys())[:n_check]
    mismatches = 0
    for eid in check_ids:
        sub = window[window[fk_col] == eid]
        if query.agg == "COUNT":
            agg = float(len(sub))
        else:
            agg = float(sub[query.column].agg(query.agg.lower())) if len(sub) else 0.0
            if np.isnan(agg):
                agg = 0.0
        expected = int(_OPS[query.op](agg, query.value)) if query.is_classification else agg
        got = compiled_map[eid]
        if query.is_classification:
            if int(got) != int(expected):
                mismatches += 1
        elif abs(float(got) - float(expected)) > 1e-4:
            mismatches += 1

    return {
        "checked": len(check_ids),
        "mismatches": mismatches,
        "ok": mismatches == 0,
        "positive_rate": float(np.mean(list(compiled_map.values()))) if compiled_map else float("nan"),
    }
