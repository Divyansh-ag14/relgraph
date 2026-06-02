"""Column stype helpers for RelBench graph construction."""

from __future__ import annotations

from typing import Dict

import torch_frame
from torch_frame import stype


def sanitize_col_to_stype_dict(
    col_to_stype_dict: Dict[str, Dict[str, stype]],
) -> Dict[str, Dict[str, stype]]:
    """Map text/embedding columns to categorical for fast CPU runs."""
    sanitized: Dict[str, Dict[str, stype]] = {}
    for table, cols in col_to_stype_dict.items():
        sanitized[table] = {}
        for col, col_stype in cols.items():
            if col_stype in (stype.text_embedded, stype.embedding, stype.multicategorical):
                sanitized[table][col] = stype.categorical
            else:
                sanitized[table][col] = col_stype
    return sanitized
