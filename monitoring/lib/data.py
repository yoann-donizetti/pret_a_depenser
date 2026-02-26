from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from core.db.repo_prod_requests import select_prod_requests
from core.db.repo_ref_dist import load_all_ref, load_one_ref

from monitoring.lib.filters import apply_time_filter, filter_rows_by_meta_ts
from monitoring.lib.security import drop_excluded_columns


def load_prod_data(
    *,
    endpoint: str,
    limit: int | None,
    time_window: str,
    excluded_features: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Retourne:
      prod_meta_df, prod_inputs_df, prod_outputs_df, raw_rows_filtered
    """
    rows = select_prod_requests(endpoint=endpoint, limit=(limit or 1_000_000))
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    prod_meta = pd.DataFrame(
        [{k: r.get(k) for k in ["ts", "endpoint", "status_code", "latency_ms", "error", "message"]} for r in rows]
    )

    prod_meta = apply_time_filter(prod_meta, time_window)
    rows = filter_rows_by_meta_ts(rows, prod_meta)

    prod_inputs = pd.DataFrame([r.get("inputs") or {} for r in rows])
    prod_outputs = pd.DataFrame([r.get("outputs") or {} for r in rows])

    prod_inputs = drop_excluded_columns(prod_inputs, excluded_features)

    return prod_meta, prod_inputs, prod_outputs, rows


def load_reference() -> List[Dict]:
    return load_all_ref()


def load_reference_one(feature: str) -> Dict | None:
    return load_one_ref(feature)