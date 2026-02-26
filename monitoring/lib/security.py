from __future__ import annotations

from typing import Iterable, Set

import pandas as pd


EXCLUDED_FEATURES: Set[str] = {"SK_ID_CURR"}
EXCLUDED_META_COLS: Set[str] = {"sk_id_curr"}


def drop_excluded_columns(df: pd.DataFrame, excluded: Iterable[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [c for c in excluded if c in df.columns]
    return df.drop(columns=cols, errors="ignore")