from __future__ import annotations

import pandas as pd


def extract_timings(prod_outputs: pd.DataFrame) -> pd.DataFrame:
    """Extrait outputs.timing -> colonnes db_ms/validation_ms/inference_ms/total_ms."""
    if "timing" not in prod_outputs.columns:
        return pd.DataFrame()

    timing_series = prod_outputs["timing"].apply(lambda x: x if isinstance(x, dict) else {})
    timing_df = pd.json_normalize(timing_series)

    for c in ["db_ms", "validation_ms", "inference_ms", "total_ms"]:
        if c not in timing_df.columns:
            timing_df[c] = pd.NA

    # cast numérique
    for c in ["db_ms", "validation_ms", "inference_ms", "total_ms"]:
        timing_df[c] = pd.to_numeric(timing_df[c], errors="coerce")

    return timing_df[["db_ms", "validation_ms", "inference_ms", "total_ms"]]