from __future__ import annotations

from typing import Iterable, Set

import pandas as pd


EXCLUDED_FEATURES: Set[str] = {"SK_ID_CURR"}
EXCLUDED_META_COLS: Set[str] = {"sk_id_curr"}


def drop_excluded_columns(df: pd.DataFrame, excluded: Iterable[str]) -> pd.DataFrame:
    """
    Supprime les colonnes spécifiées d'un DataFrame pandas si elles existent.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée dont on souhaite supprimer certaines colonnes.
    excluded : Iterable[str]
        Liste ou ensemble des noms de colonnes à exclure du DataFrame.

    Retourne
    -------
    pd.DataFrame
        DataFrame sans les colonnes spécifiées. Si le DataFrame d'entrée est None ou vide, il est retourné inchangé.

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.security import drop_excluded_columns
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    >>> df2 = drop_excluded_columns(df, excluded=['b', 'x'])
    >>> print(df2)
    """
    if df is None or df.empty:
        return df
    cols = [c for c in excluded if c in df.columns]
    return df.drop(columns=cols, errors="ignore")