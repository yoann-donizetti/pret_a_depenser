from __future__ import annotations

import pandas as pd
import plotly.express as px


def hist_latency(lat: pd.Series, title: str, nbins: int = 30):
    """
    Génère un histogramme des latences.

    Paramètres
    ----------
    lat : pd.Series
        Série contenant les valeurs de latence (format numérique).
    title : str
        Titre à afficher sur l'histogramme.
    nbins : int, optionnel
        Nombre de classes (barres) pour l'histogramme. Par défaut : 30.

    Retourne
    --------
    plotly.graph_objects.Figure
        Objet figure Plotly affichant la distribution des latences.

    Notes
    -----
    - Les valeurs non numériques sont converties en NaN puis supprimées.
    - Utilise Plotly Express pour la visualisation.

    Exemple
    -------
    >>> import pandas as pd
    >>> from charts import hist_latency
    >>> latences = pd.Series([100, 150, 120, 200, 180, 110])
    >>> fig = hist_latency(latences, title="Distribution des latences API", nbins=20)
    >>> fig.show()
    """
    return px.histogram(pd.to_numeric(lat, errors="coerce").dropna(), nbins=nbins, title=title)


def bar_status_codes(status_counts: pd.Series, title: str = "Codes HTTP"):
    """
    Génère un graphique en barres des codes de statut HTTP.

    Paramètres
    ----------
    status_counts : pd.Series
        Série pandas contenant les codes HTTP en index et leur nombre d'occurrences en valeurs.
    title : str, optionnel
        Titre du graphique. Par défaut : "Codes HTTP".

    Retourne
    --------
    plotly.graph_objs._figure.Figure
        Objet Figure Plotly représentant le graphique en barres.

    Exemple
    -------
    >>> import pandas as pd
    >>> from charts import bar_status_codes
    >>> status_counts = pd.Series([120, 30, 10], index=[200, 404, 500])
    >>> fig = bar_status_codes(status_counts, title="Répartition des codes HTTP")
    >>> fig.show()
    """
    return px.bar(status_counts, title=title)


def pie_decisions(dec_counts: pd.Series, title: str = "Décisions"):

    """
    Crée un graphique en secteurs (camembert) des décisions à partir d'une série pandas, en utilisant Plotly Express.
        dec_counts (pd.Series): Une série pandas contenant les comptes de chaque catégorie de décision. L'index représente les catégories, les valeurs représentent les comptes.
        title (str, optionnel): Le titre du graphique. Par défaut "Décisions".
        plotly.graph_objs._figure.Figure: Un objet Figure Plotly représentant le graphique en secteurs.
    Exemple:
        >>> import pandas as pd
        >>> from monitoring.lib.charts import pie_decisions
        >>> decisions = pd.Series({'Accepté': 50, 'Refusé': 30, 'En attente': 20})
        >>> fig = pie_decisions(decisions, title="Répartition des décisions")
        >>> fig.show()
    """
    return px.pie(dec_counts, values=dec_counts.values, names=dec_counts.index, title=title)


def bar_top_drift(topk_df: pd.DataFrame, title: str = "Top PSI (drift)"):
    """
    Génère un graphique en barres horizontales représentant les variables ayant le plus fort drift (PSI).

    Paramètres
    ----------
    topk_df : pd.DataFrame
        DataFrame contenant au moins deux colonnes : 'feature' (nom de la variable) et 'psi' (valeur du Population Stability Index).
    title : str, optionnel
        Titre du graphique. Par défaut : "Top PSI (drift)".

    Retourne
    --------
    plotly.graph_objs._figure.Figure
        Objet Figure Plotly représentant le graphique en barres horizontales.

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.charts import bar_top_drift
    >>> df = pd.DataFrame({'feature': ['age', 'revenu', 'score'], 'psi': [0.12, 0.08, 0.25]})
    >>> fig = bar_top_drift(df, title="Variables avec le plus de drift")
    >>> fig.show()
    """
    fig = px.bar(topk_df, x="psi", y="feature", orientation="h", title=title)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=650, margin=dict(l=260, r=40, t=60, b=40))
    return fig


def bar_ref_vs_prod(df_plot: pd.DataFrame, feature_name: str):
    """
    Génère un graphique en barres comparant la distribution d'une variable entre la référence (ref) et la production (prod).

    Paramètres
    ----------
    df_plot : pd.DataFrame
        DataFrame contenant au moins trois colonnes : 'label' (modalités ou intervalles de la variable), 'ref' (valeurs de référence), 'prod' (valeurs en production).
    feature_name : str
        Nom de la variable analysée (affiché dans le titre du graphique).

    Retourne
    --------
    plotly.graph_objs._figure.Figure
        Objet Figure Plotly représentant le graphique en barres groupées.

    Exemple
    -------
    >>> import pandas as pd
    >>> from monitoring.lib.charts import bar_ref_vs_prod
    >>> df = pd.DataFrame({'label': ['[0,
    """
    return px.bar(df_plot, x="label", y=["ref", "prod"], barmode="group", title=f"Dist: {feature_name}")