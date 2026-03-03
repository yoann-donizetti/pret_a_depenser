"""
Script d'analyse des logs de production API :
- Calcule les métriques d'exploitation (latence, taux d'erreur)
- Calcule le drift PSI entre les distributions de features en production et les distributions de référence
- Génère des rapports JSON et CSV pour le monitoring
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from core.db.repo_prod_requests import select_prod_requests
from core.db.repo_ref_dist import load_all_ref


def _psi(p: List[float], q: List[float], eps: float = 1e-6) -> float:
    """
    PSI = Σ (p - q) * ln(p/q)
    - p : distribution prod
    - q : distribution ref
    """
    """
    Calcule le Population Stability Index (PSI) entre deux distributions p et q.
    - p : distribution production
    - q : distribution référence
    Retourne :
        float : valeur du PSI
    """
    s = 0.0
    for pi, qi in zip(p, q):
        pi2 = max(eps, float(pi))
        qi2 = max(eps, float(qi))
        s += (pi2 - qi2) * math.log(pi2 / qi2)
    return float(s)


def _prod_dist_numeric(values: pd.Series, edges: List[float]) -> Tuple[List[str], List[float]]:
    """
    Calcule la distrib prod sur les bins (edges) de la ref.
    Retourne labels + p (normalisé).
    """
    """
    Calcule la distribution de la production sur les bins numériques de la référence.
    Retourne :
        (labels, p) : labels des bins et proportions normalisées
    """
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) == 0:
        return ["__EMPTY__"], [1.0]

    edges_sorted = sorted(set(float(e) for e in edges))
    if len(edges_sorted) < 2:
        return ["__EMPTY__"], [1.0]

    binned = pd.cut(x, bins=edges_sorted, include_lowest=True, duplicates="drop")
    dist = binned.value_counts(normalize=True).sort_index()

    labels = [str(idx) for idx in dist.index]
    p = [float(v) for v in dist.values]
    if not p:
        return ["__EMPTY__"], [1.0]

    return labels, p


def _prod_dist_categorical(values: pd.Series, ref_labels: List[str]) -> Tuple[List[str], List[float]]:
    """
    Calcule la distrib prod sur les labels de la ref.
    - Toute modalité inconnue -> "__OTHER__" si présent dans la ref.
    - NaN -> "__MISSING__"
    """
    """
    Calcule la distribution de la production sur les labels catégoriels de la référence.
    - Modalités inconnues mappées sur '__OTHER__' si présent dans la ref
    - NaN mappé sur '__MISSING__'
    Retourne :
        (labels, p) : labels de la ref et proportions normalisées
    """
    x = values.fillna("__MISSING__").astype(str)
    if len(x) == 0:
        return ["__EMPTY__"], [1.0]

    ref_set = set(ref_labels)
    has_other = "__OTHER__" in ref_set

    mapped = []
    for v in x.tolist():
        if v in ref_set:
            mapped.append(v)
        else:
            mapped.append("__OTHER__" if has_other else v)

    vc = pd.Series(mapped).value_counts(normalize=True)

    labels = list(ref_labels)
    p = [float(vc.get(lbl, 0.0)) for lbl in labels]

    s = sum(p)
    if s <= 0:
        return ["__EMPTY__"], [1.0]
    p = [v / s for v in p]

    return labels, p


def main() -> None:
    """
    Point d'entrée principal du script :
    - Charge les logs de production depuis la base
    - Calcule les métriques d'exploitation (latence, erreurs)
    - Calcule le drift PSI pour chaque feature
    - Génère les rapports de monitoring (JSON, CSV)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="/predict")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--outdir", default="reports/monitoring_prod")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Récupération des logs de production depuis la base de données
    logs = select_prod_requests(endpoint=args.endpoint, limit=int(args.limit))
    if not logs:
        raise RuntimeError("Aucun log trouvé (ou DB non connectée).")

    df = pd.DataFrame(logs)

    # 2) Calcul des métriques d'exploitation (latence, taux d'erreur)
    total = int(len(df))
    status = pd.to_numeric(df["status_code"], errors="coerce").fillna(0).astype(int)
    errors = int((status >= 400).sum())
    error_rate = float(errors / total) if total else 0.0

    lat = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
    ops_report = {
        "endpoint": args.endpoint,
        "n_requests": total,
        "n_errors": errors,
        "error_rate": round(error_rate, 4),
        "latency_ms": {
            "mean": round(float(lat.mean()), 2) if len(lat) else None,
            "p50": round(float(lat.quantile(0.50)), 2) if len(lat) else None,
            "p95": round(float(lat.quantile(0.95)), 2) if len(lat) else None,
            "p99": round(float(lat.quantile(0.99)), 2) if len(lat) else None,
            "max": round(float(lat.max()), 2) if len(lat) else None,
        },
    }
    (outdir / "monitoring_report.json").write_text(
        json.dumps(ops_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 3) Calcul du drift PSI entre  la production et la référence pour chaque feature
    inputs_series = df.get("inputs")
    if inputs_series is None:
        raise RuntimeError("La colonne 'inputs' est absente des logs.")

    inputs_series = inputs_series.dropna()
    prod_inputs = pd.json_normalize(inputs_series.tolist()) if len(inputs_series) else pd.DataFrame()

    refs = load_all_ref()
    if not refs:
        raise RuntimeError("ref_feature_dist est vide (build_reference_dist pas exécuté).")

    rows_out: List[Dict[str, Any]] = []

    for ref in refs:
        feat = ref["feature"]
        kind = ref["kind"]
        bins_json = ref.get("bins_json") or {}
        ref_dist = ref.get("ref_dist_json") or {}

        ref_labels = list(ref_dist.get("labels", []))
        ref_p = list(ref_dist.get("p", []))

        if feat not in prod_inputs.columns:
            rows_out.append(
                {
                    "feature": feat,
                    "kind": kind,
                    "psi": None,
                    "note": "missing_in_prod_inputs",
                }
            )
            continue

        s = prod_inputs[feat]

        if kind == "numeric":
            edges = bins_json.get("edges", [])
            if not isinstance(edges, list) or len(edges) < 2 or len(ref_p) == 0:
                rows_out.append({"feature": feat, "kind": kind, "psi": None, "note": "bad_ref_bins"})
                continue

            prod_labels, prod_p = _prod_dist_numeric(s, edges)

            # important: si le binning produit ne matche pas la ref -> on signale
            if len(prod_p) != len(ref_p):
                rows_out.append(
                    {
                        "feature": feat,
                        "kind": kind,
                        "psi": None,
                        "note": "bin_mismatch_ref_vs_prod",
                    }
                )
                continue

            psi_val = _psi(prod_p, ref_p)
            rows_out.append({"feature": feat, "kind": kind, "psi": round(psi_val, 6), "note": ""})

        else:
            if len(ref_labels) == 0 or len(ref_p) == 0:
                rows_out.append({"feature": feat, "kind": kind, "psi": None, "note": "bad_ref_labels"})
                continue

            prod_labels, prod_p = _prod_dist_categorical(s, ref_labels)

            if len(prod_p) != len(ref_p):
                rows_out.append(
                    {
                        "feature": feat,
                        "kind": kind,
                        "psi": None,
                        "note": "label_mismatch_ref_vs_prod",
                    }
                )
                continue

            psi_val = _psi(prod_p, ref_p)
            rows_out.append({"feature": feat, "kind": kind, "psi": round(psi_val, 6), "note": ""})

    # 4) Génération des rapports de drift PSI (CSV, JSON)
    psi_df = pd.DataFrame(rows_out)
    psi_df_sorted = psi_df.sort_values(by="psi", ascending=False, na_position="last")
    psi_df_sorted.to_csv(outdir / "psi_table.csv", index=False)

    def _count_gt(th: float) -> int:
        """Compte le nombre de features dont le PSI dépasse un seuil donné."""
        return int((psi_df["psi"].dropna() > th).sum())

    psi_summary = {
        "n_features_ref": int(len(refs)),
        "n_features_scored": int(psi_df["psi"].notna().sum()),
        "n_missing_in_prod": int((psi_df["note"] == "missing_in_prod_inputs").sum()),
        "psi_thresholds": {
            "gt_0_10": _count_gt(0.10),
            "gt_0_25": _count_gt(0.25),
        },
    }
    (outdir / "psi_summary.json").write_text(
        json.dumps(psi_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 5) Affichage des chemins des rapports générés
    print(f"OK: écrit {outdir / 'monitoring_report.json'}")
    print(f"OK: écrit {outdir / 'psi_table.csv'}")
    print(f"OK: écrit {outdir / 'psi_summary.json'}")


if __name__ == "__main__":
    main()