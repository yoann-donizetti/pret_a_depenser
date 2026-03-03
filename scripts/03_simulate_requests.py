"""
Script de simulation de requêtes API pour tester l'endpoint /predict.
Envoie en série des requêtes POST à partir d'un CSV de SK_ID_CURR, mesure la latence et compte les succès/échecs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()

def main():
    """
    Point d'entrée principal du script :
    - Charge un CSV de SK_ID_CURR
    - Envoie des requêtes POST à l'API /predict
    - Affiche les erreurs, la latence et le bilan final
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", default="examples/X_api.csv")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--endpoint", default="/predict")
    args = parser.parse_args()

    # 1) Chargement du CSV de SK_ID_CURR
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    if "SK_ID_CURR" not in df.columns:
        raise ValueError("Le CSV doit contenir la colonne SK_ID_CURR.")

    # 2) Préparation des paramètres de simulation
    n = min(args.n, len(df))
    url = f"{args.base_url.rstrip('/')}{args.endpoint}"

    ok = 0
    ko = 0

    # 3) Boucle d'envoi des requêtes POST
    for i in range(n):
        sk_id = int(df.iloc[i]["SK_ID_CURR"])
        payload = {"SK_ID_CURR": sk_id}

        t0 = time.time()
        try:
            r = requests.post(url, json=payload, timeout=args.timeout)
            latency_ms = round((time.time() - t0) * 1000, 2)

            if r.status_code == 200:
                ok += 1
            else:
                ko += 1
                if ko <= 5:
                    print(f"[{i+1}/{n}] KO {r.status_code} ({latency_ms}ms): {r.text[:200]}")
        except Exception as e:
            ko += 1
            if ko <= 5:
                print(f"[{i+1}/{n}] EXC: {e!r}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    # 4) Affichage du bilan final
    print(f"Done. OK={ok}  KO={ko}")


if __name__ == "__main__":
    main()