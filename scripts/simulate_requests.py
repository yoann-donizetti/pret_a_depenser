# scripts/simulate_requests.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", default="examples/X_api.csv")  # <-- évite app/assets/
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable: {csv_path}. "
            f"Conseil: mets X_api.csv dans examples/ (versionné) ou indique --csv <chemin>."
        )

    df = pd.read_csv(csv_path)
    if args.n > len(df):
        args.n = len(df)

    url = f"{args.base-url.rstrip('/')}/predict" if False else f"{args.base_url.rstrip('/')}/predict"

    ok = 0
    ko = 0

    for i in range(args.n):
        row = df.iloc[i].to_dict()

        # NaN -> None pour JSON
        payload = {k: (None if pd.isna(v) else v) for k, v in row.items()}

        t0 = time.time()
        try:
            r = requests.post(url, json=payload, timeout=10)
            latency_ms = round((time.time() - t0) * 1000, 2)

            if r.status_code == 200:
                ok += 1
            else:
                ko += 1
                if ko <= 5:
                    print(f"[{i+1}/{args.n}] KO {r.status_code} ({latency_ms}ms): {r.text[:200]}")

        except Exception as e:
            ko += 1
            if ko <= 5:
                print(f"[{i+1}/{args.n}] EXC: {e!r}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. OK={ok}  KO={ko}")
    print("Logs attendus côté API dans: prod_logs/requests.jsonl (ou PROD_LOG_DIR si défini)")


if __name__ == "__main__":
    main()