from pathlib import Path

import pandas as pd

p = Path("reports/tables/estimates.csv")
if not p.exists():
    raise SystemExit(f"missing file: {p}")

df = pd.read_csv(p)
need = {"tau_hat", "se"}
if "lcb95" not in df.columns and need.issubset(df.columns):
    Z = 1.645  # one-sided 95% (alpha_tail=0.05)
    df["lcb95"] = df["tau_hat"] - Z * df["se"]
    df.to_csv(p, index=False)
    print("[fix] added lcb95 to reports/tables/estimates.csv")
else:
    print("[ok] lcb95 present or required cols missing; no change")
