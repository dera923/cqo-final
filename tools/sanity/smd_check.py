import os

import numpy as np
import pandas as pd
from _util import load_df_for_gate, load_spec

os.makedirs("reports/tables", exist_ok=True)
sp = load_spec()
df, y, t, e, w = load_df_for_gate()

features = sp["gates"].get("smd_features", [])
rows = []
for col in features:
    if col not in df.columns:
        continue
    x = pd.to_numeric(df[col], errors="coerce").to_numpy()
    msk1 = (t == 1) & np.isfinite(x)
    msk0 = (t == 0) & np.isfinite(x)
    if not msk1.any() or not msk0.any():
        continue
    x1, x0 = x[msk1], x[msk0]
    n1, n0 = len(x1), len(x0)
    m1, m0 = x1.mean(), x0.mean()
    s1, s0 = x1.std(ddof=1), x0.std(ddof=1)
    denom = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / max(n1 + n0 - 2, 1))
    smd = abs(m1 - m0) / denom if np.isfinite(denom) and denom > 0 else np.inf
    rows.append(
        {
            "feature": col,
            "n1": int(n1),
            "n0": int(n0),
            "smd": float(smd),
            "m1": float(m1),
            "m0": float(m0),
        }
    )

# ヘッダは必ず出す
cols = ["feature", "n1", "n0", "smd", "m1", "m0"]
out = pd.DataFrame(rows, columns=cols)
out.to_csv("reports/tables/smd_metrics.csv", index=False)
print("SMD OK. max_smd:", (out["smd"].max() if len(out) else "NA"))
