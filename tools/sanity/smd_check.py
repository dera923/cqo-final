import os, numpy as np, pandas as pd
from _util import load_df_for_gate, load_spec

os.makedirs("reports/tables", exist_ok=True)
sp = load_spec()
df, y, t, e, w = load_df_for_gate()

features = sp["gates"].get("smd_features", [])
rows=[]
for col in features:
    if col not in df.columns: continue
    x = df[col].to_numpy(dtype=float)
    x1 = x[t==1]; x0 = x[t==0]
    n1, n0 = len(x1), len(x0)
    if n1<2 or n0<2: continue
    m1, m0 = x1.mean(), x0.mean()
    s1, s0 = x1.std(ddof=1), x0.std(ddof=1)
    sd_pooled = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2))
    smd = abs((m1 - m0) / sd_pooled) if sd_pooled>0 else float("inf")
    rows.append(dict(feature=col, smd=smd, n1=n1, n0=n0))
pd.DataFrame(rows).to_csv("reports/tables/smd_metrics.csv", index=False)
print("SMD OK. max_smd=", (pd.DataFrame(rows)["smd"].max() if rows else "NA"))
