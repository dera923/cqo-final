import pandas as pd, sys, os
base = "data/data_estimates.csv"
mu   = "data/interim/mu_scores.csv"   # ← モデル出力の想定パス
key  = "id"                            # ← JOINキー（無ければindexで代用）

df = pd.read_csv(base)
if os.path.exists(mu):
    mf = pd.read_csv(mu)
    if key in df.columns and key in mf.columns:
        out = df.merge(mf[[key,"t","mu0","mu1"]], on=key, how="left", validate="one_to_one")
    else:
        mf = mf[["t","mu0","mu1"]]
        mf.index = df.index[:len(mf)]
        out = df.copy()
        for c in ["t","mu0","mu1"]:
            out[c] = mf[c].values
else:
    print(f"ERROR: missing {mu}", file=sys.stderr); sys.exit(2)

need = ["y","e","t","mu0","mu1"]
miss = [c for c in need if c not in out.columns or out[c].isna().any()]
if miss:
    print(f"ERROR: columns missing or NA: {miss}", file=sys.stderr); sys.exit(3)

out.to_csv(base, index=False)
print("merged ->", base)
