import sys

import numpy as np
import pandas as pd
import yaml

sp = yaml.safe_load(open("docs/requirements/core_spec.yml"))
path = (sp.get("data_contract") or {}).get("csv_path", "data/data_estimates.csv")
df = pd.read_csv(path)


def require(cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"ERROR: missing columns {miss} in {path}", file=sys.stderr)
        sys.exit(2)


require(["y", "t", "e"])
if ("mu0" not in df.columns) or ("mu1" not in df.columns):
    print("ERROR: mu0/mu1 required in prod", file=sys.stderr)
    sys.exit(3)

# 型と値域
t = df["t"].astype(float).to_numpy()
e = df["e"].astype(float).to_numpy()
y = pd.to_numeric(df["y"], errors="coerce").to_numpy()
mu0 = pd.to_numeric(df["mu0"], errors="coerce").to_numpy()
mu1 = pd.to_numeric(df["mu1"], errors="coerce").to_numpy()

assert np.isin(np.unique(t), [0.0, 1.0]).all(), "t must be {0,1}"
assert np.isfinite(e).all() and (e > 0).all() and (e < 1).all(), "e in (0,1)"
for name, arr in [("y", y), ("mu0", mu0), ("mu1", mu1)]:
    assert np.isfinite(arr).all(), f"{name} must be finite"

print("DATA VALIDATION: OK")
