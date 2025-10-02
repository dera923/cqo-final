import numpy as np, pandas as pd
from pathlib import Path

def dr_point_and_se(df: pd.DataFrame) -> dict:
    y,t,e,mu0,mu1 = (df[c].to_numpy() for c in ["y","t","e","mu0","mu1"])
    # Doubly Robust / EIF
    term = (mu1 - mu0) + (t/e)*(y-mu1) - ((1-t)/(1-e))*(y-mu0)
    tau = term.mean()
    se  = term.std(ddof=1) / np.sqrt(len(term))
    return {"tau_hat": float(tau), "se": float(se),
            "ci_lo": float(tau - 1.96*se), "ci_hi": float(tau + 1.96*se)}

def run(input_csv: str, out_csv: str):
    df = pd.read_csv(input_csv)
    res = dr_point_and_se(df)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(out_csv, index=False)
