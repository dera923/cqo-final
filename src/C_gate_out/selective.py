from pathlib import Path

import numpy as np
import pandas as pd


def holm_adjust(ci_matrix: pd.DataFrame, alpha=0.05):
    df = ci_matrix.copy()
    z = 1.96
    # 簡易：候補数 m の多重度を SE に吸収（上界側、やや保守的）
    m = max(1, len(df))
    df["lcb95_raw"] = df["tau_hat"] - z * df["se"]
    df["lcb95_adj"] = df["tau_hat"] - z * np.sqrt(m) * df["se"]
    return df[["policy", "tau_hat", "se", "lcb95_raw", "lcb95_adj"]]


def run(in_csv: str, out_csv: str):
    cand = pd.read_csv(in_csv)  # columns: policy,tau_hat,se
    adj = holm_adjust(cand)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    adj.to_csv(out_csv, index=False)
