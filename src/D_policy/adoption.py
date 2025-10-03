from pathlib import Path

import pandas as pd


def decide(kpi_csv: str, out_csv: str):
    df = pd.read_csv(kpi_csv)
    df["adopt"] = df["lcb95_adj"] > 0
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
