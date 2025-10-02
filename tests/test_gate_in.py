import pandas as pd, numpy as np
def test_gate_in_thresholds_exist_and_tail_le_alpha():
    tbl = pd.read_csv("reports/tables/summary_metrics.csv")
    row = tbl.iloc[0]
    n = int(row.get("n", 0)) or 1
    alpha = 0.05
    assert row["tail"] <= alpha + 1.0/n + 1e-12
    assert row["w95"] >= 0 and row["w99"] >= 0
