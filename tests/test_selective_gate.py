import pandas as pd, numpy as np
def test_selection_adjustment_monotone():
    df = pd.read_csv("reports/tables/policy_kpi.csv")
    assert (df["lcb95_adj"]<=df["lcb95_raw"]+1e-12).all()
