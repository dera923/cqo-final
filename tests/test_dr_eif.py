import pandas as pd, numpy as np
def test_dr_outputs_exist_and_ci_width():
    est = pd.read_csv("reports/tables/estimates.csv")
    assert {"tau_hat","se","ci_lo","ci_hi"}.issubset(est.columns)
    # Series -> scalar にしてから float へ（FutureWarning対策）
    w = float(est["ci_hi"].iloc[0] - est["ci_lo"].iloc[0])
    assert np.isfinite(w) and w >= 0
