import pandas as pd, numpy as np, os
os.makedirs("reports/tables", exist_ok=True)
# 期待: candidates.csv に policy,tau_hat,se 列
cand = pd.read_csv("reports/tables/candidates.csv")
m = len(cand); z = 1.96
cand["lcb95_raw"] = cand["tau_hat"] - z*cand["se"]
cand["lcb95_adj"] = cand["tau_hat"] - z*np.sqrt(m)*cand["se"]
cand.to_csv("reports/tables/policy_kpi.csv", index=False)
print("Selection OK, m=", m)
