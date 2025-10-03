import os

import pandas as pd

os.makedirs("reports/tables", exist_ok=True)
df = pd.read_csv("reports/tables/policy_kpi.csv")
df["adopt"] = df["lcb95_adj"] > 0
df.to_csv("reports/tables/adoption_decision.csv", index=False)
print("Adoption OK:", df["adopt"].value_counts().to_dict())
