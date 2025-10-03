import os

import numpy as np
import pandas as pd
from _util import load_df_and_weight

os.makedirs("reports/tables", exist_ok=True)
df, y, t, e, mu0, mu1, w = load_df_and_weight()

e1 = np.clip(e, 1e-6, 1 - 1e-6)
term = (mu1 - mu0) + (t / e1) * (y - mu1) - ((1 - t) / (1 - e1)) * (y - mu0)
tau = float(term.mean())
se = float(term.std(ddof=1) / np.sqrt(len(term)))
pd.DataFrame(
    [{"tau_hat": tau, "se": se, "ci_lo": tau - 1.96 * se, "ci_hi": tau + 1.96 * se}]
).to_csv("reports/tables/estimates.csv", index=False)
print("DR OK:", tau, se)
