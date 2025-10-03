import pathlib as P

import numpy as np
import pandas as pd
import yaml

R = P.Path("reports")
T = R / "tables"
F = R / "figures"
R.mkdir(exist_ok=True, parents=True)
T.mkdir(exist_ok=True, parents=True)
F.mkdir(exist_ok=True, parents=True)

sp = yaml.safe_load(open("docs/requirements/core_spec.yml"))
cfg = sp.get("monitor") or {}
lam = float(cfg.get("ewma_lambda", 0.2))

# 直近サマリの取り込み
sm = pd.read_csv(T / "summary_metrics.csv").iloc[0]
kpi = (
    pd.read_csv(T / "policy_kpi.csv").iloc[0]
    if (T / "policy_kpi.csv").exists()
    else pd.DataFrame([{"lcb95_adj": np.nan}]).iloc[0]
)
row = {
    "date": pd.Timestamp.now().normalize(),
    "n": int(sm.get("n", np.nan)),
    "tail": float(sm["tail"]),
    "w95": float(sm["w95"]),
    "w99": float(sm["w99"]),
    "max_smd": (
        pd.read_csv(T / "smd_metrics.csv")["smd"].max()
        if (T / "smd_metrics.csv").exists() and "smd" in pd.read_csv(T / "smd_metrics.csv").columns
        else np.nan
    ),
    "tau_hat": float(pd.read_csv(T / "estimates.csv").iloc[0]["tau_hat"]),
    "se": float(pd.read_csv(T / "estimates.csv").iloc[0]["se"]),
    "lcb95": float(kpi["lcb95_adj"]),
    "adopt": (
        float(bool(pd.read_csv(T / "adoption_decision.csv").iloc[0]["adopt"]))
        if (T / "adoption_decision.csv").exists()
        else np.nan
    ),
}

M = T / "monitor_time.csv"
hist = pd.read_csv(M, parse_dates=["date"]) if M.exists() else pd.DataFrame(columns=row.keys())
hist = (
    pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    .drop_duplicates(subset=["date"], keep="last")
    .sort_values("date")
)
hist.to_csv(M, index=False)


# EWMA の簡易実装
def ewma(x, lam):
    y = []
    m = np.nan
    for v in x:
        if np.isnan(m):
            m = v
        else:
            m = lam * v + (1 - lam) * m
        y.append(m)
    return np.array(y)


# 図は matplotlib に依存させない（方針次第で追記可）。ここでは閾値カードのみ。
alpha = float((sp["gates"] or {}).get("alpha_tail", 0.05))
w95_max = float((sp["gates"] or {}).get("w95_max", np.inf))
w99_max = float((sp["gates"] or {}).get("w99_max", np.inf))
smd_max = float((sp["gates"] or {}).get("smd_max", np.inf))

tail_guard = hist["tail"].iloc[-1] <= alpha + 1.0 / max(int(hist["n"].iloc[-1]), 1) + 1e-12
w95_guard = hist["w95"].iloc[-1] <= w95_max + 1e-12
w99_guard = hist["w99"].iloc[-1] <= w99_max + 1e-12
smd_guard = np.isnan(hist["max_smd"].iloc[-1]) or hist["max_smd"].iloc[-1] <= smd_max + 1e-12

with open(F / "monitor_card.txt", "w") as f:
    f.write(
        f"tail_guard={tail_guard}, w95_guard={w95_guard}, w99_guard={w99_guard}, smd_guard={smd_guard}\n"
    )
print(
    "MONITOR OK. card:",
    {
        "tail_guard": tail_guard,
        "w95_guard": w95_guard,
        "w99_guard": w99_guard,
        "smd_guard": smd_guard,
    },
)
