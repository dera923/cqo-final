import os, numpy as np, pandas as pd
from _util import load_df_for_gate, load_spec

os.makedirs("reports/tables", exist_ok=True)
sp = load_spec()
df, y, t, e, w = load_df_for_gate()

n = len(e)
alpha = float(sp["gates"]["alpha_tail"])
mode  = str(sp["gates"].get("mode","order_stat"))

e_sorted = np.sort(e.astype(float))

if mode == "order_stat":
    # ── ここが本質：個数で固定（左右合計 floor(n*alpha) 個）
    k_total = int(np.floor(n*alpha))
    k_left  = k_total // 2
    k_right = k_total - k_left

    # 報告用の参照値（合否判定には使わない）
    eps_left  = e_sorted[k_left] if k_left>0 else 0.0
    eps_right = 1.0 - eps_left

    tail_left  = k_left
    tail_right = k_right
    tail_count = k_left + k_right
    tail = tail_count / n
    eps = eps_left
else:
    # 互換モード（未使用想定）
    try:
        eps = float(np.quantile(e, alpha/2, method="lower"))
    except TypeError:
        eps = float(np.quantile(e, alpha/2, interpolation="lower"))
    tail_left  = int((e < eps).sum())
    tail_right = int((e > 1.0 - eps).sum())
    tail_count = tail_left + tail_right
    tail = tail_count / n

w95 = float(np.quantile(w, 0.95))
w99 = float(np.quantile(w, 0.99))

# 生成
pd.DataFrame([{
    "n":n, "alpha":alpha, "mode":mode,
    "eps":float(eps), "tail":float(tail), "tail_count":int(tail_count),
    "tail_left":int(tail_left), "tail_right":int(tail_right),
    "w95":w95, "w99":w99
}]).to_csv("reports/tables/summary_metrics.csv", index=False)

print("Gate A OK:",
      {"n":n,"alpha":alpha,"mode":mode,"eps":float(eps),
       "tail":float(tail),"tail_count":int(tail_count),
       "w95":w95,"w99":w99})
