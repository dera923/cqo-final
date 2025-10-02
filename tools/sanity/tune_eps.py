import numpy as np, pandas as pd
from _util import load_df_and_weight, load_spec

sp = load_spec()
df, y, t, e, mu0, mu1, w = load_df_and_weight()
n = len(e); alpha = float(sp["gates"]["alpha_tail"])
tol = 1.0/n  # 有限標本補正

# 離散CDFを使って、候補epsを走査 → 最小epsで tail <= alpha+1/n を満たす
e_sorted = np.sort(e.astype(float))
# 候補は 0 から 0.5 まで、端点は e_sorted 上に合わせる（左右対称なので e<=0.5 だけ見れば十分）
candidates = np.concatenate([[0.0], e_sorted[(e_sorted<=0.5)] + 0.0])

def tail_for_eps(eps):
    # ランクに基づく厳密カウント
    left  = np.searchsorted(e_sorted, eps, side='left')          # #(e < eps)
    right = n - np.searchsorted(e_sorted, 1.0 - eps, side='right') # #(e > 1-eps)
    return (left + right) / n, left, right

best = None
for eps in candidates:
    tail, L, R = tail_for_eps(eps)
    if tail <= alpha + tol:
        best = (eps, tail, L, R)
        break
# もし 0 でも超える（理論上ほぼ無い）場合は eps=0 を採用
if best is None:
    eps, tail, L, R = 0.0, *tail_for_eps(0.0)

eps_star, tail, L, R = best if best is not None else (eps, tail, L, R)
w95 = float(np.quantile(w, 0.95)); w99 = float(np.quantile(w, 0.99))
pd.DataFrame([{
    "n": n, "alpha": alpha, "alpha_tol": alpha + 1.0/n,
    "eps": float(eps_star), "tail": float(tail),
    "tail_left": int(L), "tail_right": int(R),
    "w95": w95, "w99": w99
}]).to_csv("reports/tables/summary_metrics.csv", index=False)
print(f"tuned eps: {eps_star} tail: {tail} (alpha+1/n={alpha+1.0/n})")
