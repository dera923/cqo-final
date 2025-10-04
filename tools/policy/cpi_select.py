import os
import sys
from pathlib import Path

import pandas as pd

BASE = os.getenv("COMPARE_BASE", "BASE_1")
KPI_PATH = Path(f"reports/compare/{BASE}/baseline/policy_kpi.csv")
OUT = Path("reports/tables/adoption_decision.csv")

# KPI重み（必要なら環境変数で調整）
W_CTR = float(os.getenv("W_CTR", "1.0"))
W_CVR = float(os.getenv("W_CVR", "1.0"))
W_PUR = float(os.getenv("W_PUR", "1.0"))


def lcb95_series(df: pd.DataFrame) -> pd.Series:
    if "lcb95_adj" in df.columns:
        return df["lcb95_adj"]
    if {"tau_hat", "se"}.issubset(df.columns):
        return df["tau_hat"] - 1.96 * df["se"]
    raise ValueError("policy_kpi.csv に lcb95_adj か (tau_hat,se) が必要です")


def main() -> int:
    if not KPI_PATH.exists():
        print(f"[ERROR] missing: {KPI_PATH}", file=sys.stderr)
        return 2
    df = pd.read_csv(KPI_PATH)

    if "policy" not in df.columns:
        print("[ERROR] 'policy' 列が必要です", file=sys.stderr)
        return 2

    # 必須列の用意（無ければ0で埋める。CTR/CVR/PURは確率/比率想定）
    for c in ("ctr", "cvr", "pur"):
        if c not in df.columns:
            df[c] = 0.0

    # LCB95 を計算（adjusted優先）
    df["lcb95"] = lcb95_series(df)

    # LCB95>0 のみ採択候補（Fail-Closed）
    cand = df[df["lcb95"] > 0].copy()
    if cand.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"policy": None, "adopted": False, "adopt": False, "reason": "no_candidate_lcb95_le0"}]
        ).to_csv(OUT, index=False)
        print("[adopt] no candidate (LCB95<=0 only) -> adopted=False")
        return 0

    # CPI: 単純加重和（CTR/CVR/PURは[0,1]スケール前提）
    cand["cpi"] = W_CTR * cand["ctr"] + W_CVR * cand["cvr"] + W_PUR * cand["pur"]

    # ランク付け（最大CPIが採択）
    cand = cand.sort_values("cpi", ascending=False).reset_index(drop=True)
    cand["rank"] = cand.index + 1
    cand["adopted"] = cand["rank"] == 1

    # 出力（監査用に主要列を残す）
    cols = [
        c
        for c in ["policy", "cpi", "lcb95", "ctr", "cvr", "pur", "rank", "adopted"]
        if c in cand.columns
    ]
    cand["adopt"] = cand["adopted"].astype(bool)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cand[cols + ["adopt"]].to_csv(OUT, index=False)

    top = cand.iloc[0][["policy", "cpi", "lcb95"]].to_dict()
    print(f"[adopt] policy={top['policy']}  cpi={top['cpi']:.6f}  lcb95={top['lcb95']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
