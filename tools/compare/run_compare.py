import argparse
import pathlib as P
import sys

import numpy as np
import pandas as pd
import yaml


def rd(p):
    return pd.read_csv(p) if P.Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", required=True, help="baseline tag used in reports/compare/<tag>/baseline"
    )
    args = ap.parse_args()

    sp = yaml.safe_load(open("docs/requirements/core_spec.yml"))
    out_dir = P.Path((sp.get("compare") or {}).get("out_dir") or "reports/compare/")
    base_dir = out_dir / args.base / "baseline"
    new_dir = out_dir / args.base / "new"
    tbl_dir = P.Path("reports/tables")

n_sd = rd(tbl_dir / "smd_metrics.csv")
    if not base_dir.exists():
        print(f"ERROR: baseline not found: {base_dir}", file=sys.stderr)
        sys.exit(2)
    new_dir.mkdir(parents=True, exist_ok=True)

    # 読み込み
    b_sm = rd(base_dir / "summary_metrics.csv")
    n_sm = rd(tbl_dir / "summary_metrics.csv")
    b_es = rd(base_dir / "estimates.csv")
    n_es = rd(tbl_dir / "estimates.csv")
    b_kp = rd(base_dir / "policy_kpi.csv")
    n_kp = rd(tbl_dir / "policy_kpi.csv")
    b_ad = rd(base_dir / "adoption_decision.csv")
    n_ad = rd(tbl_dir / "adoption_decision.csv")

    # 要約行を取り出し
    def one(df, cols):
        if df is None or df.empty:
            return {c: np.nan for c in cols}
        s = df.iloc[0]
        return {c: float(s.get(c, np.nan)) for c in cols}

    sm_cols = ["tail", "w95", "w99"]
    es_cols = ["tau_hat", "se"]
    kp_cols = ["lcb95_adj"]
    ba, na = one(b_sm, sm_cols), one(n_sm, sm_cols)
    be, ne = one(b_es, es_cols), one(n_es, es_cols)
    bk, nk = one(b_kp, kp_cols), one(n_kp, kp_cols)

    # 差分
    summary = pd.DataFrame(
        [
            {
                "base_tag": args.base,
                "d_tau": ne["tau_hat"] - be["tau_hat"],
                "d_se": ne["se"] - be["se"],
                "d_lcb95": nk["lcb95_adj"] - bk["lcb95_adj"],
                "d_tail": na["tail"] - ba["tail"],
                "d_w95": na["w95"] - ba["w95"],
                "d_w99": na["w99"] - ba["w99"],
                "adopt_base": (
                    bool(b_ad.iloc[0]["adopt"]) if (b_ad is not None and not b_ad.empty) else np.nan
                ),
                "adopt_new": (
                    bool(n_ad.iloc[0]["adopt"]) if (n_ad is not None and not n_ad.empty) else np.nan
                ),
            }
        ]
    )
    summary.to_csv(new_dir / "summary.csv", index=False)

    # SLOガード判定（非退行）
    gates = sp.get("gates") or {}
    w95_max = float(gates.get("w95_max", np.inf))
    w99_max = float(gates.get("w99_max", np.inf))
    smd_max = float(gates.get("smd_max", np.inf))
    alpha = float(gates.get("alpha_tail", 0.05))

    # tail guard（n は new の値を使用）
    n = (
        int(n_sm.iloc[0]["n"])
        if (n_sm is not None and not n_sm.empty and "n" in n_sm.columns)
        else 1
    )
    tail_guard = na["tail"] <= alpha + 1.0 / max(n, 1) + 1e-12
    w95_guard = na["w95"] <= w95_max + 1e-12
    w99_guard = na["w99"] <= w99_max + 1e-12
    smd_guard = True
    if n_sd is not None and not n_sd.empty and "smd" in n_sd.columns:
        smd_guard = bool(n_sd["smd"].max() <= smd_max + 1e-12)

    # LCB95非退行（base>0 → new>0 を要求）
    lcb95_nr = True
    if not np.isnan(bk["lcb95_adj"]):
        if bk["lcb95_adj"] > 0 and not (nk["lcb95_adj"] > 0):
            lcb95_nr = False

    guards = pd.DataFrame(
        [
            {
                "tail_guard": tail_guard,
                "w95_guard": w95_guard,
                "w99_guard": w99_guard,
                "smd_guard": smd_guard,
                "lcb95_non_regression": lcb95_nr,
            }
        ]
    )
    guards.to_csv(new_dir / "guards.csv", index=False)

    print("COMPARE OK:", (new_dir / "summary.csv"), (new_dir / "guards.csv"))


if __name__ == "__main__":
    main()
