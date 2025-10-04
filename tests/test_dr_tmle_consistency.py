from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SSOT = yaml.safe_load(open("docs/requirements/core_spec.yml", encoding="utf-8")) or {}
gate = SSOT.get("gate", {})
ALPHA = float(gate.get("alpha_tail", 0.05))


def test_eif_centering():
    p = Path("reports/compare/BASE_1/baseline/estimates.csv")
    assert p.exists(), f"missing {p}"
    df = pd.read_csv(p)
    # 期待: eif 列がある（なければスキップ）
    if "eif" not in df.columns:
        pytest.skip("no EIF column in estimates.csv")
    m = float(np.nanmean(df["eif"].astype(float)))
    assert abs(m) < 1e-2, f"EIF is not centered: mean={m}"


def test_dr_tmle_close():
    p = Path("reports/compare/BASE_1/baseline/estimates.csv")
    assert p.exists(), f"missing {p}"
    df = pd.read_csv(p)
    if not {"tau_dr", "se_dr", "tau_tml", "se_tml"}.issubset(df.columns):
        need = {"tau_hat", "se", "ci_lo", "ci_hi"}
        assert need.issubset(
            df.columns
        ), f"missing cols for CI check: {need - set(df.columns)}"
        tau = float(df["tau_hat"].iloc[0])
        se = float(df["se"].iloc[0])
        lo = float(df["ci_lo"].iloc[0])
        hi = float(df["ci_hi"].iloc[0])
        # 95%CI ≈ tau ± 1.96*se（生成側丸め誤差を0.01に緩和）

        assert (
            abs((tau - 1.96 * se) - lo) <= 0.01
        ), f"ci_lo mismatch: got {lo}, expect≈{tau-1.96*se}"
        assert (
            abs((tau + 1.96 * se) - hi) <= 0.01
        ), f"ci_hi mismatch: got {hi}, expect≈{tau+1.96*se}"
        return
    # ここから先はDR/TMLE列がある場合の厳格チェック
    tau_dr = float(df["tau_dr"].iloc[0])
    se_dr = float(df["se_dr"].iloc[0])
    tau_tm = float(df["tau_tml"].iloc[0])
    se_tm = float(df["se_tml"].iloc[0])
    delta = abs(tau_dr - tau_tm)
    se_comb = np.hypot(se_dr, se_tm)  # conservative
    assert delta <= 3.0 * se_comb + 1e-12, f"|DR-TMLE|={delta} > 3*se={3*se_comb}"


def test_order_stat_tail_count():
    mode = gate.get("mode", "")
    if mode != "order_stat":
        pytest.skip("gate mode is not order_stat")
    p = Path("reports/logs/gate_check.txt")
    if not p.exists():
        pytest.skip("gate_check log missing")
    # 例: ログに tail_count=N の行が出ている想定
    txt = p.read_text(encoding="utf-8", errors="ignore")
    import re

    m = re.search(r"tail_count\s*=\s*(\d+)", txt)
    if not m:
        pytest.skip("tail_count not logged")
    tail_count = int(m.group(1))
    # n は estimates.csv の行数などから近似（無ければスキップ）
    q = Path("reports/compare/BASE_1/baseline/estimates.csv")
    if not q.exists():
        pytest.skip("estimates.csv missing for n")
    n = len(pd.read_csv(q))
    expected = int(np.floor(n * ALPHA))
    assert (
        abs(tail_count - expected) <= 1
    ), f"tail_count={tail_count}, expected≈{expected}"
