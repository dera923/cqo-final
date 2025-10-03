from pathlib import Path

import pandas as pd
import pytest
import yaml

SSOT = yaml.safe_load(open("docs/requirements/core_spec.yml", encoding="utf-8")) or {}
gate = SSOT.get("gate", {})
SMD_MAX = float(gate.get("smd_max", 0.10))
SMD_FEATS = [str(x) for x in gate.get("smd_features", [])]


def test_smd_guard_respects_ssot():
    p = Path("reports/compare/BASE_1/baseline/smd_metrics.csv")
    assert p.exists(), f"missing {p}"
    df = pd.read_csv(p)
    if df.empty:
        pytest.skip("smd_metrics.csv is empty — skipping SMD guard (data missing)")
    lowcols = [c.lower() for c in df.columns]
    if {"feature", "smd"}.issubset(set(lowcols)):
        smd = df.rename(columns=str.lower).set_index("feature")["smd"]
        present = [f for f in SMD_FEATS if f in smd.index]
        if not present:
            pytest.skip(f"SMD features {SMD_FEATS} not present; available={list(smd.index)[:10]}")
        for f in present:
            assert abs(float(smd.loc[f])) <= SMD_MAX + 1e-12, f"SMD({f})>{SMD_MAX}"
    else:
        present = [f for f in SMD_FEATS if f in df.columns]
        if not present:
            pytest.skip(
                f"SMD feature columns {SMD_FEATS} not found; available-cols={list(df.columns)}"
            )
        for f in present:
            assert df[f].astype(float).abs().max() <= SMD_MAX + 1e-12, f"SMD({f}) max>{SMD_MAX}"


def test_adopt_row_has_positive_lcb95():
    p = Path("reports/tables/adoption_decision.csv")
    assert p.exists(), f"missing {p}"
    df = pd.read_csv(p)
    flag = "adopt" if "adopt" in df.columns else ("adopted" if "adopted" in df.columns else None)
    assert flag is not None, "adopt/adopted col missing"
    adopted = df[df[flag]]
    if adopted.empty:
        pytest.skip("no adopted policy (all LCB<=0) — acceptable fail-closed state")
    assert "lcb95" in df.columns, "lcb95 col missing"
    assert adopted["lcb95"].min() > 0, "Adopted row must satisfy LCB95>0"
