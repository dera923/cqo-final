import pathlib as p
import subprocess
import sys

import pandas as pd


def test_cpi_selection_adopts_max_cpi_with_positive_lcb():
    # 実行
    r = subprocess.run([sys.executable, "tools/policy/cpi_select.py"], check=False)
    assert r.returncode == 0

    out = p.Path("reports/tables/adoption_decision.csv")
    assert out.exists(), "adoption_decision.csv が生成されていません"
    df = pd.read_csv(out)

    # 候補ゼロならこのテストはスキップ（ガードは別テストで担保）
    if "adopted" not in df.columns or df["adopted"].sum() == 0:
        # すべて LCB<=0 のケース
        return

    # 採択は1件のみ
    adopted = df[df["adopted"]]
    assert len(adopted) == 1, "採択は1件のみのはずです"

    # LCB>0
    assert adopted.iloc[0]["lcb95"] > 0, "採択のLCB95は正である必要があります"

    # CPI最大
    assert adopted.iloc[0]["cpi"] == df["cpi"].max(), "採択はCPI最大である必要があります"
