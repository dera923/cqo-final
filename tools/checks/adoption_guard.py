import pathlib as p
import sys

import pandas as pd

KPI = p.Path("reports/compare/BASE_1/baseline/policy_kpi.csv")
if not KPI.exists():
    print(f"[ERROR] missing: {KPI}", file=sys.stderr)
    sys.exit(2)

df = pd.read_csv(KPI)
need = "lcb95_adj"
if need not in df.columns:
    print(f"[ERROR] column '{need}' not found in {KPI}", file=sys.stderr)
    sys.exit(2)

viol = (df["lcb95_adj"] <= 0).sum()
total = len(df)
print(f"[adoption_guard] lcb95_adj<=0: {viol}/{total}")
# Fail-Closed: 1件でも LCB95_adj<=0 が「採択対象」になっていれば失敗。
# ここでは全ポリシーを監査対象とし、将来「採択候補フラグ」が加わればそこに限定する。
sys.exit(1 if viol > 0 else 0)
