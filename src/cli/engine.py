import argparse
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from src.B_dr_estimator.dr_estimator import run as run_B
from src.C_gate_out.selective import run as run_C
from src.D_policy.adoption import decide as run_D


def load_spec(p):
    return yaml.safe_load(open(p, encoding="utf-8"))


def cmd_validate(spec):
    csv = spec["data_contract"]["csv_path"]
    df = pd.read_csv(csv, nrows=5)
    need = {"y", "t", "e", "mu0", "mu1", "weight", "X0", "X1", "X2"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[contract] missing cols: {missing}")
    print(f"✓ contract ok: {csv}")


def cmd_gate(spec):
    # A: Gate-IN は cqo-viz を呼び出す想定（導入済みの場合）
    csv = spec["data_contract"]["csv_path"]
    feats = ",".join(spec["gates"]["smd_features"])
    try:
        subprocess.run(["cqo-viz", "gate-report", "--csv", csv, "--features", feats], check=True)
    except FileNotFoundError:
        raise SystemExit(
            "cqo-viz が見つかりません。external/cqo-viz を導入し、`pip install -e external/cqo-viz` を実行してください。"
        )


def cmd_estimate(spec):
    csv = spec["data_contract"]["csv_path"]
    out = Path(spec["runtime"]["out_tables"]) / "estimates.csv"
    run_B(csv, str(out))
    print("✓ estimates.csv written")


def cmd_selective(spec):
    est = pd.read_csv(Path(spec["runtime"]["out_tables"]) / "estimates.csv")
    est.insert(0, "policy", "default")  # 単一ポリシー例
    tmp = Path(spec["runtime"]["out_tables"]) / "candidates.csv"
    est[["policy", "tau_hat", "se"]].to_csv(tmp, index=False)
    out = Path(spec["runtime"]["out_tables"]) / "policy_kpi.csv"
    run_C(str(tmp), str(out))
    print("✓ policy_kpi.csv written")


def cmd_adopt(spec):
    kpi = Path(spec["runtime"]["out_tables"]) / "policy_kpi.csv"
    out = Path(spec["runtime"]["out_tables"]) / "adoption_decision.csv"
    run_D(str(kpi), str(out))
    print("✓ adoption_decision.csv written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="docs/requirements/core_spec.yml")
    ap.add_argument("fn", choices=["validate", "gate", "estimate", "selective", "adopt", "all"])
    args = ap.parse_args()
    spec = load_spec(args.spec)
    if args.fn in ["validate"]:
        cmd_validate(spec)
    if args.fn in ["gate"]:
        cmd_gate(spec)
    if args.fn in ["estimate"]:
        cmd_estimate(spec)
    if args.fn in ["selective"]:
        cmd_selective(spec)
    if args.fn in ["adopt"]:
        cmd_adopt(spec)
    if args.fn == "all":
        cmd_validate(spec)
        cmd_gate(spec)
        cmd_estimate(spec)
        cmd_selective(spec)
        cmd_adopt(spec)


if __name__ == "__main__":
    main()
