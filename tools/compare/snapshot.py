import argparse
import pathlib as P
import shutil
from datetime import datetime

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    sp = yaml.safe_load(open("docs/requirements/core_spec.yml"))
    out_dir = P.Path((sp.get("compare") or {}).get("out_dir") or "reports/compare/")
    tag_dir = out_dir / args.tag / "baseline"
    tag_dir.mkdir(parents=True, exist_ok=True)

    # 重要テーブルを保存
    src = P.Path("reports/tables")
    keep = [
        "summary_metrics.csv",
        "smd_metrics.csv",
        "estimates.csv",
        "policy_kpi.csv",
        "adoption_decision.csv",
    ]
    for k in keep:
        s = src / k
        if s.exists():
            shutil.copy2(s, tag_dir / k)

    print(f"SNAPSHOT OK: {tag_dir}")


if __name__ == "__main__":
    main()
