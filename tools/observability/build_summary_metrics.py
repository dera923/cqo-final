import argparse
import sys
from pathlib import Path

import pandas as pd

REQ = ["feature", "n1", "n0", "smd", "m1", "m0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BASE_1")
    ap.add_argument("--infile", default=None)
    ap.add_argument("--out", default="reports/tables/summary_metrics.csv")
    args = ap.parse_args()

    # 入力候補: compare/<BASE>/baseline/smd_metrics.csv を第一候補に
    if args.infile:
        src = Path(args.infile)
    else:
        src = Path(f"reports/compare/{args.base}/baseline/smd_metrics.csv")
        if not src.exists():
            # フォールバック: 生成済みを探索
            cands = sorted(Path("reports/compare").glob("*/baseline/smd_metrics.csv"))
            if cands:
                src = cands[0]

    if not src.exists():
        raise SystemExit(f"[err] missing input smd_metrics: {src}")

    df = pd.read_csv(src)
    # 列名の正規化（小文字想定）
    df.columns = [c.strip() for c in df.columns]
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise SystemExit(
            f"[err] columns missing in {src}: {miss}; have={list(df.columns)}"
        )

    # 必要列だけ、順序を保証して書き出し
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[REQ].to_csv(out, index=False)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
