from __future__ import annotations

import hashlib
import pathlib as P

import pandas as pd
import yaml


# --------- helper ----------
def card(title: str, html: str) -> str:
    return f"<section><h3>{title}</h3>{html}</section>"


def table_from_csv(path: P.Path, n: int = 20) -> str:
    if not path.exists():
        return "<p>NA</p>"
    df = pd.read_csv(path)
    if df.empty:
        return "<p>NA</p>"
    return df.head(n).to_html(index=False)


# --------- compare (safe minimal) ----------
def latest_compare() -> str:
    """
    最新タグの reports/compare/<TAG>/summary.csv があれば簡易表を返す。
    無ければ <p>NA</p> にフォールバック（安全実装）。
    """
    try:
        sp = yaml.safe_load(open("docs/requirements/core_spec.yml"))
        out_dir = P.Path((sp.get("compare") or {}).get("out_dir") or "reports/compare/")
        tags = [p for p in out_dir.iterdir() if p.is_dir()]
        if not tags:
            return "<p>NA</p>"
        new = sorted(tags, key=lambda x: x.name)[-1] / "summary.csv"
        if not new.exists():
            return "<p>NA</p>"
        df = pd.read_csv(new)
        if df.empty:
            return "<p>NA</p>"
        row = df.iloc[0].to_dict()
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in row.items())
        return f"<table>{rows}</table>"
    except Exception:
        return "<p>NA</p>"


# --------- main ----------
def main() -> None:
    r = P.Path("reports")
    r.mkdir(exist_ok=True)
    tbl = r / "tables"

    # 主要カード
    summary = table_from_csv(tbl / "summary_metrics.csv")
    smd = table_from_csv(tbl / "smd_metrics.csv")
    est = table_from_csv(tbl / "estimates.csv")
    adopt = table_from_csv(tbl / "adoption_decision.csv")

    comp = latest_compare()

    # ページ構成（シンプル）
    body = []
    body.append(card("A. Summary", summary))
    body.append(card("B. SMD", smd))
    body.append(card("C. Estimates", est))
    body.append(card("D. Adoption", adopt))
    body.append(card("F. Comparison (latest)", comp))

    summary_fingerprint = hashlib.sha256(("".join(body)).encode()).hexdigest()[:16]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>CQO Report</title>
<style>body{{font-family:system-ui, sans-serif;}} section{{margin:16px 0}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px}}</style>
</head><body>
<h2>CQO – Report</h2>
<p>Generated at: {{}}</p>
<pre>{summary_fingerprint}</pre>
{''.join(body)}
<footer><p>Generated at: build_report.py</p></footer>
</body></html>"""

    (r / "index.html").write_text(html, encoding="utf-8")
    print("wrote reports/index.html")


if __name__ == "__main__":
    main()
