import datetime as dt
import io
from pathlib import Path

import pandas as pd

root = Path("reports/compare/BASE_1")
bdir, ndir = root / "baseline", root / "new"


def r(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


html = io.StringIO()
html.write("<h2>BASE_1 – Baseline vs New</h2>")
html.write(f"<p>Generated: {dt.datetime.now().isoformat()}</p>")
# baseline
html.write("<section><h3>Baseline</h3>")
for n in [
    "summary_metrics.csv",
    "smd_metrics.csv",
    "estimates.csv",
    "policy_kpi.csv",
    "adoption_decision.csv",
]:
    p = bdir / n
    if p.exists():
        df = r(p)
        if df is not None:
            html.write(f"<h4>{n}</h4>" + df.head(50).to_html(index=False))
html.write("</section>")
# new
html.write("<section><h3>New</h3>")
for n in ["summary.csv", "guards.csv"]:
    p = ndir / n
    if p.exists():
        df = r(p)
        if df is not None:
            html.write(f"<h4>{n}</h4>" + df.head(50).to_html(index=False))
html.write("</section>")
(root / "compare_BASE_1.html").write_text(html.getvalue(), encoding="utf-8")
print("Wrote:", root / "compare_BASE_1.html")
