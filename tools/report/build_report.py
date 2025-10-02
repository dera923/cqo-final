import pandas as pd, yaml, datetime as dt
from pathlib import Path

SPEC = Path("docs/requirements/core_spec.yml")
OUT_MD = Path("reports/REPORT.md")

def read_last_csv(path):
    p = Path(path)
    if not p.exists(): return None
    df = pd.read_csv(p)
    return (df.tail(1), df)

def md_table(df: pd.DataFrame, max_rows=20):
    df2 = df.copy()
    if len(df2) > max_rows: df2 = df2.tail(max_rows)
    return df2.to_markdown(index=False)

def main():
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    figs = Path(spec["runtime"]["out_figures"])
    tabs = Path(spec["runtime"]["out_tables"])

    sm_last, _  = read_last_csv(tabs/"summary_metrics.csv")
    est_last, _ = read_last_csv(tabs/"estimates.csv")
    kpi_last, _ = read_last_csv(tabs/"policy_kpi.csv")
    adopt_last, _ = read_last_csv(tabs/"adoption_decision.csv")

    md = []
    md += [f"# CQO × Long-Term Profit — Final Report",
           f"_Generated: {now}_", ""]

    md += ["## 1. Overview (SSOT)",
           f"- Spec: `{SPEC}`",
           f"- Horizon: {spec.get('horizon_days','NA')} days, Discount: {spec.get('discount','NA')}",
           f"- Adoption rule: **{spec['policy']['adoption_rule']}**", ""]

    md += ["## 2. Gate-IN",
           f"![gate_report]({figs/'gate_report.png'})" if (figs/'gate_report.png').exists() else "_(gate_report.png not found)_",
           "### Summary Metrics (last row)"]
    md += [md_table(sm_last) if sm_last is not None else "_summary_metrics.csv not found_", ""]

    md += ["## 3. Estimation (DR/EIF)"]
    md += [md_table(est_last) if est_last is not None else "_estimates.csv not found_", ""]

    md += ["## 4. Post-selection Adjustment (Gate-OUT)"]
    md += [md_table(kpi_last) if kpi_last is not None else "_policy_kpi.csv not found_", ""]

    md += ["## 5. Adoption Decision"]
    md += [md_table(adopt_last) if adopt_last is not None else "_adoption_decision.csv not found_", ""]

    md += ["## 6. Artifacts Index",
           f"- Figures dir: `{figs}`",
           f"- Tables dir: `{tabs}`", ""]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ Wrote {OUT_MD}")

if __name__ == "__main__":
    main()
