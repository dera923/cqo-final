import pathlib as p

import pandas as pd
import yaml

SSOT = yaml.safe_load(open("docs/requirements/core_spec.yml", encoding="utf-8"))


def test_tables_exist_and_have_required_columns():
    req = SSOT.get("observability", {})
    for t in req.get("required_tables", []):
        fp = p.Path(t)
        assert fp.exists(), f"missing table: {t}"
    cols = req.get("required_columns", {})
    for name, need in cols.items():
        fp = p.Path("reports/tables") / f"{name}.csv"
        assert fp.exists(), f"missing: {fp}"
        df = pd.read_csv(fp)
        miss = set(need) - set(df.columns)
        assert not miss, f"{fp} missing cols: {sorted(miss)}"
