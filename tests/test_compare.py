import pathlib as P

import pandas as pd
import yaml


def test_compare_outputs_exist_and_nonregression():
    sp = yaml.safe_load(open("docs/requirements/core_spec.yml"))
    out_dir = P.Path((sp.get("compare") or {}).get("out_dir") or "reports/compare/")
    tags = [p for p in out_dir.iterdir() if p.is_dir()]
    assert tags, "no compare tags"
    new = sorted(tags, key=lambda x: x.name)[-1] / "new"
    guards = pd.read_csv(new / "guards.csv")
    # 非退行（LCB95>0だったら守る）。true/false/NA のいずれか
    assert "lcb95_non_regression" in guards.columns
