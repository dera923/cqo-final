import pandas as pd


def test_adoption_rule_column_and_logic():
    df = pd.read_csv("reports/tables/adoption_decision.csv")
    assert "adopt" in df.columns
    assert set(df["adopt"].unique()).issubset({True, False})
