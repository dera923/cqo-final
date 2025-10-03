import yaml


def test_enforce_steps_fixed():
    spec = yaml.safe_load(open("docs/requirements/core_spec.yml", encoding="utf-8")) or {}
    assert "pipeline" in spec, "SSOT違反: 'pipeline' が core_spec.yml にありません"
    assert "enforce_steps" in spec["pipeline"], "SSOT違反: pipeline.enforce_steps がありません"
    assert spec["pipeline"]["enforce_steps"] == [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ], f"SSOT違反: enforce_steps={spec['pipeline']['enforce_steps']}"
