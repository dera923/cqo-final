import yaml

spec = yaml.safe_load(open("docs/requirements/core_spec.yml", encoding="utf-8"))
exp = [3, 4, 5, 6, 7, 8, 9, 10, 11]
got = spec.get("pipeline", {}).get("enforce_steps")
assert got == exp, f"SSOT違反: enforce_steps={got}, expected={exp}"
print("SSOT OK: enforce_steps=[3..11]")
