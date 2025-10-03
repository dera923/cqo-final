import pathlib as P
import re
import sys

import yaml

spec = yaml.safe_load(open("docs/requirements/core_spec.yml"))
notes = P.Path(sys.argv[1]).read_text(encoding="utf-8")
blocks = re.findall(r"```yaml(.*?)```", notes, flags=re.S | re.M)
for b in blocks:
    delta = yaml.safe_load(b) or {}
    for k, v in delta.items():
        if isinstance(v, dict):
            spec.setdefault(k, {}).update(v)
        else:
            spec[k] = v
P.Path("docs/requirements/core_spec.yml").write_text(
    yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
)
print("SSOT updated from", sys.argv[1])
