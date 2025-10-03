from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Sequence

STEPS: dict[str, list[str]] = {
    "gate": ["python", "tools/sanity/gate_sanity.py"],
    "smd": ["python", "tools/sanity/smd_check.py"],
    "dr": ["python", "tools/sanity/dr_numeric.py"],
    "select": ["python", "tools/sanity/selection_numeric.py"],
    "adopt": ["python", "tools/sanity/adopt_numeric.py"],
    "monitor": ["python", "tools/monitor/roll_monitor.py"],
    "report": ["python", "tools/report/build_report.py"],
    "tests": ["pytest", "-q"],
}


def run(cmd: Sequence[str]) -> int:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(list(cmd))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cqo", description="CQO pipeline CLI")
    sp = p.add_subparsers(dest="sub")
    sp.add_parser("run")
    for k in ["gate", "smd", "dr", "select", "adopt", "monitor", "report", "tests"]:
        sp.add_parser(k)
    args = p.parse_args(argv)

    if args.sub is None:
        p.print_help()
        return 1

    if args.sub == "run":
        if not os.path.exists("reports/tables/candidates.csv"):
            os.makedirs("reports/tables", exist_ok=True)
            with open("reports/tables/candidates.csv", "w") as f:
                f.write("policy,tau_hat,se\nbase,0.02,0.01\n")
        flow = ["gate", "smd", "dr", "select", "adopt", "monitor", "report", "tests"]
        rc = 0
        for s in flow:
            rc |= run(STEPS[s])
        return int(rc)
    else:
        return run(STEPS[args.sub])


if __name__ == "__main__":
    raise SystemExit(main())
