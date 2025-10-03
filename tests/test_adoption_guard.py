import subprocess
import sys


def test_lcb95_positive_only():
    r = subprocess.run([sys.executable, "tools/checks/adoption_guard.py"])
    assert r.returncode == 0, "LCB95 guard failed. See adoption_guard output."
