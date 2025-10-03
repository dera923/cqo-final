import subprocess
import sys

from .main import main as _main


def main(argv=None):
    rc = subprocess.call([sys.executable, "tools/verification/validate_data.py"])
    if rc != 0:
        return rc
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
