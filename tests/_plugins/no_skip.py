from typing import Any

from _pytest.terminal import TerminalReporter  # type: ignore[import-not-found]


def pytest_terminal_summary(
    terminalreporter: "TerminalReporter", exitstatus: int, config: Any
) -> None:
    skipped = len(terminalreporter.getreports("skipped"))  # type: ignore[attr-defined]
    if skipped:
        terminalreporter.write_line(f"[fail-close] skipped={skipped}", red=True)
        raise SystemExit(1)
