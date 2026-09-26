"""Make optional external-data integration checks explicit in public checkouts."""

from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--require-external-data",
        action="store_true",
        help="Fail instead of skipping when a marked external dataset is absent.",
    )


def pytest_runtest_setup(item):
    for marker in item.iter_markers("external_data"):
        missing = [
            pattern
            for pattern in marker.args
            if not any(path.is_file() for path in _REPOSITORY_ROOT.glob(pattern))
        ]
        if missing:
            reason = "External data missing: " + ", ".join(missing) + "; see docs/data.md"
            if item.config.getoption("--require-external-data"):
                pytest.fail(reason, pytrace=False)
            pytest.skip(reason)
