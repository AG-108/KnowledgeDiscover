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
    parser.addoption(
        "--require-sisso", action="store_true",
        help="Fail instead of skipping SISSO integration tests when native bindings are absent.",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("native_sisso"):
        from kd.model.kd_sisso import SISSOModel

        try:
            SISSOModel.check_available()
        except ImportError as exc:
            if item.config.getoption("--require-sisso"):
                pytest.fail(str(exc), pytrace=False)
            pytest.skip(str(exc))
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


@pytest.fixture
def lightweight_cli_environment(monkeypatch):
    """Skip installed-package hashing only for explicitly opted-in CLI tests."""
    from kd import benchmark_identity

    monkeypatch.setattr(benchmark_identity, "environment_fingerprint",
                        lambda: {"test_environment": "cli-configuration-only"})
