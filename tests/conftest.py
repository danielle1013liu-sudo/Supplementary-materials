from pathlib import Path
import pytest

from src.hydrogel_immune_ml.publication_audit import forbid_fitting


@pytest.fixture(autouse=True)
def no_fitting():
    with forbid_fitting():
        yield


def pytest_addoption(parser):
    parser.addoption(
        "--replay-dir", type=Path, help="Directory produced by python -m src.pareto"
    )


@pytest.fixture(scope="session")
def replay_dir(request):
    path = request.config.getoption("--replay-dir")
    if path is None:
        pytest.skip(
            "Run numerical reproduction and supply --replay-dir for integration checks"
        )
    return path.resolve()
