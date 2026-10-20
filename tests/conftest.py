import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, data: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "fsap.cli", "--data", str(data), *args],
        check=True,
        cwd=ROOT,
        env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )


@pytest.fixture(scope="session")
def small_world(tmp_path_factory) -> Path:
    """A sixteen-month world taken through the whole pipeline once; shared by the end-to-end tests."""
    data = tmp_path_factory.mktemp("world")
    run("simulate", "--days", "500", "--plan-weeks", "2", data=data)
    run("ingest", data=data)
    run("features", data=data)
    run("train", "--rounds", "150", "--folds", "3", data=data)
    run("plan", data=data)
    yield data
    shutil.rmtree(data, ignore_errors=True)


def has_java() -> bool:
    return shutil.which("java") is not None
