"""Shared fixtures: no RSI_GATE from the shell, and one copy of the downloaded benchmarks per session."""
import os
import shutil
from pathlib import Path

import pytest

from rsi import evals


def source(name) -> Path:
    var, default = evals.SOURCES[name]
    return Path(os.environ.get(var) or default)


@pytest.fixture(autouse=True)
def no_gate_setting(monkeypatch):
    monkeypatch.delenv("RSI_GATE", raising=False)  # the default is what the tests describe, whatever the shell holds


@pytest.fixture(scope="session")
def data_home(tmp_path_factory):
    """One copy of both benchmarks for the whole session (QRData unzips to ~300 MB)."""
    home = tmp_path_factory.mktemp("evals-home")
    for name in evals.SOURCES:
        if source(name).is_dir():
            evals.fetch(name, home)
    yield home
    shutil.rmtree(home, ignore_errors=True)  # pytest keeps its last three temp roots; this one is too big for that


@pytest.fixture
def dabench(data_home):
    if not source("dabench").is_dir():
        pytest.skip("DABench source folder is not present")
    return data_home


@pytest.fixture
def qrdata(data_home):
    if not source("qrdata").is_dir():
        pytest.skip("QRData source folder is not present")
    return data_home


@pytest.fixture
def dabench_home(dabench, tmp_path):
    """A fresh home (own harnesses/ and runs/) that shares the session's DABench copy."""
    (tmp_path / "evals" / "data").mkdir(parents=True)
    os.symlink(dabench / "evals" / "data" / "dabench", tmp_path / "evals" / "data" / "dabench")
    return tmp_path
