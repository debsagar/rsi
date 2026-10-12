"""RSI: a personal science assistant with a self-improving harness.

This module holds the two lookups every other module shares: where state
lives (`home`) and how a required setting is read (`env`).

Invariants:
- Stdlib only. `rsi.runner` is imported through this package when the harness
  runs as a plain subprocess, so nothing heavy may load here.
- A missing setting raises naming the variable. A value is never printed.
- `.env` is read only when `load_env` is called (the CLI does it once), so
  importing the package in tests can never pick up real keys.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Fill os.environ from a hand-parsed KEY=VALUE file. The environment wins."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def env(name: str) -> str:
    """A required setting. litellm reads provider keys from os.environ itself;
    this is for the names RSI owns."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (environment or {ENV_FILE})")
    return value


def home() -> Path:
    """Where runs/ and harnesses/ live. Read on every call so tests can move it."""
    return Path(os.environ.get("RSI_HOME") or ".").resolve()
