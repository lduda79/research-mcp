"""Shared pytest fixtures."""
from __future__ import annotations
import textwrap
from pathlib import Path
import pytest


@pytest.fixture
def write_config(tmp_path: Path):
    def _write(body: str) -> Path:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(body), encoding="utf-8")
        return cfg
    return _write