"""Tests for research.files - the read_code security boundary.

The point of these tests: proving that "blocked" really means blocked. They set
up a temp workspace with one opt-in project and one that never opted in, then
check every path that must and must not be readable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import research.files as files
from research.config import Config, Project


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Two projects: 'open' opts into code reading, 'closed' does not."""
    open_root = tmp_path / "open"
    closed_root = tmp_path / "closed"
    (open_root / "src").mkdir(parents=True)
    (open_root / "secrets").mkdir()
    (closed_root / "src").mkdir(parents=True)

    (open_root / "src" / "model.py").write_text("def f(): pass", encoding="utf-8")
    (open_root / ".env").write_text("SECRET=1", encoding="utf-8")
    (open_root / "src" / "key.pem").write_text("-----BEGIN-----", encoding="utf-8")
    (open_root / "secrets" / "creds.py").write_text("pw = 1", encoding="utf-8")
    (closed_root / "src" / "private.py").write_text("def g(): pass", encoding="utf-8")

    def fake_config():
        return Config(
            database=tmp_path / "library.db",
            projects={
                "open": Project(name="open", root=open_root, papers=None,
                                experiments=None, thesis=None,
                                read_code_allowed=True,
                                read_code_exclude=["secrets/"]),
                "closed": Project(name="closed", root=closed_root, papers=None,
                                  experiments=None, thesis=None,
                                  read_code_allowed=False),
            },
        )

    monkeypatch.setattr(files, "load_config", fake_config)
    return {"open": open_root, "closed": closed_root}


CODE = (".py", ".pem", ".env", ".toml")


def test_allowed_project_code_is_readable(workspace):
    r = files.read_code_file(str(workspace["open"] / "src" / "model.py"), CODE)
    assert "text" in r


def test_env_is_blocked_even_when_allowed(workspace):
    r = files.read_code_file(str(workspace["open"] / ".env"), CODE)
    assert "error" in r


def test_key_file_is_blocked_even_when_allowed(workspace):
    r = files.read_code_file(str(workspace["open"] / "src" / "key.pem"), CODE)
    assert "error" in r


def test_excluded_folder_is_blocked(workspace):
    r = files.read_code_file(str(workspace["open"] / "secrets" / "creds.py"), CODE)
    assert "error" in r


def test_project_without_optin_is_blocked(workspace):
    r = files.read_code_file(str(workspace["closed"] / "src" / "private.py"), CODE)
    assert "error" in r


def test_absolute_escape_is_blocked(workspace):
    r = files.read_code_file("/etc/passwd", CODE + ("",))
    assert "error" in r


def test_dotdot_escape_is_blocked(workspace):
    escape = str(workspace["open"] / "src" / ".." / ".." / ".." / "etc" / "passwd")
    r = files.read_code_file(escape, CODE)
    assert "error" in r


def test_list_code_omits_secrets(workspace):
    listing = files.list_files(CODE)
    paths = [e.get("path", "") for e in listing]
    assert any("model.py" in p for p in paths)
    assert not any(".env" in p for p in paths)
    assert not any("creds" in p for p in paths)


def test_thesis_boundary_stays_open_for_closed_project(workspace):
    # A project without read_code_allowed must still allow thesis reading.
    thesis = workspace["closed"] / "chapter.tex"
    thesis.write_text(r"\section{X} text", encoding="utf-8")
    r = files.read_text_file(str(thesis), (".tex",))  # thesis path, no code rules
    assert "text" in r