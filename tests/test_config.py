"""Tests for research.config - read_code opt-in fields and code_roots."""
from __future__ import annotations

from research.config import load_config


def test_read_code_allowed_defaults_to_false(write_config):
    cfg_path = write_config("""
        database: library.db
        projects:
          m:
            root: /tmp/M
    """)
    cfg = load_config(cfg_path)
    assert cfg.project("m").read_code_allowed is False
    assert cfg.project("m").read_code_exclude == []


def test_read_code_allowed_is_read(write_config):
    cfg_path = write_config("""
        database: library.db
        projects:
          m:
            root: /tmp/M
            read_code_allowed: true
            read_code_exclude:
              - .env
              - secrets/
    """)
    cfg = load_config(cfg_path)
    assert cfg.project("m").read_code_allowed is True
    assert cfg.project("m").read_code_exclude == [".env", "secrets/"]


def test_code_roots_only_includes_optin(write_config):
    cfg_path = write_config("""
        database: library.db
        projects:
          yes_code:
            root: /tmp/YES
            read_code_allowed: true
          no_code:
            root: /tmp/NO
    """)
    cfg = load_config(cfg_path)
    names = {r.name for r in cfg.code_roots()}
    assert "YES" in names
    assert "NO" not in names