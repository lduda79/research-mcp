"""Tests for research.config - path resolution from config.yaml."""
from __future__ import annotations

from pathlib import Path

import pytest

from research.config import load_config


def test_defaults_are_applied(write_config):
    cfg_path = write_config("""
        database: library.db
        defaults:
          papers: papers
          experiments: experiments
          thesis: text
        projects:
          masterarbeit:
            root: /tmp/Masterarbeit
    """)
    cfg = load_config(cfg_path)
    p = cfg.project("masterarbeit")
    assert p.papers.name == "papers"
    assert p.experiments.name == "experiments"
    assert p.thesis.name == "text"


def test_per_project_override(write_config):
    cfg_path = write_config("""
        database: library.db
        defaults:
          experiments: experiments
        projects:
          rf_slam:
            root: /tmp/RF-SLAM
            experiments: runs
    """)
    cfg = load_config(cfg_path)
    assert cfg.project("rf_slam").experiments.name == "runs"


def test_null_folder_becomes_none(write_config):
    cfg_path = write_config("""
        database: library.db
        projects:
          rf_slam:
            root: /tmp/RF-SLAM
            thesis: null
    """)
    cfg = load_config(cfg_path)
    assert cfg.project("rf_slam").thesis is None


def test_tilde_is_expanded(write_config):
    cfg_path = write_config("""
        database: library.db
        projects:
          m:
            root: ~/Desktop/Masterarbeit
    """)
    cfg = load_config(cfg_path)
    root = cfg.project("m").root
    assert "~" not in str(root)
    assert root.parts[-2:] == ("Desktop", "Masterarbeit")


def test_missing_root_raises(write_config):
    cfg_path = write_config("""
        database: library.db
        projects:
          broken:
            papers: papers
    """)
    with pytest.raises(ValueError, match="missing 'root'"):
        load_config(cfg_path)


def test_missing_file_falls_back_to_data_layout(tmp_path):
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert "default" in cfg.projects
    assert cfg.project("default").papers.name == "papers"
    assert cfg.database.name == "library.db"


def test_all_roots_includes_every_folder(write_config):
    cfg_path = write_config("""
        database: library.db
        defaults:
          papers: papers
          experiments: experiments
          thesis: text
        projects:
          m:
            root: /tmp/M
    """)
    cfg = load_config(cfg_path)
    names = {r.name for r in cfg.all_roots()}
    assert "papers" in names
    assert "experiments" in names
    assert "text" in names


def test_folders_of_kind_skips_null(write_config):
    cfg_path = write_config("""
        database: library.db
        projects:
          a:
            root: /tmp/A
          b:
            root: /tmp/B
            thesis: null
    """)
    cfg = load_config(cfg_path)
    thesis_projects = [name for name, _ in cfg.folders_of_kind("thesis")]
    assert "a" in thesis_projects
    assert "b" not in thesis_projects