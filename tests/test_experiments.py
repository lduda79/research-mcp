"""Tests for research.experiments - fold statistics, correlations, aggregation.

These patch load_config so the experiment folder points at a temp directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import research.experiments as ex
from research.config import Config, Project


@pytest.fixture
def experiment_workspace(tmp_path, monkeypatch):
    """Create a temp project with runs and point load_config at it."""
    exp_root = tmp_path / "Tests"
    exp_root.mkdir()

    def make_run(name: str, hparams: dict, results: dict):
        d = exp_root / name
        d.mkdir()
        (d / "hparams.json").write_text(json.dumps(hparams), encoding="utf-8")
        (d / "results.json").write_text(json.dumps(results), encoding="utf-8")

    def fake_config():
        proj = Project(name="m", root=tmp_path, papers=None,
                       experiments=exp_root, thesis=None)
        return Config(database=tmp_path / "library.db", projects={"m": proj})

    monkeypatch.setattr(ex, "load_config", fake_config)
    return make_run


def test_stats_mean_and_std():
    s = ex._stats([0.4, 0.42, 0.38, 0.44, 0.41])
    assert s["mean"] == pytest.approx(0.41, abs=0.01)
    assert s["n"] == 5
    assert s["min"] == 0.38
    assert s["max"] == 0.44


def test_pearson_perfect_positive():
    r = ex._pearson([1, 2, 3], [2, 4, 6])
    assert r == pytest.approx(1.0)


def test_pearson_perfect_negative():
    r = ex._pearson([1, 2, 3], [6, 4, 2])
    assert r == pytest.approx(-1.0)


def test_pearson_constant_returns_none():
    assert ex._pearson([1, 1, 1], [2, 4, 6]) is None


def test_stability_warning_fires_on_high_spread():
    metrics = {"loss": {"mean": 1.0, "std": 0.5, "n": 5, "min": 0.5, "max": 1.5}}
    warnings = ex._stability_warnings(metrics)
    assert len(warnings) == 1
    assert "loss" in warnings[0]


def test_stability_warning_silent_on_low_spread():
    metrics = {"loss": {"mean": 1.0, "std": 0.01, "n": 5, "min": 0.99, "max": 1.01}}
    assert ex._stability_warnings(metrics) == []


def test_list_experiments_finds_runs(experiment_workspace):
    experiment_workspace("run_a", {"model": "MLP", "lr": 0.001}, {"status": "completed"})
    experiment_workspace("run_b", {"model": "CNN", "lr": 0.01}, {"status": "completed"})
    runs = ex.list_experiments("m")
    ids = {r["run_id"] for r in runs}
    assert ids == {"run_a", "run_b"}
    assert all(r["project"] == "m" for r in runs)


def test_get_experiment_returns_hparams(experiment_workspace):
    experiment_workspace("run_a", {"model": "MLP", "lr": 0.001}, {"status": "done"})
    exp = ex.get_experiment("run_a", "m")
    assert exp["hparams"]["lr"] == 0.001
    assert exp["project"] == "m"


def test_get_experiment_missing_run(experiment_workspace):
    assert "error" in ex.get_experiment("nope", "m")


def test_fold_summary_from_per_fold(experiment_workspace):
    experiment_workspace("cv", {"model": "MLP"},
                         {"per_fold": [{"fold": 1, "val_mse": 0.4},
                                       {"fold": 2, "val_mse": 0.42},
                                       {"fold": 3, "val_mse": 0.38}]})
    fold = ex.get_fold_summary("cv", "m")
    assert fold["n_folds"] == 3
    assert "val_mse" in fold["metrics"]
    assert "fold" not in fold["metrics"]


def test_compare_experiments_shows_only_differences(experiment_workspace):
    experiment_workspace("a", {"model": "MLP", "lr": 0.001, "dim": 128}, {"acc": 0.9})
    experiment_workspace("b", {"model": "MLP", "lr": 0.010, "dim": 128}, {"acc": 0.8})
    cmp = ex.compare_experiments(["a", "b"], "m")
    assert "lr" in cmp["differing_hparams"]
    assert "dim" in cmp["shared_hparams"]
    assert "model" in cmp["shared_hparams"]


def test_summarize_project_correlation(experiment_workspace):
    experiment_workspace("a", {"model": "MLP", "lr": 0.001}, {"accuracy": 0.9})
    experiment_workspace("b", {"model": "MLP", "lr": 0.005}, {"accuracy": 0.7})
    experiment_workspace("c", {"model": "MLP", "lr": 0.010}, {"accuracy": 0.5})
    summary = ex.summarize_project("m")
    assert summary["n_runs"] == 3
    assert "accuracy" in summary["available_metrics"]
    assert "accuracy" in summary["correlations"]


def test_summarize_project_too_few_runs_notes_it(experiment_workspace):
    experiment_workspace("a", {"model": "MLP", "lr": 0.001}, {"accuracy": 0.9})
    summary = ex.summarize_project("m")
    assert "analysis_note" in summary
    assert summary["n_runs"] == 1