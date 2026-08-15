"""Read experiment results straight from disk.

Unlike the paper side there is no database, no chunking and no embeddings:
the JSON and CSV files are already structured. The MCP server reads them on
demand and summarizes them instead of passing raw data through.

Locations come from config.yaml: each project has an experiments folder with
one subfolder per run:

    <project's experiments folder>/<run_id>/
        hparams.json     hyperparameters
        results.json     results / metrics
        folds.csv        optional, one row per fold
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config

HPARAM_NAMES = ("hparams.json", "hyperparameters.json", "config.json", "params.json")
RESULT_NAMES = ("cv_summary.json", "results.json", "result.json", "metrics.json", "scores.json", "summary.json")

NON_METRIC_KEYS = ("fold", "epoch", "epochs", "epochs_run", "step", "index", "k")


def _experiment_folders(project: str | None = None) -> list[tuple[str, Path]]:
    """(project_name, experiments folder) from the config, optionally filtered."""
    cfg = load_config()
    pairs = cfg.folders_of_kind("experiments")
    if project:
        pairs = [(name, folder) for name, folder in pairs if name == project]
    return pairs


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {"_value": data}
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        return {"_error": f"{path.name} not readable: {exc}"}


def _first_existing(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def _stats(values: list[float]) -> dict[str, float]:
    """Mean, standard deviation, min and max of a number series."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(math.sqrt(variance), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
    }


def _numeric_lists(data: dict[str, Any]) -> dict[str, list[float]]:
    """Find all fields that are a list of numbers (e.g. fold_fid)."""
    out: dict[str, list[float]] = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 1 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in value
        ):
            out[key] = [float(x) for x in value]
    return out


def _run_dirs(project: str | None = None) -> list[tuple[str, Path]]:
    """All run folders as (project_name, path), across all configured projects."""
    result: list[tuple[str, Path]] = []
    for name, folder in _experiment_folders(project):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_dir() and (
                _first_existing(path, HPARAM_NAMES) or _first_existing(path, RESULT_NAMES)
            ):
                result.append((name, path))
    return result


def list_experiments(project: str | None = None) -> list[dict[str, Any]]:
    """All runs with core info - for the overview, without details."""
    runs = []
    for proj_name, run_dir in _run_dirs(project):
        results = _read_json(_first_existing(run_dir, RESULT_NAMES)) if _first_existing(run_dir, RESULT_NAMES) else {}
        hparams = _read_json(_first_existing(run_dir, HPARAM_NAMES)) if _first_existing(run_dir, HPARAM_NAMES) else {}
        results = results or {}
        hparams = hparams or {}

        runs.append({
            "run_id": run_dir.name,
            "project": proj_name,
            "model": hparams.get("model"),
            "status": results.get("status", "unknown"),
            "timestamp": results.get("timestamp"),
            "has_folds": bool(_first_existing(run_dir, ("folds.csv",)) or _numeric_lists(results)),
        })
    return runs


def _find_run(run_id: str, project: str | None = None) -> tuple[str, Path] | None:
    for proj_name, run_dir in _run_dirs(project):
        if run_dir.name == run_id:
            return proj_name, run_dir
    return None


def get_experiment(run_id: str, project: str | None = None) -> dict[str, Any]:
    """Hyperparameters and results of a single run."""
    found = _find_run(run_id, project)
    if found is None:
        return {"error": f"No run '{run_id}' found."}
    proj_name, run_dir = found

    hparam_file = _first_existing(run_dir, HPARAM_NAMES)
    result_file = _first_existing(run_dir, RESULT_NAMES)

    return {
        "run_id": run_id,
        "project": proj_name,
        "hparams": _read_json(hparam_file) if hparam_file else None,
        "results": _read_json(result_file) if result_file else None,
        "files": [p.name for p in sorted(run_dir.iterdir()) if p.is_file()],
    }


def get_fold_summary(run_id: str, project: str | None = None) -> dict[str, Any]:
    """Summarize k-fold results: mean, spread, best/worst fold.

    Three accepted forms, in this order:
      1. folds.csv                - one row per fold
      2. "per_fold": [ {..}, .. ] - a list of objects per fold (cv_summary.json)
      3. "fold_xy": [0.1, 0.2, ..] - a list of numbers per metric
    """
    found = _find_run(run_id, project)
    if found is None:
        return {"error": f"No run '{run_id}' found."}
    _, run_dir = found

    csv_file = _first_existing(run_dir, ("folds.csv", "cv.csv", "cross_val.csv"))
    if csv_file:
        return _fold_summary_from_csv(run_id, csv_file)

    result_file = _first_existing(run_dir, RESULT_NAMES)
    results = _read_json(result_file) if result_file else None
    if not results:
        return {"run_id": run_id, "info": "No results file found."}

    per_fold = results.get("per_fold")
    if isinstance(per_fold, list) and per_fold and isinstance(per_fold[0], dict):
        columns: dict[str, list[float]] = {}
        for entry in per_fold:
            for key, value in entry.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    columns.setdefault(key, []).append(float(value))
        metrics = {
            name: _stats(vals)
            for name, vals in columns.items()
            if name.lower() not in NON_METRIC_KEYS
        }
        return {
            "run_id": run_id,
            "source": f"{result_file.name} (per_fold)",
            "n_folds": len(per_fold),
            "metrics": metrics,
            "warnings": _stability_warnings(metrics),
        }

    lists = _numeric_lists(results)
    if lists:
        metrics = {name: _stats(vals) for name, vals in lists.items()}
        return {
            "run_id": run_id,
            "source": result_file.name,
            "metrics": metrics,
            "n_folds": max(len(v) for v in lists.values()),
            "warnings": _stability_warnings(metrics),
        }

    return {"run_id": run_id, "info": "No fold data found (neither folds.csv, per_fold nor numeric lists)."}


def _fold_summary_from_csv(run_id: str, csv_file: Path) -> dict[str, Any]:
    with csv_file.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {"run_id": run_id, "info": f"{csv_file.name} is empty."}

    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                columns.setdefault(key, []).append(float(value))
            except (ValueError, TypeError):
                pass

    metrics = {
        name: _stats(vals)
        for name, vals in columns.items()
        if name.lower() not in ("fold", "epoch", "step", "index", "k")
    }

    return {
        "run_id": run_id,
        "source": csv_file.name,
        "n_folds": len(rows),
        "metrics": metrics,
        "warnings": _stability_warnings(metrics),
    }


def _stability_warnings(metrics: dict[str, dict[str, float]]) -> list[str]:
    """Flag unusually high spread across folds - often a split problem."""
    warnings = []
    for name, s in metrics.items():
        if s["mean"] and s["n"] > 1:
            rel = s["std"] / abs(s["mean"])
            if rel > 0.15:
                warnings.append(
                    f"'{name}' varies strongly across folds "
                    f"(std/mean = {rel:.0%}); possibly unstable training "
                    f"or an unfavorable split."
                )
    return warnings


def compare_experiments(run_ids: list[str], project: str | None = None) -> dict[str, Any]:
    """Compare several runs: which hyperparameters differ, and the metrics.

    The real value: only the *differing* hyperparameters are shown, not the
    whole config. So one immediately sees what makes the difference.
    """
    experiments = []
    for rid in run_ids:
        exp = get_experiment(rid, project)
        if "error" not in exp:
            experiments.append(exp)

    if len(experiments) < 2:
        return {"error": "At least two valid runs are needed to compare."}

    all_hparams = [e.get("hparams") or {} for e in experiments]
    all_keys = set().union(*(h.keys() for h in all_hparams))

    differing = {}
    shared = {}
    for key in sorted(all_keys):
        values = [h.get(key) for h in all_hparams]
        if len(set(map(str, values))) > 1:
            differing[key] = {rid: v for rid, v in zip(run_ids, values)}
        else:
            shared[key] = values[0]

    metric_comparison: dict[str, dict[str, Any]] = {}
    for e in experiments:
        res = e.get("results") or {}
        for key, value in res.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_comparison.setdefault(key, {})[e["run_id"]] = value

    return {
        "runs": [e["run_id"] for e in experiments],
        "differing_hparams": differing,
        "shared_hparams": shared,
        "metrics": metric_comparison,
    }


def _flatten_metrics(proj_name: str, run_dir: Path) -> dict[str, Any]:
    """Pull a run's summary numbers onto one level."""
    result_file = _first_existing(run_dir, RESULT_NAMES)
    results = (_read_json(result_file) or {}) if result_file else {}

    flat: dict[str, Any] = {}
    lists = _numeric_lists(results)
    for key, value in results.items():
        if key in lists:
            continue
        if key.lower() in NON_METRIC_KEYS or key.lower().startswith("cv_"):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = value

    fold = get_fold_summary(run_dir.name, proj_name)
    warnings = fold.get("warnings", [])
    for name, s in fold.get("metrics", {}).items():
        if f"{name}_mean" not in flat and f"mean_{name}" not in flat:
            flat[f"{name}_mean"] = s["mean"]
        if f"{name}_std" not in flat and f"std_{name}" not in flat:
            flat[f"{name}_std"] = s["std"]

    return {"metrics": flat, "n_folds": fold.get("n_folds"), "warnings": warnings}


def _correlations(runs: list[dict], hparam_keys: set[str], metric_key: str) -> list[dict]:
    """Pearson correlation between each numeric hyperparameter and a metric.

    Purely deterministic. Gives the model hints about which parameters relate to
    the result at all - the interpretation stays with the model.
    """
    results = []
    for hp in sorted(hparam_keys):
        pairs = [
            (r["hparams"][hp], r["metrics"][metric_key])
            for r in runs
            if isinstance(r["hparams"].get(hp), (int, float)) and not isinstance(r["hparams"].get(hp), bool)
            and metric_key in r["metrics"]
        ]
        if len(pairs) < 3:
            continue
        xs, ys = zip(*pairs)
        if len(set(xs)) < 2:
            continue
        r = _pearson(list(xs), list(ys))
        if r is not None and abs(r) >= 0.5:
            results.append({
                "hyperparameter": hp,
                "metric": metric_key,
                "pearson_r": round(r, 3),
                "direction": "higher -> larger" if r > 0 else "higher -> smaller",
                "n": len(pairs),
            })
    return sorted(results, key=lambda e: -abs(e["pearson_r"]))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def summarize_project(project: str, metric: str | None = None) -> dict[str, Any]:
    """Summarize ALL runs of a project in one object - for the overall analysis.

    Contains per run the flat hyperparameters and summarized metrics, plus
    project-wide hints: which hyperparameters were varied, correlations to every
    metric, and the most notable runs. The interpretation is left to the model.

    Args:
        project: name of the project.
        metric: optional metric to focus on (e.g. "std_val_dbm_mse"). Only affects
                ordering/display - correlation always runs over everything.
    """
    run_dirs = _run_dirs(project)
    if not run_dirs:
        return {"project": project, "info": "No runs found."}

    runs: list[dict] = []
    for proj_name, run_dir in run_dirs:
        hparam_file = _first_existing(run_dir, HPARAM_NAMES)
        hparams = (_read_json(hparam_file) or {}) if hparam_file else {}
        flat = _flatten_metrics(proj_name, run_dir)
        runs.append({
            "run_id": run_dir.name,
            "hparams": hparams,
            "metrics": flat["metrics"],
            "n_folds": flat["n_folds"],
            "warnings": flat["warnings"],
        })

    numeric_hp: set[str] = set()
    varied: dict[str, list[Any]] = {}
    constant: dict[str, Any] = {}
    all_hp = set().union(*(r["hparams"].keys() for r in runs)) if runs else set()
    for hp in sorted(all_hp):
        values = [r["hparams"].get(hp) for r in runs]
        unique = {str(w) for w in values}
        if len(unique) > 1:
            varied[hp] = sorted(unique)
            if all(isinstance(r["hparams"].get(hp), (int, float)) and not isinstance(r["hparams"].get(hp), bool)
                   for r in runs if hp in r["hparams"]):
                numeric_hp.add(hp)
        elif values:
            constant[hp] = values[0]

    metric_counter: dict[str, int] = {}
    for r in runs:
        for mk in r["metrics"]:
            metric_counter[mk] = metric_counter.get(mk, 0) + 1
    all_metrics = sorted(metric_counter)

    correlatable = [m for m, n in metric_counter.items() if n >= 3]

    if metric and metric in metric_counter:
        main_metric = metric
    else:
        candidates = [m for m in metric_counter if m.endswith("_mean")] or all_metrics
        main_metric = max(candidates, key=lambda m: metric_counter[m]) if candidates else None

    correlations: dict[str, list[dict]] = {}
    for mk in sorted(correlatable, key=lambda m: (m != main_metric, m)):
        hits = _correlations(runs, numeric_hp, mk)
        if hits:
            correlations[mk] = hits

    missing_metric = metric if (metric and metric not in metric_counter) else None

    unstable = [r["run_id"] for r in runs if r["warnings"]]

    result = {
        "project": project,
        "n_runs": len(runs),
        "runs": runs,
        "varied_hyperparameters": varied,
        "constant_hyperparameters": constant,
        "available_metrics": all_metrics,
        "main_metric": main_metric,
        "correlations": correlations,
        "unstable_runs": unstable,
    }
    if len(runs) < 3:
        result["analysis_note"] = (
            f"Only {len(runs)} run(s) - correlations need at least 3. "
            "Compare the runs directly instead (see 'runs') or use "
            "compare_experiments for a hyperparameter diff."
        )
    elif not correlations:
        result["analysis_note"] = (
            "No notable correlations found - either no numeric hyperparameters "
            "were varied, or there is no clear relationship."
        )
    else:
        result["note"] = (
            "Correlations are purely descriptive and often based on few runs - "
            "not a causal proof. Please interpret in context of the hyperparameters."
        )

    if missing_metric:
        result["warning"] = (
            f"Requested metric '{missing_metric}' does not appear in the runs. "
            f"Available: {all_metrics}."
        )
    return result