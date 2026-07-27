"""Liest Experimentergebnisse direkt von der Platte.

Anders als der Paper-Teil gibt es hier keine Datenbank, kein Chunking und
keine Embeddings: die JSON- und CSV-Dateien sind bereits strukturiert. Der
MCP-Server liest sie bei Bedarf und fasst sie zusammen, statt Rohdaten
durchzureichen.

Erwartete Ablage (Namen sind flexibel, siehe Konstanten unten):

    data/experiments/<projekt>/<run_id>/
        hparams.json     Hyperparameter
        results.json     Ergebnisse / Metriken
        folds.csv        optional, eine Zeile pro Fold

Konfigurierbar ueber die Umgebungsvariable RESEARCH_EXPERIMENTS.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from .db import PROJECT_ROOT

EXPERIMENTS_DIR = Path(
    os.environ.get("RESEARCH_EXPERIMENTS", PROJECT_ROOT / "data" / "experiments")
)

HPARAM_NAMES = ("hparams.json", "hyperparameters.json", "config.json", "params.json")
RESULT_NAMES = ("cv_summary.json", "results.json", "result.json", "metrics.json", "scores.json", "summary.json")

NON_METRIC_KEYS = ("fold", "epoch", "epochs", "epochs_run", "step", "index", "k")

def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {"_value": data}
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        return {"_error": f"{path.name} nicht lesbar: {exc}"}


def _first_existing(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def _stats(values: list[float]) -> dict[str, float]:
    """Mittelwert, Standardabweichung, Min und Max einer Zahlenreihe."""
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
    """Findet alle Felder, die eine Liste von Zahlen sind (z.B. fold_fid)."""
    out: dict[str, list[float]] = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 1 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in value
        ):
            out[key] = [float(x) for x in value]
    return out

def _run_dirs(projekt: str | None = None) -> list[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []

    dirs: list[Path] = []
    such_wurzel = EXPERIMENTS_DIR / projekt if projekt else EXPERIMENTS_DIR
    if not such_wurzel.exists():
        return []
    
    for pfad in such_wurzel.rglob("*"):
        if pfad.is_dir() and (
            _first_existing(pfad, HPARAM_NAMES) or _first_existing(pfad, RESULT_NAMES)
        ):
            dirs.append(pfad)
    return sorted(dirs)


def _projekt_von(run_dir: Path) -> str:
    rel = run_dir.resolve().relative_to(EXPERIMENTS_DIR.resolve())
    return rel.parts[0] if len(rel.parts) > 1 else "sonstiges"


def list_experiments(projekt: str | None = None) -> list[dict[str, Any]]:
    """Alle Runs mit Kerninfos - fuer den Ueberblick, ohne Details."""
    runs = []
    for run_dir in _run_dirs(projekt):
        results = _read_json(_first_existing(run_dir, RESULT_NAMES)) if _first_existing(run_dir, RESULT_NAMES) else {}
        hparams = _read_json(_first_existing(run_dir, HPARAM_NAMES)) if _first_existing(run_dir, HPARAM_NAMES) else {}
        results = results or {}
        hparams = hparams or {}

        runs.append({
            "run_id": run_dir.name,
            "projekt": _projekt_von(run_dir),
            "model": hparams.get("model"),
            "status": results.get("status", "unbekannt"),
            "timestamp": results.get("timestamp"),
            "has_folds": bool(_first_existing(run_dir, ("folds.csv",)) or _numeric_lists(results)),
        })
    return runs


def _find_run(run_id: str, projekt: str | None = None) -> Path | None:
    for run_dir in _run_dirs(projekt):
        if run_dir.name == run_id:
            return run_dir
    return None


def get_experiment(run_id: str, projekt: str | None = None) -> dict[str, Any]:
    """Hyperparameter und Ergebnisse eines einzelnen Runs."""
    run_dir = _find_run(run_id, projekt)
    if run_dir is None:
        return {"error": f"Kein Run '{run_id}' gefunden."}

    hparam_file = _first_existing(run_dir, HPARAM_NAMES)
    result_file = _first_existing(run_dir, RESULT_NAMES)

    return {
        "run_id": run_id,
        "projekt": _projekt_von(run_dir),
        "hparams": _read_json(hparam_file) if hparam_file else None,
        "results": _read_json(result_file) if result_file else None,
        "files": [p.name for p in sorted(run_dir.iterdir()) if p.is_file()],
    }


def get_fold_summary(run_id: str, projekt: str | None = None) -> dict[str, Any]:
    """Fasst k-fold-Ergebnisse zusammen: Mittelwert, Streuung, bester/schlechtester Fold.

    Drei akzeptierte Formen, in dieser Reihenfolge:
      1. folds.csv                - eine Zeile je Fold
      2. "per_fold": [ {..}, .. ] - eine Liste von Objekten je Fold (cv_summary.json)
      3. "fold_xy": [0.1, 0.2, ..] - eine Liste von Zahlen je Metrik
    """
    run_dir = _find_run(run_id, projekt)
    if run_dir is None:
        return {"error": f"Kein Run '{run_id}' gefunden."}

    csv_file = _first_existing(run_dir, ("folds.csv", "cv.csv", "cross_val.csv"))
    if csv_file:
        return _fold_summary_from_csv(run_id, csv_file)

    result_file = _first_existing(run_dir, RESULT_NAMES)
    results = _read_json(result_file) if result_file else None
    if not results:
        return {"run_id": run_id, "info": "Keine Ergebnisdatei gefunden."}

    per_fold = results.get("per_fold")
    if isinstance(per_fold, list) and per_fold and isinstance(per_fold[0], dict):
        spalten: dict[str, list[float]] = {}
        for eintrag in per_fold:
            for key, value in eintrag.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    spalten.setdefault(key, []).append(float(value))
        metriken = {
            name: _stats(werte)
            for name, werte in spalten.items()
            if name.lower() not in NON_METRIC_KEYS
        }
        return {
            "run_id": run_id,
            "source": f"{result_file.name} (per_fold)",
            "n_folds": len(per_fold),
            "metrics": metriken,
            "warnings": _stabilitaets_hinweise(metriken),
        }

    listen = _numeric_lists(results)
    if listen:
        metriken = {name: _stats(werte) for name, werte in listen.items()}
        return {
            "run_id": run_id,
            "source": result_file.name,
            "metrics": metriken,
            "n_folds": max(len(w) for w in listen.values()),
            "warnings": _stabilitaets_hinweise(metriken),
        }

    return {"run_id": run_id, "info": "Keine Fold-Daten gefunden (weder folds.csv, per_fold noch Zahlenlisten)."}


def _fold_summary_from_csv(run_id: str, csv_file: Path) -> dict[str, Any]:
    with csv_file.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {"run_id": run_id, "info": f"{csv_file.name} ist leer."}

    spalten: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                spalten.setdefault(key, []).append(float(value))
            except (ValueError, TypeError):
                pass  

    metriken = {
        name: _stats(werte)
        for name, werte in spalten.items()
        if name.lower() not in ("fold", "epoch", "step", "index", "k")
    }

    return {
        "run_id": run_id,
        "source": csv_file.name,
        "n_folds": len(rows),
        "metrics": metriken,
        "warnings": _stabilitaets_hinweise(metriken),
    }


def _stabilitaets_hinweise(metriken: dict[str, dict[str, float]]) -> list[str]:
    """Markiert auffaellig hohe Streuung ueber die Folds - oft ein Split-Problem."""
    hinweise = []
    for name, s in metriken.items():
        if s["mean"] and s["n"] > 1:
            rel = s["std"] / abs(s["mean"])
            if rel > 0.15:
                hinweise.append(
                    f"'{name}' streut stark ueber die Folds "
                    f"(std/mean = {rel:.0%}); moeglicherweise instabiles Training "
                    f"oder ein unguenstiger Split."
                )
    return hinweise


def compare_experiments(run_ids: list[str], projekt: str | None = None) -> dict[str, Any]:
    """Vergleicht mehrere Runs: welche Hyperparameter unterscheiden sich, wie die Metriken.

    Der eigentliche Nutzen: nur die *abweichenden* Hyperparameter werden gezeigt,
    nicht die komplette Config. So sieht man sofort, was den Unterschied macht.
    """
    experimente = []
    for rid in run_ids:
        exp = get_experiment(rid, projekt)
        if "error" not in exp:
            experimente.append(exp)

    if len(experimente) < 2:
        return {"error": "Mindestens zwei gueltige Runs noetig zum Vergleichen."}

    alle_hparams = [e.get("hparams") or {} for e in experimente]
    alle_keys = set().union(*(h.keys() for h in alle_hparams))

    unterschiede = {}
    gemeinsam = {}
    for key in sorted(alle_keys):
        werte = [h.get(key) for h in alle_hparams]
        if len(set(map(str, werte))) > 1:
            unterschiede[key] = {rid: w for rid, w in zip(run_ids, werte)}
        else:
            gemeinsam[key] = werte[0]

    metrik_vergleich: dict[str, dict[str, Any]] = {}
    for e in experimente:
        res = e.get("results") or {}
        for key, value in res.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrik_vergleich.setdefault(key, {})[e["run_id"]] = value

    return {
        "runs": [e["run_id"] for e in experimente],
        "unterschiedliche_hparams": unterschiede,
        "gemeinsame_hparams": gemeinsam,
        "metriken": metrik_vergleich,
    }


def _flatten_metrics(run_dir: Path) -> dict[str, Any]:
    """Zieht die zusammengefassten Zahlen eines Laufs auf eine Ebene.

    Skalare Metriken (final_fid, val_accuracy, ...) direkt, k-fold-Metriken
    als mean/std. So bekommt das Modell pro Lauf eine flache Zeile statt
    verschachtelter Objekte.
    """
    result_file = _first_existing(run_dir, RESULT_NAMES)
    results = (_read_json(result_file) or {}) if result_file else {}

    flat: dict[str, Any] = {}
    listen = _numeric_lists(results)
    for key, value in results.items():
        if key in listen:
            continue
        if key.lower() in NON_METRIC_KEYS or key.lower().startswith("cv_"):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = value  

    fold = get_fold_summary(run_dir.name, _projekt_von(run_dir))
    warnings = fold.get("warnings", [])
    for name, s in fold.get("metrics", {}).items():
        if f"{name}_mean" not in flat and f"mean_{name}" not in flat:
            flat[f"{name}_mean"] = s["mean"]
        if f"{name}_std" not in flat and f"std_{name}" not in flat:
            flat[f"{name}_std"] = s["std"]

    return {"metrics": flat, "n_folds": fold.get("n_folds"), "warnings": warnings}


def _correlations(runs: list[dict], hparam_keys: set[str], metric_key: str) -> list[dict]:
    """Pearson-Korrelation zwischen jedem numerischen Hyperparameter und einer Metrik.

    Rein deterministisch. Gibt dem Modell Anhaltspunkte, welche Parameter
    ueberhaupt mit dem Ergebnis zusammenhaengen - die Deutung bleibt beim Modell.
    """
    ergebnisse = []
    for hp in sorted(hparam_keys):
        paare = [
            (r["hparams"][hp], r["metrics"][metric_key])
            for r in runs
            if isinstance(r["hparams"].get(hp), (int, float)) and not isinstance(r["hparams"].get(hp), bool)
            and metric_key in r["metrics"]
        ]
        if len(paare) < 3:
            continue
        xs, ys = zip(*paare)
        if len(set(xs)) < 2:
            continue
        r = _pearson(list(xs), list(ys))
        if r is not None and abs(r) >= 0.5:
            ergebnisse.append({
                "hyperparameter": hp,
                "metrik": metric_key,
                "pearson_r": round(r, 3),
                "richtung": "hoeher -> groesser" if r > 0 else "hoeher -> kleiner",
                "n": len(paare),
            })
    return sorted(ergebnisse, key=lambda e: -abs(e["pearson_r"]))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def summarize_project(projekt: str, metric: str | None = None) -> dict[str, Any]:
    """Fasst ALLE Laeufe eines Projekts in einem Objekt zusammen - fuer die Gesamtanalyse.

    Enthaelt pro Lauf die flachen Hyperparameter und zusammengefassten Metriken,
    dazu projektweite Hinweise: welche Hyperparameter variiert wurden, Korrelationen
    zu jeder Metrik und die auffaelligsten Laeufe. Die Deutung ueberlaesst das Tool
    bewusst dem Modell.

    Args:
        projekt: Name des Projekts.
        metric: Optional die Metrik, die im Fokus stehen soll (z.B. "std_val_dbm_mse").
                Beeinflusst nur die Sortierung/Anzeige - korreliert wird immer alles.
    """
    run_dirs = _run_dirs(projekt)
    if not run_dirs:
        return {"projekt": projekt, "info": "Keine Laeufe gefunden."}

    runs: list[dict] = []
    for run_dir in run_dirs:
        hparam_file = _first_existing(run_dir, HPARAM_NAMES)
        hparams = (_read_json(hparam_file) or {}) if hparam_file else {}
        flat = _flatten_metrics(run_dir)
        runs.append({
            "run_id": run_dir.name,
            "hparams": hparams,
            "metrics": flat["metrics"],
            "n_folds": flat["n_folds"],
            "warnings": flat["warnings"],
        })

    numerische_hp: set[str] = set()
    variiert: dict[str, list[Any]] = {}
    konstant: dict[str, Any] = {}
    alle_hp = set().union(*(r["hparams"].keys() for r in runs)) if runs else set()
    for hp in sorted(alle_hp):
        werte = [r["hparams"].get(hp) for r in runs]
        eindeutig = {str(w) for w in werte}
        if len(eindeutig) > 1:
            variiert[hp] = sorted(eindeutig)
            if all(isinstance(r["hparams"].get(hp), (int, float)) and not isinstance(r["hparams"].get(hp), bool)
                   for r in runs if hp in r["hparams"]):
                numerische_hp.add(hp)
        elif werte:
            konstant[hp] = werte[0]

    metrik_zaehler: dict[str, int] = {}
    for r in runs:
        for mk in r["metrics"]:
            metrik_zaehler[mk] = metrik_zaehler.get(mk, 0) + 1
    alle_metriken = sorted(metrik_zaehler)

    korrelierbar = [m for m, n in metrik_zaehler.items() if n >= 3]

    if metric and metric in metrik_zaehler:
        haupt_metrik = metric
    else:
        kandidaten = [m for m in metrik_zaehler if m.endswith("_mean")] or alle_metriken
        haupt_metrik = max(kandidaten, key=lambda m: metrik_zaehler[m]) if kandidaten else None

    korrelationen: dict[str, list[dict]] = {}
    for mk in sorted(korrelierbar, key=lambda m: (m != haupt_metrik, m)):
        treffer = _correlations(runs, numerische_hp, mk)
        if treffer:
            korrelationen[mk] = treffer

    fehlende_metrik = metric if (metric and metric not in metrik_zaehler) else None

    instabil = [r["run_id"] for r in runs if r["warnings"]]

    ergebnis = {
        "projekt": projekt,
        "n_runs": len(runs),
        "runs": runs,
        "variierte_hyperparameter": variiert,
        "konstante_hyperparameter": konstant,
        "verfuegbare_metriken": alle_metriken,
        "haupt_metrik": haupt_metrik,
        "korrelationen": korrelationen,
        "instabile_laeufe": instabil,
    }
    if len(runs) < 3:
        ergebnis["analyse_hinweis"] = (
            f"Nur {len(runs)} Lauf/Laeufe - fuer Korrelationen sind mindestens 3 noetig. "
            "Vergleiche die Laeufe stattdessen direkt (siehe 'runs') oder nutze "
            "compare_experiments fuer ein Diff der Hyperparameter."
        )
    elif not korrelationen:
        ergebnis["analyse_hinweis"] = (
            "Keine nennenswerten Korrelationen gefunden - entweder wurden keine "
            "numerischen Hyperparameter variiert, oder es gibt keinen klaren Zusammenhang."
        )
    else:
        ergebnis["hinweis"] = (
            "Korrelationen sind rein deskriptiv und beruhen oft auf wenigen Laeufen - "
            "kein Kausalnachweis. Bitte im Kontext der Hyperparameter deuten."
        )

    if fehlende_metrik:
        ergebnis["warnung"] = (
            f"Gewuenschte Metrik '{fehlende_metrik}' kommt in den Laeufen nicht vor. "
            f"Verfuegbar: {alle_metriken}."
        )
    return ergebnis