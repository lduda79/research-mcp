"""MCP-Server fuer den eigenen Forschungs-Stack.

Wichtig: Bei stdio-Transport laeuft das MCP-Protokoll ueber stdout.
Niemals print() benutzen - Logging geht nach stderr.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from research.search import get_paper_text, list_papers, list_projekte, semantic_search
from research.experiments import (
    compare_experiments as _compare_experiments,
    get_experiment as _get_experiment,
    get_fold_summary as _get_fold_summary,
    list_experiments as _list_experiments,
    summarize_project as _summarize_project,
)

from research.files import read_text_file, list_files
from research.thesis import parse_thesis
 
CODE_SUFFIXES = (".py", ".toml", ".md", ".txt", ".cfg", ".ini")
THESIS_SUFFIXES = (".tex", ".md", ".markdown", ".txt")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("research-mcp")

mcp = FastMCP("research")


# ---------------------------------------------------------------------------
# Paper-Bibliothek
# ---------------------------------------------------------------------------

@mcp.tool()
def search_papers(
    query: str,
    limit: int = 5,
    projekt: str | None = None,
    bereich: str | None = None,
) -> list[dict]:
    """Durchsucht die lokale Paper-Bibliothek inhaltlich nach einem Thema.

    Findet Textstellen auch dann, wenn andere Begriffe verwendet werden als in
    der Suchanfrage. Gibt Textausschnitte mit Titel, Jahr und Seitenzahl zurueck.
    Benutze dieses Tool, um herauszufinden, was in den gelesenen Papern zu einem
    Thema steht - etwa um einen Befund aus den Experimenten mit der Literatur
    abzugleichen.

    Args:
        query: Thema oder Frage in natuerlicher Sprache, z.B. "warum Warmup beim Training"
        limit: Maximale Anzahl der Textstellen (1-20)
        projekt: Optional auf ein Projekt einschraenken, z.B. "masterarbeit".
                 Weglassen, um die gesamte Bibliothek zu durchsuchen.
        bereich: Optional auf einen Bereich innerhalb des Projekts einschraenken,
                 z.B. "baselines" oder "related-work". Gueltige Werte liefert
                 list_projects. Weglassen, um alle Bereiche zu durchsuchen.
    """
    limit = max(1, min(limit, 20))
    try:
        results = semantic_search(query, limit=limit, projekt=projekt, bereich=bereich)
    except Exception as exc:
        log.exception("Suche fehlgeschlagen")
        return [{"error": f"Suche fehlgeschlagen: {exc}"}]

    if not results:
        teile = [f"Projekt '{projekt}'" if projekt else "", f"Bereich '{bereich}'" if bereich else ""]
        eingrenzung = " in " + " / ".join(t for t in teile if t) if (projekt or bereich) else ""
        return [{"info": f"Keine Treffer{eingrenzung}. Bibliothek schon indexiert? (uv run ingest.py)"}]
    return results


@mcp.tool()
def list_projects() -> list[dict]:
    """Zeigt alle Projekte und Bereiche der Paper-Bibliothek mit Anzahl der Paper.

    Ein Projekt ist die oberste Gliederung (z.B. "masterarbeit"), ein Bereich
    eine Untergliederung darin (z.B. "baselines", "related-work"). Benutze
    dieses Tool, bevor du eine Suche einschraenkst, um die gueltigen Namen zu
    erfahren.
    """
    projekte = list_projekte()
    if not projekte:
        return [{"info": "Bibliothek ist leer. PDFs nach data/papers/<projekt>/ legen und 'uv run ingest.py' ausfuehren."}]
    return projekte


@mcp.tool()
def list_library() -> list[dict]:
    """Listet alle indexierten Paper der Bibliothek mit Titel, Jahr und Umfang auf.

    Benutze dieses Tool, um einen Ueberblick zu bekommen, welche Paper ueberhaupt
    verfuegbar sind, bevor du inhaltlich suchst.
    """
    papers = list_papers()
    if not papers:
        return [{"info": "Bibliothek ist leer. PDFs nach data/papers/ legen und 'uv run ingest.py' ausfuehren."}]
    return papers


@mcp.tool()
def read_paper(paper_id: int, max_chars: int = 6000) -> dict:
    """Liest den Volltext eines Papers, um Details nachzuschlagen.

    Nutze zuerst search_papers oder list_library, um die paper_id zu bekommen.

    Args:
        paper_id: Die numerische ID aus search_papers oder list_library
        max_chars: Maximale Textlaenge, die zurueckgegeben wird
    """
    return get_paper_text(paper_id, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Experimente
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_project(projekt: str, metric: str | None = None) -> dict:
    """Fasst ALLE Laeufe eines Projekts in einem Aufruf zusammen - fuer die Gesamtanalyse.

    Das ist das richtige Tool fuer Fragen wie "analysiere alle meine Testlaeufe",
    "welche Hyperparameter haengen mit dem Ergebnis zusammen", "gibt es Ausreisser"
    oder "was sollte ich als naechstes testen". Liefert in einem Objekt:

    - jeden Lauf mit flachen Hyperparametern und zusammengefassten Metriken
    - welche Hyperparameter ueberhaupt variiert wurden und welche konstant sind
    - Korrelationen zwischen numerischen Hyperparametern und JEDER Metrik
    - die verfuegbaren Metriknamen und die Laeufe mit hoher Fold-Streuung

    Die Korrelationen sind deskriptiv und beruhen oft auf wenigen Laeufen - sie
    sind Anhaltspunkte, kein Kausalnachweis. Deute sie im Kontext.

    Fuer Vorschlaege, was als Naechstes zu testen ist, kannst du die Befunde
    anschliessend mit search_papers gegen die Literatur abgleichen.

    Args:
        projekt: Name des Projekts, z.B. "masterarbeit"
        metric: Optional die Zielmetrik, die im Fokus stehen soll, z.B.
                "std_val_dbm_mse". Wird sie weggelassen, waehlt das Tool selbst
                eine aus - korreliert wird ohnehin gegen alle Metriken. Die
                gueltigen Namen stehen im Feld "verfuegbare_metriken".
    """
    return _summarize_project(projekt, metric=metric)


@mcp.tool()
def list_experiments(projekt: str | None = None) -> list[dict]:
    """Listet Trainings- und Testlaeufe mit Modell, Status und Datum auf.

    Nur der Ueberblick. Fuer eine Gesamtanalyse aller Laeufe nutze
    analyze_project, fuer einen einzelnen Lauf get_experiment.

    Args:
        projekt: Optional auf ein Projekt einschraenken, z.B. "masterarbeit".
                 Weglassen, um alle Laeufe zu sehen.
    """
    runs = _list_experiments(projekt)
    if not runs:
        return [{"info": "Keine Experimente gefunden. Ergebnisse nach data/experiments/<projekt>/<run_id>/ legen."}]
    return runs


@mcp.tool()
def get_experiment(run_id: str, projekt: str | None = None) -> dict:
    """Gibt Hyperparameter und Ergebnisse eines einzelnen Laufs vollstaendig zurueck.

    Nutze zuerst list_experiments oder analyze_project, um gueltige run_ids zu
    bekommen.

    Args:
        run_id: Name des Laufs, z.B. "dcgan_run_005"
        projekt: Optional, um die Suche einzugrenzen
    """
    return _get_experiment(run_id, projekt)


@mcp.tool()
def get_fold_summary(run_id: str, projekt: str | None = None) -> dict:
    """Fasst k-fold-Cross-Validation-Ergebnisse eines Laufs statistisch zusammen.

    Gibt pro Metrik Mittelwert, Standardabweichung, Minimum und Maximum ueber
    alle Folds zurueck - nicht die Rohwerte. Warnt automatisch, wenn eine
    Metrik stark ueber die Folds streut (Hinweis auf instabiles Training oder
    einen unguenstigen Split).

    Args:
        run_id: Name des Laufs, z.B. "dcgan_run_005"
        projekt: Optional, um die Suche einzugrenzen
    """
    return _get_fold_summary(run_id, projekt)


@mcp.tool()
def compare_experiments(run_ids: list[str], projekt: str | None = None) -> dict:
    """Vergleicht mehrere Laeufe und hebt hervor, was sie unterscheidet.

    Zeigt nur die *abweichenden* Hyperparameter (nicht die ganze Config) und
    stellt die Ergebnis-Metriken nebeneinander. Ideal fuer die gezielte Frage,
    welche einzelne Konfigurationsaenderung welchen Effekt hatte. Fuer den
    Gesamtueberblick ueber alle Laeufe nutze stattdessen analyze_project.

    Args:
        run_ids: Liste von Laufnamen, z.B. ["dcgan_run_005", "dcgan_run_006"]
        projekt: Optional, um die Suche einzugrenzen
    """
    return _compare_experiments(run_ids, projekt)


@mcp.tool()
def read_code(path: str, max_chars: int = 100_000) -> dict:
    """Reads a source file from the project so its current content is available.
 
    Use this to see the up-to-date version of a file in the research-mcp
    project (e.g. "server.py", "research/thesis.py", "pyproject.toml")
    instead of relying on a pasted copy. Only files inside the project and
    of an allowed type can be read.
 
    Args:
        path: Project-relative path, e.g. "research/search.py"
        max_chars: Maximum number of characters to return
    """
    return read_text_file(path, CODE_SUFFIXES, max_chars=max_chars)
 
@mcp.tool()
def list_code() -> list[dict]:
    """Lists the source files of the project that read_code can open."""
    files = list_files(CODE_SUFFIXES)
    return files or [{"info": "No source files found."}]
 
@mcp.tool()
def read_thesis(path: str, markdown: bool = False, max_chars: int = 100_000) -> dict:
    """Reads a thesis file (LaTeX or Markdown) and splits it into sentences.
 
    Returns each sentence with whether it carries a citation and which cite
    keys, so the model can separate uncited claims from cited ones. This is
    the entry point for the citation assistant: read the thesis, then judge
    which uncited sentences are citation-worthy and search the library for
    support.
 
    Args:
        path: Project-relative path to the .tex or .md file
        markdown: Set true for Markdown/pandoc ([@key]) instead of LaTeX
        max_chars: Maximum characters of the file to parse
    """
    raw = read_text_file(path, THESIS_SUFFIXES, max_chars=max_chars)
    if "error" in raw:
        return raw
 
    sentences = parse_thesis(raw["text"], markdown=markdown)
    cited = sum(1 for s in sentences if s.has_citation)
    return {
        "path": raw["path"],
        "truncated": raw["truncated"],
        "n_sentences": len(sentences),
        "n_cited": cited,
        "n_uncited": len(sentences) - cited,
        "sentences": [
            {"index": s.index, "text": s.text,
             "has_citation": s.has_citation, "cite_keys": s.cite_keys}
            for s in sentences
        ],
    }


if __name__ == "__main__":
    mcp.run()