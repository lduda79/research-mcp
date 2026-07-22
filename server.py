"""MCP-Server fuer den eigenen Forschungs-Stack.

Wichtig: Bei stdio-Transport laeuft das MCP-Protokoll ueber stdout.
Niemals print() benutzen - Logging geht nach stderr.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from research.search import get_paper_text, list_papers, list_projekte, semantic_search

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("research-mcp")

mcp = FastMCP("research")


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
    Thema steht.

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
    """Zeigt alle Projekte und Bereiche der Bibliothek mit Anzahl der Paper.

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


@mcp.tool()
def get_run_metrics(run_id: str) -> dict:
    """Holt die Metriken eines Trainingslaufs.

    Args:
        run_id: Die ID des Laufs, z.B. "abc123"
    """
    # TODO: gegen MLflow / W&B austauschen
    return {"run_id": run_id, "final_loss": 2.31, "steps": 5000, "status": "completed"}


if __name__ == "__main__":
    mcp.run()