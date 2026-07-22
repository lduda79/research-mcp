"""Holt verlaessliche Metadaten von der arXiv-API.

Die Heuristiken auf der PDF-Titelseite liegen oft daneben (Lizenztexte,
Revisionsdaten). Wenn eine arXiv-ID gefunden wurde, ist die API die
bessere Quelle.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import httpx

log = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_metadata(arxiv_id: str, timeout: float = 10.0) -> dict | None:
    """Fragt Titel, Autoren und Jahr zu einer arXiv-ID ab.

    Gibt None zurueck, wenn die ID unbekannt ist oder das Netz nicht geht -
    die Indexierung darf daran nie scheitern.
    """
    try:
        response = httpx.get(
            ARXIV_API,
            params={"id_list": arxiv_id},
            timeout=timeout,
            headers={"User-Agent": "research-mcp/0.1"},
            follow_redirects=True,
        )
        response.raise_for_status()
        entry = ET.fromstring(response.text).find("a:entry", NS)
    except Exception as exc:
        log.warning("arXiv-Abfrage fuer %s fehlgeschlagen: %s", arxiv_id, exc)
        return None

    if entry is None:
        return None

    raw_title = entry.findtext("a:title", default="", namespaces=NS)
    title = " ".join(raw_title.split())

    # Unbekannte IDs liefern einen Fehler-Eintrag statt eines Papers
    if not title or title.lower().startswith("error"):
        return None

    authors = [
        name.strip()
        for author in entry.findall("a:author", NS)
        if (name := author.findtext("a:name", default="", namespaces=NS))
    ]
    published = entry.findtext("a:published", default="", namespaces=NS)

    return {
        "title": title,
        "authors": ", ".join(authors) or None,
        "year": int(published[:4]) if published[:4].isdigit() else None,
    }