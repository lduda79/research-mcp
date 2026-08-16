"""Fetch reliable metadata from the arXiv API.

The heuristics on a PDF title page are often wrong (license notices,
revision dates). When an arXiv id is found, the API is the better source.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import httpx

log = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_metadata(arxiv_id: str, timeout: float = 10.0) -> dict | None:
    """Look up title, authors and year for an arXiv id.

    Returns None when the id is unknown or the network is down - indexing must
    never fail because of this.
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
        log.warning("arXiv lookup for %s failed: %s", arxiv_id, exc)
        return None

    if entry is None:
        return None

    raw_title = entry.findtext("a:title", default="", namespaces=NS)
    title = " ".join(raw_title.split())

    # Unknown ids return an error entry instead of a paper
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