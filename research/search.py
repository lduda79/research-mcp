"""Lesende Zugriffe auf die Bibliothek - wird vom MCP-Server benutzt."""

from __future__ import annotations

import sqlite3
from typing import Any

import sqlite_vec

from .db import connect


def _snippet(text: str, max_len: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0] + " ..."


def semantic_search(
    query: str,
    limit: int = 5,
    projekt: str | None = None,
    bereich: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Vektorsuche ueber alle Chunks. Gibt Treffer mit Quellenangabe zurueck."""
    from .embeddings import embed_query

    own_conn = conn is None
    conn = conn or connect()
    try:
        vector = sqlite_vec.serialize_float32(embed_query(query))

        # Die Vektorsuche kennt die Projekt-Spalte nicht - sie liefert die k
        # aehnlichsten Chunks der ganzen Datenbank. Wird danach gefiltert,
        # bleiben ohne Ueberholen zu wenige uebrig.
        k = limit * 8 if (projekt or bereich) else limit

        rows = conn.execute(
            """
            SELECT p.id            AS paper_id,
                   p.title         AS title,
                   p.authors       AS authors,
                   p.year          AS year,
                   p.projekt       AS projekt,
                   p.bereich       AS bereich,
                   p.file_path     AS file_path,
                   c.page_start    AS page_start,
                   c.page_end      AS page_end,
                   c.text          AS text,
                   v.distance      AS distance
            FROM (
                SELECT chunk_id, distance
                FROM chunk_vectors
                WHERE embedding MATCH ? AND k = ?
            ) AS v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN papers p ON p.id = c.paper_id
            WHERE (? IS NULL OR p.projekt = ?)
              AND (? IS NULL OR p.bereich = ?)
            ORDER BY v.distance
            LIMIT ?
            """,
            (vector, k, projekt, projekt, bereich, bereich, limit),
        ).fetchall()

        return [
            {
                "paper_id": r["paper_id"],
                "title": r["title"],
                "authors": r["authors"],
                "year": r["year"],
                "projekt": r["projekt"],
                "bereich": r["bereich"],
                "pages": f"{r['page_start']}-{r['page_end']}",
                "snippet": _snippet(r["text"]),
                "score": round(1.0 - r["distance"] / 2.0, 3),
                "source": r["file_path"],
            }
            for r in rows
        ]
    finally:
        if own_conn:
            conn.close()


def list_projekte(conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """Alle Projekte mit Anzahl der Paper."""
    own_conn = conn is None
    conn = conn or connect()
    try:
        rows = conn.execute(
            """
            SELECT projekt, bereich, COUNT(*) AS n_papers
            FROM papers
            GROUP BY projekt, bereich
            ORDER BY projekt, n_papers DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own_conn:
            conn.close()


def list_papers(conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """Alle indexierten Paper mit Anzahl der Chunks."""
    own_conn = conn is None
    conn = conn or connect()
    try:
        rows = conn.execute(
            """
            SELECT p.id, p.title, p.authors, p.year, p.projekt, p.bereich, p.arxiv_id, p.n_pages,
                   COUNT(c.id) AS n_chunks
            FROM papers p
            LEFT JOIN chunks c ON c.paper_id = p.id
            GROUP BY p.id
            ORDER BY p.projekt, p.bereich, p.year DESC, p.title
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own_conn:
            conn.close()


def get_paper_text(paper_id: int, max_chars: int = 6000, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Zusammenhaengender Text eines Papers, fuer tieferes Nachlesen."""
    own_conn = conn is None
    conn = conn or connect()
    try:
        paper = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if paper is None:
            return {"error": f"Kein Paper mit id={paper_id}"}

        rows = conn.execute(
            "SELECT text FROM chunks WHERE paper_id = ? ORDER BY chunk_index",
            (paper_id,),
        ).fetchall()

        full = "\n\n".join(r["text"] for r in rows)
        truncated = len(full) > max_chars
        return {
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "text": full[:max_chars],
            "truncated": truncated,
        }
    finally:
        if own_conn:
            conn.close()