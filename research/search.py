"""Read access to the library - used by the MCP server."""

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
    project: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Vector search over all chunks. Returns hits with their source."""
    from .embeddings import embed_query

    own_conn = conn is None
    conn = conn or connect()
    try:
        vector = sqlite_vec.serialize_float32(embed_query(query))

        # The vector search does not know the project column - it returns the k
        # nearest chunks of the whole database. When filtering afterwards, too
        # few would remain without overfetching.
        k = limit * 8 if project else limit

        rows = conn.execute(
            """
            SELECT p.id            AS paper_id,
                   p.title         AS title,
                   p.authors       AS authors,
                   p.year          AS year,
                   p.project       AS project,
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
            WHERE (? IS NULL OR p.project = ?)
            ORDER BY v.distance
            LIMIT ?
            """,
            (vector, k, project, project, limit),
        ).fetchall()

        return [
            {
                "paper_id": r["paper_id"],
                "title": r["title"],
                "authors": r["authors"],
                "year": r["year"],
                "project": r["project"],
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


def list_projects(conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """All projects with their paper count."""
    own_conn = conn is None
    conn = conn or connect()
    try:
        rows = conn.execute(
            """
            SELECT project, COUNT(*) AS n_papers
            FROM papers
            GROUP BY project
            ORDER BY project, n_papers DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own_conn:
            conn.close()


def list_papers(conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """All indexed papers with their chunk count."""
    own_conn = conn is None
    conn = conn or connect()
    try:
        rows = conn.execute(
            """
            SELECT p.id, p.title, p.authors, p.year, p.project, p.arxiv_id, p.n_pages,
                   COUNT(c.id) AS n_chunks
            FROM papers p
            LEFT JOIN chunks c ON c.paper_id = p.id
            GROUP BY p.id
            ORDER BY p.project, p.year DESC, p.title
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own_conn:
            conn.close()


def get_paper_text(paper_id: int, max_chars: int = 6000, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Contiguous text of a paper, for deeper reading."""
    own_conn = conn is None
    conn = conn or connect()
    try:
        paper = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if paper is None:
            return {"error": f"No paper with id={paper_id}"}

        rows = conn.execute(
            "SELECT text FROM chunks WHERE paper_id = ? ORDER BY chunk_index",
            (paper_id,),
        ).fetchall()

        full = "\n\n".join(r["text"] for r in rows)
        return {
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "text": full[:max_chars],
            "truncated": len(full) > max_chars,
        }
    finally:
        if own_conn:
            conn.close()


def find_evidence(
    statement: str,
    limit: int = 5,
    project: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Find supporting passages for a SINGLE statement - core of the citation assistant.

    Unlike semantic_search this returns the FULL chunk text (not the shortened
    preview), so the model in the chat can judge whether the passage really
    supports the statement. Each hit also carries the arxiv_id and chunk_index -
    for later linking to a .bib key and for evaluation.

    The result is ordered by similarity, best first, which makes recall@k and
    MRR directly measurable.
    """
    from .embeddings import embed_query

    own_conn = conn is None
    conn = conn or connect()
    try:
        vector = sqlite_vec.serialize_float32(embed_query(statement))
        k = limit * 8 if project else limit

        rows = conn.execute(
            """
            SELECT p.id            AS paper_id,
                   p.title         AS title,
                   p.authors       AS authors,
                   p.year          AS year,
                   p.arxiv_id      AS arxiv_id,
                   p.project       AS project,
                   p.file_path     AS file_path,
                   c.chunk_index   AS chunk_index,
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
            WHERE (? IS NULL OR p.project = ?)
            ORDER BY v.distance
            LIMIT ?
            """,
            (vector, k, project, project, limit),
        ).fetchall()

        return [
            {
                "rank": i + 1,
                "paper_id": r["paper_id"],
                "title": r["title"],
                "authors": r["authors"],
                "year": r["year"],
                "arxiv_id": r["arxiv_id"],
                "project": r["project"],
                "page": (
                    str(r["page_start"])
                    if r["page_start"] == r["page_end"]
                    else f"{r['page_start']}-{r['page_end']}"
                ),
                "chunk_index": r["chunk_index"],
                "passage": " ".join(r["text"].split()),
                "score": round(1.0 - r["distance"] / 2.0, 3),
                "source": r["file_path"],
            }
            for i, r in enumerate(rows)
        ]
    finally:
        if own_conn:
            conn.close()