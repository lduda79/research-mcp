"""SQLite access and schema for the paper library."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from .config import PROJECT_ROOT, load_config

EMBEDDING_DIM = 384


def _db_path() -> Path:
    """Database location comes from config.yaml (falls back to data/library.db)."""
    try:
        return load_config().database
    except Exception:
        return PROJECT_ROOT / "data" / "library.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY,
    file_path   TEXT    NOT NULL UNIQUE,
    file_hash   TEXT    NOT NULL,
    project     TEXT    NOT NULL DEFAULT 'misc',
    title       TEXT,
    authors     TEXT,
    year        INTEGER,
    arxiv_id    TEXT,
    n_pages     INTEGER,
    indexed_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    paper_id     INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    page_start   INTEGER,
    page_end     INTEGER,
    text         TEXT    NOT NULL,
    UNIQUE (paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_papers_project ON papers(project);

CREATE TABLE IF NOT EXISTS notes (
    id           INTEGER PRIMARY KEY,
    created_at   TEXT NOT NULL,
    topic        TEXT,
    observation  TEXT NOT NULL,
    evidence     TEXT,
    synthesis    TEXT
);

CREATE TABLE IF NOT EXISTS note_links (
    note_id      INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    target_type  TEXT    NOT NULL CHECK (target_type IN ('paper', 'run', 'note')),
    target_id    TEXT    NOT NULL,
    PRIMARY KEY (note_id, target_type, target_id)
);
"""

VECTOR_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[{EMBEDDING_DIM}]
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the database and load the sqlite-vec extension."""
    path = Path(db_path) if db_path else _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables. Idempotent - safe to call repeatedly."""
    conn.executescript(SCHEMA)
    conn.executescript(VECTOR_SCHEMA)
    conn.executescript(FTS_SCHEMA)
    conn.commit()