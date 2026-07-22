"""Indexiert PDFs in die lokale Bibliothek.

Aufruf:
    uv run ingest.py                    # data/papers/ einlesen
    uv run ingest.py --path C:/pfad     # anderer Ordner
    uv run ingest.py --force            # alles neu indexieren
    uv run ingest.py --stats            # nur Bestand anzeigen
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import sqlite_vec

from research.chunking import chunk_pages
from research.db import PROJECT_ROOT, connect, init_db
from research.metadata import fetch_arxiv_metadata
from research.embeddings import embed_texts

DEFAULT_PAPER_DIR = PROJECT_ROOT / "data" / "papers"

ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def projekt_fuer(pdf: Path, root: Path) -> tuple[str, str | None]:
    """Leitet Projekt und Bereich aus der Ordnerstruktur ab.

    data/papers/masterarbeit/baselines/su.pdf -> ("masterarbeit", "baselines")
    data/papers/masterarbeit/vaswani.pdf      -> ("masterarbeit", None)
    data/papers/lose.pdf                      -> ("sonstiges", None)
    """
    rel = pdf.resolve().relative_to(root.resolve())
    parts = rel.parts

    if len(parts) == 1:
        return "sonstiges", None
    if len(parts) == 2:
        return parts[0], None
    return parts[0], parts[1]


def title_by_fontsize(doc) -> str | None:
    """Der Titel ist auf der ersten Seite fast immer der groesste Text.

    Deutlich verlaesslicher als 'nimm die erste lange Zeile' - Lizenztexte
    und Kopfzeilen stehen naemlich oft ueber dem Titel, aber immer kleiner.
    """
    if doc.page_count == 0:
        return None

    spans: list[tuple[float, float, str]] = []
    for block in doc[0].get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if len(text) > 3:
                    spans.append((round(span["size"], 1), span["bbox"][1], text))

    if not spans:
        return None

    groesste = max(size for size, _, _ in spans)
    # Titel koennen ueber mehrere Zeilen laufen -> alle Spans dieser Groesse,
    # von oben nach unten zusammensetzen
    teile = sorted((s for s in spans if s[0] >= groesste - 0.5), key=lambda s: s[1])
    titel = " ".join(t[2] for t in teile).strip()

    return titel if len(titel) > 8 else None


def guess_title(doc, first_page: str, path: Path) -> str:
    per_layout = title_by_fontsize(doc)
    if per_layout:
        return per_layout

    meta_title = ((doc.metadata or {}).get("title") or "").strip()
    if len(meta_title) > 8 and not meta_title.lower().endswith(".pdf"):
        return meta_title

    for line in first_page.splitlines():
        line = line.strip()
        if len(line) > 12 and not ARXIV_RE.search(line) and "@" not in line:
            return line

    return path.stem.replace("_", " ")


def extract(path: Path, arxiv_lookup: bool = True) -> dict:
    with fitz.open(path) as doc:
        pages = [page.get_text() for page in doc]
        first = pages[0] if pages else ""
        arxiv = ARXIV_RE.search(first)
        years = YEAR_RE.findall(first)

        info = {
            "pages": pages,
            "n_pages": len(pages),
            "title": guess_title(doc, first, path),
            "authors": ((doc.metadata or {}).get("author") or "").strip() or None,
            # min statt max: auf der Titelseite steht neben dem Erscheinungsjahr
            # oft ein spaeteres Revisionsdatum des arXiv-Stempels
            "year": int(min(years)) if years else None,
            "arxiv_id": arxiv.group(1) if arxiv else None,
        }

    # Die API kennt den echten Titel - Heuristik nur als Rueckfallebene
    if arxiv_lookup and info["arxiv_id"]:
        echte = fetch_arxiv_metadata(info["arxiv_id"])
        if echte:
            info["title"] = echte["title"]
            info["authors"] = echte["authors"] or info["authors"]
            info["year"] = echte["year"] or info["year"]

    return info


def ingest_file(conn, path: Path, projekt: str, bereich: str | None,
                force: bool = False, arxiv_lookup: bool = True) -> str:
    rel_path = str(path.resolve())
    digest = file_hash(path)

    existing = conn.execute(
        "SELECT id, file_hash FROM papers WHERE file_path = ?", (rel_path,)
    ).fetchone()

    if existing and existing["file_hash"] == digest and not force:
        return "unveraendert"

    info = extract(path, arxiv_lookup=arxiv_lookup)
    chunks = chunk_pages(info["pages"])
    if not chunks:
        return "kein Text (gescanntes PDF?)"

    if existing:
        # ON DELETE CASCADE raeumt chunks weg, die Vektoren muessen wir selbst loeschen
        old_ids = [
            r["id"] for r in conn.execute("SELECT id FROM chunks WHERE paper_id = ?", (existing["id"],))
        ]
        conn.executemany("DELETE FROM chunk_vectors WHERE chunk_id = ?", [(i,) for i in old_ids])
        conn.execute("DELETE FROM papers WHERE id = ?", (existing["id"],))

    cursor = conn.execute(
        """INSERT INTO papers (file_path, file_hash, projekt, bereich, title, authors, year, arxiv_id, n_pages, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            rel_path,
            digest,
            projekt,
            bereich,
            info["title"],
            info["authors"],
            info["year"],
            info["arxiv_id"],
            info["n_pages"],
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    paper_id = cursor.lastrowid

    vectors = embed_texts([c.text for c in chunks])

    for chunk, vector in zip(chunks, vectors):
        chunk_cursor = conn.execute(
            """INSERT INTO chunks (paper_id, chunk_index, page_start, page_end, text)
               VALUES (?, ?, ?, ?, ?)""",
            (paper_id, chunk.index, chunk.page_start, chunk.page_end, chunk.text),
        )
        conn.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_cursor.lastrowid, sqlite_vec.serialize_float32(vector)),
        )

    conn.commit()
    return f"{'neu indexiert' if not existing else 'aktualisiert'} ({len(chunks)} Chunks)"


def show_stats(conn) -> None:
    papers = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
    chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    print(f"{papers} Paper, {chunks} Chunks\n")

    projekte = conn.execute(
        "SELECT projekt, COUNT(*) AS n FROM papers GROUP BY projekt ORDER BY projekt"
    ).fetchall()

    for p in projekte:
        print(f"[{p['projekt']}]  {p['n']} Paper")
        rows = conn.execute(
            "SELECT title, year, bereich, n_pages FROM papers WHERE projekt = ? ORDER BY bereich, year DESC",
            (p["projekt"],),
        )
        for row in rows:
            tag = f"{row['bereich']:<14}" if row["bereich"] else " " * 14
            print(f"    {tag} {row['year'] or '????'}  {(row['title'] or '?')[:50]}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="PDFs in die Bibliothek indexieren")
    parser.add_argument("--path", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--projekt", default=None,
                        help="ueberschreibt den Ordnernamen als Projekt")
    parser.add_argument("--force", action="store_true", help="auch unveraenderte Dateien neu indexieren")
    parser.add_argument("--no-arxiv", action="store_true", help="keine Metadaten von arXiv holen (offline)")
    parser.add_argument("--stats", action="store_true", help="nur Bestand anzeigen")
    args = parser.parse_args()

    conn = connect()
    init_db(conn)

    if args.stats:
        show_stats(conn)
        return 0

    if not args.path.exists():
        args.path.mkdir(parents=True, exist_ok=True)
        print(f"Ordner {args.path} angelegt. Leg dort PDFs ab und starte erneut.")
        return 0

    pdfs = sorted(args.path.rglob("*.pdf"))
    if not pdfs:
        print(f"Keine PDFs in {args.path}")
        return 0

    print(f"{len(pdfs)} PDF(s) gefunden\n")
    for pdf in pdfs:
        projekt, bereich = projekt_fuer(pdf, args.path)
        if args.projekt:
            projekt = args.projekt
        try:
            result = ingest_file(conn, pdf, projekt, bereich,
                                 force=args.force, arxiv_lookup=not args.no_arxiv)
        except Exception as exc:  # eine kaputte Datei darf den Lauf nicht stoppen
            result = f"FEHLER: {exc}"
        label = f"{projekt}/{bereich}" if bereich else projekt
        print(f"  [{label}] {pdf.name[:42]:<45} {result}")

    print()
    show_stats(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())