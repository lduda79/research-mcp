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

import fitz
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
            "year": int(min(years)) if years else None,
            "arxiv_id": arxiv.group(1) if arxiv else None,
        }

    if arxiv_lookup and info["arxiv_id"]:
        echte = fetch_arxiv_metadata(info["arxiv_id"])
        if echte:
            info["title"] = echte["title"]
            info["authors"] = echte["authors"] or info["authors"]
            info["year"] = echte["year"] or info["year"]

    return info


def ingest_file(conn, path: Path, projekt: str, bereich: str | None,
                force: bool = False, arxiv_lookup: bool = True,
                allow_duplicates: bool = False) -> str:
    rel_path = str(path.resolve())
    digest = file_hash(path)

    existing = conn.execute(
        "SELECT id, file_hash FROM papers WHERE file_path = ?", (rel_path,)
    ).fetchone()

    if existing and existing["file_hash"] == digest and not force:
        return "unveraendert"
    
    if not existing and not allow_duplicates:
        anderswo = conn.execute(
            "SELECT id, file_path FROM papers WHERE file_hash = ? AND projekt = ? LIMIT 1",
            (digest, projekt),
        ).fetchone()
        if anderswo:
            if not Path(anderswo["file_path"]).exists():
                conn.execute(
                    "UPDATE papers SET file_path = ? WHERE id = ?",
                    (rel_path, anderswo["id"]),
                )
                conn.commit()
                return f"umbenannt (war {Path(anderswo['file_path']).name})"
            return f"SKIP:{anderswo['file_path']}"

    info = extract(path, arxiv_lookup=arxiv_lookup)
    chunks = chunk_pages(info["pages"])
    if not chunks:
        return "kein Text (gescanntes PDF?)"

    if existing:
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


def normalize_title(title: str | None) -> str:
    """Titel auf eine vergleichbare Form bringen (fuer Duplikatserkennung)."""
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _paper_rows(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT p.id, p.projekt, p.bereich, p.title, p.arxiv_id, p.file_hash,
               p.file_path, COUNT(c.id) AS n_chunks
        FROM papers p
        LEFT JOIN chunks c ON c.paper_id = p.id
        GROUP BY p.id
        ORDER BY p.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def find_duplicates(conn) -> list[dict]:
    """Findet Paper, die mehrfach in derselben Sammlung stecken.

    Drei Stufen, jedes Paper landet in hoechstens einer Gruppe:
      1. identische Datei - gleicher SHA-256 unter zwei Pfaden
      2. gleiche arXiv-ID - dieselbe Arbeit, andere Datei (v1 vs v7)
      3. gleicher Titel   - dieselbe Arbeit ohne arXiv-ID

    Zwei Eintraege in *verschiedenen* Projekten gelten nicht als Dublette.
    Behalten wird immer der zuerst indexierte Eintrag (kleinste id).
    """
    rows = _paper_rows(conn)
    treffer: list[dict] = []
    vergeben: set[int] = set()

    stufen = [
        ("identische Datei", lambda r: r["file_hash"] or ""),
        ("gleiche arXiv-ID", lambda r: r["arxiv_id"] or ""),
        ("gleicher Titel", lambda r: normalize_title(r["title"])),
    ]

    for grund, schluessel_von in stufen:
        gruppen: dict[tuple, list[dict]] = {}
        for r in rows:
            if r["id"] in vergeben:
                continue
            schluessel = schluessel_von(r)
            if not schluessel:
                continue
            gruppen.setdefault((r["projekt"], schluessel), []).append(r)

        for (projekt, _schluessel), eintraege in gruppen.items():
            if len(eintraege) < 2:
                continue
            treffer.append({
                "grund": grund,
                "projekt": projekt,
                "titel": eintraege[0]["title"],
                "behalten": eintraege[0],
                "entfernen": eintraege[1:],
            })
            vergeben.update(e["id"] for e in eintraege)

    return treffer


def find_cross_project_copies(conn) -> list[dict]:
    """Identische Dateien, die in mehreren Projekten liegen - keine Dubletten."""
    gruppen: dict[str, list[dict]] = {}
    for r in _paper_rows(conn):
        if r["file_hash"]:
            gruppen.setdefault(r["file_hash"], []).append(r)

    return [
        {"titel": eintraege[0]["title"], "eintraege": eintraege}
        for eintraege in gruppen.values()
        if len({e["projekt"] for e in eintraege}) > 1
    ]


def frage_ja_nein(frage: str, default: bool = True) -> bool:
    """Rueckfrage auf der Kommandozeile. Ohne Terminal wird nichts geaendert."""
    if not sys.stdin.isatty():
        print(f"{frage} -> kein interaktives Terminal, es wird nichts entfernt.")
        return False

    suffix = "[J/n]" if default else "[j/N]"
    while True:
        try:
            antwort = input(f"{frage} {suffix} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not antwort:
            return default
        if antwort in ("j", "ja", "y", "yes"):
            return True
        if antwort in ("n", "nein", "no"):
            return False
        print("Bitte 'j' oder 'n' eingeben.")


def remove_papers(conn, paper_ids: list[int]) -> int:
    """Entfernt Eintraege samt Chunks und Vektoren. PDF-Dateien bleiben liegen."""
    if not paper_ids:
        return 0

    for paper_id in paper_ids:
        chunk_ids = [
            c["id"] for c in conn.execute("SELECT id FROM chunks WHERE paper_id = ?", (paper_id,))
        ]
        conn.executemany("DELETE FROM chunk_vectors WHERE chunk_id = ?", [(c,) for c in chunk_ids])
        conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))

    conn.commit()
    return len(paper_ids)


def handle_duplicates(conn, auto_yes: bool = False) -> None:
    """Zeigt Dubletten in der Datenbank an und fragt, ob sie entfernt werden."""
    quer = find_cross_project_copies(conn)
    if quer:
        print(f"{len(quer)} Datei(en) liegen absichtlich in mehreren Projekten")
        print("und bleiben unangetastet:")
        for q in quer:
            projekte = ", ".join(sorted({e["projekt"] for e in q["eintraege"]}))
            print(f"    {(q['titel'] or '?')[:52]}  ({projekte})")
        print()

    dubletten = find_duplicates(conn)
    if not dubletten:
        print("Keine Dubletten innerhalb eines Projekts gefunden.")
        return

    zu_entfernen: list[int] = []
    dateien: list[str] = []

    print(f"{len(dubletten)} Dublette(n) gefunden:\n")
    for d in dubletten:
        print(f"[{d['projekt']}] {(d['titel'] or '?')[:58]}  ({d['grund']})")
        b = d["behalten"]
        print(f"    behalten   id={b['id']:<4} {b['n_chunks']:>3} Chunks  {b['file_path']}")
        for e in d["entfernen"]:
            print(f"    entfernen  id={e['id']:<4} {e['n_chunks']:>3} Chunks  {e['file_path']}")
            zu_entfernen.append(e["id"])
            dateien.append(e["file_path"])
        print()

    print(f"{len(zu_entfernen)} Eintrag/Eintraege wuerden aus der Datenbank entfernt.")
    print("Die PDF-Dateien selbst werden NICHT geloescht.\n")

    if auto_yes or frage_ja_nein("Dubletten aus der Datenbank entfernen?", default=True):
        n = remove_papers(conn, zu_entfernen)
        print(f"\n{n} Eintrag/Eintraege entfernt.")
        print("Damit sie beim naechsten Lauf nicht zurueckkommen, verschieb oder")
        print("loesche diese Dateien:")
        for pfad in dateien:
            print(f"    {pfad}")
    else:
        print("\nNichts entfernt.")


def prune_missing(conn) -> int:
    """Entfernt Eintraege, deren PDF nicht mehr auf der Platte liegt."""
    rows = conn.execute("SELECT id, file_path FROM papers").fetchall()
    weg = [r["id"] for r in rows if not Path(r["file_path"]).exists()]
    return remove_papers(conn, weg)


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
    parser.add_argument("--duplicates", action="store_true",
                        help="Dubletten anzeigen und nach Rueckfrage entfernen")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Rueckfragen automatisch mit Ja beantworten")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="identische Dateien trotzdem doppelt indexieren")
    parser.add_argument("--prune", action="store_true", help="Eintraege ohne zugehoerige PDF entfernen")
    args = parser.parse_args()

    conn = connect()
    init_db(conn)

    if args.stats:
        show_stats(conn)
        return 0

    if args.duplicates:
        handle_duplicates(conn, auto_yes=args.yes)
        return 0

    if args.prune:
        n = prune_missing(conn)
        print(f"{n} verwaiste Eintraege entfernt." if n else "Nichts zu bereinigen.")
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
    uebersprungen: list[tuple[str, str]] = []

    for pdf in pdfs:
        projekt, bereich = projekt_fuer(pdf, args.path)
        if args.projekt:
            projekt = args.projekt
        try:
            result = ingest_file(conn, pdf, projekt, bereich,
                                 force=args.force, arxiv_lookup=not args.no_arxiv,
                                 allow_duplicates=args.allow_duplicates)
        except Exception as exc:
            result = f"FEHLER: {exc}"

        label = f"{projekt}/{bereich}" if bereich else projekt

        if result.startswith("SKIP:"):
            original = result[5:]
            uebersprungen.append((pdf.name, Path(original).name))
            anzeige = f"uebersprungen (identisch mit {Path(original).name})"
        else:
            anzeige = result
        print(f"  [{label}] {pdf.name[:42]:<45} {anzeige}")

    print()
    show_stats(conn)

    if uebersprungen:
        print("=" * 64)
        print(f"{len(uebersprungen)} Datei(en) wurden als identisch uebersprungen:\n")
        for neu, alt in uebersprungen:
            print(f"    {neu}")
            print(f"        identisch mit bereits indexiertem  {alt}")
        print()
        print("Bitte manuell pruefen. Falls die Dateien wirklich identisch sind,")
        print("koennen die ueberzaehligen geloescht werden - die Bibliothek enthaelt sie")
        print("bereits. Danach 'uv run ingest.py --prune' fuer ein sauberes Aufraeumen.")
        print()

    if find_duplicates(conn):
        handle_duplicates(conn, auto_yes=args.yes)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())