"""Index PDFs into the local library.

Usage:
    uv run ingest.py                       # index all projects from config.yaml
    uv run ingest.py --project masterarbeit  # only one project
    uv run ingest.py --path C:/folder --project x  # an ad-hoc folder
    uv run ingest.py --force               # re-index everything
    uv run ingest.py --stats               # show current contents
    uv run ingest.py --duplicates          # find duplicates, remove after confirm
    uv run ingest.py --prune               # remove entries whose PDF is gone
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
from research.config import load_config
from research.db import connect, init_db
from research.metadata import fetch_arxiv_metadata
from research.embeddings import embed_texts

ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def title_by_fontsize(doc) -> str | None:
    """The title is almost always the largest text on the first page.

    Far more reliable than 'take the first long line' - license notices and
    headers often sit above the title, but always in a smaller size.
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

    largest = max(size for size, _, _ in spans)
    parts = sorted((s for s in spans if s[0] >= largest - 0.5), key=lambda s: s[1])
    title = " ".join(t[2] for t in parts).strip()

    return title if len(title) > 8 else None


def guess_title(doc, first_page: str, path: Path) -> str:
    by_layout = title_by_fontsize(doc)
    if by_layout:
        return by_layout

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
        real = fetch_arxiv_metadata(info["arxiv_id"])
        if real:
            info["title"] = real["title"]
            info["authors"] = real["authors"] or info["authors"]
            info["year"] = real["year"] or info["year"]

    return info


def ingest_file(conn, path: Path, project: str,
                force: bool = False, arxiv_lookup: bool = True,
                allow_duplicates: bool = False) -> str:
    rel_path = str(path.resolve())
    digest = file_hash(path)

    existing = conn.execute(
        "SELECT id, file_hash FROM papers WHERE file_path = ?", (rel_path,)
    ).fetchone()

    if existing and existing["file_hash"] == digest and not force:
        return "unchanged"

    if not existing and not allow_duplicates:
        elsewhere = conn.execute(
            "SELECT id, file_path FROM papers WHERE file_hash = ? AND project = ? LIMIT 1",
            (digest, project),
        ).fetchone()
        if elsewhere:
            if not Path(elsewhere["file_path"]).exists():
                conn.execute(
                    "UPDATE papers SET file_path = ? WHERE id = ?",
                    (rel_path, elsewhere["id"]),
                )
                conn.commit()
                return f"renamed (was {Path(elsewhere['file_path']).name})"
            return f"SKIP:{elsewhere['file_path']}"

    info = extract(path, arxiv_lookup=arxiv_lookup)
    chunks = chunk_pages(info["pages"])
    if not chunks:
        return "no text (scanned PDF?)"

    if existing:
        old_ids = [
            r["id"] for r in conn.execute("SELECT id FROM chunks WHERE paper_id = ?", (existing["id"],))
        ]
        conn.executemany("DELETE FROM chunk_vectors WHERE chunk_id = ?", [(i,) for i in old_ids])
        conn.execute("DELETE FROM papers WHERE id = ?", (existing["id"],))

    cursor = conn.execute(
        """INSERT INTO papers (file_path, file_hash, project, title, authors, year, arxiv_id, n_pages, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            rel_path,
            digest,
            project,
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
    return f"{'indexed' if not existing else 'updated'} ({len(chunks)} chunks)"


def normalize_title(title: str | None) -> str:
    """Reduce a title to a comparable form (for duplicate detection)."""
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _paper_rows(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT p.id, p.project, p.title, p.arxiv_id, p.file_hash,
               p.file_path, COUNT(c.id) AS n_chunks
        FROM papers p
        LEFT JOIN chunks c ON c.paper_id = p.id
        GROUP BY p.id
        ORDER BY p.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def find_duplicates(conn) -> list[dict]:
    """Find papers that sit in the same collection more than once.

    Three stages, each paper lands in at most one group:
      1. identical file - same SHA-256 under two paths
      2. same arXiv id  - same work, different file (v1 vs v7)
      3. same title     - same work without an arXiv id

    Two entries in *different* projects do not count as a duplicate.
    The first-indexed entry (smallest id) is always kept.
    """
    rows = _paper_rows(conn)
    hits: list[dict] = []
    taken: set[int] = set()

    stages = [
        ("identical file", lambda r: r["file_hash"] or ""),
        ("same arXiv id", lambda r: r["arxiv_id"] or ""),
        ("same title", lambda r: normalize_title(r["title"])),
    ]

    for reason, key_of in stages:
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            if r["id"] in taken:
                continue
            key = key_of(r)
            if not key:
                continue
            groups.setdefault((r["project"], key), []).append(r)

        for (project, _key), entries in groups.items():
            if len(entries) < 2:
                continue
            hits.append({
                "reason": reason,
                "project": project,
                "title": entries[0]["title"],
                "keep": entries[0],
                "remove": entries[1:],
            })
            taken.update(e["id"] for e in entries)

    return hits


def find_cross_project_copies(conn) -> list[dict]:
    """Identical files that live in several projects - not duplicates."""
    groups: dict[str, list[dict]] = {}
    for r in _paper_rows(conn):
        if r["file_hash"]:
            groups.setdefault(r["file_hash"], []).append(r)

    return [
        {"title": entries[0]["title"], "entries": entries}
        for entries in groups.values()
        if len({e["project"] for e in entries}) > 1
    ]


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Command-line confirmation. Without a terminal nothing is changed."""
    if not sys.stdin.isatty():
        print(f"{question} -> no interactive terminal, nothing will be removed.")
        return False

    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            answer = input(f"{question} {suffix} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def remove_papers(conn, paper_ids: list[int]) -> int:
    """Remove entries with their chunks and vectors. PDF files stay on disk."""
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
    """Show duplicates in the database and ask whether to remove them."""
    cross = find_cross_project_copies(conn)
    if cross:
        print(f"{len(cross)} file(s) intentionally live in several projects")
        print("and are left untouched:")
        for c in cross:
            projects = ", ".join(sorted({e["project"] for e in c["entries"]}))
            print(f"    {(c['title'] or '?')[:52]}  ({projects})")
        print()

    duplicates = find_duplicates(conn)
    if not duplicates:
        print("No duplicates found within a project.")
        return

    to_remove: list[int] = []
    files: list[str] = []

    print(f"{len(duplicates)} duplicate(s) found:\n")
    for d in duplicates:
        print(f"[{d['project']}] {(d['title'] or '?')[:58]}  ({d['reason']})")
        k = d["keep"]
        print(f"    keep    id={k['id']:<4} {k['n_chunks']:>3} chunks  {k['file_path']}")
        for e in d["remove"]:
            print(f"    remove  id={e['id']:<4} {e['n_chunks']:>3} chunks  {e['file_path']}")
            to_remove.append(e["id"])
            files.append(e["file_path"])
        print()

    print(f"{len(to_remove)} entry/entries would be removed from the database.")
    print("The PDF files themselves are NOT deleted.\n")

    if auto_yes or ask_yes_no("Remove duplicates from the database?", default=True):
        n = remove_papers(conn, to_remove)
        print(f"\n{n} entry/entries removed.")
        print("So they do not come back on the next run, move or delete these files:")
        for path in files:
            print(f"    {path}")
    else:
        print("\nNothing removed.")


def prune_missing(conn) -> int:
    """Remove entries whose PDF is no longer on disk."""
    rows = conn.execute("SELECT id, file_path FROM papers").fetchall()
    gone = [r["id"] for r in rows if not Path(r["file_path"]).exists()]
    return remove_papers(conn, gone)


def show_stats(conn) -> None:
    papers = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
    chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    print(f"{papers} papers, {chunks} chunks\n")

    projects = conn.execute(
        "SELECT project, COUNT(*) AS n FROM papers GROUP BY project ORDER BY project"
    ).fetchall()

    for p in projects:
        print(f"[{p['project']}]  {p['n']} papers")
        rows = conn.execute(
            "SELECT title, year, n_pages FROM papers WHERE project = ? ORDER BY year DESC",
            (p["project"],),
        )
        for row in rows:
            print(f"    {row['year'] or '????'}  {(row['title'] or '?')[:55]}")
        print()


def _folders_to_index(args) -> list[tuple[str, Path]]:
    """Resolve which (project, papers-folder) pairs to index from args/config."""
    # Ad-hoc folder wins if given.
    if args.path:
        project = args.project or "misc"
        return [(project, Path(args.path))]

    cfg = load_config()
    pairs = cfg.folders_of_kind("papers")
    if args.project:
        pairs = [(name, folder) for name, folder in pairs if name == args.project]
        if not pairs:
            print(f"No project '{args.project}' with a papers folder in config.yaml.")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Index PDFs into the library")
    parser.add_argument("--path", type=Path, default=None,
                        help="index this ad-hoc folder instead of the config projects")
    parser.add_argument("--project", default=None,
                        help="only index this project (or name the ad-hoc folder's project)")
    parser.add_argument("--force", action="store_true", help="re-index even unchanged files")
    parser.add_argument("--no-arxiv", action="store_true", help="do not fetch metadata from arXiv (offline)")
    parser.add_argument("--stats", action="store_true", help="only show current contents")
    parser.add_argument("--duplicates", action="store_true",
                        help="show duplicates and remove after confirmation")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="answer confirmations automatically with yes")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="index identical files twice anyway")
    parser.add_argument("--prune", action="store_true", help="remove entries without a matching PDF")
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
        print(f"{n} orphaned entries removed." if n else "Nothing to clean up.")
        return 0

    folders = _folders_to_index(args)
    if not folders:
        print("Nothing to index. Add projects to config.yaml or pass --path.")
        return 0

    skipped: list[tuple[str, str]] = []

    for project, folder in folders:
        if not folder.exists():
            print(f"[{project}] folder does not exist: {folder}")
            continue

        pdfs = sorted(folder.rglob("*.pdf"))
        if not pdfs:
            print(f"[{project}] no PDFs in {folder}")
            continue

        print(f"[{project}] {len(pdfs)} PDF(s) in {folder}\n")

        for pdf in pdfs:
            try:
                result = ingest_file(conn, pdf, project,
                                     force=args.force, arxiv_lookup=not args.no_arxiv,
                                     allow_duplicates=args.allow_duplicates)
            except Exception as exc:
                result = f"ERROR: {exc}"

            if result.startswith("SKIP:"):
                original = result[5:]
                skipped.append((pdf.name, Path(original).name))
                shown = f"skipped (identical to {Path(original).name})"
            else:
                shown = result
            print(f"  [{project}] {pdf.name[:42]:<45} {shown}")
        print()

    show_stats(conn)

    if skipped:
        print("=" * 64)
        print(f"{len(skipped)} file(s) were skipped as identical:\n")
        for new, old in skipped:
            print(f"    {new}")
            print(f"        identical to already indexed  {old}")
        print()
        print("Please check manually. If they really are identical, you can delete")
        print("the extra ones - the library already contains them. Then run")
        print("'uv run ingest.py --prune' for a clean tidy-up.")
        print()

    if find_duplicates(conn):
        handle_duplicates(conn, auto_yes=args.yes)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())