"""Zerlegt Paper-Text in ueberlappende Chunks mit Seitenzuordnung."""

from __future__ import annotations

import re
from dataclasses import dataclass

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK = 120

_REFERENCES = re.compile(
    r"^\s*(references|bibliography|literaturverzeichnis)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int


def normalize(text: str) -> str:
    """Repariert typische PDF-Artefakte."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\n(?=[a-zaeoeue])", "", text)
    text = re.sub(r"(?<![\n])\n(?![\n])", " ", text) 
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_references(pages: list[str]) -> list[str]:
    """Schneidet das Literaturverzeichnis weg - es verzerrt sonst jede Suche."""
    n = len(pages)
    for i in range(max(0, n // 2), n):
        match = _REFERENCES.search(pages[i])
        if match:
            return pages[:i] + [pages[i][: match.start()]]
    return pages


def _flatten(pages: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Fuegt Seiten zu einem Text zusammen und merkt sich die Zeichen-Offsets."""
    parts: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for page_no, raw in enumerate(pages, start=1):
        cleaned = normalize(raw)
        if not cleaned:
            continue
        parts.append(cleaned)
        offsets.append((cursor, cursor + len(cleaned), page_no))
        cursor += len(cleaned) + 2
    return "\n\n".join(parts), offsets


def _page_for(offsets: list[tuple[int, int, int]], pos: int) -> int:
    for start, end, page in offsets:
        if start <= pos < end:
            return page
    return offsets[-1][2] if offsets else 1


def chunk_pages(pages: list[str]) -> list[Chunk]:
    """Sliding Window ueber den Volltext, Schnitt an Absatz- oder Satzgrenzen."""
    full, offsets = _flatten(strip_references(pages))
    if not full:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(full):
        end = min(start + CHUNK_SIZE, len(full))

        if end < len(full):
            window = full[start:end]
            for separator in ("\n\n", ". ", " "):
                cut = window.rfind(separator)
                if cut > CHUNK_SIZE * 0.5:
                    end = start + cut + len(separator)
                    break

        text = full[start:end].strip()
        if len(text) >= MIN_CHUNK:
            chunks.append(
                Chunk(
                    index=index,
                    text=text,
                    page_start=_page_for(offsets, start),
                    page_end=_page_for(offsets, max(start, end - 1)),
                )
            )
            index += 1

        if end >= len(full):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks