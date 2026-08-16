"""Tests for research.chunking - text normalization and windowing."""
from __future__ import annotations

from research.chunking import (
    CHUNK_SIZE, MIN_CHUNK, chunk_pages, normalize, strip_references,
)


def test_normalize_joins_hard_wrapped_lines():
    assert normalize("hello\nworld") == "hello world"


def test_normalize_dehyphenates():
    assert normalize("exam-\nple") == "example"


def test_normalize_keeps_paragraph_breaks():
    assert "\n\n" in normalize("para one\n\npara two")


def test_strip_references_cuts_bibliography():
    pages = ["intro text", "more body text", "References\n[1] Foo et al."]
    out = strip_references(pages)
    assert "[1] Foo" not in " ".join(out)


def test_strip_references_keeps_body_when_absent():
    pages = ["intro", "body", "conclusion"]
    assert strip_references(pages) == pages


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([]) == []
    assert chunk_pages(["", "   "]) == []


def test_short_text_below_min_is_dropped():
    assert chunk_pages(["tiny"]) == []


def test_long_text_is_split_into_multiple_chunks():
    page = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_pages([page])
    assert len(chunks) > 1


def test_chunks_are_indexed_sequentially():
    page = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_pages([page])
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_size_bound():
    page = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_pages([page])
    assert all(len(c.text) <= CHUNK_SIZE + 5 for c in chunks)


def test_consecutive_chunks_overlap():
    page = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_pages([page])
    assert len(chunks) >= 2
    tail = chunks[0].text[-30:]
    assert any(word in chunks[1].text for word in tail.split())


def test_page_numbers_are_tracked_across_pages():
    p1 = " ".join(f"alpha{i}" for i in range(200))
    p2 = " ".join(f"beta{i}" for i in range(200))
    chunks = chunk_pages([p1, p2])
    pages_seen = {c.page_start for c in chunks} | {c.page_end for c in chunks}
    assert 1 in pages_seen
    assert 2 in pages_seen