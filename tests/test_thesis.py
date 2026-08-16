"""Tests for research.thesis - citation extraction and sentence splitting."""
from __future__ import annotations

from research.thesis import (
    find_citations, parse_thesis, split_sentences, strip_latex,
)


def test_find_latex_citation():
    cites = find_citations(r"Transformers prevail \citep{vaswani2017}.")
    assert len(cites) == 1
    assert cites[0].keys == ["vaswani2017"]


def test_find_multi_key_citation():
    cites = find_citations(r"\cite{a2020, b2021}")
    assert cites[0].keys == ["a2020", "b2021"]


def test_citation_with_optional_page_arg():
    cites = find_citations(r"see \citep[p. 5]{tay2022}")
    assert cites[0].keys == ["tay2022"]


def test_markdown_citation():
    cites = find_citations("Result [@smith2019] holds.", markdown=True)
    assert cites[0].keys == ["smith2019"]


def test_no_false_positive_on_plain_text():
    assert find_citations("no citations here at all") == []


def test_strip_latex_removes_commands_keeps_text():
    out = strip_latex(r"\section{Intro} This is \emph{important} text.")
    assert "Intro" in out
    assert "important" in out
    assert "\\emph" not in out


def test_strip_latex_drops_equation_environment():
    out = strip_latex(r"Before \begin{equation} E=mc^2 \end{equation} after.")
    assert "before" in out.lower()
    assert "after" in out.lower()
    assert "mc^2" not in out


def test_split_respects_et_al():
    result = split_sentences("Smith et al. found gains. Next sentence here.")
    assert len(result) == 2


def test_split_respects_eg():
    result = split_sentences("Many methods (e.g. dropout) help. Second one.")
    assert len(result) == 2


def test_split_plain_sentences():
    result = split_sentences("First sentence. Second sentence. Third one.")
    assert len(result) == 3


def test_parse_marks_cited_and_uncited():
    tex = r"""
    Gaussian splatting renders in real time \citep{kerbl2023}.
    In this chapter we present our method.
    """
    sentences = parse_thesis(tex)
    cited = [s for s in sentences if s.has_citation]
    uncited = [s for s in sentences if not s.has_citation]
    assert any("kerbl2023" in s.cite_keys for s in cited)
    assert any("present our method" in s.text for s in uncited)


def test_parse_replaces_citation_with_marker():
    sentences = parse_thesis(r"Result holds \citep{x2020}.")
    assert "[CITE]" in sentences[0].text
    assert "x2020" not in sentences[0].text


def test_parse_markdown_mode():
    sentences = parse_thesis("Claim one [@a2019]. Claim two without.", markdown=True)
    assert sentences[0].has_citation
    assert not sentences[1].has_citation