"""Parse a thesis (LaTeX or Markdown) into sentences with citation info.

Pure, deterministic text processing - no LLM. The MCP server uses this module
to hand the model clean sentences in the chat; the judgement (citation-worthy?
supported?) is then made by the model itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_LATEX_CITE = re.compile(
    r"\\(?:cite|citep|citet|citealt|citealp|parencite|textcite|autocite|citeauthor|footcite)"
    r"\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}",
    re.IGNORECASE,
)
_MD_CITE = re.compile(r"@([A-Za-z0-9_:\-]+)")


@dataclass
class Citation:
    keys: list[str]
    start: int
    end: int
    raw: str


def _split_keys(raw_keys: str) -> list[str]:
    return [k.strip() for k in raw_keys.split(",") if k.strip()]


def find_citations(text: str, markdown: bool = False) -> list[Citation]:
    """Find all citations in the text with position and keys."""
    matches: list[Citation] = []
    for m in _LATEX_CITE.finditer(text):
        matches.append(Citation(_split_keys(m.group(1)), m.start(), m.end(), m.group(0)))
    if markdown:
        taken = [(c.start, c.end) for c in matches]
        for m in _MD_CITE.finditer(text):
            pos = m.start()
            if any(s <= pos < e for s, e in taken):
                continue
            matches.append(Citation([m.group(1)], m.start(), m.end(), m.group(0)))
    return sorted(matches, key=lambda c: c.start)


_DROP_ENVIRONMENTS = (
    "figure", "table", "tabular", "equation", "align", "displaymath",
    "lstlisting", "verbatim", "minted", "tikzpicture", "algorithm",
)
_ENV_RE = {
    env: re.compile(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", re.DOTALL)
    for env in _DROP_ENVIRONMENTS
}
_CITE_PLACEHOLDER = "\x00CITE\x00"


def strip_latex(text: str) -> str:
    """Remove LaTeX commands while keeping readable prose."""
    text = re.sub(r"(?<!\\)%.*", "", text)
    for env_re in _ENV_RE.values():
        text = env_re.sub(" ", text)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$[^$]*\$", " ", text)
    text = _LATEX_CITE.sub(_CITE_PLACEHOLDER, text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph|chapter)\*?"
                  r"\{([^}]*)\}", r" \1. ", text)
    text = re.sub(r"\\(?:emph|textbf|textit|texttt|textsc|underline)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:label|ref|eqref|cref|Cref|includegraphics|input|usepackage|"
                  r"documentclass|bibliography|bibliographystyle)\s*(?:\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


_ABBREV = {
    "e.g", "i.e", "cf", "et al", "al", "vs", "approx", "resp", "no", "pp",
    "ch", "eqn", "eq", "fig", "sec", "tab", "ca", "s", "abb", "kap", "nr",
    "z.b", "d.h", "u.a", "o.ae", "bzw", "etc", "vgl", "usw", "ggf", "sog", "evtl",
    "dr", "prof", "mr", "mrs", "ms", "phd", "inc",
}
_SENT_END = re.compile(r'([.!?])(["\')\]]?)(\s+)')


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, robust against common abbreviations."""
    text = " ".join(text.split())
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        candidate = text[start:end].strip()
        word = re.split(r"[\s(]", candidate.rstrip(".!?"))[-1].lower().rstrip(".")
        last_two = " ".join(candidate.lower().replace(".", "").split()[-2:])
        if word in _ABBREV or last_two in _ABBREV:
            continue
        if len(candidate) < 2:
            continue
        sentences.append(text[start:m.end(2)].strip())
        start = m.end()
    rest = text[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences


@dataclass
class ThesisSentence:
    index: int
    text: str
    has_citation: bool
    cite_keys: list[str] = field(default_factory=list)


def parse_thesis(text: str, markdown: bool = False) -> list[ThesisSentence]:
    """Full path: raw text -> clean sentences with citation info."""
    raw_citations = find_citations(text, markdown=markdown)
    if markdown:
        clean = text
        for c in sorted(raw_citations, key=lambda c: c.start, reverse=True):
            clean = clean[:c.start] + _CITE_PLACEHOLDER + clean[c.end:]
    else:
        clean = strip_latex(text)
    key_queue = [c.keys for c in raw_citations]
    sentences = split_sentences(clean)
    result: list[ThesisSentence] = []
    marker_index = 0
    for i, sentence in enumerate(sentences):
        n_markers = sentence.count(_CITE_PLACEHOLDER)
        keys: list[str] = []
        for _ in range(n_markers):
            if marker_index < len(key_queue):
                keys.extend(key_queue[marker_index])
                marker_index += 1
        display = sentence.replace(_CITE_PLACEHOLDER, "[CITE]").strip()
        result.append(ThesisSentence(index=i, text=display,
                                     has_citation=n_markers > 0, cite_keys=keys))
    return result