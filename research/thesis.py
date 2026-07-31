"""Parse a thesis (LaTeX or Markdown) into sentences with citation info.

Pure, deterministic text processing - no LLM. The MCP server uses this module
to hand the model clean sentences in the chat; the judgement (citation-worthy?
supported?) is then made by the model itself.

Three building blocks, each testable on its own:
    strip_latex(text)     -> plain prose without command apparatus, citations kept as markers
    split_sentences(text) -> list of sentences (robust against "et al.", "e.g." ...)
    find_citations(text)  -> all \\cite{...} / [@key] with position and keys
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Find citations
# ---------------------------------------------------------------------------

# LaTeX: \cite, \citep, \citet, \parencite, \textcite, \autocite, \citeauthor ...
# an optional [..] before the brace is consumed too: \citep[p. 5]{key}
_LATEX_CITE = re.compile(
    r"\\(?:cite|citep|citet|citealt|citealp|parencite|textcite|autocite|citeauthor|footcite)"
    r"\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}",
    re.IGNORECASE,
)

# Markdown/pandoc: [@key], [@key1; @key2], or a bare @key
_MD_CITE = re.compile(r"@([A-Za-z0-9_:\-]+)")


@dataclass
class Citation:
    keys: list[str]          # e.g. ["vaswani2017", "su2021"]
    start: int               # character position in the given text
    end: int
    raw: str                 # the full match, e.g. "\citep{vaswani2017}"


def _split_keys(raw_keys: str) -> list[str]:
    """'a, b , c' -> ['a','b','c'] (LaTeX separates multiple keys with commas)."""
    return [k.strip() for k in raw_keys.split(",") if k.strip()]


def find_citations(text: str, markdown: bool = False) -> list[Citation]:
    """Find all citations in the text with position and keys."""
    matches: list[Citation] = []

    for m in _LATEX_CITE.finditer(text):
        matches.append(Citation(_split_keys(m.group(1)), m.start(), m.end(), m.group(0)))

    if markdown:
        # only outside citations already found as LaTeX
        taken = [(c.start, c.end) for c in matches]
        for m in _MD_CITE.finditer(text):
            pos = m.start()
            if any(s <= pos < e for s, e in taken):
                continue
            matches.append(Citation([m.group(1)], m.start(), m.end(), m.group(0)))

    return sorted(matches, key=lambda c: c.start)


# ---------------------------------------------------------------------------
# Clean up LaTeX
# ---------------------------------------------------------------------------

# Environments whose content is not prose and should be dropped entirely.
_DROP_ENVIRONMENTS = (
    "figure", "table", "tabular", "equation", "align", "displaymath",
    "lstlisting", "verbatim", "minted", "tikzpicture", "algorithm",
)

_ENV_RE = {
    env: re.compile(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", re.DOTALL)
    for env in _DROP_ENVIRONMENTS
}

_CITE_PLACEHOLDER = "\x00CITE\x00"  # marks citations so they survive sentence splitting


def strip_latex(text: str) -> str:
    """Remove LaTeX commands while keeping readable prose.

    Citations are replaced by a neutral marker so that a sentence stays
    recognisable as "cited" without \\cite{...} disturbing sentence splitting.
    """
    # 1. comments (% to end of line, but \% is a literal percent sign)
    text = re.sub(r"(?<!\\)%.*", "", text)

    # 2. remove whole non-prose environments
    for env_re in _ENV_RE.values():
        text = env_re.sub(" ", text)

    # 3. strip math: $$...$$, $...$, \[...\]
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$[^$]*\$", " ", text)

    # 4. replace citations with a marker (BEFORE the general command strip)
    text = _LATEX_CITE.sub(_CITE_PLACEHOLDER, text)

    # 5. structural commands whose argument is text -> keep the argument
    #    \section{Introduction} -> Introduction , \emph{important} -> important
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph|chapter)\*?"
                  r"\{([^}]*)\}", r" \1. ", text)
    text = re.sub(r"\\(?:emph|textbf|textit|texttt|textsc|underline)\{([^}]*)\}", r"\1", text)

    # 6. \label{}, \ref{}, \eqref{}, \includegraphics{} etc. -> drop
    text = re.sub(r"\\(?:label|ref|eqref|cref|Cref|includegraphics|input|usepackage|"
                  r"documentclass|bibliography|bibliographystyle)\s*(?:\[[^\]]*\])?\{[^}]*\}", " ", text)

    # 7. remaining bare commands without arguments: \newpage, \noindent, \item ...
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)

    # 8. leftover braces and collapsed whitespace
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Split sentences
# ---------------------------------------------------------------------------

# Abbreviations after whose period there is NO sentence boundary.
_ABBREV = {
    "e.g", "i.e", "cf", "et al", "al", "vs", "approx", "resp", "no", "pp",
    "ch", "eqn", "eq", "fig", "sec", "tab", "ca", "s", "abb", "kap", "nr",
    "z.b", "d.h", "u.a", "o.ae", "bzw", "etc", "vgl", "usw", "ggf", "sog", "evtl",
    "dr", "prof", "mr", "mrs", "ms", "phd", "inc",
}

_SENT_END = re.compile(r'([.!?])(["\')\]]?)(\s+)')


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, robust against common abbreviations.

    No NLP model - an abbreviation list is enough for scientific text and keeps
    the module dependency-free.
    """
    text = " ".join(text.split())
    if not text:
        return []

    sentences: list[str] = []
    start = 0

    for m in _SENT_END.finditer(text):
        end = m.end(1)  # position right after the punctuation mark
        candidate = text[start:end].strip()

        # look at the last "word" before the period
        word = re.split(r"[\s(]", candidate.rstrip(".!?"))[-1].lower().rstrip(".")
        # check compound abbreviations like "et al": last two words
        last_two = " ".join(candidate.lower().replace(".", "").split()[-2:])

        if word in _ABBREV or last_two in _ABBREV:
            continue  # not a real sentence boundary
        if len(candidate) < 2:
            continue

        sentences.append(text[start:m.end(2)].strip())
        start = m.end()

    rest = text[start:].strip()
    if rest:
        sentences.append(rest)

    return sentences


# ---------------------------------------------------------------------------
# Put it all together
# ---------------------------------------------------------------------------

@dataclass
class ThesisSentence:
    index: int
    text: str                        # sentence with citation markers shown as [CITE]
    has_citation: bool
    cite_keys: list[str] = field(default_factory=list)


def parse_thesis(text: str, markdown: bool = False) -> list[ThesisSentence]:
    """Full path: raw text -> clean sentences with citation info.

    For each sentence we record whether it contains a citation and which keys.
    This lets the model in the chat separate uncited from cited sentences.
    """
    # find citations on the RAW text first (keys are lost during stripping)
    raw_citations = find_citations(text, markdown=markdown)

    if markdown:
        clean = text  # Markdown needs no LaTeX stripping
        # replace citations with a marker so sentence splitting keeps them intact
        for c in sorted(raw_citations, key=lambda c: c.start, reverse=True):
            clean = clean[:c.start] + _CITE_PLACEHOLDER + clean[c.end:]
    else:
        clean = strip_latex(text)

    # keys in original order, to map them onto the marker-bearing sentences
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
        result.append(
            ThesisSentence(
                index=i,
                text=display,
                has_citation=n_markers > 0,
                cite_keys=keys,
            )
        )

    return result