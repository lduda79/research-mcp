"""MCP server for the personal research stack.

Important: with stdio transport the MCP protocol runs over stdout.
Never use print() - logging goes to stderr.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from research.search import find_evidence, get_paper_text, list_papers, list_projects as _list_projects, semantic_search
from research.experiments import (
    compare_experiments as _compare_experiments,
    get_experiment as _get_experiment,
    get_fold_summary as _get_fold_summary,
    list_experiments as _list_experiments,
    summarize_project as _summarize_project,
)
from research.files import read_text_file, list_files
from research.thesis import parse_thesis

CODE_SUFFIXES = (".py", ".toml", ".md", ".txt", ".cfg", ".ini", ".yaml", ".yml")
THESIS_SUFFIXES = (".tex", ".md", ".markdown", ".txt")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("research-mcp")

mcp = FastMCP("research")


# ---------------------------------------------------------------------------
# Paper library
# ---------------------------------------------------------------------------

@mcp.tool()
def search_papers(query: str, limit: int = 5, project: str | None = None) -> list[dict]:
    """Search the local paper library by topic.

    Finds passages even when they use different words than the query. Returns
    text snippets with title, year and page. Use this to find what the read
    papers say about a topic - e.g. to check a finding from the experiments
    against the literature.

    Args:
        query: topic or question in natural language, e.g. "why warmup during training"
        limit: maximum number of passages (1-20)
        project: optional, restrict to one project, e.g. "masterarbeit".
                 Omit to search the whole library.
    """
    limit = max(1, min(limit, 20))
    try:
        results = semantic_search(query, limit=limit, project=project)
    except Exception as exc:
        log.exception("Search failed")
        return [{"error": f"Search failed: {exc}"}]

    if not results:
        scope = f" in project '{project}'" if project else ""
        return [{"info": f"No hits{scope}. Is the library indexed? (uv run ingest.py)"}]
    return results


@mcp.tool()
def list_projects() -> list[dict]:
    """Show all projects of the paper library with their paper count.

    Use this before restricting a search, to learn the valid project names.
    """
    projects = _list_projects()
    if not projects:
        return [{"info": "Library is empty. Add PDFs and run 'uv run ingest.py'."}]
    return projects


@mcp.tool()
def list_library() -> list[dict]:
    """List all indexed papers with title, year and size.

    Use this for an overview of which papers are available before searching.
    """
    papers = list_papers()
    if not papers:
        return [{"info": "Library is empty. Add PDFs and run 'uv run ingest.py'."}]
    return papers


@mcp.tool()
def read_paper(paper_id: int, max_chars: int = 6000) -> dict:
    """Read the full text of a paper to look up details.

    Use search_papers or list_library first to get the paper_id.

    Args:
        paper_id: the numeric id from search_papers or list_library
        max_chars: maximum text length to return
    """
    return get_paper_text(paper_id, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_project(project: str, metric: str | None = None) -> dict:
    """Summarize ALL runs of a project in one call - for the overall analysis.

    This is the right tool for questions like "analyze all my test runs",
    "which hyperparameters relate to the result", "are there outliers" or
    "what should I test next". Returns in one object:

    - each run with flat hyperparameters and summarized metrics
    - which hyperparameters were varied and which are constant
    - correlations between numeric hyperparameters and EVERY metric
    - the available metric names and the runs with high fold spread

    Correlations are descriptive and often based on few runs - they are hints,
    not causal proof. Interpret them in context.

    For suggestions on what to test next, you can afterwards match the findings
    against the literature with search_papers.

    Args:
        project: name of the project, e.g. "masterarbeit"
        metric: optional target metric to focus on, e.g. "std_val_dbm_mse".
                If omitted the tool picks one - correlation runs over all metrics
                anyway. Valid names are in the "available_metrics" field.
    """
    return _summarize_project(project, metric=metric)


@mcp.tool()
def list_experiments(project: str | None = None) -> list[dict]:
    """List training and test runs with model, status and date.

    Overview only. For an overall analysis of all runs use analyze_project,
    for a single run use get_experiment.

    Args:
        project: optional, restrict to one project, e.g. "masterarbeit".
                 Omit to see all runs.
    """
    runs = _list_experiments(project)
    if not runs:
        return [{"info": "No experiments found. Put results under the project's experiments folder."}]
    return runs


@mcp.tool()
def get_experiment(run_id: str, project: str | None = None) -> dict:
    """Return hyperparameters and results of a single run in full.

    Use list_experiments or analyze_project first to get valid run_ids.

    Args:
        run_id: name of the run, e.g. "dcgan_run_005"
        project: optional, to narrow the search
    """
    return _get_experiment(run_id, project)


@mcp.tool()
def get_fold_summary(run_id: str, project: str | None = None) -> dict:
    """Summarize k-fold cross-validation results of a run statistically.

    Returns per metric the mean, standard deviation, minimum and maximum across
    all folds - not the raw values. Warns automatically when a metric varies
    strongly across folds (a hint of unstable training or an unfavorable split).

    Args:
        run_id: name of the run, e.g. "dcgan_run_005"
        project: optional, to narrow the search
    """
    return _get_fold_summary(run_id, project)


@mcp.tool()
def compare_experiments(run_ids: list[str], project: str | None = None) -> dict:
    """Compare several runs and highlight what differs.

    Shows only the *differing* hyperparameters (not the whole config) and puts
    the result metrics side by side. Ideal for the targeted question of which
    single configuration change had which effect. For the overall picture across
    all runs use analyze_project instead.

    Args:
        run_ids: list of run names, e.g. ["dcgan_run_005", "dcgan_run_006"]
        project: optional, to narrow the search
    """
    return _compare_experiments(run_ids, project)


# ---------------------------------------------------------------------------
# Citation assistant
# ---------------------------------------------------------------------------

@mcp.tool()
def find_citation_candidates(statement: str, limit: int = 5, project: str | None = None) -> list[dict]:
    """Find supporting passages in the library for a SINGLE statement.

    Core tool of the citation assistant. For a concrete factual statement from
    the thesis (e.g. "warmup stabilizes training") this searches the most
    similar passages from the user's own papers and returns them ordered by
    relevance - each with the full passage text, paper, page, arXiv id and score.

    Important for judging: check against the returned passage text whether the
    source REALLY supports the statement before proposing it. Only propose hits
    with a clear topical match and a sufficiently high score - better no
    proposal than a weak one. Use this only for citation-worthy statements
    (claims about the state of research), not for meta-sentences like "in this
    chapter we show ...".

    Args:
        statement: the single statement to find support for
        limit: maximum number of candidates (ordered, best first)
        project: optional, restrict to one project, e.g. "masterarbeit"
    """
    limit = max(1, min(limit, 20))
    try:
        candidates = find_evidence(statement, limit=limit, project=project)
    except Exception as exc:
        log.exception("Evidence search failed")
        return [{"error": f"Evidence search failed: {exc}"}]

    if not candidates:
        return [{"info": "No supporting passages found. Is the library indexed?"}]
    return candidates


@mcp.tool()
def audit_thesis(
    path: str,
    markdown: bool = False,
    limit_per_statement: int = 3,
    max_statements: int = 40,
) -> dict:
    """Scan a thesis for uncited statements and suggest supporting papers.

    Combines read_thesis and find_citation_candidates in one pass: it reads the
    thesis, collects the sentences that carry NO citation, groups them by
    paragraph, and for each uncited sentence looks up candidate passages from
    the library. The result lets the model go through the thesis and propose,
    per statement or per paragraph, where a citation could be added and which
    paper (with page and passage) would support it.

    Judgement stays with the model: not every uncited sentence is
    citation-worthy (skip meta-sentences like "in this chapter we ..."), and a
    whole paragraph may deserve a single citation rather than one per sentence -
    the paragraph grouping is provided for exactly that decision.

    Args:
        path: path to the .tex or .md file (inside a configured project folder)
        markdown: set true for Markdown/pandoc ([@key]) instead of LaTeX
        limit_per_statement: how many candidate passages to return per statement
        max_statements: safety cap on how many uncited statements to look up
    """
    raw = read_text_file(path, THESIS_SUFFIXES, max_chars=200_000)
    if "error" in raw:
        return raw

    sentences = parse_thesis(raw["text"], markdown=markdown)
    uncited = [s for s in sentences if not s.has_citation and len(s.text) > 40]

    capped = uncited[:max_statements]
    limit_per_statement = max(1, min(limit_per_statement, 10))

    findings = []
    for s in capped:
        try:
            candidates = find_evidence(s.text, limit=limit_per_statement)
        except Exception as exc:
            log.exception("Evidence lookup failed")
            candidates = [{"error": f"lookup failed: {exc}"}]
        findings.append({
            "sentence_index": s.index,
            "paragraph": s.paragraph,
            "text": s.text,
            "candidates": candidates,
        })

    return {
        "path": raw["path"],
        "n_sentences": len(sentences),
        "n_cited": sum(1 for s in sentences if s.has_citation),
        "n_uncited_considered": len(capped),
        "n_uncited_total": len(uncited),
        "truncated_statements": len(uncited) > max_statements,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# File / code access
# ---------------------------------------------------------------------------

@mcp.tool()
def read_code(path: str, max_chars: int = 100_000) -> dict:
    """Read a source file from the project so its current content is available.

    Use this to see the up-to-date version of a file in the research-mcp
    project (e.g. "server.py", "research/thesis.py", "pyproject.toml") instead
    of relying on a pasted copy. Only files inside allowed directories and of an
    allowed type can be read.

    Args:
        path: project-relative path, e.g. "research/search.py"
        max_chars: maximum number of characters to return
    """
    return read_text_file(path, CODE_SUFFIXES, max_chars=max_chars)


@mcp.tool()
def list_code() -> list[dict]:
    """List the source files of the project that read_code can open."""
    files = list_files(CODE_SUFFIXES)
    return files or [{"info": "No source files found."}]


@mcp.tool()
def read_thesis(path: str, markdown: bool = False, max_chars: int = 100_000) -> dict:
    """Read a thesis file (LaTeX or Markdown) and split it into sentences.

    Returns each sentence with whether it carries a citation and which cite
    keys, so the model can separate uncited claims from cited ones. This is the
    entry point for the citation assistant: read the thesis, then judge which
    uncited sentences are citation-worthy and search the library for support.

    Args:
        path: path to the .tex or .md file (inside a configured project folder)
        markdown: set true for Markdown/pandoc ([@key]) instead of LaTeX
        max_chars: maximum characters of the file to parse
    """
    raw = read_text_file(path, THESIS_SUFFIXES, max_chars=max_chars)
    if "error" in raw:
        return raw

    sentences = parse_thesis(raw["text"], markdown=markdown)
    cited = sum(1 for s in sentences if s.has_citation)
    return {
        "path": raw["path"],
        "truncated": raw["truncated"],
        "n_sentences": len(sentences),
        "n_cited": cited,
        "n_uncited": len(sentences) - cited,
        "sentences": [
            {"index": s.index, "text": s.text,
             "has_citation": s.has_citation, "cite_keys": s.cite_keys,
             "paragraph": s.paragraph}
            for s in sentences
        ],
    }


if __name__ == "__main__":
    mcp.run()