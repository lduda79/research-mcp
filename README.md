# research-mcp

![tests](https://github.com/lduda79/research-mcp/actions/workflows/tests.yml/badge.svg)

A local Model Context Protocol server that turns a personal research workspace —
papers *and* experiment results — into tools any MCP-capable client (Claude Desktop,
Cursor, VS Code) can query.

Ask *"analyse all my thesis runs, which hyperparameters drive instability, and does my
paper library back that up?"* and the model works across both sources: it reads the
experiment summaries, correlates hyperparameters against metrics, and cross-checks the
findings against the PDFs on your own disk.

## The problem

Research context is scattered. Papers sit in one folder, training runs log JSON and CSV
somewhere else, notes live in a third place. Questions that span those sources can't be
answered without manual digging, and pasting file after file into a chat window does not
scale.

This server exposes each source as a set of tools. The language model decides at runtime
which to call and how to combine them. Retrieval and aggregation are deterministic
Python; only the interpretation happens in the model.

## Architecture

Papers and experiments are handled differently on purpose.

```
                indexing (offline)            reading (on demand)
PDFs ──► ingest.py ──► library.db ◄──┐
                                     ├──► server.py ◄──► Claude Desktop
JSON / CSV runs ─────────────────────┘        (stdio MCP)
```

**Papers** are unstructured text, so they need preparation. `ingest.py` extracts text,
strips references, splits it into overlapping chunks, computes embeddings and writes
everything to a single SQLite file. You run it when you add papers.

**Experiments** are already structured. There is no database and no preprocessing: the
server reads the JSON/CSV files straight from disk when a tool is called and summarises
them on the fly. Drop a new results file in place and it is instantly queryable.

`server.py` is read-only and starts automatically when Claude Desktop launches. It
contains no LLM — it just serves data over stdio.

### Tools

**Paper library**

| Tool | Purpose |
|---|---|
| `search_papers` | Semantic search across all chunks, optionally scoped to a project |
| `list_projects` | Available paper projects with counts |
| `list_library` | All indexed papers |
| `read_paper` | Full text of a single paper |

**Experiments**

| Tool | Purpose |
|---|---|
| `analyze_project` | Summarises *all* runs of a project in one call: per-run metrics, which hyperparameters were varied, correlations against every metric, and flagged unstable runs |
| `list_experiments` | Overview of runs with model, status and date |
| `get_experiment` | Full hyperparameters and results of a single run |
| `get_fold_summary` | k-fold results reduced to mean/std per metric, with a stability warning on high spread |
| `compare_experiments` | Diffs runs, showing only the hyperparameters that differ alongside the metrics |

**Citation assistant**

| Tool | Purpose |
|---|---|
| `find_citation_candidates` | For a single statement, returns the most similar passages from your own papers — full passage text, page and score — so the model can judge whether a source really supports the claim |
| `read_thesis` | Reads a LaTeX/Markdown thesis and splits it into sentences, marking which already carry a citation |

**Source access**

| Tool | Purpose |
|---|---|
| `read_code` | Reads a source file of the project, confined to configured directories |
| `list_code` | Lists the readable source files |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lduda79/research-mcp
cd research-mcp
uv sync
```

Register the server in `claude_desktop_config.json` (on Linux:
`~/.config/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "research": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/research-mcp", "run", "server.py"]
    }
  }
}
```

Verify without a model in the loop:

```bash
npx @modelcontextprotocol/inspector uv run server.py
```

## Configuration

Paths live in `config.yaml` at the project root. Copy the example and edit it:

```bash
cp config.example.yaml config.yaml
```

```yaml
# The vector database stays inside research-mcp — it is a derived index.
database: library.db

# Default subfolder names inside each project. Name your folders the same way
# everywhere and you only need each project's "root" below.
defaults:
  papers: papers
  experiments: experiments
  thesis: text

projects:
  my_project:
    root: ~/Desktop/my_project
    # uses the defaults

  # A second project may override folder names or omit a folder:
  # my_other_project:
  #   root: ~/Desktop/my_other_project
  #   experiments: runs
  #   thesis: null
```

`~` expands to your home directory; relative paths are taken from the research-mcp
folder. Each project keeps its papers, experiment runs and thesis text wherever you
work — nothing has to live inside research-mcp. Papers from every project share one
database, separated by the `project` column.

If no `config.yaml` is present, the server falls back to the classic layout
(`data/papers`, `data/experiments`, `data/thesis`, `data/library.db`).

The read tools (`read_code`, `read_thesis`) are confined to the research-mcp checkout
plus the folders named in `config.yaml` — nothing outside can be read.

## Papers

Put PDFs in a project's papers folder (e.g. `~/Desktop/my_project/papers`). All
papers of a project are indexed under that project's name. Index them:

```bash
uv run ingest.py                       # index all projects from config.yaml
uv run ingest.py --project my_project  # only one project
uv run ingest.py --path ~/some/folder --project scratch  # an ad-hoc folder
uv run ingest.py --force               # re-index everything
uv run ingest.py --stats               # show current contents
uv run ingest.py --duplicates          # find duplicates, remove after confirmation
uv run ingest.py --no-arxiv            # skip metadata lookup (offline)
```

## Experiments

Store each run under a project's experiments folder, one subfolder per run, with a
hyperparameter file and a results file:

```
~/Desktop/my_project/experiments/
└── run1_lower_lr/
    ├── hparams.json
    └── results.json
```

`hparams.json` holds flat, numeric hyperparameters. The results file carries summary
values (`mean_<metric>`, `std_<metric>`) and an optional `per_fold` list with the raw
per-fold values; from those the server computes spread and flags unstable runs itself.
Filenames and metric names are flexible — several common names are accepted, and each
project may use its own metrics.

The `templates/` directory contains annotated templates and `save_run.py`, a helper you
call at the end of training that writes both files consistently (it derives the
summary values from the per-fold data, so they can never disagree).

## Tests

The deterministic core is covered by a pytest suite: configuration resolution,
text chunking, the thesis/citation parser and the experiment analysis. These
modules need neither the embedding model nor a database, so the tests run in a
fraction of a second.

```bash
uv run pytest            # run everything
uv run pytest -v         # list each test
```

The suite runs automatically on every push via GitHub Actions
(`.github/workflows/tests.yml`) against Python 3.12.

## Design decisions

**Paths are configurable, data lives where you work.** A `config.yaml` maps each project
to a real folder on disk, so papers, runs and thesis text stay in your workspace instead
of being copied into the server. One module resolves every path; nothing else hardcodes
a location.

**Papers and experiments take different paths.** Unstructured PDFs are embedded into a
vector store ahead of time; structured run files are read and aggregated on demand. Two
problems, two mechanisms — forcing them through one pipeline would help neither.

**Aggregation happens before the model sees anything.** k-fold runs can hold thousands
of raw numbers. The server returns mean, std and outlier flags instead, so the model
reasons over a handful of meaningful figures rather than a flood of noise. Correlations
between hyperparameters and metrics are computed deterministically (Pearson) and labelled
as descriptive, not causal.

**Ingestion is separate from the server.** The server is read-only and loads the
embedding model lazily, so Claude Desktop starts in milliseconds instead of waiting for
PyTorch.

**Reads are confined to configured directories.** The file-reading tools resolve every
path and reject anything outside the checkout or the folders named in `config.yaml`, so a
stray or malicious path cannot escape the project.

**No `print()` anywhere in the server.** With stdio transport the MCP protocol occupies
stdout — a single stray print corrupts the message stream. All logging goes to stderr.

**References are stripped before chunking.** Bibliographies are dense clusters of domain
vocabulary with no propositional content; leaving them in hijacks semantic search.

**Filtered vector search overfetches.** The KNN query is unaware of the metadata columns
and returns the *k* globally nearest chunks; the project filter is applied afterwards.
Without overfetching (`k = limit * 8`) a filtered query can return almost nothing — the
standard pre- vs post-filtering tradeoff in ANN search.

**Titles come from font size, not the first line.** Paper title pages often carry licence
notices above the title. Taking the largest text span on page one is far more reliable;
when an arXiv ID is present, the arXiv API overrides the heuristic entirely.

**Content hashing drives re-indexing.** Each PDF is fingerprinted with SHA-256, so
`ingest.py` is idempotent — unchanged files are skipped, renamed files are detected and
moved rather than re-embedded, and changed files are replaced along with their orphaned
vectors (virtual tables are not covered by `ON DELETE CASCADE`).

## Stack

Python MCP SDK (FastMCP) · SQLite + sqlite-vec · sentence-transformers
(`all-MiniLM-L6-v2`) · PyMuPDF · httpx · PyYAML

## Status

Working: configurable project paths, PDF ingestion with duplicate and rename handling,
semantic search with project scoping, arXiv metadata lookup, full experiment analysis
(per-run summaries, k-fold statistics, cross-run comparison and hyperparameter
correlations), a citation assistant that proposes supporting passages for uncited
statements, and a pytest suite for the deterministic core running in CI.

Planned:

- Hybrid retrieval (BM25 via FTS5 + dense, combined with reciprocal rank fusion)
- External paper discovery (arXiv / Semantic Scholar) so literature cross-checks can
  reach beyond the local library
- Citation checking: verify that an existing `\cite{...}` is actually supported by the
  cited source, via a .bib lookup

## Licence

MIT