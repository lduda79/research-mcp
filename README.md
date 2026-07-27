# research-mcp

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
     data/experiments/
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
| `search_papers` | Semantic search across all chunks, optionally scoped to a project or area |
| `list_projects` | Available paper projects and areas with counts |
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

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<user>/research-mcp
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

Optional environment variables let one codebase serve different data locations:

| Variable | Default | Purpose |
|---|---|---|
| `RESEARCH_DB` | `data/library.db` | Path to the paper database |
| `RESEARCH_EXPERIMENTS` | `data/experiments` | Root of the experiment folders |

Verify without a model in the loop:

```bash
npx @modelcontextprotocol/inspector uv run server.py
```

## Papers

Organise PDFs by project; the folder structure becomes queryable metadata. The first
level is the project, an optional second level is an area within it:

```
data/papers/
├── thesis/
│   ├── baselines/
│   └── related-work/
└── general/
```

Index them:

```bash
uv run ingest.py               # index new, changed or renamed files
uv run ingest.py --force       # re-index everything
uv run ingest.py --stats       # show current contents
uv run ingest.py --duplicates  # find duplicates, remove after confirmation
uv run ingest.py --no-arxiv    # skip metadata lookup (offline)
```

## Experiments

Store each run under `data/experiments/<project>/<run_id>/` with a hyperparameter file
and a results file:

```
data/experiments/
└── thesis/
    └── cv17_lower_lr/
        ├── hparams.json
        └── results.json
```

`hparams.json` holds flat, numeric hyperparameters. The results file carries summary
values (`mean_<metric>`, `std_<metric>`) and a optional `per_fold` list with the raw per-fold
values; from those the server computes spread and flags unstable runs itself. Filenames
and metric names are flexible — several common names are accepted, and each project may
use its own metrics.

The `templates/` directory contains annotated templates and `save_run.py`, a helper you
call at the end of training that writes both files consistently (it derives the
summary values from the per-fold data, so they can never disagree).

## Design decisions

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
(`all-MiniLM-L6-v2`) · PyMuPDF · httpx

## Status

Working: PDF ingestion with duplicate and rename handling, semantic search,
project/area scoping, arXiv metadata lookup, and full experiment analysis (per-run
summaries, k-fold statistics, cross-run comparison and hyperparameter correlations).

Planned:

- Test suite covering chunking, retrieval and experiment aggregation, running in CI
- Hybrid retrieval (BM25 via FTS5 + dense, combined with reciprocal rank fusion)
- External paper discovery (arXiv / Semantic Scholar) so literature cross-checks can
  reach beyond the local library
- Note enrichment: a tool that gathers evidence for rough notes, keeping retrieved
  facts and model-generated synthesis in separate, clearly labelled sections

## Licence

MIT