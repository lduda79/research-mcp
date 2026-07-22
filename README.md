# research-mcp

A local Model Context Protocol server that makes a personal paper library searchable
from any MCP-capable client (Claude Desktop, Cursor, VS Code).

Ask *"what do my baseline papers say about positional encoding?"* and get answers
grounded in the PDFs on your own disk, with page-level citations.

## The problem

Research context is scattered. Papers live in one folder, experiment runs in MLflow,
notes in a third place. Questions that span those sources — *"which of my runs used the
configuration from the paper I read last month?"* — can't be answered without manual
digging.

This server exposes each source as a tool. The language model decides at runtime which
one to query and how to combine the results.

## Architecture

Two programs, one SQLite file. They never talk to each other.

```
PDFs ──► ingest.py ──► library.db ◄── server.py ◄──► Claude Desktop
         (offline)                    (stdio MCP)
```

`ingest.py` extracts text, splits it into overlapping chunks, computes embeddings and
writes everything to SQLite. You run it when you add papers.

`server.py` exposes read-only tools over stdio. Claude Desktop starts it automatically.
It contains no LLM — retrieval is deterministic Python, synthesis happens in the host.

### Tools

| Tool | Purpose |
|---|---|
| `search_papers` | Semantic search across all chunks, optionally scoped to a project or area |
| `list_projects` | Available projects and areas with paper counts |
| `list_library` | All indexed papers |
| `read_paper` | Full text of a single paper |
| `get_run_metrics` | Training run metrics (placeholder, MLflow integration pending) |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<user>/research-mcp
cd research-mcp
uv sync
```

Organise papers by project. The folder structure becomes queryable metadata:

```
data/papers/
├── thesis/
│   ├── baselines/
│   └── related-work/
└── general/
```

Index them:

```bash
uv run ingest.py            # index new or changed files
uv run ingest.py --force    # re-index everything
uv run ingest.py --stats    # show current contents
uv run ingest.py --no-arxiv # skip metadata lookup (offline)
```

Register the server in `claude_desktop_config.json`:

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

## Design decisions

**Ingestion is separate from the server.** The server is read-only and loads the
embedding model lazily, so Claude Desktop starts in milliseconds instead of waiting for
PyTorch. A write path inside the server would also mean an embedding model resident in
memory for every session.

**No `print()` anywhere in the server.** With stdio transport the MCP protocol occupies
stdout — a single stray print corrupts the message stream. All logging goes to stderr.

**References are stripped before chunking.** Bibliographies are dense clusters of domain
vocabulary with no propositional content; leaving them in hijacks semantic search.

**Filtered vector search overfetches.** The KNN query is unaware of the metadata columns
and returns the *k* globally nearest chunks; the project filter is applied afterwards.
Without overfetching (`k = limit * 8`) a filtered query can return almost nothing. This
is the standard pre- vs post-filtering tradeoff in ANN search.

**Titles come from font size, not from the first line.** Paper title pages frequently
carry licence notices above the title. Taking the largest text span on page one is
substantially more reliable. When an arXiv ID is present, the arXiv API overrides the
heuristic entirely.

**Content hashing drives re-indexing.** Each PDF is fingerprinted with SHA-256, so
`ingest.py` is idempotent — unchanged files are skipped, changed files are replaced
along with their orphaned vectors (virtual tables are not covered by
`ON DELETE CASCADE`).

## Stack

Python MCP SDK (FastMCP) · SQLite + sqlite-vec · sentence-transformers
(`all-MiniLM-L6-v2`) · PyMuPDF · httpx

## Status

Working: PDF ingestion, semantic search, project/area scoping, arXiv metadata lookup.

Planned:

- Test suite covering chunking and retrieval, running in CI
- Hybrid retrieval (BM25 via FTS5 + dense, combined with reciprocal rank fusion)
- MLflow integration replacing the `get_run_metrics` placeholder
- Note enrichment: a tool that gathers evidence for rough notes, keeping retrieved
  facts and model-generated synthesis in separate, clearly labelled sections

## Licence

MIT