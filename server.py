from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research")


@mcp.tool()
def search_papers(query: str, limit: int = 5) -> list[dict]:
    """Durchsucht die lokale Paper-Bibliothek nach einem Thema.

    Args:
        query: Suchbegriff in natürlicher Sprache, z.B. "positional encoding"
        limit: Maximale Anzahl der Treffer
    """
    # Noch Platzhalter-Daten – echte Suche kommt später
    return [
        {"title": "Attention Is All You Need", "year": 2017, "authors": "Vaswani et al."},
        {"title": "RoFormer: Rotary Position Embedding", "year": 2021, "authors": "Su et al."},
    ][:limit]


@mcp.tool()
def get_run_metrics(run_id: str) -> dict:
    """Holt die Metriken eines Trainingslaufs.

    Args:
        run_id: Die ID des Laufs, z.B. "abc123"
    """
    return {"run_id": run_id, "final_loss": 2.31, "steps": 5000, "status": "completed"}


if __name__ == "__main__":
    mcp.run()