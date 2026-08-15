"""Safe file reading confined to configured project directories.

Shared base for tools that read files from disk (source code, thesis text).
Its single job: a tool can never read outside an allowed root - no matter what
path a caller passes in. Allowed roots are the research-mcp checkout plus every
folder named in config.yaml.

Attacks it protects against:
    read_code("../../.ssh/id_rsa")        -> rejected
    read_thesis("/etc/passwd")            -> rejected
    read_code("data/../../secret.txt")    -> rejected (resolved, then checked)
"""

from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT, load_config

# Directories that are never readable, even inside an allowed root.
_BLOCKED_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


def _allowed_roots() -> list[Path]:
    """All roots a read may touch: the checkout plus every configured folder."""
    try:
        return load_config().all_roots()
    except Exception:
        # If the config is broken, fall back to the checkout only - never widen.
        return [PROJECT_ROOT.resolve()]


def _within(resolved: Path, roots: list[Path]) -> Path | None:
    """Return the root that contains `resolved`, or None if none does."""
    for root in roots:
        if resolved == root or root in resolved.parents:
            return root
    return None


def safe_resolve(user_path: str, allowed_suffixes: tuple[str, ...],
                 roots: list[Path] | None = None) -> Path:
    """Resolve a user path and confirm it stays inside an allowed root.

    Returns the resolved absolute Path, or raises ValueError with a readable
    reason. Callers should catch ValueError and turn it into a tool error dict.
    """
    if not user_path or not user_path.strip():
        raise ValueError("Empty path.")

    roots = roots or _allowed_roots()

    raw = Path(user_path).expanduser()
    # Relative paths are interpreted against the checkout (keeps "server.py" working).
    candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)

    # resolve() collapses '..' and symlinks - do the containment check AFTER this.
    resolved = candidate.resolve()

    if _within(resolved, roots) is None:
        raise ValueError("Path is outside the allowed project directories.")

    if any(part in _BLOCKED_DIRS for part in resolved.parts):
        raise ValueError(f"Path lies inside a blocked directory ({', '.join(sorted(_BLOCKED_DIRS))}).")

    if resolved.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            f"File type '{resolved.suffix}' not allowed here. "
            f"Allowed: {', '.join(allowed_suffixes)}."
        )

    return resolved


def _display_path(resolved: Path, roots: list[Path]) -> str:
    """Show the path relative to whichever allowed root contains it."""
    root = _within(resolved, roots)
    if root is not None:
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            pass
    return str(resolved)


def read_text_file(user_path: str, allowed_suffixes: tuple[str, ...],
                   max_chars: int = 100_000) -> dict:
    """Read a UTF-8 text file safely. Returns a dict ready to hand to the model."""
    roots = _allowed_roots()
    try:
        path = safe_resolve(user_path, allowed_suffixes, roots=roots)
    except ValueError as exc:
        return {"error": str(exc)}

    if not path.exists():
        return {"error": f"File not found: {_display_path(path, roots)}"}
    if not path.is_file():
        return {"error": "Path is not a file."}

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": "File is not valid UTF-8 text."}
    except OSError as exc:
        return {"error": f"Could not read file: {exc}"}

    return {
        "path": _display_path(path, roots),
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
        "n_chars": len(text),
    }


def list_files(allowed_suffixes: tuple[str, ...], subdir: str | None = None) -> list[dict]:
    """List readable files of the given types inside the checkout (or a subdir).

    Note: this lists source files in the research-mcp checkout (for read_code).
    It deliberately does not walk the whole configured project tree.
    """
    root = PROJECT_ROOT.resolve()
    base = root
    if subdir:
        base = (root / subdir).resolve()
        if base != root and root not in base.parents:
            base = root

    out: list[dict] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        if any(part in _BLOCKED_DIRS for part in path.parts):
            continue
        out.append({
            "path": str(path.relative_to(root)),
            "n_chars": path.stat().st_size,
        })
    return out