"""Safe file reading confined to the project directory.

Shared base for tools that read files from disk (source code, thesis text).
The single job of this module is to make sure a tool can never read outside
the project root - no matter what path a caller passes in.

Attack it protects against:
    read_code("../../.ssh/id_rsa")        -> rejected
    read_thesis("/etc/passwd")            -> rejected
    read_code("data/../../secret.txt")    -> rejected (resolved, then checked)
"""

from __future__ import annotations

from pathlib import Path

from .db import PROJECT_ROOT

# Directories that are never readable, even inside the project.
_BLOCKED_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


def safe_resolve(user_path: str, allowed_suffixes: tuple[str, ...]) -> Path:
    """Resolve a user-supplied path and confirm it stays inside the project.

    Returns the resolved absolute Path, or raises ValueError with a readable
    reason. Callers should catch ValueError and turn it into a tool error dict.
    """
    if not user_path or not user_path.strip():
        raise ValueError("Empty path.")

    raw = Path(user_path)
    # Interpret relative paths against the project root, not the process CWD.
    candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)

    # resolve() collapses '..' and symlinks - do the containment check AFTER this,
    # otherwise 'data/../../x' would slip through.
    resolved = candidate.resolve()
    root = PROJECT_ROOT.resolve()

    if root != resolved and root not in resolved.parents:
        raise ValueError("Path is outside the project directory.")

    if any(part in _BLOCKED_DIRS for part in resolved.parts):
        raise ValueError(f"Path lies inside a blocked directory ({', '.join(sorted(_BLOCKED_DIRS))}).")

    if resolved.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            f"File type '{resolved.suffix}' not allowed here. "
            f"Allowed: {', '.join(allowed_suffixes)}."
        )

    return resolved


def read_text_file(user_path: str, allowed_suffixes: tuple[str, ...],
                   max_chars: int = 100_000) -> dict:
    """Read a UTF-8 text file safely. Returns a dict ready to hand to the model."""
    try:
        path = safe_resolve(user_path, allowed_suffixes)
    except ValueError as exc:
        return {"error": str(exc)}

    if not path.exists():
        rel = path.relative_to(PROJECT_ROOT.resolve()) if PROJECT_ROOT.resolve() in path.parents else path
        return {"error": f"File not found: {rel}"}
    if not path.is_file():
        return {"error": "Path is not a file."}

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": "File is not valid UTF-8 text."}
    except OSError as exc:
        return {"error": f"Could not read file: {exc}"}

    truncated = len(text) > max_chars
    rel = path.relative_to(PROJECT_ROOT.resolve())
    return {
        "path": str(rel),
        "text": text[:max_chars],
        "truncated": truncated,
        "n_chars": len(text),
    }


def list_files(allowed_suffixes: tuple[str, ...], subdir: str | None = None) -> list[dict]:
    """List readable files of the given types inside the project (or a subdir)."""
    root = PROJECT_ROOT.resolve()
    base = root
    if subdir:
        try:
            base = safe_resolve(subdir, allowed_suffixes=()) if False else (root / subdir).resolve()
        except ValueError:
            base = root
        if root != base and root not in base.parents:
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