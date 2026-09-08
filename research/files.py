"""Safe file reading confined to configured project directories.

Shared base for tools that read files from disk (source code, thesis text).
Its single job: a tool can never read outside an allowed root - no matter what
path a caller passes in.

Two boundaries, on purpose:
  - Thesis reading (read_thesis) may touch every configured project folder.
  - Code reading (read_code) may touch only the checkout plus projects that
    opt in with `read_code_allowed: true`, minus per-project excludes and a
    hard-blocked set of secret files that can never be read.

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

# Secret-bearing files that are NEVER readable, even with read_code_allowed and
# even if the user forgot to list them in read_code_exclude. This is the safety
# net that makes the opt-in safe: forgetting to exclude a secret does not leak it.
_BLOCKED_NAMES = {".env"}
_BLOCKED_SUFFIXES = {".pem", ".key", ".pfx", ".p12", ".crt"}
_BLOCKED_STEMS = {"id_rsa", "id_ed25519", "id_dsa", "id_ecdsa"}


def _thesis_roots() -> list[Path]:
    """Roots read_thesis may touch: the checkout plus every configured folder."""
    try:
        return load_config().all_roots()
    except Exception:
        # If the config is broken, fall back to the checkout only - never widen.
        return [PROJECT_ROOT.resolve()]


def _code_roots() -> list[Path]:
    """Roots read_code may touch: the checkout plus opt-in project roots only."""
    try:
        return load_config().code_roots()
    except Exception:
        return [PROJECT_ROOT.resolve()]


def _within(resolved: Path, roots: list[Path]) -> Path | None:
    """Return the root that contains `resolved`, or None if none does."""
    for root in roots:
        if resolved == root or root in resolved.parents:
            return root
    return None


def _is_blocked_secret(resolved: Path) -> bool:
    """True if the file is a secret we never read, regardless of any allow-list."""
    if resolved.name in _BLOCKED_NAMES:
        return True
    if resolved.suffix.lower() in _BLOCKED_SUFFIXES:
        return True
    if resolved.stem in _BLOCKED_STEMS:
        return True
    return False


def _is_excluded(resolved: Path, roots: list[Path]) -> bool:
    """True if the file falls under a project's read_code_exclude entry.

    An exclude entry is matched relative to the project root, so
    'secrets/' blocks everything under <project>/secrets, and 'config/x.py'
    blocks exactly that file. Matching is done on path parts, so it works the
    same on Windows and Unix.
    """
    try:
        cfg = load_config()
    except Exception:
        return False

    for project in cfg.projects.values():
        if not project.read_code_allowed:
            continue
        root = project.root
        if root != resolved and root not in resolved.parents:
            continue  # this file is not inside this project
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        rel_parts = rel.parts
        for entry in project.read_code_exclude:
            entry_parts = Path(entry).parts
            # file matches if its path starts with the exclude entry's parts
            if rel_parts[:len(entry_parts)] == entry_parts:
                return True
    return False


def safe_resolve(user_path: str, allowed_suffixes: tuple[str, ...],
                 roots: list[Path], check_code_rules: bool = False) -> Path:
    """Resolve a user path and confirm it stays inside an allowed root.

    Returns the resolved absolute Path, or raises ValueError with a readable
    reason. Callers should catch ValueError and turn it into a tool error dict.

    When check_code_rules is True (used by read_code), the secret-file block and
    the per-project read_code_exclude list are enforced as well.
    """
    if not user_path or not user_path.strip():
        raise ValueError("Empty path.")

    raw = Path(user_path).expanduser()
    # Relative paths are interpreted against the checkout (keeps "server.py" working).
    candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)

    # resolve() collapses '..' and symlinks - do the containment check AFTER this.
    resolved = candidate.resolve()

    if _within(resolved, roots) is None:
        raise ValueError("Path is outside the allowed project directories.")

    if any(part in _BLOCKED_DIRS for part in resolved.parts):
        raise ValueError(f"Path lies inside a blocked directory ({', '.join(sorted(_BLOCKED_DIRS))}).")

    if check_code_rules:
        if _is_blocked_secret(resolved):
            raise ValueError("This file is blocked as a secret and cannot be read.")
        if _is_excluded(resolved, roots):
            raise ValueError("This file is excluded from code reading in config.yaml.")

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
                   max_chars: int = 100_000, roots: list[Path] | None = None,
                   check_code_rules: bool = False) -> dict:
    """Read a UTF-8 text file safely. Returns a dict ready to hand to the model.

    roots selects the boundary: pass _code_roots() for code, or leave it None to
    default to the thesis boundary. check_code_rules turns on the secret/exclude
    checks (used by read_code).
    """
    if roots is None:
        roots = _thesis_roots()
    try:
        path = safe_resolve(user_path, allowed_suffixes, roots=roots,
                            check_code_rules=check_code_rules)
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


def read_code_file(user_path: str, allowed_suffixes: tuple[str, ...],
                   max_chars: int = 100_000) -> dict:
    """Read a source file with the stricter code boundary and rules."""
    return read_text_file(user_path, allowed_suffixes, max_chars=max_chars,
                          roots=_code_roots(), check_code_rules=True)


def list_files(allowed_suffixes: tuple[str, ...], subdir: str | None = None) -> list[dict]:
    """List readable code files across the checkout and every opt-in project.

    Honours the same rules as read_code: blocked dirs, blocked secret files and
    each project's read_code_exclude list.
    """
    out: list[dict] = []
    for root in _code_roots():
        root = root.resolve()
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            if any(part in _BLOCKED_DIRS for part in path.parts):
                continue
            if _is_blocked_secret(path):
                continue
            if _is_excluded(path, [root]):
                continue
            out.append({
                "path": _display_path(path, [root]),
                "root": root.name,
                "n_chars": path.stat().st_size,
            })
    return out