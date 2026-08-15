"""Central configuration: where each project's papers, experiments and thesis live.

Instead of a fixed data/ folder, the user points the server at real project
folders via config.yaml. One place resolves all paths so the rest of the code
never hardcodes a location again.

Example config.yaml:

    database: ~/research-mcp/library.db

    defaults:
      papers: Literatur
      experiments: Tests
      thesis: Text

    projects:
      masterarbeit:
        root: ~/Desktop/Masterarbeit
      rf_slam:
        root: ~/Desktop/RF-SLAM
        experiments: runs      # override a default
        thesis: null           # this project has no thesis text

If no config.yaml exists, the code falls back to the classic layout
(data/papers, data/experiments, data/thesis, data/library.db) so nothing
breaks for an existing setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# PROJECT_ROOT = the research-mcp checkout itself (two levels up from this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Folder roles a project can define. Value = default subfolder name.
_DEFAULT_FOLDERS = {"papers": "papers", "experiments": "experiments", "thesis": "thesis"}


@dataclass
class Project:
    name: str
    root: Path
    papers: Path | None
    experiments: Path | None
    thesis: Path | None

    def folder(self, kind: str) -> Path | None:
        return getattr(self, kind, None)


@dataclass
class Config:
    database: Path
    projects: dict[str, Project]

    def project(self, name: str) -> Project | None:
        return self.projects.get(name)

    def all_roots(self) -> list[Path]:
        """Every configured folder - used to widen the safe-read boundary."""
        roots: list[Path] = [PROJECT_ROOT.resolve()]
        for p in self.projects.values():
            roots.append(p.root)
            for kind in _DEFAULT_FOLDERS:
                f = p.folder(kind)
                if f:
                    roots.append(f)
        # de-duplicate while keeping order
        seen: set[str] = set()
        unique: list[Path] = []
        for r in roots:
            key = str(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def folders_of_kind(self, kind: str) -> list[tuple[str, Path]]:
        """All (project_name, folder) pairs that define the given role."""
        out = []
        for name, p in self.projects.items():
            f = p.folder(kind)
            if f:
                out.append((name, f))
        return out


def _expand(path_str: str) -> Path:
    """Resolve ~ and make absolute. Relative paths are taken against PROJECT_ROOT."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _fallback_config() -> Config:
    """Classic data/ layout when no config.yaml is present."""
    data = PROJECT_ROOT / "data"
    proj = Project(
        name="default",
        root=data.resolve(),
        papers=(data / "papers").resolve(),
        experiments=(data / "experiments").resolve(),
        thesis=(data / "thesis").resolve(),
    )
    return Config(database=(data / "library.db").resolve(), projects={"default": proj})


def load_config(path: Path | None = None) -> Config:
    """Read config.yaml and resolve every path. Falls back to data/ if absent."""
    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        return _fallback_config()

    import yaml  # local import so the module loads even if pyyaml is missing

    with cfg_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    database = _expand(raw.get("database", "data/library.db"))

    defaults = {**_DEFAULT_FOLDERS, **(raw.get("defaults") or {})}

    projects: dict[str, Project] = {}
    for name, spec in (raw.get("projects") or {}).items():
        spec = spec or {}
        if "root" not in spec:
            raise ValueError(f"Project '{name}' in config.yaml is missing 'root'.")
        root = _expand(spec["root"])

        folders: dict[str, Path | None] = {}
        for kind, default_name in defaults.items():
            if kind not in _DEFAULT_FOLDERS:
                continue
            # explicit value wins; None means "this project has no such folder"
            if kind in spec:
                value = spec[kind]
                folders[kind] = None if value is None else (root / value).resolve()
            else:
                folders[kind] = (root / default_name).resolve()

        projects[name] = Project(
            name=name,
            root=root,
            papers=folders.get("papers"),
            experiments=folders.get("experiments"),
            thesis=folders.get("thesis"),
        )

    if not projects:
        return _fallback_config()

    return Config(database=database, projects=projects)