"""Central configuration (Lukas' version, paired)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
_DEFAULT_FOLDERS = {"papers": "papers", "experiments": "experiments", "thesis": "thesis"}


@dataclass
class Project:
    name: str
    root: Path
    papers: Path | None
    experiments: Path | None
    thesis: Path | None
    read_code_allowed: bool = False
    read_code_exclude: list[str] = field(default_factory=list)

    def folder(self, kind: str) -> Path | None:
        return getattr(self, kind, None)


@dataclass
class Config:
    database: Path
    projects: dict[str, Project]
    def project(self, name: str) -> Project | None:
        return self.projects.get(name)
    def all_roots(self) -> list[Path]:
        roots: list[Path] = [PROJECT_ROOT.resolve()]
        for p in self.projects.values():
            roots.append(p.root)
            for kind in _DEFAULT_FOLDERS:
                f = p.folder(kind)
                if f:
                    roots.append(f)
        seen: set[str] = set()
        unique: list[Path] = []
        for r in roots:
            key = str(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
    def code_roots(self) -> list[Path]:
        """Every project folder where reading code is allowed in config.yaml."""
        roots: list[Path] = [PROJECT_ROOT.resolve()]
        for p in self.projects.values():
            if p.read_code_allowed:
                roots.append(p.root)
        seen: set[str] = set()
        unique: list[Path] = []
        for r in roots:
            key = str(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def folders_of_kind(self, kind: str) -> list[tuple[str, Path]]:
        out = []
        for name, p in self.projects.items():
            f = p.folder(kind)
            if f:
                out.append((name, f))
        return out


def _expand(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _fallback_config() -> Config:
    data = PROJECT_ROOT / "data"
    proj = Project(name="default", root=data.resolve(),
                   papers=(data / "papers").resolve(),
                   experiments=(data / "experiments").resolve(),
                   thesis=(data / "thesis").resolve())
    return Config(database=(data / "library.db").resolve(), projects={"default": proj})


def load_config(path: Path | None = None) -> Config:
    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        return _fallback_config()
    import yaml
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
            if kind in spec:
                value = spec[kind]
                folders[kind] = None if value is None else (root / value).resolve()
            else:
                folders[kind] = (root / default_name).resolve()
        read_code_allowed = spec.get("read_code_allowed", False)
        read_code_exclude = spec.get("read_code_exclude", [])
        projects[name] = Project(
            name=name, root=root,
            papers=folders.get("papers"),
            experiments=folders.get("experiments"),
            thesis=folders.get("thesis"),
            read_code_allowed=read_code_allowed,
            read_code_exclude=read_code_exclude,
        )
    if not projects:
        return _fallback_config()
    return Config(database=database, projects=projects)