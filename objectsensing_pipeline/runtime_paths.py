"""Runtime paths and environment configuration for local and headless modes."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent


def _flag(value, default=False):
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _path(value, fallback):
    return Path(value or fallback).expanduser().resolve()


def path_is_within(path, roots):
    candidate = Path(path).expanduser().resolve()
    for root in roots:
        try:
            candidate.relative_to(Path(root).expanduser().resolve())
            return True
        except ValueError:
            continue
    return False


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    scene_dir: Path
    database_dir: Path
    run_dir: Path
    state_dir: Path
    debug_dir: Path
    headless: bool = False

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] = None,
        project_root: Path = None,
    ):
        env = os.environ if environ is None else environ
        root = _path(project_root, PROJECT_ROOT)
        default_database = root / "db_dir"
        state_dir = _path(
            env.get("OBJECTSENSING_STATE_DIR"), root / "work" / "run_console",
        )
        return cls(
            project_root=root,
            scene_dir=_path(
                env.get("OBJECTSENSING_SCENE_DIR"), root / "scenes",
            ),
            database_dir=_path(
                env.get("OBJECTSENSING_DB_DIR"), default_database,
            ),
            run_dir=_path(
                env.get("OBJECTSENSING_RUN_DIR"), root / "work" / "cli_run",
            ),
            state_dir=state_dir,
            debug_dir=_path(
                env.get("OBJECTSENSING_DEBUG_DIR"),
                state_dir.parent / "debug_visualizer",
            ),
            headless=_flag(env.get("OBJECTSENSING_HEADLESS"), False),
        )

    def browse_roots(self, environ: Mapping[str, str] = None) -> Tuple[Path, ...]:
        env = os.environ if environ is None else environ
        configured = str(env.get("OBJECTSENSING_ALLOWED_ROOTS", "")).strip()
        if configured:
            roots = [_path(value, self.project_root) for value in configured.split(os.pathsep)]
        else:
            roots = [
                self.scene_dir,
                self.database_dir,
                self.database_dir.parent,
                self.run_dir,
                self.state_dir,
            ]
            if not self.headless:
                roots.extend([self.project_root, Path.home().resolve()])
        unique = []
        for root in roots:
            resolved = Path(root).resolve()
            if resolved not in unique:
                unique.append(resolved)
        return tuple(unique)

    def ensure_writable_directories(self):
        for path in (self.run_dir, self.state_dir, self.debug_dir):
            path.mkdir(parents=True, exist_ok=True)

    def public_summary(self):
        return {
            "scene_dir": str(self.scene_dir),
            "database_dir": str(self.database_dir),
            "run_dir": str(self.run_dir),
            "state_dir": str(self.state_dir),
            "debug_dir": str(self.debug_dir),
            "headless": self.headless,
        }


RUNTIME_PATHS = RuntimePaths.from_env()
