"""Canonical paths for the browser interfaces and their shared assets."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
WEB_SHARED_ROOT = WEB_ROOT / "shared"
WEB_RUN_CONSOLE_ROOT = WEB_ROOT / "run-console"
WEB_DEBUG_ROOT = WEB_ROOT / "debug"
WEB_VIEWER_ROOT = WEB_ROOT / "viewer"

PAGE_ASSETS = {
    "/": WEB_RUN_CONSOLE_ROOT / "run_console.html",
    "/run_console.html": WEB_RUN_CONSOLE_ROOT / "run_console.html",
    "/debug": WEB_DEBUG_ROOT / "debug_visualizer.html",
    "/debug_visualizer.html": WEB_DEBUG_ROOT / "debug_visualizer.html",
}

STATIC_ASSET_PATHS = {
    "/run_console.css": WEB_RUN_CONSOLE_ROOT / "run_console.css",
    "/run_console.js": WEB_RUN_CONSOLE_ROOT / "run_console.js",
    "/ui_shared.css": WEB_SHARED_ROOT / "ui_shared.css",
    "/ui_shared.js": WEB_SHARED_ROOT / "ui_shared.js",
    "/ui_scene3d.js": WEB_SHARED_ROOT / "ui_scene3d.js",
    "/ui_timeline.js": WEB_SHARED_ROOT / "ui_timeline.js",
    "/debug_visualizer.css": WEB_DEBUG_ROOT / "debug_visualizer.css",
    "/debug_visualizer.js": WEB_DEBUG_ROOT / "debug_visualizer.js",
}
STATIC_ASSETS = frozenset(STATIC_ASSET_PATHS)

_ASSETS_BY_NAME = {
    path.name: path
    for path in (*PAGE_ASSETS.values(), *STATIC_ASSET_PATHS.values())
}
_ASSETS_BY_NAME["timeline_viewer.css"] = WEB_VIEWER_ROOT / "timeline_viewer.css"


def web_asset(name: str) -> Path:
    """Resolve a unique web asset by filename for embedding and tests."""
    try:
        return _ASSETS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unknown ObjectSensing web asset: {name}") from exc
