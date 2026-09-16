from pathlib import Path

from tools.check_docs import broken_documentation_links
from web_assets import PAGE_ASSETS, STATIC_ASSET_PATHS, web_asset


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_assets_are_grouped_and_registered():
    assert not list(ROOT.glob("*.html"))
    assert not list(ROOT.glob("*.css"))
    assert not list(ROOT.glob("*.js"))
    assert all(path.is_file() for path in PAGE_ASSETS.values())
    assert all(path.is_file() for path in STATIC_ASSET_PATHS.values())
    assert web_asset("timeline_viewer.css").is_file()


def test_local_documentation_links_resolve():
    assert broken_documentation_links(ROOT) == []
