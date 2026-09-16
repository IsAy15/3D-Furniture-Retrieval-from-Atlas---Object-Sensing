"""Validate local Markdown links in the maintained project documentation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def documentation_files(root: Path = PROJECT_ROOT):
    root = Path(root)
    files = [root / "README.md", root / "web" / "README.md"]
    files.extend(sorted((root / "docs").glob("*.md")))
    files.append(root / "vendor" / "VERSIONS.md")
    return [path for path in files if path.is_file()]


def broken_documentation_links(root: Path = PROJECT_ROOT):
    root = Path(root).resolve()
    errors = []
    for source in documentation_files(root):
        text = source.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or target.startswith(("#", "/")):
                continue
            local = unquote(target.split("#", 1)[0]).strip()
            if not local:
                continue
            destination = (source.parent / local).resolve()
            if not destination.exists():
                errors.append(
                    f"{source.relative_to(root)} -> {target}"
                )
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Check local ObjectSensing documentation links.",
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    errors = broken_documentation_links(args.root)
    if errors:
        for error in errors:
            print(f"BROKEN: {error}")
        raise SystemExit(1)
    print(
        f"Documentation valid: "
        f"{len(documentation_files(args.root))} file(s)."
    )


if __name__ == "__main__":
    main()
