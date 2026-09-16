"""Versioned storage helpers for ShapeNet target annotations."""
from __future__ import annotations

import json
import os
from pathlib import Path


SCHEMA_VERSION = 2


def normalize_quality(payload):
    """Return the canonical scene -> target -> model -> quality mapping."""
    if not isinstance(payload, dict):
        raise ValueError('Quality levels must be a JSON object')
    if payload.get("schema_version") == SCHEMA_VERSION:
        payload = payload.get("scenes", {})
    normalized = {}
    for scene, groups in payload.items():
        if not isinstance(groups, dict):
            raise ValueError(f"Scene {scene!r} must contain groups")
        scene_quality = {}
        for group, models in groups.items():
            if not isinstance(models, dict):
                raise ValueError(
                    f"Quality group {group!r} must be an object"
                )
            values = {}
            for name, quality in models.items():
                try:
                    quality = int(quality)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid quality for {name!r}: {quality!r}"
                    ) from exc
                if quality not in {0, 1, 2}:
                    raise ValueError(
                        f"Quality for {name!r} must be 0, 1 or 2"
                    )
                if str(name).strip():
                    values[str(name)] = quality
            scene_quality[str(group)] = values
        normalized[str(scene)] = scene_quality
    return normalized


def normalize_accepted(payload):
    """Normalize the legacy scene -> target -> accepted model list mapping."""
    if not isinstance(payload, dict):
        raise ValueError('Annotations must be a JSON object')
    if payload.get("schema_version") == SCHEMA_VERSION:
        return accepted_from_quality(normalize_quality(payload))
    normalized = {}
    for scene, groups in payload.items():
        if not isinstance(groups, dict):
            raise ValueError(f"Scene {scene!r} must contain groups")
        normalized[str(scene)] = {}
        for group, names in groups.items():
            if not isinstance(names, list):
                raise ValueError(f"Group {group!r} must be a list")
            normalized[str(scene)][str(group)] = list(dict.fromkeys(
                str(name) for name in names if str(name).strip()
            ))
    return normalized


def accepted_from_quality(quality):
    return {
        scene: {
            group: [name for name, value in models.items() if int(value) > 0]
            for group, models in groups.items()
        }
        for scene, groups in normalize_quality(quality).items()
    }


def merge_legacy(annotations, quality):
    merged = normalize_quality(quality or {})
    for scene, groups in normalize_accepted(annotations or {}).items():
        scene_quality = merged.setdefault(scene, {})
        for group, names in groups.items():
            values = scene_quality.setdefault(group, {})
            for name in names:
                values.setdefault(name, 1)
    return merged


def read_store(path, legacy_quality_path=None):
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = {}
    if isinstance(payload, dict) and payload.get("schema_version") == SCHEMA_VERSION:
        return normalize_quality(payload), False
    legacy_quality = {}
    if legacy_quality_path:
        try:
            legacy_quality = json.loads(
                Path(legacy_quality_path).read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            pass
    return merge_legacy(payload, legacy_quality), bool(payload or legacy_quality)


def write_store(path, quality):
    path = Path(path)
    normalized = normalize_quality(quality)
    payload = {"schema_version": SCHEMA_VERSION, "scenes": normalized}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))
    return normalized


def migrate_store(path, legacy_quality_path=None):
    quality, needs_migration = read_store(path, legacy_quality_path)
    if not needs_migration:
        return quality, False
    write_store(path, quality)
    # Preserve the obsolete data outside the active JSON namespace.
    legacy = Path(legacy_quality_path) if legacy_quality_path else None
    if legacy and legacy.exists():
        backup = legacy.with_suffix(legacy.suffix + ".legacy.bak")
        os.replace(str(legacy), str(backup))
    return quality, True


def accepted_annotations_from_payload(payload):
    """Accept both the legacy sweep format and the unified v2 format."""
    if isinstance(payload, dict) and payload.get("schema_version") == SCHEMA_VERSION:
        return accepted_from_quality(payload)
    return normalize_accepted(payload)
