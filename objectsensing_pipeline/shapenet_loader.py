'\nShapeNetCore loader supporting extracted <synset>/<model_id>/models/model_normalized.obj, per-synset ZIP archives and arbitrary OBJ files under each synset. ModelSource loads geometry from disk or archive without extracting the entire dataset.\n'

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelSource:
    model_id: str
    synset: str
    kind: str          # "file" | "zip"
    path: str          # chemin du .obj (file) ou du .zip (zip)
    member: str = ""   # nom du membre .obj dans l'archive (zip)

    def load(self):
        'Load mesh geometry using trimesh.'
        import trimesh
        if self.kind == "file":
            mesh = trimesh.load(self.path, force="mesh", process=False)
        else:
            with zipfile.ZipFile(self.path) as zf:
                data = zf.read(self.member)
            mesh = trimesh.load(io.BytesIO(data), file_type="obj", force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        return mesh


def _model_id_from_member(member: str) -> str:
    parts = [p for p in member.replace("\\", "/").split("/") if p]
    # .../<synset>/<model_id>/models/model_normalized.obj
    if "models" in parts:
        k = parts.index("models")
        if k >= 1:
            return parts[k - 1]
    return parts[-1].rsplit(".", 1)[0]


def find_model_sources(root, synset: str, limit: Optional[int] = None) -> List[ModelSource]:
    root = Path(root)
    sources: List[ModelSource] = []

    # (a) extrait : model_normalized.obj
    for p in sorted(root.glob(f"{synset}/*/models/model_normalized.obj")):
        sources.append(ModelSource(p.parent.parent.name, synset, "file", str(p)))

    # (b) zip par synset
    if not sources:
        zp = root / f"{synset}.zip"
        if zp.exists():
            with zipfile.ZipFile(zp) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith("model_normalized.obj")]
                if not names:
                    names = [n for n in zf.namelist() if n.lower().endswith(".obj")]
            for n in sorted(names):
                sources.append(ModelSource(_model_id_from_member(n), synset, "zip", str(zp), n))

    # (c) repli : tout .obj sous <root>/<synset>/
    if not sources:
        for p in sorted(root.glob(f"{synset}/**/*.obj")):
            sources.append(ModelSource(p.stem, synset, "file", str(p)))

    if limit:
        sources = sources[:limit]
    return sources


def list_synsets(root) -> List[str]:
    'Available synsets (directories or ZIP archives) under the root.'
    root = Path(root)
    out = set()
    for d in root.iterdir() if root.exists() else []:
        if d.is_dir() and d.name.isdigit():
            out.add(d.name)
        elif d.suffix == ".zip" and d.stem.isdigit():
            out.add(d.stem)
    return sorted(out)
