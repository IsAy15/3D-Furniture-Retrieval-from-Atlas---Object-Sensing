"""Teste shapenet_loader sur the deux dispositions : .zip par synset (format
HuggingFace ShapeNetCore) et dossier extrait (ShapeNetCore.v2)."""
import os
import sys

import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from shapenet_loader import find_model_sources, list_synsets  # noqa: E402


def make_L_mesh():
    a = trimesh.creation.box(extents=[0.6, 0.5, 0.2]); a.apply_translation([0.3, 0.25, 0.1])
    b = trimesh.creation.box(extents=[0.2, 0.5, 0.6]); b.apply_translation([0.1, 0.25, 0.3])
    return trimesh.util.concatenate([a, b])


def test_zip_layout(shapenet_zip_root):
    root = shapenet_zip_root
    src = find_model_sources(root, "03001627")
    print(f"[zip] sources trouvees: {len(src)} (kind={src[0].kind}, id={src[0].model_id})")
    assert len(src) == 2 and src[0].kind == "zip"
    mesh = src[0].load()
    assert hasattr(mesh, "vertices") and len(mesh.vertices) > 0
    print(f"[zip] maillage charge: {len(mesh.vertices)} sommets")
    assert "03001627" in list_synsets(root)


def test_extracted_layout():
    root = os.path.join(HERE, "..", "data", "v2_layout")
    p = os.path.join(root, "04379243", "tbl1", "models", "model_normalized.obj")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    make_L_mesh().export(p)
    src = find_model_sources(root, "04379243")
    print(f"[v2 ] sources trouvees: {len(src)} (kind={src[0].kind}, id={src[0].model_id})")
    assert len(src) == 1 and src[0].kind == "file" and src[0].model_id == "tbl1"
    mesh = src[0].load()
    assert len(mesh.vertices) > 0


if __name__ == "__main__":
    test_zip_layout()
    test_extracted_layout()
    print("OK shapenet_loader")
