import zipfile

import pytest
import trimesh


def _make_l_mesh():
    a = trimesh.creation.box(extents=[0.6, 0.5, 0.2])
    a.apply_translation([0.3, 0.25, 0.1])
    b = trimesh.creation.box(extents=[0.2, 0.5, 0.6])
    b.apply_translation([0.1, 0.25, 0.3])
    return trimesh.util.concatenate([a, b])


@pytest.fixture
def shapenet_zip_root(tmp_path):
    """Disposition HuggingFace minimale, indépendante de l'ordre des tests."""
    obj_bytes = _make_l_mesh().export(file_type="obj").encode("utf-8")
    zip_path = tmp_path / "03001627.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for model_id in ("Lmodel", "Lmodel2"):
            archive.writestr(
                f"03001627/{model_id}/models/model_normalized.obj",
                obj_bytes,
            )
    return str(tmp_path)
