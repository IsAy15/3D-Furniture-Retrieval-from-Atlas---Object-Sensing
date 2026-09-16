"""Teste le visualiseur : chargement maillage (chemin et zip!membre) ->
normalisation -> placement par la transformation du .json -> écriture .ply."""
import json
import os
import hashlib
import sys
import tempfile
import zipfile

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import viz  # noqa: E402


def test_visualize_roundtrip():
    tmp = tempfile.mkdtemp()
    # un maillage chaise factice (boîte) sur disque + un dans un zip
    box = trimesh.creation.box(extents=[0.4, 0.8, 0.4])
    obj_path = os.path.join(tmp, "chair.obj")
    box.export(obj_path)
    zip_path = os.path.join(tmp, "models.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("04379243/T/models/model_normalized.obj", box.export(file_type="obj"))

    summary = [
        {"model": "chair_a", "mesh": obj_path, "theta_deg": 30.0, "scale": 1.0,
         "translation": [1.0, 0.0, -0.5]},
        {"model": "table_b", "mesh": f"{zip_path}!04379243/T/models/model_normalized.obj",
         "theta_deg": -90.0, "scale": 1.2, "translation": [-0.5, 0.0, 0.3]},
    ]
    json_path = os.path.join(tmp, "scene.json")
    with open(json_path, "w") as f:
        json.dump(summary, f)

    # scan factice (gris) à overlay
    scan = np.random.default_rng(0).uniform(-1, 1, (1000, 3)).astype(np.float32)
    scan_ply = os.path.join(tmp, "scene_scan.ply")
    viz.write_points_ply(scan_ply, scan)

    out = os.path.join(tmp, "viz.ply")
    viz.build_visualization(json_path, out, scan_ply=scan_ply, n_per_model=5000, progress=print)
    pts, col = viz.read_points_ply(out)
    print(f"points totaux: {len(pts)}  couleurs distinctes: {len(np.unique(col, axis=0))}")
    assert len(pts) > 10000              # scan + 2 models
    # le modèle chaise (hauteur cible 0.9) doit être autour de translation [1,0,-0.5]
    # -> on vérifie qu'il existe des points colorés (non gris) près de x=1
    non_grey = col[:, 0] != 150
    mp = pts[non_grey]
    assert len(mp) > 0
    # au moins un modèle centré horizontalement près de sa translation
    near_chair = np.abs(mp[:, 0] - 1.0) < 0.6
    assert near_chair.any(), "modèle chaise pas placé près de sa translation"
    # hauteurs positives (posé sur le sol y>=0 puis translaté y=0)
    print(f"y range des modeles: [{mp[:,1].min():.2f}, {mp[:,1].max():.2f}]")
    assert mp[:, 1].max() > 0.5          # ~0.9 m de haut

    out2 = os.path.join(tmp, "viz_again.ply")
    viz.build_visualization(json_path, out2, scan_ply=scan_ply, n_per_model=5000)
    assert hashlib.sha256(open(out, "rb").read()).hexdigest() == hashlib.sha256(
        open(out2, "rb").read()
    ).hexdigest()


if __name__ == "__main__":
    test_visualize_roundtrip()
    print("OK viz")
