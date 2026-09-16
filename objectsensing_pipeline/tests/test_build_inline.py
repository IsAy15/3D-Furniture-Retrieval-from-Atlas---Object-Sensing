"""Reproduit la logique de build_database avec des modules frais (le bac à sable
sert une version cachée obsolète de database.py). Valide : loader zip -> mesh ->
normalisation -> virtual scan -> key points -> features. Logique identique à
database.build_database / preprocess_mesh."""
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from config import PipelineConfig  # noqa: E402
from geometry import make_point_cloud  # noqa: E402
from keypoints import detect_keypoints  # noqa: E402
from matching import build_model_features  # noqa: E402
from shapenet_loader import find_model_sources  # noqa: E402

CATEGORY_HEIGHT = {"03001627": 0.90, "04379243": 0.75, "04256520": 0.80}


def normalize_mesh(mesh, synset):
    V = np.asarray(mesh.vertices, float)
    h = V[:, 1].max() - V[:, 1].min()
    s = CATEGORY_HEIGHT.get(synset, 0.8) / h if h > 1e-6 else 1.0
    V = V * s
    V[:, 1] -= V[:, 1].min()
    V[:, 0] -= (V[:, 0].max() + V[:, 0].min()) / 2
    V[:, 2] -= (V[:, 2].max() + V[:, 2].min()) / 2
    mesh.vertices = V
    return mesh


def virtual_scan(mesh, n, rng, add_ground=True):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    pts = np.asarray(pts)
    if add_ground:
        lo, hi = pts.min(0), pts.max(0)
        m = max(hi[0] - lo[0], hi[2] - lo[2]) * 0.6 + 0.1
        g = rng.uniform([-m, 0, -m], [m, 0, m], size=(n // 4, 3))
        g[:, 0] += (lo[0] + hi[0]) / 2; g[:, 2] += (lo[2] + hi[2]) / 2; g[:, 1] = lo[1]
        pts = np.vstack([pts, g])
    return pts


def test_build_inline(shapenet_zip_root):
    root = shapenet_zip_root
    cfg = PipelineConfig()
    cfg.keypoint.neighbor_radius = 0.07
    cfg.keypoint.nms_radius = 0.12
    cfg.keypoint.dedup_radius = 0.08
    cfg.keypoint.curvature_threshold = 0.03
    up = np.asarray(cfg.up_axis, float); up /= np.linalg.norm(up)

    sources = find_model_sources(root, "03001627", limit=2)
    assert len(sources) == 2 and sources[0].kind == "zip"
    models = []
    for src in sources:
        mesh = normalize_mesh(src.load(), src.synset)
        pts = virtual_scan(mesh, 4000, np.random.default_rng(0))
        cloud = make_point_cloud(pts, radius=cfg.keypoint.neighbor_radius)
        ground = float((pts @ up).min())
        kps = detect_keypoints(cloud, cfg.keypoint)
        feats = build_model_features(cloud, kps, cfg, up, ground)
        print(f"  {src.model_id}: {cloud.size} pts, {kps.size} key points, {len(feats)} features")
        assert kps.size > 0 and len(feats) == kps.size
        models.append((cloud, kps, feats))
    print(f"modeles prétraités depuis le .zip: {len(models)}")
    assert len(models) == 2


if __name__ == "__main__":
    test_build_inline()
    print("OK build (logique identique a database.build_database)")
