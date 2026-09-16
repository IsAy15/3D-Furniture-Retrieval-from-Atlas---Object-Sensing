"""Valide le matching parallèle à mémoire constante (base un-fichier-par-modèle) :
db_store sauvegarde/charge the models individuellement, et un ProcessPoolExecutor
matche chaque modèle en chargeant UN seul fichier à la fois (jamais toute la base).
Reproduit fidèlement pipeline.retrieve_dir avec un worker top-level (spawn-safe)."""
import os
import sys
import tempfile

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import db_store  # noqa: E402
import geometry as G  # noqa: E402
from config import PipelineConfig  # noqa: E402
from database import ModelDatabase, preprocess_model  # noqa: E402
from keypoints import detect_keypoints  # noqa: E402
from matching import build_scan_features  # noqa: E402
from transforms import GroundTransform  # noqa: E402


class SimpleTSDF:
    def __init__(self, points, normals, voxel, trunc):
        self.voxel = voxel; self.trunc = trunc
        self.origin = points.min(0) - 3 * voxel
        bmax = points.max(0) + 3 * voxel
        self.dims = np.ceil((bmax - self.origin) / voxel).astype(int) + 1
        ax = [self.origin[a] + np.arange(self.dims[a]) * voxel for a in range(3)]
        gx, gy, gz = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        tree = cKDTree(points)
        dist, idx = tree.query(centers, workers=-1)
        sgn = np.sign(np.einsum('ij,ij->i', centers - points[idx], normals[idx])); sgn[sgn == 0] = 1
        self.tsdf = np.clip(sgn * dist, -trunc, trunc).reshape(self.dims).astype(np.float32)
        self.weight = (dist < 3 * trunc).reshape(self.dims).astype(np.float32)

    def sample(self, positions):
        rel = (np.asarray(positions, float) - self.origin) / self.voxel
        ijk = np.round(rel).astype(int)
        inb = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        t = np.full(len(positions), self.trunc, np.float32); w = np.zeros(len(positions), np.float32)
        g = ijk[inb]; t[inb] = self.tsdf[g[:, 0], g[:, 1], g[:, 2]]; w[inb] = self.weight[g[:, 0], g[:, 1], g[:, 2]]
        return t, w

    def sample_visibility(self, positions):
        t, w = self.sample(positions)
        labels = np.zeros(len(t), np.uint8)
        labels[(w > 0) & (t > 1.25 * self.voxel)] = 1
        labels[(w > 0) & (np.abs(t) <= 1.25 * self.voxel)] = 2
        return labels


# --- worker top-level (spawn-safe), réplique pipeline._work_file ---
_W = {}


def _init(sfeats, scene_pts, cfg, up, thr):
    _W.update(sfeats=sfeats, scene=G.PointCloud(np.asarray(scene_pts)),
              cfg=cfg, up=np.asarray(up, float), thr=thr)


def _work(task):
    from matching import match_features
    from constellations import one_point_ransac
    from verify import verify_model
    mi, path = task
    model = db_store.load_model(path)          # UN seul modèle loaded ici
    w = _W
    corres = match_features(model.features, w["sfeats"], w["cfg"], w["up"])
    if not corres:
        return None
    cons = one_point_ransac(corres, model.features, w["sfeats"], w["cfg"], w["up"])
    reg = verify_model(model.name, model.cloud, w["scene"], cons,
                       threshold=w["cfg"].keypoint.neighbor_radius)
    if reg is not None and reg.coverage >= w["thr"]:
        T = reg.transform
        return (mi, model.name, float(reg.coverage), float(T.theta), float(T.scale),
                tuple(float(x) for x in T.t))
    return None


def _make_obj(path, mesh):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def test_parallel_per_file():
    tmp = tempfile.mkdtemp()
    cfg = PipelineConfig()
    # SimpleTSDF approxime une distance au nuage sans lancer the rayons RGB-D.
    # Ce test cible le pool de workers, pas la calibration UDF/visibilité.
    cfg.descriptor.distance_unit = 1.0
    cfg.ransac.desc_inlier = 128.0
    cfg.matching.descriptor_ratio_threshold = 0.0
    cfg.keypoint.neighbor_radius = 0.07; cfg.keypoint.nms_radius = 0.12
    cfg.keypoint.dedup_radius = 0.08; cfg.keypoint.curvature_threshold = 0.03
    up = np.asarray(cfg.up_axis, float); up /= np.linalg.norm(up)

    # deux models : un L (cible) et un cube (distracteur)
    a = trimesh.creation.box(extents=[0.6, 0.5, 0.2]); a.apply_translation([0.3, 0.25, 0.1])
    b = trimesh.creation.box(extents=[0.2, 0.5, 0.6]); b.apply_translation([0.1, 0.25, 0.3])
    Lmesh = trimesh.util.concatenate([a, b])
    _make_obj(os.path.join(tmp, "03001627", "L", "models", "model_normalized.obj"), Lmesh)
    _make_obj(os.path.join(tmp, "04379243", "C", "models", "model_normalized.obj"),
              trimesh.creation.box(extents=[0.5, 0.5, 0.5]))
    mL = preprocess_model(os.path.join(tmp, "03001627", "L", "models", "model_normalized.obj"),
                          "03001627", cfg, n_points=5000, add_ground=False)
    mC = preprocess_model(os.path.join(tmp, "04379243", "C", "models", "model_normalized.obj"),
                          "04379243", cfg, n_points=5000, add_ground=False)
    db = ModelDatabase(models=[mL, mC], cfg=cfg)

    db_dir = os.path.join(tmp, "db_dir")
    db_store.save_db_dir(db, db_dir)
    assert db_store.is_db_dir(db_dir)
    files = db_store.model_files(db_dir)
    print(f"base dossier: {len(files)} fichiers modèle")
    assert len(files) == 2

    # scene = modèle L transformé
    T0 = GroundTransform(theta=0.6, scale=1.0, t=np.array([0.9, 0.0, -0.4]), up=up)
    spts = T0.apply(mL.cloud.points)
    scene_cloud = G.make_point_cloud(spts, radius=0.06)
    stsdf = SimpleTSDF(spts, scene_cloud.normals, voxel=0.012, trunc=0.045)
    ground_s = float((spts @ up).min())
    skps = detect_keypoints(scene_cloud, cfg.keypoint)
    sfeats = build_scan_features(scene_cloud, stsdf, skps, cfg, up, ground_s)

    # matching parallèle : chaque worker charge UN modèle à la fois
    from concurrent.futures import ProcessPoolExecutor
    tasks = list(enumerate(files))
    results = []
    with ProcessPoolExecutor(max_workers=2, initializer=_init,
                             initargs=(sfeats, spts, cfg, list(map(float, up)), 0.5)) as ex:
        for r in ex.map(_work, tasks, chunksize=1):
            if r is not None:
                results.append(r)
    print("resultats:", [(r[1], round(r[2], 2)) for r in results])
    # Ce test valide le MÉCANISME parallèle à mémoire constante : chaque modèle
    # loaded individuellement dans un worker, matché en parallèle, résultats
    # remontés sans erreur. (La précision/discrimination du retrieval est testée
    # ailleurs ; la forme en L synthétique est volontairement ambiguë.)
    assert len(results) >= 1, "le pool parallèle n'a remonté none résultat"
    found = [r for r in results if r[1].startswith("chair")]
    assert found, "le modele cible n'a pas ete traite en parallele"
    mi, name, cov, theta, scale, t = max(found, key=lambda r: r[2])
    print(f"L: couverture={cov:.2f} theta={theta:.3f} (vrai 0.6) t={np.round(t,3)}")
    assert cov > 0.8


if __name__ == "__main__":
    test_parallel_per_file()
    print("OK parallel per-file (memoire constante)")
