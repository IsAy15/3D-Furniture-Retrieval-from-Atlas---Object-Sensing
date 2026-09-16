'Test database + retrieve : on prétraite un maillage (forme en L) comme un\nmodèle ShapeNet (chargement + virtual scan + key points + descripteurs), on le\nplace dans une scene avec une transformation connue, et on vérifie que retrieve()\nle retrouve et le recale correctement.'
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import geometry as G  # noqa: E402
from config import PipelineConfig  # noqa: E402
from database import ModelDatabase, preprocess_model  # noqa: E402
from pipeline import SceneScan, retrieve  # noqa: E402
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
        sgn = np.sign(np.einsum('ij,ij->i', centers - points[idx], normals[idx]))
        sgn[sgn == 0] = 1.0
        self.tsdf = np.clip(sgn * dist, -trunc, trunc).reshape(self.dims).astype(np.float32)
        self.weight = (dist < 3 * trunc).reshape(self.dims).astype(np.float32)

    def sample(self, positions):
        rel = (np.asarray(positions, float) - self.origin) / self.voxel
        ijk = np.round(rel).astype(int)
        inb = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        t = np.full(len(positions), self.trunc, np.float32)
        w = np.zeros(len(positions), np.float32)
        g = ijk[inb]
        t[inb] = self.tsdf[g[:, 0], g[:, 1], g[:, 2]]
        w[inb] = self.weight[g[:, 0], g[:, 1], g[:, 2]]
        return t, w

    def sample_visibility(self, positions):
        t, w = self.sample(positions)
        labels = np.zeros(len(t), np.uint8)
        labels[(w > 0) & (t > 1.25 * self.voxel)] = 1
        labels[(w > 0) & (np.abs(t) <= 1.25 * self.voxel)] = 2
        return labels


def make_L_mesh(path):
    a = trimesh.creation.box(extents=[0.6, 0.5, 0.2])
    a.apply_translation([0.3, 0.25, 0.1])
    b = trimesh.creation.box(extents=[0.2, 0.5, 0.6])
    b.apply_translation([0.1, 0.25, 0.3])
    mesh = trimesh.util.concatenate([a, b])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def angle_diff(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def test_database_retrieve():
    cfg = PipelineConfig()
    # SimpleTSDF approxime une distance au nuage sans lancer the rayons RGB-D.
    # Ce test cible la récupération de pose, pas la calibration UDF/visibilité.
    cfg.descriptor.distance_unit = 1.0
    cfg.ransac.desc_inlier = 128.0
    cfg.matching.descriptor_ratio_threshold = 0.0
    cfg.keypoint.neighbor_radius = 0.07
    cfg.keypoint.nms_radius = 0.12
    cfg.keypoint.dedup_radius = 0.08
    cfg.keypoint.curvature_threshold = 0.03
    cfg.matching.floor_height = 0.0   # scene synthetique sans sol (objet pose a y=0)
    up = np.asarray(cfg.up_axis, float)

    obj = os.path.join(HERE, "..", "data", "03001627", "Lmodel", "models", "model_normalized.obj")
    make_L_mesh(obj)
    model = preprocess_model(obj, "03001627", cfg, n_points=6000, add_ground=False)
    assert model is not None and model.keypoints.size > 0
    db = ModelDatabase(models=[model], cfg=cfg)
    print(f"modele DB: {model.cloud.size} pts, {model.keypoints.size} key points, "
          f"hauteur ~{model.cloud.points[:,1].max():.2f} m")

    # scene = modèle transformé (transformation sol connue)
    T0 = GroundTransform(theta=-0.5, scale=1.0, t=np.array([0.8, 0.0, 0.3]), up=up)
    spts = T0.apply(model.cloud.points)
    scene_cloud = G.make_point_cloud(spts, radius=0.06)
    stsdf = SimpleTSDF(spts, scene_cloud.normals, voxel=0.012, trunc=0.045)
    ground_s = float((spts @ (up / np.linalg.norm(up))).min())
    scene = SceneScan(scene_cloud, stsdf, ground_s, up / np.linalg.norm(up))

    regs = retrieve(scene, db, cfg, coverage_threshold=0.5)
    print(f"objets retrouves: {len(regs)}")
    assert len(regs) >= 1, "none modele retrouve"
    r = regs[0]
    T = r.transform
    print(f"VRAI : theta={T0.theta:.3f} scale={T0.scale:.3f} t={np.round(T0.t,3)}")
    print(f"ESTIM: theta={T.theta:.3f} scale={T.scale:.3f} t={np.round(T.t,3)}")
    print(f"couverture={r.coverage:.2f}")
    assert r.coverage > 0.8
    assert angle_diff(T.theta, T0.theta) < np.radians(10)
    assert np.linalg.norm(T.t - T0.t) < 0.12


if __name__ == "__main__":
    test_database_retrieve()
    print("OK database + retrieve")
