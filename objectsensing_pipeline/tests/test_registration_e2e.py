"Test d'intégration bout-en-bout : on place un modèle (forme en L, asymétrique)\ndans une scene avec une transformation sol connue, puis on la recouvre via\nmatching -> 1-Point RANSAC -> vérification géométrique."
import os
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry as G  # noqa: E402
from config import PipelineConfig  # noqa: E402
from keypoints import detect_keypoints  # noqa: E402
from matching import build_model_features, build_scan_features, match_features  # noqa: E402
from constellations import one_point_ransac  # noqa: E402
from verify import verify_model  # noqa: E402
from transforms import GroundTransform  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


class SimpleTSDF:
    """Mini-TSDF auto-suffisant (interface .sample compatible avec build_scan_descriptor)."""

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
        labels[(w > 0) & (t > 0.75 * self.voxel)] = 1
        labels[(w > 0) & (np.abs(t) <= 0.75 * self.voxel)] = 2
        return labels


def box_faces(lo, hi, n, rng):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    pts = []
    for axis in range(3):
        for val in (lo[axis], hi[axis]):
            uv = rng.uniform(0, 1, (n, 2))
            p = np.zeros((n, 3))
            others = [a for a in range(3) if a != axis]
            for k, a in enumerate(others):
                p[:, a] = lo[a] + uv[:, k] * (hi[a] - lo[a])
            p[:, axis] = val
            pts.append(p)
    return np.vstack(pts)


def inside(pts, lo, hi, eps=0.01):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return np.all((pts > lo + eps) & (pts < hi - eps), axis=1)


def sample_L(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    A = ([0, 0, 0], [0.6, 0.5, 0.2])
    B = ([0, 0, 0], [0.2, 0.5, 0.6])
    pa = box_faces(*A, n, rng); pa = pa[~inside(pa, *B)]
    pb = box_faces(*B, n, rng); pb = pb[~inside(pb, *A)]
    return np.vstack([pa, pb])


def angle_diff(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def test_recover_transform():
    cfg = PipelineConfig()
    # Ce test cible la registration géométrique avec son SimpleTSDF historique.
    # La calibration en cellules est couverte séparément par test_descriptors.
    cfg.descriptor.distance_unit = 1.0
    cfg.ransac.desc_inlier = 128.0
    cfg.matching.descriptor_ratio_threshold = 0.0
    cfg.keypoint.neighbor_radius = 0.07
    cfg.keypoint.nms_radius = 0.12
    cfg.keypoint.dedup_radius = 0.08
    cfg.keypoint.curvature_threshold = 0.03
    up = np.array([0.0, 1.0, 0.0])

    mpts = sample_L(seed=0)
    model = G.make_point_cloud(mpts, radius=0.06)
    mkps = detect_keypoints(model, cfg.keypoint)
    ground_m = mpts[:, 1].min()
    mfeats = build_model_features(model, mkps, cfg, up, ground_m)
    print(f"modele: {model.size} pts, {mkps.size} key points")

    T0 = GroundTransform(theta=0.7, scale=1.15, t=np.array([1.2, 0.0, -0.6]), up=up)
    R = G.rotation_about_axis(up, T0.theta)
    spts = T0.apply(mpts)
    scene = G.make_point_cloud(spts, radius=0.06)
    ground_s = spts[:, 1].min()
    stsdf = SimpleTSDF(spts, scene.normals, voxel=0.01, trunc=0.04)
    skps = detect_keypoints(scene, cfg.keypoint)
    sfeats = build_scan_features(scene, stsdf, skps, cfg, up, ground_s)
    print(f"scene : {scene.size} pts, {skps.size} key points")

    corres = match_features(mfeats, sfeats, cfg, up)
    print(f"correspondences putatives: {len(corres)}")
    cons = one_point_ransac(corres, mfeats, sfeats, cfg, up)
    print(f"constellations: {len(cons)}")
    reg = verify_model("L", model, scene, cons, threshold=0.05)
    assert reg is not None, "none registration trouvee"
    T = reg.transform
    print(f"VRAI : theta={T0.theta:.3f} scale={T0.scale:.3f} t={np.round(T0.t,3)}")
    print(f"ESTIM: theta={T.theta:.3f} scale={T.scale:.3f} t={np.round(T.t,3)}")
    print(f"couverture={reg.coverage:.2f} dist_surface={reg.mean_surface_dist*1000:.1f}mm")

    assert reg.coverage > 0.8, f"couverture trop faible: {reg.coverage}"
    assert angle_diff(T.theta, T0.theta) < np.radians(8), "theta mal recouvre"
    assert abs(T.scale - T0.scale) < 0.12, "echelle mal recouvree"
    assert np.linalg.norm(T.t - T0.t) < 0.1, "translation mal recouvree"


if __name__ == "__main__":
    test_recover_transform()
    print("OK pipeline registration")
