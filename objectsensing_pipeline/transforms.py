'\nModel-to-scene transformations constrained by the ground plane: yaw around up, translation and uniform scale.\n'

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import rotation_about_axis


@dataclass
class GroundTransform:
    theta: float            # rotation horizontale (rad)
    scale: float            # scale uniforme
    t: np.ndarray           # translation (3,)
    up: np.ndarray          # axe vertical

    def matrix(self) -> np.ndarray:
        R = rotation_about_axis(self.up, self.theta) * self.scale
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = self.t
        return M

    def apply(self, points: np.ndarray) -> np.ndarray:
        R = rotation_about_axis(self.up, self.theta)
        return self.scale * (points @ R.T) + self.t


def estimate_from_correspondence(model_kp, scan_kp, theta, scale, up) -> GroundTransform:
    'Translation aligning a rotated/scaled model keypoint with a scan keypoint: t = Kj - s R_theta Ki.'
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    R = rotation_about_axis(up, theta)
    t = np.asarray(scan_kp, float) - scale * (R @ np.asarray(model_kp, float))
    return GroundTransform(theta, scale, t, up)


def refine_transform(model_pts, scan_pts, weights, up, fixed_scale=None,
                     scale_bounds=None) -> GroundTransform:
    'Weighted least-squares refinement of yaw, scale and translation from paired keypoints. Estimate horizontal rotation with weighted 2D Procrustes, scale from radii and translation from weighted centroids.\n    '
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    w = np.asarray(weights, float)
    w = w / (w.sum() + 1e-12)
    mc = (model_pts * w[:, None]).sum(0)
    sc = (scan_pts * w[:, None]).sum(0)
    M = model_pts - mc
    S = scan_pts - sc
    # base horizontale
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    Mh = np.column_stack([M @ e1, M @ e2])      # (N,2)
    Sh = np.column_stack([S @ e1, S @ e2])
    # theta par Procrustes 2D pondéré
    A = (w[:, None] * Sh).T @ Mh                # 2x2
    num = A[1, 0] - A[0, 1]
    den = A[0, 0] + A[1, 1]
    theta = float(np.arctan2(num, den))
    # scale : rapport des rayons horizontaux pondérés
    rm = np.sqrt((w * (Mh ** 2).sum(1)).sum())
    rs = np.sqrt((w * (Sh ** 2).sum(1)).sum())
    scale = float(rs / rm) if rm > 1e-9 else 1.0
    if fixed_scale is not None:
        scale = fixed_scale
    if scale_bounds is not None:
        scale = float(np.clip(scale, scale_bounds[0], scale_bounds[1]))
    R = rotation_about_axis(up, theta)
    t = sc - scale * (R @ mc)
    return GroundTransform(theta, scale, t, up)
