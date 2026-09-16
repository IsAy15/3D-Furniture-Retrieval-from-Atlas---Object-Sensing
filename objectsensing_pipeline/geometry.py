'\nShared geometric structures and utilities: point clouds, neighborhood PCA normals/curvature and KD-trees for scans and sampled database models.\n'

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from parallelism import kdtree_workers


@dataclass
class PointCloud:
    points: np.ndarray                 # (N, 3) float64
    normals: Optional[np.ndarray] = None      # (N, 3) unit
    curvature: Optional[np.ndarray] = None    # (N,) surface variation [0, 1/3]
    confidence: Optional[np.ndarray] = None   # (N,) poids d'observation TSDF
    colors: Optional[np.ndarray] = None       # (N, 3) RGB normalise [0, 1]
    _tree: Optional[cKDTree] = None

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float64)
        if self.curvature is not None:
            self.curvature = np.ascontiguousarray(self.curvature, dtype=np.float64)
        if self.confidence is not None:
            self.confidence = np.ascontiguousarray(
                self.confidence, dtype=np.float64,
            ).reshape(-1)
            if len(self.confidence) != len(self.points):
                raise ValueError('confidence must contain one value per point')
        if self.colors is not None:
            self.colors = np.ascontiguousarray(
                self.colors, dtype=np.float64,
            ).reshape(-1, 3)
            if len(self.colors) != len(self.points):
                raise ValueError('colors must contain one RGB color per point')
            self.colors = np.clip(self.colors, 0.0, 1.0)

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    def neighbors(self, idx_or_point, radius: float):
        'Neighbor indices within a radius around a point or point index.'
        p = self.points[idx_or_point] if np.isscalar(idx_or_point) else np.asarray(idx_or_point)
        return self.tree.query_ball_point(p, radius)

    def subset(self, idx) -> "PointCloud":
        idx = np.asarray(idx)
        colors = getattr(self, "colors", None)
        return PointCloud(
            self.points[idx],
            None if self.normals is None else self.normals[idx],
            None if self.curvature is None else self.curvature[idx],
            None if self.confidence is None else self.confidence[idx],
            None if colors is None else colors[idx],
        )


def estimate_normals_curvature(
    points: np.ndarray,
    radius: float = 0.05,
    min_neighbors: int = 6,
    orient_up: Optional[np.ndarray] = None,
):
    'Neighborhood PCA normals and surface variation: smallest eigenvalue divided by the sum of eigenvalues. Values approach zero on planes and can reach one third in isotropic neighborhoods.\n    '
    points = np.ascontiguousarray(points, dtype=np.float64)
    n = len(points)
    tree = cKDTree(points)
    normals = np.zeros((n, 3))
    curvature = np.zeros(n)
    # Query groupée des voisins.
    neigh = tree.query_ball_point(points, radius, workers=kdtree_workers())
    for i in range(n):
        idx = neigh[i]
        if len(idx) < min_neighbors:
            normals[i] = (0.0, 0.0, 1.0)
            curvature[i] = 0.0
            continue
        q = points[idx]
        q = q - q.mean(0)
        cov = (q.T @ q) / len(idx)
        w, v = np.linalg.eigh(cov)          # valeurs propres croissantes
        normals[i] = v[:, 0]                 # vecteur propre de plus petite valeur
        s = w.sum()
        curvature[i] = (w[0] / s) if s > 1e-12 else 0.0
    # Orientation cohérente des normales (vers le haut/observateur si fourni).
    if orient_up is not None:
        flip = (normals @ np.asarray(orient_up, float)) < 0
        normals[flip] *= -1
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.clip(norm, 1e-12, None)
    return normals, curvature


def make_point_cloud(points, radius=0.05, normals=None, orient_up=None) -> PointCloud:
    'Build a PointCloud, estimating normals and curvature when absent.'
    points = np.ascontiguousarray(points, dtype=np.float64)
    if normals is None:
        normals, curvature = estimate_normals_curvature(points, radius=radius, orient_up=orient_up)
    else:
        # courbure quand même utile pour la détection de key points
        _, curvature = estimate_normals_curvature(points, radius=radius)
        normals = np.ascontiguousarray(normals, dtype=np.float64)
        normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)
    return PointCloud(points, normals, curvature)


def estimate_ground_plane(points: np.ndarray, up_hint=(0, 1, 0), bins: int = 200):
    'Estimate ground using the vertical hint and a lower height-density peak. Return (unit_up, ground_height) for horizontal transformation constraints.\n    '
    up = np.asarray(up_hint, float)
    up = up / np.linalg.norm(up)
    h = points @ up
    hist, edges = np.histogram(h, bins=bins)
    # Le sol = grande couche dense dans le quart inférieur des hauteurs.
    lo = np.searchsorted(edges, np.percentile(h, 2))
    hi = np.searchsorted(edges, np.percentile(h, 40))
    lo = max(lo, 0); hi = max(hi, lo + 1)
    k = lo + int(np.argmax(hist[lo:hi]))
    ground = 0.5 * (edges[k] + edges[k + 1])
    return up, float(ground)


def detect_gravity_rotation(points: np.ndarray, iters: int = 1000,
                            thr: float = 0.03, seed: int = 0,
                            up_hint: Optional[np.ndarray] = None) -> np.ndarray:
    'Estimate a 3x3 rotation to Y-up from the dominant ground plane. Orient the RANSAC plane normal toward the centroid and align it with +Y.\n    '
    rng = np.random.default_rng(seed)
    prior = None
    if up_hint is not None:
        prior = np.asarray(up_hint, float)
        magnitude = np.linalg.norm(prior)
        if magnitude > 1e-9:
            prior = prior / magnitude
        else:
            prior = None
    p = points if len(points) <= 6000 else points[rng.choice(len(points), 6000, replace=False)]
    cen = p.mean(0)
    best = None
    for _ in range(iters):
        i = rng.choice(len(p), 3, replace=False)
        a, b, c = p[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        alignment = 1.0
        if prior is not None:
            alignment = abs(float(n @ prior))
            if alignment < math.cos(math.radians(40.0)):
                continue
        d = n @ a
        inl = int((np.abs(p @ n - d) < thr).sum())
        # léger bonus aux plans dont le nuage est majoritairement d'un côté (sol/plafond)
        side = abs(float(np.sign(p @ n - d).mean()))
        score = inl * (0.5 + side) * alignment
        if best is None or score > best[0]:
            best = (score, n, d)
    if best is None:
        n = prior if prior is not None else np.array([0.0, 1.0, 0.0])
        d = 0.0
    else:
        _, n, d = best
    if prior is not None:
        if n @ prior < 0:
            n = -n
    elif (cen @ n - d) < 0:             # normale orientée vers le nuage
        n = -n
    # Rotation amenant n -> +Y tout en conservant un cap horizontal stable.
    # La formule de Rodrigues devient mal conditionnée lorsque n est presque
    # opposée à +Y : de petites variations de normale provoquent alors un grand
    # changement de lacet. Une base explicite supprime cette ambiguïté.
    y = np.array([0.0, 1.0, 0.0])
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ n)) > 0.95:
        reference = np.array([0.0, 0.0, 1.0])
    source_horizontal = reference - (reference @ n) * n
    source_horizontal /= np.linalg.norm(source_horizontal)
    source_side = np.cross(source_horizontal, n)
    target_horizontal = reference - (reference @ y) * y
    target_horizontal /= np.linalg.norm(target_horizontal)
    target_side = np.cross(target_horizontal, y)
    source_basis = np.column_stack([
        source_horizontal, n, source_side,
    ])
    target_basis = np.column_stack([
        target_horizontal, y, target_side,
    ])
    return target_basis @ source_basis.T


def rotation_about_axis(axis, angle: float) -> np.ndarray:
    """3x3 rotation matrix around a unit axis (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = a
    return np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
