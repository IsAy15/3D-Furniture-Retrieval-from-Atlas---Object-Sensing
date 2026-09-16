'\nGlobal keypoint support primitives (paper section 5.1): horizontal and vertical planes and lines. Store directions and sizes; derive the six-component vector (Ah, Av1, Av2, Lh, Lv1, Lv2) for matching filters.\n'

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import PrimitiveConfig


@dataclass
class Primitive:
    kind: str                 # "plane" | "line"
    direction: np.ndarray     # normale (plan) ou direction (line), unitaire
    size: float               # aire (plan) ou longueur (line)
    orientation: str          # "horizontal" | "vertical"


@dataclass
class PrimitiveSet:
    horizontal_plane: Optional[Primitive] = None
    verticals: List[Primitive] = field(default_factory=list)   # 0..2

    def size_vector(self, cfg: PrimitiveConfig) -> np.ndarray:
        'Six-component vector (Ah, Av1, Av2, Lh, Lv1, Lv2) with default bounds.'
        Ah = cfg.plane_min_area
        Lh = cfg.line_min_length
        Av = [cfg.plane_min_area, cfg.plane_min_area]
        Lv = [cfg.line_min_length, cfg.line_min_length]
        if self.horizontal_plane is not None:
            Ah = max(Ah, self.horizontal_plane.size)
        for i, p in enumerate(self.verticals[:2]):
            if p.kind == "plane":
                Av[i] = max(Av[i], p.size)
            else:
                Lv[i] = max(Lv[i], p.size)
        return np.array([Ah, Av[0], Av[1], Lh, Lv[0], Lv[1]], float)

    def directions(self) -> List[np.ndarray]:
        'Candidate directions for descriptor alignment (paper section 5.2).'
        dirs = []
        if self.horizontal_plane is not None:
            dirs.append(self.horizontal_plane.direction)
        for p in self.verticals:
            dirs.append(p.direction)
        return dirs


def _plane_area(points: np.ndarray, normal: np.ndarray) -> float:
    'Approximate area of coplanar points using a 2D convex hull.'
    if len(points) < 3:
        return 0.0
    a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(normal, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    uv = np.column_stack([points @ t1, points @ t2])
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(uv).volume)        # aire en 2D
    except Exception:
        # repli : aire de la boîte englobante 2D
        ext = uv.max(0) - uv.min(0)
        return float(ext[0] * ext[1])


def _grow_planes(points, normals, cfg: PrimitiveConfig, up) -> List[Primitive]:
    'Group points by similar normals to propose planes.'
    n = len(points)
    if n < 6:
        return []
    used = np.zeros(n, bool)
    order = np.arange(n)
    planes: List[Primitive] = []
    cos_n = cfg.plane_grow_normal
    for seed in order:
        if used[seed]:
            continue
        nd = normals[seed]
        same = (np.abs(normals @ nd) > cos_n)       # normales ~ parallèles
        # parmi celles-ci, garder the ~coplanaires (même offset)
        offs = points @ nd
        o0 = offs[seed]
        member = same & (np.abs(offs - o0) < cfg.plane_grow_dist * 3)
        if member.sum() < 6:
            used[seed] = True
            continue
        used[member] = True
        pts = points[member]
        nrm_mean = normals[member].mean(0)
        nrm_mean /= np.linalg.norm(nrm_mean) + 1e-12
        area = _plane_area(pts, nrm_mean)
        if area < cfg.plane_min_area:
            continue
        cosu = abs(float(nrm_mean @ up))
        if cosu > cfg.horizontal_cos:
            orient = "horizontal"
        elif cosu < cfg.vertical_cos:
            orient = "vertical"
        else:
            orient = "oblique"
        planes.append(Primitive("plane", nrm_mean, area, orient))
    return planes


def _detect_line(points, up, cfg: PrimitiveConfig) -> Optional[Primitive]:
    'Detect a dominant line from point PCA.'
    if len(points) < 6:
        return None
    q = points - points.mean(0)
    w, v = np.linalg.eigh(q.T @ q)
    direction = v[:, 2]                              # axe principal
    # élongation : valeur propre dominante très supérieure aux autres
    if w[2] < 1e-9 or w[1] / w[2] > 0.25:
        return None                                  # pas assez linéaire
    proj = q @ direction
    length = float(proj.max() - proj.min())
    if length < cfg.line_min_length:
        return None
    cosu = abs(float(direction @ up))
    orient = "vertical" if cosu > 0.7 else ("horizontal" if cosu < 0.3 else "oblique")
    return Primitive("line", direction / np.linalg.norm(direction), length, orient)


def detect_primitives(neigh_points: np.ndarray, neigh_normals: np.ndarray,
                      cfg: PrimitiveConfig, up) -> PrimitiveSet:
    'Compute keypoint support primitives from neighboring positions and normals.'
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    planes = _grow_planes(neigh_points, neigh_normals, cfg, up)

    horiz = sorted([p for p in planes if p.orientation == "horizontal"],
                   key=lambda p: -p.size)
    vert = sorted([p for p in planes if p.orientation == "vertical"],
                  key=lambda p: -p.size)

    pset = PrimitiveSet()
    if horiz:
        pset.horizontal_plane = horiz[0]
    pset.verticals = vert[:2]

    # Si pas assez de plans verticaux, on tente une line (déclenché si plan échoue).
    if len(pset.verticals) < 2:
        line = _detect_line(neigh_points, up, cfg)
        if line is not None and line.orientation in ("vertical", "oblique"):
            pset.verticals.append(line)
    return pset
