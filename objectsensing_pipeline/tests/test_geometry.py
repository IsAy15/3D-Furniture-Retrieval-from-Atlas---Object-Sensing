"""Tests synthétiques de geometry.py : courbure faible sur the faces, forte sur
the arêtes/coins d'un cube ; estimation du plan du sol."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry as G  # noqa: E402


def sample_cube_surface(n_per_face=2000, size=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    h = size / 2
    for axis in range(3):
        for sign in (-h, h):
            uv = rng.uniform(-h, h, size=(n_per_face, 2))
            p = np.zeros((n_per_face, 3))
            others = [a for a in range(3) if a != axis]
            p[:, others[0]] = uv[:, 0]
            p[:, others[1]] = uv[:, 1]
            p[:, axis] = sign
            pts.append(p)
    return np.vstack(pts)


def test_curvature_edges_vs_faces():
    pts = sample_cube_surface()
    normals, curv = G.estimate_normals_curvature(pts, radius=0.08)
    # distance d'un point à l'arête la plus proche du cube (min sur 2 des 3 coords ~ 0.5)
    a = np.abs(pts)
    # un point d'arête a deux coords proches de 0.5 ; un point de face une seule
    near_half = (np.abs(a - 0.5) < 0.06).sum(axis=1)
    face_mask = near_half <= 1
    edge_mask = near_half >= 2
    cf = curv[face_mask].mean()
    ce = curv[edge_mask].mean()
    print(f"curvature face={cf:.4f}  edge={ce:.4f}  (edge doit etre >> face)")
    assert cf < 0.02, f"courbure de face trop haute: {cf}"
    assert ce > cf * 3, f"courbure d'arete pas assez marquee: {ce} vs {cf}"


def test_ground_plane():
    rng = np.random.default_rng(1)
    floor = np.column_stack([rng.uniform(-2, 2, 5000), np.full(5000, -1.0), rng.uniform(-2, 2, 5000)])
    obj = np.column_stack([rng.uniform(-0.3, 0.3, 800), rng.uniform(-1.0, 0.0, 800), rng.uniform(-0.3, 0.3, 800)])
    pts = np.vstack([floor, obj])
    up, ground = G.estimate_ground_plane(pts, up_hint=(0, 1, 0))
    print(f"ground height estimee={ground:.3f} (attendu ~ -1.0)")
    assert abs(ground + 1.0) < 0.1


if __name__ == "__main__":
    test_curvature_edges_vs_faces()
    test_ground_plane()
    print("OK geometry")
