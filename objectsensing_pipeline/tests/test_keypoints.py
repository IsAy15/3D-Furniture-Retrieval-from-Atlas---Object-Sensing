"""Test synthétique : sur un cube échantillonné, the key points doivent se
concentrer près des 8 coins (3 faces -> 3 directions de normales -> Harris fort)."""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry as G  # noqa: E402
from config import KeypointConfig  # noqa: E402
from keypoints import (  # noqa: E402
    _covariance_normals,
    _convex_hull_area_2d,
    _has_large_planar_support,
    _harris_response,
    _passes_2d_corner,
    _prefilter_2d_candidates,
    detect_keypoints,
)


def sample_cube_surface(n_per_face=4000, size=1.0, seed=0):
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


def test_cube_corners():
    pts = sample_cube_surface()
    cloud = G.make_point_cloud(pts, radius=0.08)
    cfg = KeypointConfig(neighbor_radius=0.12, curvature_threshold=0.03,
                         nms_radius=0.25, dedup_radius=0.15)
    kps = detect_keypoints(cloud, cfg)
    corners = np.array([[sx, sy, sz] for sx in (-0.5, 0.5)
                        for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)])
    print(f"key points detectes: {kps.size}")
    assert kps.size >= 6, f"trop peu de key points: {kps.size}"
    # chaque coin doit avoir un key point proche
    from scipy.spatial import cKDTree
    tree = cKDTree(kps.positions)
    d, _ = tree.query(corners)
    print("distance coin -> key point la plus proche:", np.round(d, 3))
    covered = (d < 0.2).sum()
    print(f"coins couverts: {covered}/8")
    assert covered >= 7, f"coins couverts insuffisant: {covered}/8"
    # pas trop de key points parasites (sur the faces planes)
    assert kps.size <= 40, f"trop de key points parasites: {kps.size}"


def test_paper_2d_corner_area_threshold():
    cfg = KeypointConfig()
    assert np.isclose(cfg.convex_hull_ratio, math.pi / 3.0)

    grid = np.linspace(0.0, 0.5, 4)
    xx, yy = np.meshgrid(grid, grid)
    partial_plane = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    assert _passes_2d_corner(
        np.zeros(3), np.array([0.0, 0.0, 1.0]),
        partial_plane, radius=1.0, hull_ratio=cfg.convex_hull_ratio,
    )

    angle = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    full_plane = np.column_stack([
        0.8 * np.cos(angle), 0.8 * np.sin(angle), np.zeros_like(angle),
    ])
    assert not _passes_2d_corner(
        np.zeros(3), np.array([0.0, 0.0, 1.0]),
        full_plane, radius=1.0, hull_ratio=cfg.convex_hull_ratio,
    )


def test_2d_corner_requires_large_planar_support():
    cfg = KeypointConfig()
    grid = np.linspace(-0.1, 0.1, 9)
    xx, yy = np.meshgrid(grid, grid)
    plane = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    normals = np.tile([0.0, 0.0, 1.0], (len(plane), 1))
    assert _has_large_planar_support(
        np.zeros(3), normals[0], plane, normals,
        radius=0.15, min_area=cfg.corner_plane_min_area,
        normal_cos=cfg.corner_plane_normal_cos,
    )

    tiny = plane * 0.15
    assert not _has_large_planar_support(
        np.zeros(3), normals[0], tiny, normals,
        radius=0.15, min_area=cfg.corner_plane_min_area,
        normal_cos=cfg.corner_plane_normal_cos,
    )


def test_harris_response_is_density_invariant():
    base = np.eye(3)
    sparse = np.repeat(base, 2, axis=0)
    dense = np.repeat(base, 20, axis=0)
    sparse_response = _harris_response(_covariance_normals(sparse), 0.04)
    dense_response = _harris_response(_covariance_normals(dense), 0.04)
    assert np.isclose(sparse_response, dense_response)
    assert sparse_response > 0.008

    planar = np.tile([0.0, 0.0, 1.0], (60, 1))
    assert _harris_response(_covariance_normals(planar), 0.04) < 0.008


def test_2d_prefilter_keeps_best_responses_per_cell():
    positions = np.array([
        [0.01, 0.01, 0.01],
        [0.02, 0.01, 0.01],
        [0.03, 0.01, 0.01],
        [0.20, 0.01, 0.01],
    ])
    responses = np.array([-3.0, -1.0, -2.0, -4.0])
    selected = _prefilter_2d_candidates(
        np.arange(4), positions, responses, cell_size=0.05, per_cell=2,
    )
    assert set(selected) == {1, 2, 3}


def test_in_memory_convex_hull_area():
    square = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.5, 0.5], [0.0, 0.0],
    ])
    assert np.isclose(_convex_hull_area_2d(square), 1.0)
    assert _convex_hull_area_2d(np.array([[0.0, 0.0], [1.0, 0.0]])) == 0.0


def test_debug_trace_preserves_detector_result():
    pts = sample_cube_surface(n_per_face=600)
    cloud = G.make_point_cloud(pts, radius=0.08)
    cfg = KeypointConfig(
        neighbor_radius=0.12, curvature_threshold=0.03,
        nms_radius=0.25, dedup_radius=0.15,
    )

    reference = detect_keypoints(cloud, cfg)
    trace = {}
    instrumented = detect_keypoints(cloud, cfg, trace=trace)

    assert np.allclose(instrumented.positions, reference.positions)
    assert np.array_equal(instrumented.source_index, reference.source_index)
    assert trace["schema_version"] == 1
    assert trace["counts"]["input"] == len(pts)
    assert trace["counts"]["final"] == instrumented.size
    assert np.array_equal(trace["final_source_index"], instrumented.source_index)
    assert set(trace["adjustment_rejected"]) == {
        "neighbors", "singular", "non_finite", "jitter",
    }


if __name__ == "__main__":
    test_cube_corners()
    print("OK keypoints")
