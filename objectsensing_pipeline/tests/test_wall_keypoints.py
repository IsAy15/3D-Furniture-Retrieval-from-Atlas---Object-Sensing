import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import KeypointConfig  # noqa: E402
from geometry import PointCloud  # noqa: E402
from keypoints import KeyPoints  # noqa: E402
from wall_keypoints import (  # noqa: E402
    analyze_wall_planes,
    apply_wall_keypoint_filter,
    detect_dominant_wall_planes,
)


def _wall_with_small_object():
    wall_x = np.linspace(-1.5, 1.5, 55)
    wall_y = np.linspace(0.0, 2.2, 42)
    xx, yy = np.meshgrid(wall_x, wall_y)
    wall = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    wall_normals = np.tile([0.0, 0.0, 1.0], (len(wall), 1))

    object_x = np.linspace(-0.2, 0.2, 10)
    object_y = np.linspace(0.2, 0.9, 16)
    ox, oy = np.meshgrid(object_x, object_y)
    object_face = np.column_stack([
        ox.ravel(), oy.ravel(), np.full(ox.size, 0.45),
    ])
    object_normals = np.tile([0.0, 0.0, 1.0], (len(object_face), 1))

    points = np.vstack([wall, object_face])
    normals = np.vstack([wall_normals, object_normals])
    return PointCloud(points, normals, np.zeros(len(points)))


def test_dominant_wall_detector_rejects_small_vertical_object_face():
    cloud = _wall_with_small_object()
    cfg = KeypointConfig(
        wall_plane_min_points=80,
        wall_plane_min_width=0.8,
        wall_plane_min_height=0.8,
        wall_plane_min_area=1.0,
    )

    planes = detect_dominant_wall_planes(cloud, cfg, [0.0, 1.0, 0.0])

    assert len(planes) == 1
    assert planes[0].support_count > 1000
    assert planes[0].width > 2.5
    assert planes[0].height > 1.8
    assert abs(planes[0].offset) < 0.06


def test_parallel_room_walls_remain_distinct_after_layer_deduplication():
    cloud = _wall_with_small_object()
    opposite = cloud.points[:55 * 42].copy()
    opposite[:, 2] = 2.0
    opposite_normals = np.tile([0.0, 0.0, -1.0], (len(opposite), 1))
    cloud = PointCloud(
        np.vstack([cloud.points, opposite]),
        np.vstack([cloud.normals, opposite_normals]),
        np.zeros(cloud.size + len(opposite)),
    )

    planes = detect_dominant_wall_planes(
        cloud, KeypointConfig(wall_plane_min_points=80), [0.0, 1.0, 0.0],
    )

    assert len(planes) == 2
    assert abs(abs(planes[0].offset) - abs(planes[1].offset)) > 1.5


def test_wall_plane_affinity_is_diagnostic_and_localized():
    cloud = _wall_with_small_object()
    keypoints = KeyPoints(
        positions=np.asarray([[0.8, 1.0, 0.01], [0.0, 0.5, 0.45]]),
        normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        responses=np.asarray([0.02, 0.03]),
        source_index=np.asarray([100, cloud.size - 1]),
    )
    cfg = KeypointConfig(
        wall_plane_min_points=80,
        wall_small_radius=0.12,
        wall_large_radius=0.25,
    )

    analysis = analyze_wall_planes(
        cloud, keypoints, cfg, [0.0, 1.0, 0.0],
    )

    assert len(analysis.planes) == 1
    assert analysis.plane_affinity[0] > 0.80
    assert analysis.plane_affinity[1] < 0.05
    assert analysis.keypoint_plane_index.tolist() == [0, -1]
    assert np.count_nonzero(analysis.surface_plane_index >= 0) > 1000
    assert analysis.small_support[0] > 0.70
    assert analysis.large_support[0] > 0.70
    assert analysis.wall_score[0] > 0.60


def _wall_with_local_protrusion():
    cloud = _wall_with_small_object()
    rng = np.random.default_rng(8)
    protrusion = np.column_stack([
        rng.uniform(-0.10, 0.10, 180),
        rng.uniform(0.40, 0.62, 180),
        rng.uniform(0.10, 0.18, 180),
    ])
    protrusion_normals = np.tile([1.0, 0.0, 0.0], (len(protrusion), 1))
    cloud = PointCloud(
        np.vstack([cloud.points, protrusion]),
        np.vstack([cloud.normals, protrusion_normals]),
        np.zeros(cloud.size + len(protrusion)),
    )
    return cloud


def test_object_protrusion_protects_keypoint_anchored_in_front_of_wall():
    cloud = _wall_with_local_protrusion()
    keypoints = KeyPoints(
        positions=np.asarray([[0.0, 0.5, 0.06]]),
        normals=np.asarray([[0.0, 0.0, 1.0]]),
        responses=np.asarray([0.02]),
        source_index=np.asarray([500]),
    )
    unprotected_cfg = KeypointConfig(
        wall_plane_min_points=80, wall_object_protection=0.0,
        wall_small_radius=0.12,
    )
    protected_cfg = KeypointConfig(
        wall_plane_min_points=80, wall_object_protection=0.9,
        wall_small_radius=0.12,
    )

    unprotected = analyze_wall_planes(
        cloud, keypoints, unprotected_cfg, [0.0, 1.0, 0.0],
    )
    protected = analyze_wall_planes(
        cloud, keypoints, protected_cfg, [0.0, 1.0, 0.0],
    )

    assert protected.protrusion_score[0] > 0.70
    assert protected.discontinuity_score[0] > 0.50
    assert protected.object_anchor_score[0] > 0.70
    assert protected.object_score[0] > 0.50
    assert protected.wall_score[0] < 0.50 * unprotected.wall_score[0]


def test_nearby_object_does_not_protect_keypoint_on_wall():
    cloud = _wall_with_local_protrusion()
    keypoints = KeyPoints(
        positions=np.asarray([[0.0, 0.5, 0.005]]),
        normals=np.asarray([[0.0, 0.0, 1.0]]),
        responses=np.asarray([0.02]),
        source_index=np.asarray([500]),
    )
    cfg = KeypointConfig(
        wall_plane_min_points=80,
        wall_small_radius=0.12,
        wall_large_radius=0.25,
        wall_reject_threshold=0.30,
    )

    result = apply_wall_keypoint_filter(
        cloud, keypoints, cfg, [0.0, 1.0, 0.0],
    )

    assert result.analysis.protrusion_score[0] > 0.70
    assert result.analysis.object_anchor_score[0] < 0.10
    assert result.analysis.object_score[0] < 0.10
    assert result.rejected_mask.tolist() == [True]


def test_wall_filter_rejects_plain_wall_but_keeps_non_wall_keypoint():
    cloud = _wall_with_small_object()
    keypoints = KeyPoints(
        positions=np.asarray([[0.8, 1.0, 0.01], [0.0, 0.5, 0.45]]),
        normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        responses=np.asarray([0.02, 0.03]),
        source_index=np.asarray([100, cloud.size - 1]),
    )
    cfg = KeypointConfig(
        wall_plane_min_points=80,
        wall_small_radius=0.12,
        wall_large_radius=0.25,
        wall_reject_threshold=0.30,
    )

    result = apply_wall_keypoint_filter(
        cloud, keypoints, cfg, [0.0, 1.0, 0.0],
    )

    assert result.rejected_mask.tolist() == [True, False]
    assert result.keypoints.source_index.tolist() == [cloud.size - 1]
    assert result.keypoints.responses.tolist() == [0.03]
    assert result.keypoints.selection_scores[0] == 0.03
    assert result.keypoints.wall_plane_indices.tolist() == [-1]
    assert result.keypoints.wall_affinities[0] < 0.05
    assert result.keypoints.wall_proximities[0] < 0.05


def test_disabling_wall_filter_is_exact_passthrough():
    cloud = _wall_with_small_object()
    keypoints = KeyPoints(
        positions=np.asarray([[0.8, 1.0, 0.01]]),
        normals=np.asarray([[0.0, 0.0, 1.0]]),
        responses=np.asarray([0.02]),
        source_index=np.asarray([100]),
    )

    result = apply_wall_keypoint_filter(
        cloud, keypoints, KeypointConfig(wall_filter_enabled=False),
        [0.0, 1.0, 0.0],
    )

    assert result.analysis is None
    assert not result.rejected_mask.any()
    assert np.array_equal(result.keypoints.positions, keypoints.positions)
    assert np.array_equal(
        result.keypoints.selection_scores, keypoints.responses,
    )
    assert result.keypoints.wall_plane_indices.tolist() == [-1]
    assert result.keypoints.wall_proximities.tolist() == [0.0]
