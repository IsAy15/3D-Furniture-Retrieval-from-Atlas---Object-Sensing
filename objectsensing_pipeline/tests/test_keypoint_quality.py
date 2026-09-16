from types import SimpleNamespace

import numpy as np

from config import PipelineConfig
from descriptors import LocalDescriptor
from geometry import PointCloud
from keypoint_quality import filter_scene_keypoints
from keypoints import KeyPoints
from matching import _filter_scan_feature_quality


def _keypoints(positions, responses=None):
    positions = np.asarray(positions, float)
    count = len(positions)
    responses = np.asarray(
        responses if responses is not None else np.full(count, 0.02), float,
    )
    return KeyPoints(
        positions, np.tile([0.0, 0.0, 1.0], (count, 1)), responses,
        np.arange(count), selection_scores=responses.copy(),
    )


def _disable_final_quality_filters(cfg):
    cfg.quality_response_ratio = 0.0
    cfg.geometric_nms_radius = 0.0


def test_planar_filter_rejects_interior_but_preserves_plane_boundary():
    axis = np.linspace(-0.3, 0.3, 25)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    cloud = PointCloud(points, normals, np.zeros(len(points)))
    cfg = PipelineConfig().keypoint
    cfg.quality_filter_min_features = 0
    cfg.hole_boundary_filter_enabled = False
    cfg.repeatability_filter_enabled = False
    _disable_final_quality_filters(cfg)
    cfg.planar_filter_radius = 0.14
    cfg.planar_filter_min_neighbors = 12
    scene = SimpleNamespace(cloud=cloud, volume=None)

    result = filter_scene_keypoints(
        scene, _keypoints([[0, 0, 0], [0.3, 0, 0]]), cfg,
    )

    assert result.planar_rejected == 1
    np.testing.assert_allclose(result.keypoints.positions, [[0.3, 0, 0]])


def test_hole_boundary_filter_uses_visibility_probes():
    class Volume:
        def sample_visibility_details(self, positions):
            boundary = np.zeros(len(positions), bool)
            boundary[:7] = True
            return np.zeros(len(positions), np.uint8), boundary

    points = np.asarray([[0, 0, 0], [1, 0, 0]], float)
    cloud = PointCloud(points, np.tile([0, 1, 0], (2, 1)), np.ones(2))
    cfg = PipelineConfig().keypoint
    cfg.quality_filter_min_features = 0
    cfg.planar_filter_enabled = False
    cfg.repeatability_filter_enabled = False
    _disable_final_quality_filters(cfg)
    scene = SimpleNamespace(cloud=cloud, volume=Volume())
    scene.sample_visibility_details = scene.volume.sample_visibility_details

    result = filter_scene_keypoints(scene, _keypoints(points), cfg)

    assert result.hole_rejected == 1
    np.testing.assert_allclose(result.keypoints.positions, [[1, 0, 0]])


def test_repeatability_filter_preserves_paper_2d_corners():
    axis = np.linspace(-0.2, 0.2, 12)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    cloud = PointCloud(points, normals, np.zeros(len(points)))
    cfg = PipelineConfig().keypoint
    cfg.quality_filter_min_features = 0
    cfg.planar_filter_enabled = False
    cfg.hole_boundary_filter_enabled = False
    _disable_final_quality_filters(cfg)
    scene = SimpleNamespace(cloud=cloud, volume=None)

    result = filter_scene_keypoints(
        scene, _keypoints([[0, 0, 0], [0.1, 0, 0]], [0.02, 0.0]), cfg,
    )

    assert result.repeatability_rejected == 1
    np.testing.assert_allclose(result.keypoints.positions, [[0.1, 0, 0]])


def test_relative_harris_score_preserves_2d_corner():
    points = np.asarray([
        [0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0, 0], [2, 0, 0],
    ], float)
    cloud = PointCloud(points, np.tile([0, 1, 0], (5, 1)), np.ones(5))
    cfg = PipelineConfig().keypoint
    cfg.quality_filter_min_features = 0
    cfg.planar_filter_enabled = False
    cfg.hole_boundary_filter_enabled = False
    cfg.repeatability_filter_enabled = False
    cfg.geometric_nms_radius = 0.0
    cfg.quality_response_ratio = 0.5
    cfg.quality_score_radius = 3.0

    result = filter_scene_keypoints(
        SimpleNamespace(cloud=cloud, volume=None),
        _keypoints(points, [1.0, 0.8, 0.7, 0.2, 0.0]), cfg,
    )

    assert result.score_rejected == 1
    np.testing.assert_allclose(
        result.keypoints.positions, [points[0], points[1], points[2], points[4]],
    )


def test_geometric_nms_keeps_best_local_keypoint():
    points = np.asarray([[0, 0, 0], [0.04, 0, 0], [1, 0, 0]], float)
    cloud = PointCloud(points, np.tile([0, 1, 0], (3, 1)), np.ones(3))
    cfg = PipelineConfig().keypoint
    cfg.quality_filter_min_features = 0
    cfg.planar_filter_enabled = False
    cfg.hole_boundary_filter_enabled = False
    cfg.repeatability_filter_enabled = False
    cfg.quality_response_ratio = 0.0
    cfg.geometric_nms_radius = 0.1

    result = filter_scene_keypoints(
        SimpleNamespace(cloud=cloud, volume=None),
        _keypoints(points, [0.03, 0.02, 0.025]), cfg,
    )

    assert result.nms_rejected == 1
    np.testing.assert_allclose(result.keypoints.positions, [points[0], points[2]])


def test_descriptor_quality_rejects_empty_and_redundant_features():
    cfg = PipelineConfig()
    cfg.matching.descriptor_keypoint_min_features = 0
    cfg.matching.quality_nms_radius = 0.08
    cfg.matching.quality_min_score_ratio = 0.0

    def feature(position, occupied, response):
        return SimpleNamespace(
            position=np.asarray(position, float), response=response,
            selection_score=response,
            descriptor=LocalDescriptor(
                "scan", 4, 0.1, np.asarray(position, float),
                n_occupied=occupied, n_unknown=0, n_total=64,
            ),
        )

    trace = {}
    result = _filter_scan_feature_quality([
        feature([0, 0, 0], 20, 0.03),
        feature([0.02, 0, 0], 20, 0.02),
        feature([1, 0, 0], 2, 0.04),
    ], cfg, trace=trace)

    assert len(result) == 1
    np.testing.assert_allclose(result[0].position, [0, 0, 0])
    assert trace["descriptor_rejected"] == 1
    assert trace["quality_nms_rejected"] == 1
