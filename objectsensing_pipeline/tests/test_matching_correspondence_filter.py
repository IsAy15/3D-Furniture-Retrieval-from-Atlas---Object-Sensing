import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import matching  # noqa: E402
from config import PipelineConfig  # noqa: E402
from constellations import one_point_ransac  # noqa: E402
from descriptors import LocalDescriptor  # noqa: E402
from matching import Correspondence, KeyPointFeature  # noqa: E402
from primitives import Primitive, PrimitiveSet  # noqa: E402
from transforms import GroundTransform  # noqa: E402


UP = np.array([0.0, 1.0, 0.0])


def _feature(position, distance=0.0):
    descriptor = LocalDescriptor(
        "model", 2, 0.1, np.asarray(position, float),
        udf=np.full((2, 2, 2), distance, np.float32),
        occupied_offsets=np.zeros((1, 3)),
        n_occupied=1, n_unknown=1, n_total=8,
    )
    return KeyPointFeature(
        np.asarray(position, float), UP.copy(), 1.0,
        descriptor, PrimitiveSet(), np.ones(6),
    )


def test_match_features_applies_absolute_descriptor_threshold(monkeypatch):
    cfg = PipelineConfig()
    cfg.ransac.desc_inlier = 128.0
    cfg.matching.descriptor_ratio_threshold = 0.0
    model_features = [_feature([i, 0, 0], distance) for i, distance in enumerate((10, 20, 200))]
    scan_features = [_feature([0, 0, 0])]

    monkeypatch.setattr(
        matching, "descriptor_distance_batch",
        lambda model, scan, config, thetas, up: np.full(len(thetas), model.udf.flat[0]),
    )

    correspondences = matching.match_features(model_features, scan_features, cfg, UP)

    assert [item.model_idx for item in correspondences] == [0, 1]
    assert all(item.desc_dist < cfg.ransac.desc_inlier for item in correspondences)


def test_matching_descriptor_angle_produces_model_to_scan_pose():
    """An asymmetric local surface must score the inverse of the pose angle."""
    from scipy.spatial import cKDTree
    from descriptors import build_model_descriptor
    from geometry import rotation_about_axis

    cfg = PipelineConfig()
    cfg.descriptor.udf_grid_res = 33
    offsets = np.array([
        [-0.04, 0.0, 0.02], [0.03, 0.015, 0.06], [0.07, -0.02, -0.01],
    ])
    theta = np.radians(40)
    rotation = rotation_about_axis(UP, theta)
    model = _feature([0.2, 0.7, 0.1])
    scan = _feature(rotation @ model.position + np.array([1.0, 0.0, -0.5]))
    model.descriptor = build_model_descriptor(cKDTree(offsets), np.zeros(3), cfg.descriptor)
    scan.descriptor = LocalDescriptor(
        "scan", 33, cfg.descriptor.udf_extent, np.zeros(3),
        occupied_offsets=offsets @ rotation.T,
        n_occupied=len(offsets), n_unknown=model.descriptor.n_occupied,
    )

    pairs = matching.match_features([model], [scan], cfg, UP)

    assert len(pairs) == 1
    assert abs((pairs[0].theta - theta + np.pi) % (2 * np.pi) - np.pi) < 1e-6
    assert np.allclose(pairs[0].transform.apply(model.position[None, :])[0], scan.position)
    # The same pose must align an off-center landmark, not only the seed point.
    landmark = model.position + offsets[0]
    expected = scan.position + rotation @ offsets[0]
    assert np.allclose(pairs[0].transform.apply(landmark[None, :])[0], expected)


def test_match_features_ratio_test_rejects_equal_best_matches(monkeypatch):
    cfg = PipelineConfig()
    cfg.matching.descriptor_ratio_threshold = 0.9
    model_features = [_feature([i, 0, 0], 10.0) for i in range(2)]
    scan_features = [_feature([0, 0, 0])]
    monkeypatch.setattr(
        matching, "descriptor_distance_batch",
        lambda model, scan, config, thetas, up: np.full(len(thetas), model.udf.flat[0]),
    )

    assert matching.match_features(model_features, scan_features, cfg, UP) == []


def test_missing_scan_primitive_does_not_reject_partial_observation(monkeypatch):
    cfg = PipelineConfig()
    cfg.matching.descriptor_ratio_threshold = 0.0
    model_feature = _feature([0, 0, 0])
    scan_feature = _feature([0, 0, 0])
    model_feature.size6d = np.array([1.0, 0.4, 0.0, 0.0, 0.0, 0.0])
    scan_feature.size6d = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        matching, "descriptor_distance_batch",
        lambda model, scan, config, thetas, up: np.ones(len(thetas)),
    )

    diagnostics = {}
    correspondences = matching.match_features(
        [model_feature], [scan_feature], cfg, UP, diagnostics=diagnostics,
    )

    assert len(correspondences) == 1
    assert diagnostics["scan_with_size_candidates"] == 1
    assert diagnostics["scan_with_descriptor_candidates"] == 1


def test_observed_incompatible_primitive_sizes_are_rejected(monkeypatch):
    cfg = PipelineConfig()
    model_feature = _feature([0, 0, 0])
    scan_feature = _feature([0, 0, 0])
    model_feature.size6d = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scan_feature.size6d = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        matching, "descriptor_distance_batch",
        lambda model, scan, config, thetas, up: np.ones(len(thetas)),
    )

    diagnostics = {}
    assert matching.match_features(
        [model_feature], [scan_feature], cfg, UP, diagnostics=diagnostics,
    ) == []
    assert diagnostics["scan_with_scale_candidates"] == 1
    assert diagnostics["scan_with_size_candidates"] == 0


def test_candidate_rotations_keep_uniform_fallback_with_primitives():
    cfg = PipelineConfig()
    cfg.matching.n_uniform_rotations = 12
    cfg.matching.primitive_rotation_fallback = True
    model = _feature([0, 0, 0])
    scan = _feature([0, 0, 0])
    model.primitives.verticals = [Primitive(
        "line", np.array([1.0, 0.0, 0.0]), 1.0, "vertical",
    )]
    scan.primitives.verticals = [Primitive(
        "line", np.array([0.0, 0.0, 1.0]), 1.0, "vertical",
    )]

    thetas = matching._candidate_thetas(
        model, scan, cfg, UP,
        np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
    )

    assert len(thetas) >= cfg.matching.n_uniform_rotations


def test_constellation_requires_distinct_spread_scan_keypoints():
    cfg = PipelineConfig()
    cfg.ransac.min_inliers = 4
    cfg.ransac.min_scan_inliers = 4
    cfg.ransac.min_scan_spread = 0.12
    model_features = [_feature([0.1 * i, 0, 0]) for i in range(4)]
    scan_features = [_feature([0.1 * i, 0, 0]) for i in range(4)]
    identity = GroundTransform(0.0, 1.0, np.zeros(3), UP)

    degenerate = [
        Correspondence(i, 0, 0.0, 1.0, 1.0, identity)
        for i in range(4)
    ]
    diagnostics = {}
    assert one_point_ransac(
        degenerate, model_features, scan_features, cfg, UP,
        diagnostics=diagnostics,
    ) == []
    assert diagnostics["accepted"] == 0
    assert sum(diagnostics["rejected"].values()) > 0

    valid = [
        Correspondence(i, i, 0.0, 1.0, 1.0, identity)
        for i in range(4)
    ]
    constellations = one_point_ransac(valid, model_features, scan_features, cfg, UP)
    assert constellations
    assert len({item.scan_idx for item in constellations[0].inliers}) == 4


def test_constellation_requires_multiple_spatial_cells_on_both_sides():
    cfg = PipelineConfig()
    cfg.ransac.min_scan_spread = 0.0
    cfg.ransac.scan_cell_size = 0.10
    cfg.ransac.model_cell_size = 0.10
    cfg.ransac.min_scan_cells = 3
    cfg.ransac.min_model_cells = 3
    identity = GroundTransform(0.0, 1.0, np.zeros(3), UP)

    clustered = np.asarray([
        [0.00, 0.00, 0.00], [0.02, 0.00, 0.00],
        [0.04, 0.00, 0.00], [0.06, 0.00, 0.00],
    ])
    features = [_feature(point) for point in clustered]
    correspondences = [
        Correspondence(i, i, 0.0, 1.0, 1.0, identity) for i in range(4)
    ]
    assert one_point_ransac(
        correspondences, features, features, cfg, UP,
    ) == []

    distributed = np.asarray([
        [0.00, 0.00, 0.00], [0.12, 0.00, 0.00],
        [0.24, 0.00, 0.00], [0.36, 0.00, 0.00],
    ])
    features = [_feature(point) for point in distributed]
    assert one_point_ransac(
        correspondences, features, features, cfg, UP,
    )
