import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constellations import Constellation  # noqa: E402
from geometry import PointCloud  # noqa: E402
from sdf_fusion import VIS_FREE, VIS_OCCUPIED, VIS_UNKNOWN  # noqa: E402
from transforms import GroundTransform  # noqa: E402
from verify import (  # noqa: E402
    icp_refine,
    model_support_distribution,
    verify_model,
    verify_model_candidates,
    visibility_coverage_score,
)
import verify  # noqa: E402


def test_visibility_coverage_ignores_unknown_and_counts_free_space():
    cloud = PointCloud(np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ]))
    transform = GroundTransform(
        0.0, 1.0, np.zeros(3), np.asarray([0.0, 1.0, 0.0]),
    )

    class Volume:
        def sample_visibility(self, points):
            assert np.allclose(points, cloud.points)
            return np.asarray([
                VIS_OCCUPIED, VIS_UNKNOWN, VIS_FREE, VIS_OCCUPIED,
            ])

    score = visibility_coverage_score(cloud, transform, Volume())

    assert score == (2.0 / 3.0, 3.0 / 4.0, 1.0 / 3.0, 3)


def test_model_support_distribution_rejects_a_single_local_patch():
    axis = np.linspace(0.0, 1.0, 8)
    model_points = np.asarray([
        [x, y, z] for x in axis for y in axis for z in axis
    ])
    model = PointCloud(model_points)
    local_scan = PointCloud(model_points[np.all(model_points <= 0.40, axis=1)])
    transform = GroundTransform(
        0.0, 1.0, np.zeros(3), np.asarray([0.0, 1.0, 0.0]),
    )

    zone_fraction, extent_ratio, supported, total = model_support_distribution(
        model, local_scan, transform, threshold=0.03,
        zone_grid=2, zone_min_support=0.15,
    )

    assert supported == 1
    assert total == 8
    assert zone_fraction == 0.125
    assert extent_ratio < 0.4


def test_verify_rejects_pose_supported_by_too_few_model_zones(monkeypatch):
    axis = np.linspace(0.0, 1.0, 8)
    model_points = np.asarray([
        [x, y, z] for x in axis for y in axis for z in axis
    ])
    model = PointCloud(model_points)
    scan = PointCloud(model_points[np.all(model_points <= 0.40, axis=1)])
    up = np.asarray([0.0, 1.0, 0.0])
    transform = GroundTransform(0.0, 1.0, np.zeros(3), up)
    monkeypatch.setattr(verify, "icp_refine", lambda *args, **kwargs: transform)

    registration = verify_model(
        "table", model, scan, [Constellation(transform, [], quality=1.0)],
        threshold=0.03, up=up, reverse_gate=0.0,
        min_structure=0.0, min_thickness=0.0,
        min_model_zone_fraction=0.35, min_support_extent_ratio=0.2,
    )

    assert registration is None


def test_icp_rejects_scale_update_outside_paper_bounds(monkeypatch):
    up = np.array([0.0, 1.0, 0.0])
    pts = np.column_stack([
        np.linspace(0.0, 1.0, 40),
        np.zeros(40),
        np.zeros(40),
    ])
    scan = PointCloud(pts.copy())
    initial = GroundTransform(0.0, 1.0, np.zeros(3), up)
    collapsed = GroundTransform(0.0, 0.5, np.zeros(3), up)
    monkeypatch.setattr(verify, "refine_transform", lambda *args, **kwargs: collapsed)

    refined = icp_refine(
        pts, scan, initial, up,
        scale_bounds=(1.0 / 1.5, 1.5),
    )

    assert refined is None


def test_surface_distance_breaks_equal_coverage_ties():
    up = np.array([0.0, 1.0, 0.0])
    x = np.linspace(0.0, 1.0, 40)
    pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    model = PointCloud(pts)
    scan = PointCloud(pts.copy())

    worse = GroundTransform(theta=0.0, scale=1.0, t=np.array([0.04, 0.0, 0.0]), up=up)
    exact = GroundTransform(theta=0.0, scale=1.0, t=np.zeros(3), up=up)
    cons = [
        Constellation(worse, [], quality=10.0),
        Constellation(exact, [], quality=1.0),
    ]

    reg = verify_model("line", model, scan, cons, threshold=0.05, up=None,
                       surface_distance_weight=0.1)

    assert reg is not None
    assert np.allclose(reg.transform.t, exact.t)
    assert reg.coverage == 1.0


def test_verify_model_prefers_harmonically_balanced_pose(monkeypatch):
    up = np.array([0.0, 1.0, 0.0])
    model = PointCloud(np.zeros((20, 3)))
    scan = PointCloud(np.zeros((20, 3)))
    partial = GroundTransform(0.0, 1.0, np.zeros(3), up)
    balanced = GroundTransform(1.0, 1.0, np.zeros(3), up)
    cons = [
        Constellation(partial, [], quality=2.0),
        Constellation(balanced, [], quality=1.0),
    ]

    monkeypatch.setattr(verify, "icp_refine", lambda *args, **kwargs: args[2])

    def fake_coverage(*args, **kwargs):
        transform = args[2]
        return ((0.90, 0.20, 0.01, 1.0, 1.0)
                if transform.theta == 0.0
                else (0.60, 0.50, 0.01, 1.0, 1.0))

    monkeypatch.setattr(verify, "coverage_score", fake_coverage)

    reg = verify_model(
        "chair", model, scan, cons, top=2, up=up,
        reverse_gate=0.0, min_structure=0.0, min_thickness=0.0,
        surface_distance_weight=0.0,
    )

    assert reg is not None
    assert reg.transform is balanced


@pytest.mark.parametrize("mode", ["single", "delegated", "multiple"])
@pytest.mark.parametrize("coverage_threshold, expected_count", [(.45, 1), (.46, 0)])
def test_coverage_threshold_filters_poses_before_ranking_and_separation(
        monkeypatch, mode, coverage_threshold, expected_count):
    up = np.array([0., 1., 0.])
    cloud = PointCloud(np.zeros((20, 3)))
    below_threshold = GroundTransform(0., 1., np.zeros(3), up)
    acceptable = GroundTransform(1., 1., np.zeros(3), up)
    constellations = [
        Constellation(below_threshold, [], quality=2.),
        Constellation(acceptable, [], quality=1.),
    ]
    monkeypatch.setattr(verify, "icp_refine", lambda *args, **kwargs: args[2])
    monkeypatch.setattr(verify, "coverage_score", lambda *args, **kwargs:
                        ((.44, .95, .01, 1., 1.) if args[2] is below_threshold
                         else (.45, .45, .01, 1., 1.)))
    monkeypatch.setattr(verify, "model_support_distribution",
                        lambda *args, **kwargs: (1., 1., 8, 8))

    def run(**extra):
        kwargs = dict(top=2, up=up, surface_distance_weight=0., **extra)
        if mode == "single":
            result = verify_model("desk", cloud, cloud, constellations, **kwargs)
            return [] if result is None else [result]
        return verify_model_candidates(
            "desk", cloud, cloud, constellations,
            max_results=1 if mode == "delegated" else 2,
            min_center_distance=.35, **kwargs,
        )

    # The rejected pose has the higher unchanged ranking score. Its identical
    # center would also suppress the admissible pose in the multiple-result path.
    baseline = run()
    assert len(baseline) == 1
    assert baseline[0].transform is below_threshold

    diagnostics = {}
    results = run(coverage_threshold=coverage_threshold, diagnostics=diagnostics)

    assert len(results) == expected_count
    assert diagnostics["coverage_threshold"] == coverage_threshold
    assert diagnostics["rejection_counts"] == {"coverage": 2 - expected_count}
    assert diagnostics["poses"][0]["status"] == "rejected_coverage"
    assert diagnostics["poses"][1]["status"] == (
        "accepted_pose" if expected_count else "rejected_coverage"
    )
    if expected_count:
        assert results[0].transform is acceptable
        assert results[0].coverage == coverage_threshold
    else:
        assert diagnostics["status"] == "rejected"


def test_verify_model_candidates_keeps_distinct_poses():
    up = np.array([0.0, 1.0, 0.0])
    x = np.linspace(0.0, 1.0, 80)
    pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    model = PointCloud(pts)
    scan = PointCloud(np.vstack([pts, pts + np.array([0.0, 0.0, 1.0])]))

    cons = [
        Constellation(GroundTransform(0.0, 1.0, np.zeros(3), up), [], quality=2.0),
        Constellation(GroundTransform(0.0, 1.0, np.array([0.0, 0.0, 1.0]), up), [], quality=1.5),
    ]

    regs = verify_model_candidates(
        "line", model, scan, cons,
        threshold=0.05, up=up,
        reverse_gate=0.0,
        min_structure=0.0,
        min_thickness=0.0,
        max_results=2,
        min_center_distance=0.5,
    )

    assert len(regs) == 2
    zs = sorted(round(float(r.transform.t[2]), 3) for r in regs)
    assert zs == [0.0, 1.0]


def test_paper_profile_accepts_pose_rejected_only_by_added_guards(monkeypatch):
    up = np.array([0.0, 1.0, 0.0])
    model = PointCloud(np.zeros((20, 3)))
    scan = PointCloud(np.zeros((20, 3)))
    transform = GroundTransform(0.0, 1.0, np.zeros(3), up)
    constellations = [Constellation(transform, [], quality=3.0)]

    monkeypatch.setattr(verify, "icp_refine", lambda *args, **kwargs: transform)
    monkeypatch.setattr(
        verify, "coverage_score",
        lambda *args, **kwargs: (0.62, 0.10, 0.01, 0.04, 0.01),
    )

    guarded = verify_model(
        "desk", model, scan, constellations, up=up,
        reverse_gate=0.42, min_structure=0.15, min_thickness=0.06,
    )
    paper = verify_model(
        "desk", model, scan, constellations, up=up,
        reverse_gate=0.0, min_structure=0.0, min_thickness=0.0,
        surface_distance_weight=0.05,
    )

    assert guarded is None
    assert paper is not None
    assert paper.coverage == 0.62
