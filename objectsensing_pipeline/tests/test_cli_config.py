import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli import _apply_overrides, _limit_from_arg  # noqa: E402
from annotation_store import accepted_annotations_from_payload  # noqa: E402
from config import PipelineConfig  # noqa: E402


def test_unified_annotations_remain_compatible_with_sweep_targets():
    payload = {
        "schema_version": 2,
        "scenes": {
            "office": {"desk": {"table_good": 2, "table_ok": 1, "bad": 0}},
        },
    }

    assert accepted_annotations_from_payload(payload) == {
        "office": {"desk": ["table_good", "table_ok"]},
    }


def test_voxel_override_updates_truncation_band():
    cfg = PipelineConfig()
    args = Namespace(voxel=0.03, truncation=None, frame_stride=None, max_frames=None,
                     all_frames=False,
                     max_scan_keypoints=None, max_model_keypoints=None,
                     spatial_keypoints=False, spatial_keypoint_cell=None,
                     reverse_gate=None, paper_verification=False,
                     surface_distance_weight=None,
                     verification_top_constellations=None,
                     max_registrations_per_model=None,
                     registration_min_center_distance=None,
                     registration_top_constellations=None,
                     multi_registration_synsets=None)

    out = _apply_overrides(cfg, args)

    assert out.sdf.voxel_size == 0.03
    assert out.sdf.truncation == 0.09


def test_fusion_backend_override():
    cfg = PipelineConfig()
    args = Namespace(
        fusion_backend="cuda", voxel=None, truncation=None,
        frame_stride=None, max_frames=None, all_frames=False,
        max_scan_keypoints=None, max_model_keypoints=None,
        spatial_keypoints=False, spatial_keypoint_cell=None,
        reverse_gate=None, paper_verification=False,
        surface_distance_weight=None,
        verification_top_constellations=None,
        max_registrations_per_model=None,
        registration_min_center_distance=None,
        registration_top_constellations=None,
        multi_registration_synsets=None,
    )

    out = _apply_overrides(cfg, args)

    assert out.sdf.backend == "cuda"


def test_volume_layout_override():
    cfg = PipelineConfig()
    args = Namespace(
        volume_layout="sparse", voxel=None, truncation=None,
        frame_stride=None, max_frames=None, all_frames=False,
        max_scan_keypoints=None, max_model_keypoints=None,
        spatial_keypoints=False, spatial_keypoint_cell=None,
        reverse_gate=None, paper_verification=False,
        surface_distance_weight=None,
        verification_top_constellations=None,
        max_registrations_per_model=None,
        registration_min_center_distance=None,
        registration_top_constellations=None,
        multi_registration_synsets=None,
    )

    out = _apply_overrides(cfg, args)

    assert out.sdf.volume_layout == "sparse"


def test_explicit_truncation_wins_over_voxel_default():
    cfg = PipelineConfig()
    args = Namespace(voxel=0.03, truncation=0.05, frame_stride=4,
                     min_frames=50, max_frames=80,
                     max_surface_points=350000,
                     all_frames=False,
                     max_scan_keypoints=800, max_model_keypoints=150,
                     spatial_keypoints=True, spatial_keypoint_cell=0.4,
                     reverse_gate=0.55, paper_verification=False,
                     surface_distance_weight=0.08,
                     verification_top_constellations=50,
                     max_registrations_per_model=2,
                     registration_min_center_distance=0.4,
                     registration_top_constellations=8,
                     multi_registration_synsets=["04379243"])

    out = _apply_overrides(cfg, args)

    assert out.sdf.voxel_size == 0.03
    assert out.sdf.truncation == 0.05
    assert out.sdf.frame_stride == 4
    assert out.sdf.min_frames == 50
    assert out.sdf.max_frames == 80
    assert out.sdf.max_surface_points == 350000
    assert out.matching.max_scan_keypoints == 800
    assert out.matching.max_model_keypoints == 150
    assert out.matching.spatial_keypoint_balance is True
    assert out.matching.spatial_keypoint_cell == 0.4
    assert out.matching.reverse_gate == 0.55
    assert out.matching.surface_distance_weight == 0.08
    assert out.matching.verification_top_constellations == 50
    assert out.matching.max_registrations_per_model == 2
    assert out.matching.registration_min_center_distance == 0.4
    assert out.matching.registration_top_constellations == 8
    assert out.matching.multi_registration_synsets == ("04379243",)


def test_zero_limit_means_all_models():
    assert _limit_from_arg(0) is None
    assert _limit_from_arg(-1) is None
    assert _limit_from_arg(None) is None
    assert _limit_from_arg(200) == 200


def test_quality_overrides_accept_zero_limits_and_all_frames():
    cfg = PipelineConfig()
    cfg.sdf.frame_stride = 8
    cfg.sdf.min_frames = 50
    cfg.sdf.max_frames = 80
    args = Namespace(
        voxel=0.02, truncation=None, frame_stride=None, max_frames=None,
        all_frames=True,
        max_scan_keypoints=0, max_model_keypoints=0,
        spatial_keypoints=False, spatial_keypoint_cell=None,
        reverse_gate=None, paper_verification=False,
        surface_distance_weight=None,
        verification_top_constellations=None,
        max_registrations_per_model=None,
        registration_min_center_distance=None,
        registration_top_constellations=None,
        multi_registration_synsets=None,
    )

    out = _apply_overrides(cfg, args)

    assert out.sdf.voxel_size == 0.02
    assert out.sdf.truncation == 0.06
    assert out.sdf.frame_stride == 1
    assert out.sdf.min_frames == 0
    assert out.sdf.max_frames == 0
    assert out.matching.max_scan_keypoints == 0
    assert out.matching.max_model_keypoints == 0


def test_paper_verification_disables_added_guards():
    cfg = PipelineConfig()
    args = Namespace(
        voxel=None, truncation=None, frame_stride=None, max_frames=None,
        all_frames=False,
        max_scan_keypoints=None, max_model_keypoints=None,
        spatial_keypoints=False, spatial_keypoint_cell=None,
        reverse_gate=None, paper_verification=True,
        surface_distance_weight=0.05,
        verification_top_constellations=None,
        max_registrations_per_model=None,
        registration_min_center_distance=None,
        registration_top_constellations=None,
        multi_registration_synsets=None,
    )

    out = _apply_overrides(cfg, args)

    assert out.matching.reverse_gate == 0.0
    assert out.matching.min_structure_height == 0.0
    assert out.matching.min_thickness == 0.0
    assert out.matching.surface_distance_weight == 0.05


def test_keypoint_detector_overrides_are_applied_explicitly():
    cfg = PipelineConfig()
    cfg.matching.spatial_keypoint_balance = True
    args = Namespace(
        voxel=None, truncation=None, frame_stride=None, max_frames=None,
        all_frames=False, max_scan_keypoints=None, max_model_keypoints=None,
        spatial_keypoints=False, no_spatial_keypoints=True,
        spatial_keypoint_cell=0.35, reverse_gate=None,
        paper_verification=False, surface_distance_weight=None,
        verification_top_constellations=None,
        max_registrations_per_model=None,
        registration_min_center_distance=None,
        registration_top_constellations=None,
        multi_registration_synsets=None,
        neighbor_radius=0.075, curvature_threshold=0.082,
        harris_k=0.05, harris_threshold=0.011,
        harris_reference_neighbors=8.0, convex_hull_ratio=0.9,
        corner_plane_radius_factor=4.0, corner_plane_min_area=0.03,
        corner_plane_normal_cos=0.94, nms_radius=0.08,
        adjust_iterations=3, jitter_reject="auto", dedup_radius=0.025,
        floor_height=0.12, min_2d_corner_fraction=0.4,
    )

    out = _apply_overrides(cfg, args)

    assert out.keypoint.neighbor_radius == 0.075
    assert out.keypoint.curvature_threshold == 0.082
    assert out.keypoint.harris_threshold == 0.011
    assert out.keypoint.corner_plane_normal_cos == 0.94
    assert out.keypoint.adjust_iterations == 3
    assert out.keypoint.jitter_reject is None
    assert out.matching.floor_height == 0.12
    assert out.matching.min_2d_corner_fraction == 0.4
    assert out.matching.spatial_keypoint_cell == 0.35
    assert out.matching.spatial_keypoint_balance is False


def test_wall_keypoint_filter_overrides_are_applied_explicitly():
    cfg = PipelineConfig()
    args = Namespace(
        voxel=None, truncation=None, frame_stride=None, max_frames=None,
        all_frames=False, max_scan_keypoints=None, max_model_keypoints=None,
        spatial_keypoints=False, spatial_keypoint_cell=None,
        reverse_gate=None, paper_verification=False,
        surface_distance_weight=None,
        verification_top_constellations=None,
        max_registrations_per_model=None,
        registration_min_center_distance=None,
        registration_top_constellations=None,
        multi_registration_synsets=None,
        wall_filter_enabled=False, wall_hard_reject=False,
        wall_penalty_weight=0.075, wall_reject_threshold=0.42,
        wall_reject_max_object_score=0.18,
        component_budget_enabled=False, component_budget_radius=0.22,
        component_budget_max_per_component=18,
        component_budget_min_features=32,
    )

    out = _apply_overrides(cfg, args)

    assert out.keypoint.wall_filter_enabled is False
    assert out.keypoint.wall_hard_reject is False
    assert out.keypoint.wall_penalty_weight == 0.075
    assert out.keypoint.wall_reject_threshold == 0.42
    assert out.keypoint.wall_reject_max_object_score == 0.18
    assert out.keypoint.component_budget_enabled is False
    assert out.keypoint.component_budget_radius == 0.22
    assert out.keypoint.component_budget_max_per_component == 18
    assert out.keypoint.component_budget_min_features == 32
