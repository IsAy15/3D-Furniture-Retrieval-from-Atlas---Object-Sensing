'ObjectSensing Debug Visualizer backend. Each stage exposes sources, parameters, substeps, 3D layers and metrics. Short-lived worker processes load saved code changes on the next computation without restarting the server.\n'
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from config import KeypointConfig, PipelineConfig
from ui_help_en import english_parameter_help
from geometry import PointCloud, estimate_ground_plane, make_point_cloud
from keypoints import KeyPoints, detect_keypoints
from keypoint_groups import build_object_keypoint_groups
from keypoint_quality import filter_scene_keypoints
from matching import _cap_by_response
from runtime_paths import RUNTIME_PATHS, path_is_within
from wall_keypoints import apply_wall_keypoint_filter


ROOT = Path(__file__).resolve().parent
DEBUG_ROOT = RUNTIME_PATHS.debug_dir
CACHE_ROOT = DEBUG_ROOT / "cloud_cache"
RUN_ROOT = DEBUG_ROOT / "runs"
SETTINGS_PATH = DEBUG_ROOT / "settings.json"
SCHEMA_VERSION = 1

# Debug grouping controls are separate from detector parameters and annotations.
GROUP_CONTROLS = {
    "keypoint_link_radius": (.30, 'Keypoint links', .05, .6, .01, 'A larger radius joins distant parts but may connect neighboring furniture objects.'),
    "part_merge_gap": (.45, 'Part merging', .05, .8, .01, 'A smaller distance separates neighbors more clearly but may fragment one object.'),
    "component_merge_gap": (.35, 'Component merging', .05, .8, .01, 'Allowed gap when merging compatible surface components.'),
    "wall_clearance": (.075, 'Wall clearance', 0., .2, .005, 'Exclude near-wall surface from the grouping graph without directly deleting keypoints.'),
    "surface_connection": (.095, "Connexion de surface", .03, .2, .005, 'Connection distance between surface voxels. Large values merge touching objects.'),
    "assignment_radius": (.14, 'Surface assignment', .03, .4, .01, 'Maximum distance for assigning a keypoint to a surface component.'),
    "orphan_attach_radius": (.18, 'Orphan attachment', .03, .5, .01, 'Recover points near a group; large values may reintroduce wall points.'),
}


def json_compatible(value):
    """Return JSON data with non-finite and NumPy scalars normalized."""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def _serialize_object_groups(groups, keypoints):
    source_indices = np.asarray(keypoints.source_index, int)
    return [{
        "id": int(group_id),
        "member_indices": np.asarray(group.member_indices, int).tolist(),
        "source_indices": source_indices[
            np.asarray(group.member_indices, int)
        ].tolist(),
        "centroid": np.asarray(group.centroid, float).tolist(),
        "extent": np.asarray(group.extent, float).tolist(),
        "score": float(group.score),
        "descriptor": np.asarray(group.descriptor, float).tolist(),
    } for group_id, group in enumerate(groups)]


def _append_group_preview(layers, steps, keypoints, records, up, ground):
    """Render the exact disjoint groups saved for descriptor computation."""
    layers[:] = [layer for layer in layers if layer["id"] not in ("keypoint_groups", "keypoint_group_links")]
    steps[:] = [step for step in steps if step["id"] != "keypoint_groups"]
    points = _to_display(keypoints.positions, up, ground)
    palette = [[.35,.78,.86], [.96,.63,.30], [.53,.82,.51], [.72,.52,.94], [.94,.45,.52]]
    vertices, colors, sources, links = [], [], [], []
    for index, record in enumerate(records):
        members = np.asarray(record["member_indices"], int)
        center = points[members].mean(axis=0).tolist()
        for member in members:
            point = points[member].tolist()
            vertices.append(point); colors.append(palette[index % len(palette)])
            sources.append(int(keypoints.source_index[member])); links.append([center, point])
    layers.extend([
        dict(id="keypoint_groups", label='Passed groups', kind="points", role="candidate", points=vertices, colors=colors, source_index=sources, total=len(vertices), sampled=len(vertices)),
        dict(id="keypoint_group_links", label='Group links', kind="lines", role="vector", segments=links, total=len(links)),
    ])
    steps.append(dict(id="keypoint_groups", label='Object groups', explanation='Disjoint groups passed to descriptors. One color per group; lines connect members to the centroid rather than a matching constellation.', input=keypoints.size, kept=len(vertices), rejected=keypoints.size-len(vertices), visible_layers=["surface", "keypoint_groups", "keypoint_group_links"]))


def _feature_groups_from_keypoints(group_records, keypoints, features):
    """Map retained-keypoint groups after descriptor quality filtering."""
    if not group_records or not features:
        return []
    keypoint_positions = np.asarray(keypoints.positions, float).reshape(-1, 3)
    feature_positions = np.asarray([
        feature.position for feature in features
    ], float).reshape(-1, 3)
    feature_tree = cKDTree(feature_positions)
    distance, nearest = feature_tree.query(keypoint_positions, k=1)
    keypoint_to_feature = {
        int(index): int(feature_index)
        for index, (value, feature_index) in enumerate(zip(distance, nearest))
        if float(value) <= 1e-6
    }
    output = []
    for record in group_records:
        feature_indices = sorted({
            keypoint_to_feature[int(index)]
            for index in record.get("member_indices", [])
            if int(index) in keypoint_to_feature
        })
        if len(feature_indices) < 2:
            continue
        output.append({
            **record,
            "feature_indices": feature_indices,
            "descriptor_count": len(feature_indices),
        })
    return output


def _query_group_weights(group_records, feature_groups, enabled=True):
    """Estimate query reliability without requiring new database fields."""
    if not feature_groups:
        return []
    if not enabled:
        return [1.0] * len(feature_groups)
    counts = np.asarray([len(group) for group in feature_groups], dtype=float)
    scores = np.asarray([
        max(0.0, float(record.get("score", 0.0)))
        for record in group_records
    ], dtype=float)
    count_support = np.clip(counts / 20.0, 0.15, 1.0)
    if float(scores.max(initial=0.0)) > 1e-9:
        score_support = np.log1p(scores) / max(
            float(np.log1p(scores).max()), 1e-9,
        )
    else:
        score_support = np.ones(len(scores), dtype=float)
    raw = 0.25 + 0.55 * count_support + 0.20 * score_support
    raw /= max(float(raw.mean()), 1e-9)
    return np.clip(raw, 0.35, 2.5).tolist()


PIPELINE_STAGES = (
    {"id": "fusion", "label": "RGB-D fusion", "available": True},
    {"id": "keypoints", "label": "Keypoints", "available": True},
    {"id": "descriptors", "label": "Descriptors", "available": True},
    {"id": "query", "label": "Top-k", "available": True},
    {"id": "matching", "label": "Matching", "available": True},
    {"id": "verification", "label": 'Verification', "available": True},
    {"id": "selection", "label": 'Selection', "available": True},
)

FUSION_STEPS = (
    ("frame-selection", 'Frame selection', 'Apply temporal stride and the frame limit.'),
    ("frames", 'RGB-D integration', 'Back-project and integrate each depth measurement.'),
    ("volume", "TSDF volume", 'Distinguish free, occupied and unknown voxels.'),
    ("surface-raw", "Raw iso-surface", 'Extract zero crossings in capture coordinates.'),
    ("gravity", 'Gravity alignment', 'Rotate the scan to make Y the vertical axis.'),
    ("ground", 'Ground placement', 'Estimate the dominant lower plane and place it at Y=0.'),
    ("normals", 'SDF normals', 'Expose signed field gradients on the surface.'),
    ("curvature", 'Curvature', "Measure local variation of the reconstructed surface."),
)

KEYPOINT_STEPS = (
    ("surface", 'Input surface', 'Fusion reference: positions, normals, curvature and TSDF confidence.', 'Context', 'Essential: holes and fusion noise limit every subsequent stage.'),
    ("wall_planes", "Vertical planes", 'Detect large coherent vertical surfaces. This diagnostic stage does not remove points.', 'Context', 'Useful indoors, but must protect objects in front of walls.'),
    ("wall_scores", 'Wall / object evidence', 'Assign wall and object evidence to each seed using local and wider support.', 'Context', 'Important for Office: proximity-only rejection would also remove the desk and shelves.'),
    ("curvature", 'Curvature candidates', 'Retain points whose local PCA variation exceeds curvature_threshold.', 'Detection', 'Necessary but insufficient: TSDF noise and hole boundaries also produce high curvature.'),
    ("pre_harris_visibility", 'Pre-Harris visibility', 'Remove only candidates on definite free/unknown boundaries before Harris evaluation.', 'Detection', 'Reduce false RGB-D corners and computation. Keep this threshold more conservative than the final hole filter.'),
    ("harris", '3D Harris corners', 'Measure normal variation in several directions and retain responses above harris_threshold.', 'Detection', 'Core 3D detector; compare thresholds across scenes after density normalization.'),
    ("corner_2d", '2D corner recovery', 'Test weak Harris candidates in the tangent plane to recover silhouettes and thin structures.', 'Recall', 'Useful for legs and thin tabletops, but can introduce hole-boundary keypoints.'),
    ("nms", 'Local maxima', 'Keep only the strongest seed within each nms_radius neighborhood.', "Consolidation", 'Avoid near-identical responses around the same corner.'),
    ("adjustment", 'Geometric adjustment', 'Move each seed toward a local plane or line intersection over several iterations.', "Consolidation", 'Useful on dense surfaces; inspect movements toward hole boundaries.'),
    ("dedup", 'Deduplication', 'Merge adjusted seeds that converge to the same position.', "Consolidation", 'Necessary after adjustment; keep the radius smaller than the details of interest.'),
    ("wall_filter", 'Wall rejection', 'Reject keypoints with strong wall evidence and weak object evidence.', 'Filtering', 'Important for Office. Aggressive rejection removes objects adjacent to walls.'),
    ("planar_filter", 'Planar interiors', 'Reject maxima whose wider neighborhood remains a coherent plane in all directions.', 'Filtering', 'Reduce wall noise while preserving intersections and genuine edges.'),
    ("hole_boundary_filter", 'RGB-D hole boundaries', 'Use free, occupied and unknown space to remove contours caused by missing measurements.', 'Filtering', 'Important for Office, where occlusion and dark surfaces produce holes around desks and legs.'),
    ("repeatability_filter", 'Multi-scale repeatability', 'Check that a corner remains strong as the analysis radius increases.', 'Quality', 'Noise often disappears at a larger scale while stable geometric corners remain.'),
    ("quality_score", 'Relative strength', 'Compare each response with strong nearby corners and reject weak responses.', 'Quality', 'Reduces cost, but a global threshold can favor large walls over small objects.'),
    ("geometric_nms", 'Final spacing', 'Keep the best keypoint within each quality_nms_radius neighborhood after quality filtering.', 'Quality', 'Provides distinct points for constellation support.'),
    ("floor", 'Ground filter', 'Remove points below floor_height above the estimated ground plane.', "Budget", 'Useful for ground points; does not remove walls or horizontal tabletops.'),
    ("wall_budget", 'Wall quota', 'Limit the final budget share originating from walls and distribute it across cells.', "Budget", 'Keep a few diagnostic wall points without allowing them to dominate matching.'),
    ("component_budget", 'Per-region quota', 'Cap dense spatial components so that one region cannot consume the entire budget.', "Budget", 'Controls distribution, not geometric quality; a quota cannot make an unreliable point reliable.'),
    ("cap", 'Output budget', 'Rank survivors, optionally balance spatial cells, and pass at most max_scan_keypoints to matching.', 'Output', 'Controls runtime; optimize the budget after assessing geometric quality.'),
)


def _step_definition(item):
    return {
        "id": item[0], "label": item[1], "description": item[2],
        "group": item[3] if len(item) > 3 else "Pipeline",
        "assessment": item[4] if len(item) > 4 else "",
    }

DESCRIPTOR_STEPS = (
    ("occupation", 'Occupancy grids', 'Build free, occupied and unknown cells around each keypoint.'),
    ("seed_fill", "Seed-fill", 'Keep only the occupied component supporting the keypoint.'),
    ("groups", 'Keypoint groups', 'Color descriptors by object group and connect members to the centroid.'),
    ("candidate_pool", 'Diagnostic pool', 'Preselect a small, diverse ShapeNet set.'),
    ("distance", 'Local distance', 'Compare each scan grid with local model UDFs.'),
    ("ambiguity", 'Discriminative power', 'Separate informative, ambiguous and unmatched descriptors.'),
)

QUERY_STEPS = (
    ("database", 'Candidate database', 'Load ShapeNet models and their compact index.'),
    ("object_groups", 'Object queries', 'Divide local descriptors into disjoint geometric groups.'),
    ("global_pool", 'Per-group pools', 'Rank the database independently for each group, then merge ranks.'),
    ("local_rerank", "Local reranking", 'Rerank the pool using the computed local descriptors.'),
    ("top_k", 'Final Top-k', 'Retain models passed to constellation matching.'),
)


def _existing(*paths):
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate.resolve()
    return Path(paths[0]).resolve()


def default_sources():
    """Discover supplied ZIP files; otherwise show disabled setup examples."""
    archives = sorted(RUNTIME_PATHS.scene_dir.glob("*.zip"))
    if not archives:
        archives = [RUNTIME_PATHS.scene_dir / (name + ".zip")
                    for name in ("office", "single-chair", "chairs", "ikea-table")]
    sources = []
    for archive in archives:
        scan = RUNTIME_PATHS.run_dir / (archive.stem + "_scan.ply")
        sources.append({
            "id": archive.stem, "label": archive.stem.replace("-", " ").title(),
            "enabled": archive.is_file(), "path": str(scan),
            "stage_paths": {"fusion": str(archive), "keypoints": str(scan)},
        })
    return sources


def default_parameters(stage="keypoints"):
    if stage in ("matching", "verification", "selection"):
        from debug_retrieval import defaults
        return defaults(stage)
    cfg = PipelineConfig()
    if stage == "fusion":
        values = asdict(cfg.sdf)
        values.update({
            "surface_preview_points": 45000,
            "volume_preview_points": 12000,
            "normal_preview_points": 1200,
            "parallel_scenes": min(3, max(1, (os.cpu_count() or 2) // 2)),
        })
        return values
    if stage == "descriptors":
        return {
            "occ_grid_res": cfg.descriptor.occ_grid_res,
            "occ_extent": cfg.descriptor.occ_extent,
            "distance_unit": cfg.descriptor.distance_unit,
            "distance_exponent": cfg.descriptor.distance_exponent,
            "max_occupied": cfg.descriptor.max_occupied,
            "visibility_surface_band_factor": (
                cfg.sdf.visibility_surface_band_factor
            ),
            "utility_candidate_models": 24,
            "utility_model_features": 24,
            "utility_rotations": 8,
            "utility_distance_threshold": cfg.ransac.desc_inlier,
            "descriptor_ratio_threshold": cfg.matching.descriptor_ratio_threshold,
            "utility_margin_threshold": 0.12,
            "utility_max_valid_models": 6,
            "utility_entropy_threshold": 0.80,
            "surface_preview_points": 45000,
            "layer_preview_points": 18000,
            "parallel_scenes": min(3, max(1, (os.cpu_count() or 2) // 2)),
        }
    if stage == "query":
        return {
            "candidate_top_k": 100,
            "candidate_pool_k": 500,
            "candidate_diverse": True,
            "candidate_diversity": 0.5,
            "local_rerank": True,
            "grouped_query": True,
            "candidate_min_group_descriptors": 6,
            "candidate_max_groups": 1,
            "candidate_group_weighting": True,
            "candidate_pool_min_per_group": 20,
            "candidate_local_extra_k": 64,
            "candidate_global_pool_fraction": 0.70,
            "candidate_top_min_per_group": 6,
            "candidate_descriptor_ratio_threshold": 0.97,
            "candidate_local_weight": 0.80,
            "candidate_extent_weight": 0.90,
            "candidate_extent_quota_fraction": 0.70,
            "candidate_target_matches": 4,
            "candidate_target_coverage": 0.10,
            "surface_preview_points": 45000,
            "layer_preview_points": 18000,
            "parallel_scenes": min(3, max(1, (os.cpu_count() or 2) // 2)),
        }
    values = asdict(cfg.keypoint)
    values.update({"group_"+key: item[0] for key,item in GROUP_CONTROLS.items()})
    values.update({
        "floor_height": cfg.matching.floor_height,
        "max_scan_keypoints": cfg.matching.max_scan_keypoints,
        "spatial_keypoint_balance": True,
        "spatial_keypoint_cell": cfg.matching.spatial_keypoint_cell,
        "min_2d_corner_fraction": cfg.matching.min_2d_corner_fraction,
        "descriptor_keypoint_filter_enabled": cfg.matching.descriptor_keypoint_filter_enabled,
        "descriptor_min_occupied": cfg.matching.descriptor_min_occupied,
        "descriptor_max_hole_fraction": cfg.matching.descriptor_max_hole_fraction,
        "descriptor_keypoint_min_features": cfg.matching.descriptor_keypoint_min_features,
        "descriptor_keypoint_min_absolute": cfg.matching.descriptor_keypoint_min_absolute,
        "descriptor_keypoint_min_ratio": cfg.matching.descriptor_keypoint_min_ratio,
        "descriptor_keypoint_max_reintroduced_fraction": (
            cfg.matching.descriptor_keypoint_max_reintroduced_fraction
        ),
        "quality_nms_radius": cfg.matching.quality_nms_radius,
        "quality_min_score_ratio": cfg.matching.quality_min_score_ratio,
        "surface_preview_points": 45000,
        "layer_preview_points": 18000,
        "parallel_scenes": min(3, max(1, (os.cpu_count() or 2) // 2)),
    })
    return values


PARAMETER_SCHEMA = (
    ("neighbor_radius", 'Neighborhood radius', "float", 0.01, 0.20, 0.005, "m"),
    ("curvature_threshold", 'Curvature threshold', "float", 0.0, 0.20, 0.002, ""),
    ("harris_k", "Harris k", "float", 0.005, 0.12, 0.002, ""),
    ("harris_threshold", 'Harris threshold', "float", -0.05, 0.10, 0.001, ""),
    ("harris_reference_neighbors", 'Reference neighbors', "float", 3, 30, 1, ""),
    ("convex_hull_ratio", '2D hull ratio', "float", 0.1, 3.2, 0.05, ""),
    ("corner_plane_radius_factor", '2D support range', "float", 1, 8, 0.25, "×r"),
    ("corner_plane_min_area", '2D support area', "float", 0, 0.15, 0.005, "m²"),
    ("corner_plane_normal_cos", 'Normal consistency', "float", 0.5, 1, 0.01, ""),
    ("nms_radius", 'NMS radius', "float", 0.005, 0.30, 0.005, "m"),
    ("adjust_iterations", 'Adjustment iterations', "int", 0, 12, 1, ""),
    ("jitter_reject", 'Maximum displacement', "nullable_float", 0, 0.30, 0.005, "m"),
    ("dedup_radius", 'Deduplication radius', "float", 0.001, 0.15, 0.002, "m"),
    ("pre_harris_visibility_filter_enabled", 'Pre-Harris visibility', "bool", None, None, None, ""),
    ("pre_harris_visibility_probe_radius", 'Visibility prefilter radius', "float", 0, 0.20, 0.005, "m"),
    ("pre_harris_visibility_max_fraction", 'Visibility prefilter fraction', "float", 0, 1, 0.05, ""),
    ("wall_vertical_normal_cos", 'Wall verticality', "float", 0.05, 0.70, 0.01, ""),
    ("wall_plane_normal_cos", 'Wall plane consistency', "float", 0.70, 0.999, 0.005, ""),
    ("wall_plane_angle_degrees", 'Wall orientation step', "float", 2, 30, 1, "°"),
    ("wall_plane_distance", 'Wall plane tolerance', "float", 0.01, 0.20, 0.005, "m"),
    ("wall_plane_min_points", 'Minimum wall support', "int", 20, 5000, 20, "pts"),
    ("wall_plane_min_width", 'Minimum wall width', "float", 0.20, 4, 0.10, "m"),
    ("wall_plane_min_height", 'Minimum wall height', "float", 0.20, 4, 0.10, "m"),
    ("wall_plane_min_area", 'Minimum wall area', "float", 0.10, 12, 0.10, "m²"),
    ("wall_plane_max_count", 'Maximum walls', "int", 1, 16, 1, ""),
    ("wall_plane_sample_points", 'Wall voting sample', "int", 1000, 100000, 1000, "pts"),
    ("wall_small_radius", 'Local wall support', "float", 0.02, 0.30, 0.01, "m"),
    ("wall_large_radius", 'Wider wall support', "float", 0.05, 0.60, 0.01, "m"),
    ("wall_protrusion_radius", 'Object protection radius', "float", 0.05, 0.80, 0.01, "m"),
    ("wall_protrusion_distance", 'Object-wall separation', "float", 0.02, 0.30, 0.01, "m"),
    ("wall_object_protection", 'Object protection', "float", 0, 1, 0.05, ""),
    ("wall_filter_enabled", 'Enable wall filter', "bool", None, None, None, ""),
    ("wall_penalty_weight", 'Wall penalty weight', "float", 0, 0.30, 0.01, ""),
    ("wall_hard_reject", 'Hard wall rejection', "bool", None, None, None, ""),
    ("wall_reject_threshold", 'Wall rejection threshold', "float", 0, 1, 0.05, ""),
    ("wall_reject_max_object_score", 'Maximum object evidence for rejection', "float", 0, 1, 0.05, ""),
    ("wall_budget_enabled", 'Enable wall budget', "bool", None, None, None, ""),
    ("wall_budget_affinity_threshold", 'Wall budget affinity', "float", 0, 1, 0.01, ""),
    ("wall_budget_proximity_threshold", 'Wall budget proximity', "float", 0, 1, 0.01, ""),
    ("wall_budget_max_fraction", 'Maximum wall fraction', "float", 0, 1, 0.01, ""),
    ("wall_budget_cell_size", 'Wall budget cell', "float", 0.05, 2, 0.05, "m"),
    ("wall_budget_max_per_cell", "Wall points per cell", "int", 1, 20, 1, ""),
    ("wall_budget_min_features", 'Post-quota minimum', "int", 0, 600, 5, ""),
    ("component_budget_enabled", 'Enable regional budget', "bool", None, None, None, ""),
    ("component_budget_radius", 'Region radius', "float", 0.05, 1.0, 0.05, "m"),
    ("component_budget_max_per_component", 'Points per region', "int", 1, 200, 1, ""),
    ("component_budget_min_features", 'Global regional minimum', "int", 0, 600, 5, ""),
    ("planar_filter_enabled", 'Planar interior filter', "bool", None, None, None, ""),
    ("planar_filter_radius", 'Planar radius', "float", 0.03, 0.50, 0.01, "m"),
    ("planar_filter_max_residual", 'Maximum plane residual', "float", 0.001, 0.08, 0.001, "m"),
    ("planar_filter_normal_cos", 'Planar consistency', "float", 0.5, 1, 0.01, ""),
    ("planar_filter_angular_coverage", 'Angular coverage', "float", 0, 1, 0.05, ""),
    ("planar_filter_min_neighbors", "Planar neighbors", "int", 6, 500, 1, ""),
    ("hole_boundary_filter_enabled", "Hole boundary filter", "bool", None, None, None, ""),
    ("hole_boundary_probe_radius", 'Hole probe radius', "float", 0, 0.20, 0.005, "m"),
    ("hole_boundary_max_fraction", 'Maximum hole fraction', "float", 0, 1, 0.05, ""),
    ("repeatability_filter_enabled", 'Multi-scale repeatability', "bool", None, None, None, ""),
    ("repeatability_radius_factor", 'Repeatability radius factor', "float", 1, 4, 0.1, ""),
    ("repeatability_response_ratio", 'Repeatable response ratio', "float", 0, 2, 0.05, ""),
    ("quality_response_ratio", "Relative Harris score", "float", 0, 1, 0.05, ""),
    ("quality_score_radius", 'Local score radius', "float", 0.05, 2, 0.05, "m"),
    ("geometric_nms_radius", 'Geometric NMS', "float", 0, 0.50, 0.01, "m"),
    ("quality_filter_min_features", 'Quality filter minimum', "int", 0, 600, 5, ""),
    ("quality_filter_min_absolute", 'Absolute quality minimum', "int", 0, 600, 5, ""),
    ("quality_filter_min_ratio", 'Relative quality minimum', "float", 0, 1, 0.05, ""),
    ("quality_filter_max_reintroduced_fraction", 'Maximum quality reintroduction', "float", 0, 1, 0.05, ""),
    ("floor_height", 'Margin above ground', "float", 0, 0.50, 0.01, "m"),
    ("max_scan_keypoints", "Keypoint budget", "int", 0, 5000, 50, ""),
    ("spatial_keypoint_balance", 'Spatial balancing', "bool", None, None, None, ""),
    ("spatial_keypoint_cell", "Spatial cell", "float", 0.05, 2, 0.05, "m"),
    ("min_2d_corner_fraction", 'Reserved 2D budget', "float", 0, 1, 0.05, ""),
    ("descriptor_keypoint_filter_enabled", "Descriptor filter", "bool", None, None, None, ""),
    ("descriptor_min_occupied", 'Minimum occupied cells', "int", 1, 200, 1, ""),
    ("descriptor_max_hole_fraction", "Maximum hole fraction", "float", 0, 1, 0.05, ""),
    ("descriptor_keypoint_min_features", "Descriptor minimum", "int", 0, 600, 5, ""),
    ("descriptor_keypoint_min_absolute", 'Absolute descriptor minimum', "int", 0, 600, 5, ""),
    ("descriptor_keypoint_min_ratio", 'Relative descriptor minimum', "float", 0, 1, 0.05, ""),
    ("descriptor_keypoint_max_reintroduced_fraction", 'Maximum descriptor reintroduction', "float", 0, 1, 0.05, ""),
    ("quality_nms_radius", 'Quality NMS', "float", 0, 0.50, 0.01, "m"),
    ("quality_min_score_ratio", 'Minimum relative score', "float", 0, 1, 0.05, ""),
    ("parallel_scenes", 'Parallel scenes', "int", 1, 3, 1, ""),
)

FUSION_PARAMETER_SCHEMA = (
    ("voxel_size", 'Voxel size', "float", 0.005, 0.05, 0.005, "m"),
    ("truncation", "TSDF band", "float", 0.01, 0.20, 0.005, "m"),
    ("depth_trunc", 'Maximum depth', "float", 0.5, 10.0, 0.1, "m"),
    ("frame_stride", 'Maximum frame stride', "int", 1, 64, 1, "frames"),
    ("min_frames", "Minimum frames", "int", 0, 5000, 10, ""),
    ("max_frames", "Maximum frames", "int", 0, 5000, 10, ""),
    ("iso_sampling", "Iso-surface spacing", "float", 0.002, 0.05, 0.002, "m"),
    ("max_surface_points", "Surface point limit", "int",
     10000, 1000000, 10000, "points"),
    ("surface_field", 'Surface field', "enum", None, None, None, "",
     ("raw", "smoothed")),
    ("surface_min_support", "Surface support", "float", 0.0, 1.0, 0.05, ""),
    ("surface_extraction", "Surface extraction", "enum",
     None, None, None, "", ("zero_crossing", "hybrid", "near_zero")),
    ("surface_band_factor", 'Recovery band', "float",
     0.25, 3.0, 0.05, "voxel"),
    ("surface_rescue_distance_factor", 'Recovery distance', "float",
     0.0, 3.0, 0.05, "voxel"),
    ("surface_rescue_min_weight", 'Minimum recovery weight', "float",
     0.0, 64.0, 1.0, "observations"),
    ("gradient_smoothing_sigma", 'Normal smoothing', "float",
     0.0, 2.0, 0.1, "voxel"),
    ("depth_edge_threshold", 'Silhouette threshold', "float",
     0.0, 0.20, 0.01, "m"),
    ("depth_edge_background_weight", 'Background weight at depth edges', "float",
     0.0, 1.0, 0.05, ""),
    ("depth_edge_radius", 'Protection radius', "int",
     0, 4, 1, "pixel"),
    ("visibility_surface_band_factor", 'Occupied half-band', "float",
     0.25, 2.0, 0.05, "voxel"),
    ("volume_layout", 'Volume layout', "enum",
     None, None, None, "", ("auto", "dense", "sparse")),
    ("sparse_block_resolution", 'Block resolution', "int",
     4, 32, 4, "voxels"),
    ("sparse_block_count", 'Block capacity', "int",
     1000, 200000, 1000, "blocks"),
    ("visibility_depth_stride", 'Visibility stride', "int",
     1, 8, 1, "pixels"),
    ("backend", "Fusion backend", "enum", None, None, None, "",
     ("auto", "cpu", "cuda")),
    ("parallel_scenes", 'Parallel scenes', "int", 1, 3, 1, ""),
)

DESCRIPTOR_PARAMETER_SCHEMA = (
    ("occ_grid_res", 'Occupancy resolution', "int", 4, 32, 1, "cells"),
    ("occ_extent", 'Local half-extent', "float", 0.04, 0.40, 0.01, "m"),
    ("distance_unit", 'Distance unit', "float", 0.0, 0.05, 0.001, "m"),
    ("distance_exponent", "Distance exponent", "float", 1, 8, 0.5, ""),
    ("max_occupied", 'Maximum occupied cells', "int", 0, 1000, 25, "cells"),
    ("visibility_surface_band_factor", 'Occupied half-band', "float",
     0.25, 2.0, 0.05, "voxel"),
    ("utility_candidate_models", 'Diagnostic models', "int", 3, 200, 1, 'models'),
    ("utility_model_features", 'Keypoints per model', "int", 4, 120, 4, "points"),
    ("utility_rotations", "Diagnostic rotations", "int", 1, 36, 1, "angles"),
    ("utility_distance_threshold", 'Correspondence threshold', "float", 1, 512, 4, ""),
    ("descriptor_ratio_threshold", "Best/second ratio", "float", 0, 1, 0.005, ""),
    ("utility_margin_threshold", "Informative margin", "float", 0, 1, 0.02, ""),
    ("utility_max_valid_models", 'Maximum compatible models', "int", 1, 100, 1, 'models'),
    ("utility_entropy_threshold", "Maximum informative entropy", "float", 0, 1, 0.05, ""),
    ("parallel_scenes", 'Parallel scenes', "int", 1, 3, 1, ""),
)

QUERY_PARAMETER_SCHEMA = (
    ("candidate_top_k", 'Top-k size', "int", 1, 1000, 1, 'models'),
    ("candidate_pool_k", 'Global pool size', "int", 1, 5000, 10, 'models'),
    ("candidate_local_extra_k", 'Additional local candidates (0 = disabled)', "int", 0, 500, 8, 'models'),
    ("candidate_diverse", "Diversify pool", "bool", None, None, None, ""),
    ("candidate_diversity", 'Diversity fraction', "float", 0, 1, 0.05, ""),
    ("local_rerank", "Local reranking", "bool", None, None, None, ""),
    ("grouped_query", 'Grouped query', "bool", None, None, None, ""),
    ("candidate_min_group_descriptors", "Minimum group support", "int", 2, 100, 1, "descriptors"),
    ("candidate_max_groups", 'Source groups tested (0 = all)', "int", 0, 20, 1, "groups"),
    ("candidate_group_weighting", 'Weight group confidence', "bool", None, None, None, ""),
    ("candidate_pool_min_per_group", 'Pool minimum per group', "int", 0, 200, 1, 'models'),
    ("candidate_global_pool_fraction", 'Global pool reserve', "float", 0.0, 1.0, 0.05, "fraction"),
    ("candidate_top_min_per_group", 'Top-k minimum per group', "int", 0, 100, 1, 'models'),
    ("candidate_descriptor_ratio_threshold", "Local best/second ratio", "float", 0, 1, 0.005, ""),
    ("candidate_local_weight", "Local evidence weight", "float", 0, 1, 0.05, ""),
    ("candidate_extent_weight", 'Object extent weight', "float", 0, 1, 0.05, ""),
    ("candidate_extent_quota_fraction", 'Extent-based Top-k quota', "float", 0, 1, 0.05, "fraction"),
    ("candidate_target_matches", 'Matches for full support', "int", 1, 20, 1, "points"),
    ("candidate_target_coverage", 'Coverage for full support', "float", 0.01, 1, 0.01, ""),
    ("surface_preview_points", 'Displayed surface points', "int", 5000, 200000, 5000, "points"),
    ("parallel_scenes", 'Parallel scenes', "int", 1, 3, 1, ""),
)


PARAMETER_HELP = {key: value for key, value in english_parameter_help("PARAMETER_HELP").items() if key in {row[0] for row in PARAMETER_SCHEMA}}

PARAMETER_HELP.update({
    "planar_filter_enabled": {
        "summary": 'Remove maxima inside a coherent planar surface.',
        "detail": 'Fit a local plane. Remove a point only when residuals are small, normals agree and neighbors surround it, preserving edges.',
        "higher": 'Enabled: fewer wall and tabletop points, with some risk on very smooth objects.',
        "lower": 'Disabled: greater raw recall, but large-surface noise reaches matching.',
        "cost": 'Adds local PCA per keypoint, cheaper than the matching it may avoid.',
    },
    "planar_filter_radius": {
        "summary": 'Scale used to recognize a planar interior.',
        "detail": 'Neighbors within this radius support the plane fit and angular coverage measurement.',
        "higher": 'Stabilizes wall decisions but may combine an edge with a neighboring plane.',
        "lower": 'Preserves small details but becomes sensitive to local noise.',
        "cost": 'A larger radius increases the number of neighbors examined.',
    },
    "planar_filter_max_residual": {
        "summary": 'Maximum RMS residual for a neighborhood to be considered planar.',
        "detail": 'The smallest PCA variance measures point-cloud thickness around the local plane.',
        "higher": 'Rejects noisier planes but may remove slightly curved shapes.',
        "lower": 'Removes only very clean planes and leaves more wall noise.',
        "cost": 'Changes selectivity rather than test cost.',
    },
    "planar_filter_normal_cos": {
        "summary": 'Minimum normal agreement with the local plane.',
        "detail": 'Protects neighborhoods whose positions appear flat but whose orientations reveal a real corner.',
        "higher": 'More conservative decisions and better corner recall.',
        "lower": 'Rejects more noisy surfaces, with greater risk to objects.',
        "cost": 'Low cost; normals are already available.',
    },
    "planar_filter_angular_coverage": {
        "summary": 'Fraction of directions around the keypoint that must contain planar support.',
        "detail": 'An interior point is surrounded by surface; an edge has neighbors mainly on one side.',
        "higher": 'Protects edges more strongly but leaves some incomplete planar interiors.',
        "lower": 'Removes more incomplete planes, risking confusion with their boundaries.',
        "cost": 'Low cost after neighborhood search.',
    },
    "planar_filter_min_neighbors": {
        "summary": 'Minimum support before classifying a point as planar.',
        "detail": 'Undersampled neighborhoods remain undecidable and are retained.',
        "higher": 'More conservative filtering in sparse regions.',
        "lower": 'Filters more often using less reliable planes.',
        "cost": 'Little effect on computation cost.',
    },
    "hole_boundary_filter_enabled": {
        "summary": 'Remove keypoints on unconfirmed invalid-depth boundaries.',
        "detail": 'Probe fusion visibility near the keypoint without treating all unknown space as an error.',
        "higher": 'Enabled: reduces false corners around RGB-D holes.',
        "lower": 'Disabled: preserves these boundaries, which may produce false matches.',
        "cost": 'A few volume probes per keypoint.',
    },
    "pre_harris_visibility_filter_enabled": {
        "summary": 'Remove definite RGB-D boundaries before Harris.',
        "detail": 'Probe the same free/unknown state as the hole filter just after curvature, with a more conservative threshold.',
        "higher": 'Enabled: fewer false corners and less Harris computation.',
        "lower": 'Disabled: maximum recall, but holes reach Harris and the 2D branch.',
        "cost": 'Seven inexpensive volume probes per curvature candidate.',
    },
    "pre_harris_visibility_probe_radius": {
        "summary": 'Probe distance used before Harris.',
        "detail": 'Six axial probes around the central cell detect nearby visibility boundaries.',
        "higher": 'Captures wider holes but may reach a neighboring boundary.',
        "lower": 'Very local decisions, less effective on thick holes.',
        "cost": "Constant probe count.",
    },
    "pre_harris_visibility_max_fraction": {
        "summary": 'Evidence needed for rejection before Harris.',
        "detail": 'A higher value makes this prefilter conservative. The post-Harris hole filter remains stricter.',
        "higher": 'Preserves more candidates and saves less computation.',
        "lower": 'Saves more computation but may remove incomplete object contours.',
        "cost": 'No effect on probe cost.',
    },
    "hole_boundary_probe_radius": {
        "summary": 'Visibility probe distance around the keypoint.',
        "detail": 'Six axial probes supplement the central cell to detect nearby boundaries.',
        "higher": 'Captures wider holes but may reach an unrelated nearby boundary.',
        "lower": 'More local decisions, sometimes insufficient on thick surfaces.',
        "cost": "The probe count remains constant.",
    },
    "hole_boundary_max_fraction": {
        "summary": 'Maximum fraction of probes marked as hole boundary.',
        "detail": 'A flagged center always rejects the point; this threshold controls boundaries found only around it.',
        "higher": "More conservative filtering and higher recall.",
        "lower": 'Rejects neighborhoods around holes more broadly.',
        "cost": 'No significant runtime effect.',
    },
    "repeatability_filter_enabled": {
        "summary": 'Require a 3D Harris corner to remain strong at a second scale.',
        "detail": 'Recompute the response in a wider neighborhood. 2D corners use their own support test and are preserved.',
        "higher": 'Enabled: remove maxima caused by noise at a single scale.',
        "lower": 'Disabled: greater recall but more unstable points.',
        "cost": 'A second normal covariance per 3D keypoint.',
    },
    "repeatability_radius_factor": {
        "summary": 'Ratio between the verification scale and the initial Harris radius.',
        "detail": 'A value above one tests persistence as the neighborhood grows.',
        "higher": 'Requires wider structure and may remove fine details.',
        "lower": "Closer to the initial test, with less noise discrimination.",
        "cost": 'Neighbor count grows with this factor.',
    },
    "repeatability_response_ratio": {
        "summary": 'Minimum Harris response expected at the larger scale.',
        "detail": 'Multiply the initial threshold by this ratio to assess stability.',
        "higher": 'Retains only strong multi-scale corners.',
        "lower": 'Retains weaker structures and more noise.',
        "cost": 'No effect on computation cost.',
    },
    "quality_response_ratio": {
        "summary": 'Remove weak 3D Harris responses relative to strong scene corners.',
        "detail": 'Threshold equals this ratio times the local 90th percentile of 3D Harris responses. 2D corners use separate validation.',
        "higher": 'Strongly reduces noise but may remove poorly reconstructed objects.',
        "lower": 'Preserves recall at the cost of less distinctive points.',
        "cost": 'Reduces downstream descriptor and matching cost.',
    },
    "quality_score_radius": {
        "summary": 'Spatial range used to compare Harris corner strength.',
        "detail": 'Compute the reference locally so a weakly reconstructed object is not compared with strong corners elsewhere in the room.',
        "higher": 'Smooths quality more broadly but can introduce bias between objects.',
        "lower": 'Protects small structures; very low values barely filter.',
        "cost": 'Low neighborhood-search cost.',
    },
    "geometric_nms_radius": {
        "summary": 'Minimum distance between keypoints after quality filtering.',
        "detail": 'Keep the highest-scoring point in each neighborhood. This NMS runs before descriptor computation.',
        "higher": 'Produces a smaller, more distributed selection, with some risk to small objects.',
        "lower": 'Retains nearby point clusters and increases matching cost.',
        "cost": 'Speeds up subsequent stages; NMS itself is inexpensive.',
    },
    "quality_filter_min_features": {
        "summary": 'Upper bound on the adaptive minimum after quality filtering.',
        "detail": 'The effective minimum combines an absolute minimum and an input fraction, capped by this value. Definite free/unknown boundaries are never restored.',
        "higher": 'Protects incomplete objects but reduces precision and runtime gains.',
        "lower": 'Allows compact selections but may miss objects.',
        "cost": 'Indirectly controls downstream cost.',
    },
    "quality_filter_min_absolute": {
        "summary": 'Absolute target minimum for soft filters.',
        "detail": 'Protects small scenes, subject to the restoration cap and rejection eligibility.',
        "higher": 'More conservative recall, with more weak points restored.',
        "lower": 'More compact selections in small scenes.',
        "cost": 'Controls downstream work.',
    },
    "quality_filter_min_ratio": {
        "summary": 'Input fraction targeted by the adaptive minimum.',
        "detail": 'Adapts the minimum to scene complexity instead of always requiring 80 points.',
        "higher": "Restore more soft rejections.",
        "lower": 'Allows a stronger reduction.',
        "cost": 'Indirectly controls downstream cost.',
    },
    "quality_filter_max_reintroduced_fraction": {
        "summary": 'Maximum fraction of soft rejections eligible for restoration.',
        "detail": 'Prevents the minimum from nearly undoing a filter. Hard rejections always remain excluded.',
        "higher": "Meilleur rappel, filtre potentiellement moins effectif.",
        "lower": 'More stable decisions and a smaller selection.',
        "cost": 'No direct computation cost.',
    },
    "descriptor_keypoint_filter_enabled": {
        "summary": 'Use intrinsic local-grid quality before matching.',
        "detail": 'Remove empty descriptors or descriptors dominated by removed hole-boundary cells before model comparison.',
        "higher": 'Enabled: fewer wasted comparisons and false matches.',
        "lower": 'Disabled: retain all descriptors, including weakly informative ones.',
        "cost": 'Descriptors must already exist; savings mainly affect matching.',
    },
    "descriptor_min_occupied": {
        "summary": 'Minimum surface-cell count in a scan descriptor.',
        "detail": 'A nearly empty grid has insufficient geometry for reliable correspondence.',
        "higher": 'Richer descriptors, with a risk to legs and very thin objects.',
        "lower": 'Better thin-structure recall but greater ambiguity.',
        "cost": 'Reduces the number of descriptors passed to matching.',
    },
    "descriptor_max_hole_fraction": {
        "summary": 'Maximum support fraction assigned to removed hole boundaries.',
        "detail": 'Compare reliable occupied cells with cells invalidated by RGB-D visibility.',
        "higher": 'Tolerates more incomplete reconstructions.',
        "lower": 'More strictly excludes descriptors dominated by holes.',
        "cost": 'No additional computation once the local grid exists.',
    },
    "descriptor_keypoint_min_features": {
        "summary": 'Upper bound on the adaptive descriptor minimum.',
        "detail": 'Empty or hole-dominated descriptors remain hard rejections. Only score or redundancy rejections can be restored.',
        "higher": 'Protects recall for incomplete desk observations.',
        "lower": 'Maximizes matching-time reduction.',
        "cost": 'Sets a lower target for the number of compared features.',
    },
    "descriptor_keypoint_min_absolute": {
        "summary": 'Absolute target minimum of admissible descriptors.',
        "detail": 'Protects small scenes without restoring intrinsically invalid descriptors.',
        "higher": "Davantage d'ancres faibles peuvent revenir.",
        "lower": "Matching plus compact.",
        "cost": 'Controls matching cost.',
    },
    "descriptor_keypoint_min_ratio": {
        "summary": 'Target fraction of input descriptors.',
        "detail": 'Adapts the minimum to what remains after geometric filtering.',
        "higher": "More recall and redundancy.",
        "lower": 'Greater selectivity.',
        "cost": 'Controls matching cost.',
    },
    "descriptor_keypoint_max_reintroduced_fraction": {
        "summary": 'Maximum fraction of recoverable descriptor rejections.',
        "detail": 'Restore only the best soft rejections after scoring and descriptor NMS.',
        "higher": 'Recall plus prudent.',
        "lower": 'More stable and faster filtering.',
        "cost": 'No direct computation cost.',
    },
    "quality_nms_radius": {
        "summary": 'Second non-maximum suppression after descriptor computation.',
        "detail": 'Between nearby anchors, retain the one combining stronger response and more reliable volume support.',
        "higher": 'Fewer, better-spaced points, but distinct details may be merged.',
        "lower": 'Preserves nearby details and more redundancy.',
        "cost": 'Low cost, offset by reduced matching work.',
    },
    "quality_min_score_ratio": {
        "summary": 'Relative utility threshold compared with strong Harris keypoints.',
        "detail": 'Combine geometric response, occupied-cell count and hole contamination. 2D corners remain separate.',
        "higher": 'Compact selection dominated by the strongest corners.',
        "lower": "Preserve more weak points and increase recall.",
        "cost": 'Directly reduces the number of compared features.',
    },
})

FUSION_PARAMETER_HELP = {key: value for key, value in english_parameter_help("FUSION_PARAMETER_HELP").items() if key in {row[0] for row in FUSION_PARAMETER_SCHEMA}}

DESCRIPTOR_PARAMETER_HELP = {key: value for key, value in english_parameter_help("DESCRIPTOR_PARAMETER_HELP").items() if key in {row[0] for row in DESCRIPTOR_PARAMETER_SCHEMA}}

QUERY_PARAMETER_HELP = {key: value for key, value in english_parameter_help("QUERY_PARAMETER_HELP").items() if key in {row[0] for row in QUERY_PARAMETER_SCHEMA}}


PARAMETER_HELP.update({"group_"+key: dict(summary=item[5], detail=item[5],
    higher=item[5], lower=item[5], cost='Recompute groups after detection without repeating fusion.')
    for key,item in GROUP_CONTROLS.items()})


def parameter_schema(stage="keypoints"):
    if stage in ("matching", "verification", "selection"):
        from debug_retrieval import schema
        return schema(stage)
    if stage == "fusion":
        schema, help_text = FUSION_PARAMETER_SCHEMA, FUSION_PARAMETER_HELP
    elif stage == "descriptors":
        schema, help_text = (
            DESCRIPTOR_PARAMETER_SCHEMA, DESCRIPTOR_PARAMETER_HELP,
        )
    elif stage == "query":
        schema, help_text = QUERY_PARAMETER_SCHEMA, QUERY_PARAMETER_HELP
    else:
        schema, help_text = PARAMETER_SCHEMA, PARAMETER_HELP
    result = [
        {
            "id": item[0], "label": item[1], "type": item[2],
            "min": item[3], "max": item[4], "step": item[5], "unit": item[6],
            **({"choices": list(item[7])} if len(item) > 7 else {}),
            **help_text[item[0]],
        }
        for item in schema
    ]
    if stage == "keypoints":
        result.extend(dict(id="group_"+key, label=item[1], type="float", min=item[2],
            max=item[3], step=item[4], unit="m", **PARAMETER_HELP["group_"+key])
            for key,item in GROUP_CONTROLS.items())
    return result


def normalized_parameters(raw=None, stage="keypoints"):
    parameters = default_parameters(stage)
    parameters.update(raw or {})
    defaults = default_parameters(stage)
    for spec in parameter_schema(stage):
        key = spec["id"]
        value = parameters.get(key, defaults.get(key))
        if spec["type"] == "bool":
            parameters[key] = bool(value)
            continue
        if spec["type"] == "enum":
            value = str(value or defaults.get(key))
            if value not in spec.get("choices", []):
                raise ValueError(
                    f"{key} must be one of {', '.join(spec.get('choices', []))}"
                )
            parameters[key] = value
            continue
        if spec["type"] == "nullable_float" and value in (None, "", "auto"):
            parameters[key] = None
            continue
        caster = int if spec["type"] == "int" else float
        value = caster(value)
        if spec["min"] is not None and value < spec["min"]:
            raise ValueError(f"{key} must be >= {spec['min']}")
        if spec["max"] is not None and value > spec["max"]:
            raise ValueError(f"{key} must be <= {spec['max']}")
        parameters[key] = value
    return parameters


def normalized_payload(raw=None):
    raw = dict(raw or {})
    stage = str(raw.get("stage") or "keypoints")
    available = {
        item["id"] for item in PIPELINE_STAGES if item.get("available")
    }
    if stage not in available:
        raise ValueError(f"Stage de debug indisponible : {stage}")
    parameters = normalized_parameters(raw.get("parameters"), stage=stage)
    sources = []
    for index, item in enumerate(raw.get("sources") or default_sources()):
        if not item.get("enabled", True):
            continue
        default_path = (item.get("stage_paths") or {}).get(stage)
        path = Path(str(default_path or item.get("path") or "")).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source not found : {path}")
        if stage == "fusion" and path.suffix.lower() != ".zip":
            raise ValueError(
                f"RGB-D fusion requires a .zip archive: {path}"
            )
        if stage == "descriptors" and path.suffix.lower() != ".pkl":
            raise ValueError(
                "The Descriptors stage expects a Keypoints .pkl artifact"
            )
        source_id = re.sub(r"[^a-z0-9-]+", "-", str(item.get("id") or path.stem).lower()).strip("-")
        source = {
            "id": source_id or f"scene-{index + 1}",
            "label": str(item.get("label") or path.stem),
            "path": str(path), "enabled": True,
        }
        upstream = item.get("upstream")
        if isinstance(upstream, dict) and upstream.get("run_id"):
            source["upstream"] = {
                "run_id": str(upstream["run_id"]),
                "stage": str(upstream.get("stage") or "fusion"),
                "scene_id": str(upstream.get("scene_id") or source["id"]),
                "artifact": str(upstream.get("artifact") or "geometry"),
            }
        sources.append(source)
    if not sources:
        raise ValueError('Enable at least one debug scene')
    return {"stage": stage, "parameters": parameters, "sources": sources}


def _source_signature(path):
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|v3".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def _atomic_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(str(temporary), str(path))


def _load_scene_state(path):
    path = Path(path)
    if path.suffix.lower() != ".pkl":
        return None
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    scene = payload.get("scene", payload) if isinstance(payload, dict) else payload
    return scene if hasattr(scene, "cloud") and hasattr(scene, "volume") else None


def _subset_keypoints(keypoints, local_indices):
    indices = np.asarray(local_indices, int)

    def subset(name):
        values = getattr(keypoints, name, None)
        return None if values is None else np.asarray(values)[indices]

    return KeyPoints(
        positions=np.asarray(keypoints.positions)[indices],
        normals=np.asarray(keypoints.normals)[indices],
        responses=np.asarray(keypoints.responses)[indices],
        source_index=np.asarray(keypoints.source_index)[indices],
        selection_scores=subset("selection_scores"),
        wall_scores=subset("wall_scores"),
        object_scores=subset("object_scores"),
        wall_affinities=subset("wall_affinities"),
        wall_proximities=subset("wall_proximities"),
        wall_plane_indices=subset("wall_plane_indices"),
    )


def _load_cloud(source, progress=None):
    source_path = Path(source["path"])
    cache_path = CACHE_ROOT / f"{source['id']}_{_source_signature(source_path)}.npz"
    if cache_path.exists():
        cached = np.load(str(cache_path))
        confidence = (
            cached["confidence"] if "confidence" in cached else None
        )
        colors = cached["colors"] if "colors" in cached else None
        return (
            PointCloud(
                cached["points"], cached["normals"], cached["curvature"],
                confidence, colors,
            ),
            cached["up"], float(cached["ground"]), True,
        )

    if progress:
        progress('Preparing geometry cache')
    suffix = source_path.suffix.lower()
    if suffix == ".pkl":
        with source_path.open("rb") as stream:
            payload = pickle.load(stream)
        scene = payload.get("scene", payload)
        cloud = scene.cloud
        up = np.asarray(getattr(scene, "up", [0.0, 1.0, 0.0]), float)
        ground = float(getattr(scene, "ground", estimate_ground_plane(cloud.points)[1]))
    elif suffix == ".npz":
        payload = np.load(str(source_path))
        confidence = payload["confidence"] if "confidence" in payload else None
        colors = payload["colors"] if "colors" in payload else None
        cloud = PointCloud(
            payload["points"], payload["normals"], payload["curvature"],
            confidence, colors,
        )
        up = np.asarray(payload["up"] if "up" in payload else [0.0, 1.0, 0.0], float)
        ground = float(payload["ground"] if "ground" in payload else estimate_ground_plane(cloud.points)[1])
    elif suffix == ".ply":
        from viz import read_points_ply
        points, _ = read_points_ply(str(source_path))
        cloud = make_point_cloud(points, radius=KeypointConfig().neighbor_radius)
        up = np.array([0.0, 1.0, 0.0])
        _, ground = estimate_ground_plane(cloud.points, up_hint=up)
    else:
        raise ValueError(
            f'Unsupported source format: {source_path.suffix}. Use a scene_*.pkl cache, an .npz file or a .ply scan.'
        )

    arrays = {
        "points": cloud.points,
        "normals": cloud.normals,
        "curvature": cloud.curvature,
        "up": np.asarray(up),
        "ground": np.asarray(ground),
    }
    if getattr(cloud, "confidence", None) is not None:
        arrays["confidence"] = cloud.confidence
    if getattr(cloud, "colors", None) is not None:
        arrays["colors"] = cloud.colors
    _atomic_npz(cache_path, **arrays)
    return cloud, np.asarray(up), float(ground), False


def _display_basis(up):
    up = np.asarray(up, float)
    up /= np.clip(np.linalg.norm(up), 1e-12, None)
    x = np.array([1.0, 0.0, 0.0])
    x -= (x @ up) * up
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 0.0, 1.0])
        x -= (x @ up) * up
    x /= np.linalg.norm(x)
    z = np.cross(x, up)
    z /= np.clip(np.linalg.norm(z), 1e-12, None)
    return x, up, z


def _to_display(points, up, ground):
    points = np.asarray(points, float).reshape(-1, 3)
    x, y, z = _display_basis(up)
    return np.column_stack([points @ x, points @ y - ground, points @ z])


def _sample_rows(count, limit, seed=0):
    if count <= limit or limit <= 0:
        return np.arange(count, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, int(limit), replace=False))


def _project_rgb_to_surface(surface_points, measurement_points,
                            measurement_colors, max_distance):
    """Project sparse RGB-D samples onto surface points for debug display."""
    surface_points = np.asarray(surface_points, float).reshape(-1, 3)
    measurement_points = np.asarray(
        measurement_points, float,
    ).reshape(-1, 3)
    measurement_colors = np.asarray(
        measurement_colors, float,
    ).reshape(-1, 3)
    fallback = np.asarray([0.57, 0.60, 0.62], float)
    colors = np.tile(fallback, (len(surface_points), 1))
    valid = np.zeros(len(surface_points), bool)
    if not len(surface_points) or not len(measurement_points):
        return colors, valid

    neighbor_count = min(3, len(measurement_points))
    distances, neighbors = cKDTree(measurement_points).query(
        surface_points, k=neighbor_count,
    )
    distances = np.asarray(distances, float).reshape(-1, neighbor_count)
    neighbors = np.asarray(neighbors, int).reshape(-1, neighbor_count)
    valid = distances[:, 0] <= float(max_distance)
    weights = 1.0 / np.maximum(distances, 1e-5)
    projected = np.sum(
        measurement_colors[neighbors] * weights[..., None], axis=1,
    ) / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    colors[valid] = projected[valid]
    return np.clip(colors, 0.0, 1.0), valid


def _rounded(values, digits=4):
    return np.round(np.asarray(values, float), digits).tolist()


def _point_layer(layer_id, label, points, source_index, cloud, response_by_source,
                 up, ground, limit, role="active", reasons=None,
                 positions_are_display=False, attributes=None):
    points = np.asarray(points, float).reshape(-1, 3)
    source_index = np.asarray(source_index, int).reshape(-1)
    selection = _sample_rows(len(points), int(limit), seed=sum(map(ord, layer_id)))
    selected_sources = source_index[selection]
    display = points[selection] if positions_are_display else _to_display(points[selection], up, ground)
    payload = {
        "id": layer_id, "label": label, "kind": "points", "role": role,
        "total": int(len(points)), "sampled": int(len(selection)),
        "points": _rounded(display),
        "source_index": selected_sources.astype(int).tolist(),
        "curvature": _rounded(cloud.curvature[selected_sources], 6),
        "response": _rounded(response_by_source[selected_sources], 6),
        "height": _rounded(points[selection] @ up - ground, 4),
    }
    confidence = getattr(cloud, "confidence", None)
    if confidence is not None:
        payload["confidence"] = _rounded(
            np.asarray(confidence)[selected_sources], 4,
        )
    if reasons is not None:
        reasons = np.asarray(reasons, object).reshape(-1)
        payload["reason"] = reasons[selection].tolist()
    for key, values in (attributes or {}).items():
        values = np.asarray(values).reshape(-1)
        selected_values = values[selection]
        if np.issubdtype(selected_values.dtype, np.integer):
            payload[key] = selected_values.astype(int).tolist()
        elif np.issubdtype(selected_values.dtype, np.number):
            payload[key] = _rounded(selected_values, 6)
        else:
            payload[key] = selected_values.tolist()
    return payload


def _index_layer(layer_id, label, indices, cloud, response_by_source, up, ground,
                 limit, role="active", reasons=None, attributes=None):
    indices = np.asarray(indices, int).reshape(-1)
    return _point_layer(
        layer_id, label, cloud.points[indices], indices, cloud,
        response_by_source, up, ground, limit, role=role, reasons=reasons,
        attributes=attributes,
    )


def _line_layer(origins, targets, up, ground, limit=3000):
    origins = np.asarray(origins, float).reshape(-1, 3)
    targets = np.asarray(targets, float).reshape(-1, 3)
    selection = _sample_rows(len(origins), limit, seed=41)
    segments = np.stack([
        _to_display(origins[selection], up, ground),
        _to_display(targets[selection], up, ground),
    ], axis=1)
    return {
        "id": "adjustment_vectors", "label": 'Displacements', "kind": "lines",
        "role": "vector", "total": int(len(origins)),
        "segments": _rounded(segments),
    }


def _descriptor_grid_layers(feature, visibility_volume, cloud, source_index,
                            up, ground, context_limit=12000):
    """Build an inspectable local occupancy grid for one scan descriptor."""
    from descriptors import EMPTY, OCCUPIED, UNKNOWN
    from sdf_fusion import VIS_FREE, VIS_OCCUPIED

    descriptor = getattr(feature, "descriptor", None)
    final_grid = getattr(descriptor, "occ", None)
    if final_grid is None:
        return [], None
    final_grid = np.asarray(final_grid, np.int8)
    res = int(getattr(descriptor, "res", final_grid.shape[0]))
    extent = float(getattr(descriptor, "extent", 0.0))
    if final_grid.size != res ** 3 or extent <= 0:
        return [], None

    keypoint = np.asarray(feature.position, float).reshape(3)
    axis = np.linspace(-extent, extent, res)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    offsets = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    grid_points = keypoint + offsets
    final_occ = final_grid.reshape(-1)

    # Rebuild the visibility classification before seed-fill. This keeps the
    # visual trace faithful to build_scan_descriptor without changing it.
    raw_occ = final_occ.copy()
    if hasattr(visibility_volume, "sample_visibility"):
        try:
            if hasattr(visibility_volume, "sample_visibility_details"):
                visibility, hole_boundary = (
                    visibility_volume.sample_visibility_details(grid_points)
                )
            else:
                visibility = visibility_volume.sample_visibility(grid_points)
                hole_boundary = np.zeros(len(grid_points), bool)
            raw_occ = np.full(len(grid_points), UNKNOWN, np.int8)
            raw_occ[np.asarray(visibility) == VIS_FREE] = EMPTY
            raw_occ[np.asarray(visibility) == VIS_OCCUPIED] = OCCUPIED
            raw_occ[
                np.asarray(hole_boundary, bool) & (raw_occ == OCCUPIED)
            ] = UNKNOWN
        except (AttributeError, RuntimeError, ValueError):
            raw_occ = final_occ.copy()

    disconnected = (raw_occ == OCCUPIED) & (final_occ != OCCUPIED)

    def cell_layer(layer_id, label, mask, role):
        points = grid_points[np.asarray(mask, bool)]
        return {
            "id": layer_id, "label": label, "kind": "points", "role": role,
            "total": int(len(points)), "sampled": int(len(points)),
            "points": _rounded(_to_display(points, up, ground)),
            "source_index": [int(source_index)] * len(points),
        }

    delta = cloud.points - keypoint
    context_indices = np.flatnonzero(
        np.max(np.abs(delta), axis=1) <= extent * 1.35
    )
    context_selection = context_indices[
        _sample_rows(len(context_indices), context_limit, seed=83)
    ]
    context_layer = {
        "id": "descriptor_context", "label": "Surface locale",
        "kind": "surface", "role": "surface",
        "total": int(len(context_indices)),
        "sampled": int(len(context_selection)),
        "points": _rounded(_to_display(
            cloud.points[context_selection], up, ground,
        )),
        "curvature": _rounded(cloud.curvature[context_selection], 6),
        "source_index": context_selection.astype(int).tolist(),
    }

    corners = keypoint + np.asarray([
        [x, y, z]
        for x in (-extent, extent)
        for y in (-extent, extent)
        for z in (-extent, extent)
    ], float)
    edge_pairs = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    bounds = np.asarray([
        [_to_display(corners[start:start + 1], up, ground)[0],
         _to_display(corners[end:end + 1], up, ground)[0]]
        for start, end in edge_pairs
    ])
    bounds_layer = {
        "id": "descriptor_grid_bounds", "label": "Volume local",
        "kind": "lines", "role": "vector", "total": len(edge_pairs),
        "segments": _rounded(bounds),
    }
    seed_layer = {
        "id": "descriptor_seed", "label": "Keypoint support",
        "kind": "points", "role": "retained", "total": 1, "sampled": 1,
        "points": _rounded(_to_display(keypoint.reshape(1, 3), up, ground)),
        "source_index": [int(source_index)],
    }
    layers = [
        context_layer, bounds_layer, seed_layer,
        cell_layer(
            "descriptor_raw_unknown", 'Unknown before seed-fill',
            raw_occ == UNKNOWN, "unknown",
        ),
        cell_layer(
            "descriptor_raw_free", 'Free before seed-fill',
            raw_occ == EMPTY, "free",
        ),
        cell_layer(
            "descriptor_raw_occupied", 'Occupied before seed-fill',
            raw_occ == OCCUPIED, "occupied",
        ),
        cell_layer(
            "descriptor_final_unknown", 'Unknown after seed-fill',
            final_occ == UNKNOWN, "unknown",
        ),
        cell_layer(
            "descriptor_final_free", 'Free after seed-fill',
            final_occ == EMPTY, "free",
        ),
        cell_layer(
            "descriptor_final_occupied", 'Occupied after seed-fill',
            final_occ == OCCUPIED, "occupied",
        ),
        cell_layer(
            "descriptor_seed_removed", 'Disconnected components',
            disconnected, "rejected",
        ),
    ]
    return layers, {
        "source_index": int(source_index),
        "resolution": res,
        "extent": round(extent, 4),
        "raw_occupied": int(np.count_nonzero(raw_occ == OCCUPIED)),
        "final_occupied": int(np.count_nonzero(final_occ == OCCUPIED)),
        "free": int(np.count_nonzero(final_occ == EMPTY)),
        "unknown": int(np.count_nonzero(final_occ == UNKNOWN)),
        "seed_removed": int(np.count_nonzero(disconnected)),
    }


def _wall_plane_bounds_layer(planes, up, ground):
    segments = []
    for plane in planes:
        corners = plane.corners(up)
        for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
            segments.append([
                _to_display(corners[start:start + 1], up, ground)[0],
                _to_display(corners[end:end + 1], up, ground)[0],
            ])
    return {
        "id": "wall_plane_bounds", "label": 'Vertical plane bounds',
        "kind": "lines", "role": "wall", "total": int(len(planes)),
        "segments": _rounded(segments),
    }


def _combine_rejections(groups):
    indices = []
    reasons = []
    for reason, values in groups:
        values = np.asarray(values, int).reshape(-1)
        indices.extend(values.tolist())
        reasons.extend([reason] * len(values))
    return np.asarray(indices, int), np.asarray(reasons, object)


def _fusion_scene_result(
        source, parameters, progress=None, artifact_path=None):
    from pipeline import scan_from_zip
    from sdf_fusion import VIS_FREE, VIS_OCCUPIED, VIS_UNKNOWN

    started = time.perf_counter()
    cfg = PipelineConfig()
    for key in asdict(cfg.sdf):
        if key in parameters:
            setattr(cfg.sdf, key, parameters[key])

    trace = {}
    scene = scan_from_zip(
        source["path"], cfg, progress=progress, fusion_trace=trace,
    )
    up = np.asarray(scene.up, float)
    ground = float(scene.ground)
    frames = list(trace.get("frames") or [])
    alignment_rotation = getattr(scene, "alignment_rotation", None)
    rotation = np.asarray(
        alignment_rotation if alignment_rotation is not None else np.eye(3),
        float,
    )

    measurement_points_raw = []
    measurement_colors = []
    measurement_frame_indices = []
    camera_positions_raw = []
    frame_offsets = []
    measurement_count = 0
    for frame_index, frame in enumerate(frames):
        points = np.asarray(frame.get("points") or [], float).reshape(-1, 3)
        colors = np.asarray(frame.get("colors") or [], float).reshape(-1, 3)
        if len(points):
            measurement_points_raw.append(points @ rotation)
            if len(colors) != len(points):
                colors = np.full((len(points), 3), 117.0)
            measurement_colors.append(np.clip(colors / 255.0, 0.0, 1.0))
            measurement_frame_indices.extend(
                [frame_index + 1] * len(points)
            )
            measurement_count += len(points)
        camera = np.asarray(frame.get("camera_position") or [], float)
        if camera.shape == (3,):
            camera_positions_raw.append(camera @ rotation)
        frame_offsets.append(measurement_count)

    if measurement_points_raw:
        measurement_points_raw = np.vstack(measurement_points_raw)
        measurement_colors = np.vstack(measurement_colors)
    else:
        measurement_points_raw = np.zeros((0, 3), float)
        measurement_colors = np.zeros((0, 3), float)

    camera_segments_raw = []
    for start, end in zip(
            camera_positions_raw[:-1], camera_positions_raw[1:]):
        camera_segments_raw.extend([start, end])

    surface_limit = int(parameters.get("surface_preview_points", 45000))
    surface_selection = _sample_rows(
        scene.cloud.size, surface_limit, seed=11,
    )
    aligned_surface = np.asarray(scene.cloud.points, float)
    raw_surface = aligned_surface @ rotation
    raw_surface_preview = raw_surface[surface_selection]
    gravity_surface_preview = _to_display(
        aligned_surface[surface_selection], up, 0.0,
    )
    ground_surface_preview = _to_display(
        aligned_surface[surface_selection], up, ground,
    )
    rgb_projection_distance = max(float(cfg.sdf.voxel_size) * 6.0, 0.06)
    surface_rgb_colors, surface_rgb_valid = _project_rgb_to_surface(
        raw_surface, measurement_points_raw, measurement_colors,
        rgb_projection_distance,
    )
    rgb_surface_available = bool(np.any(surface_rgb_valid))
    if rgb_surface_available:
        scene.cloud.colors = surface_rgb_colors

    artifacts = {}
    if artifact_path is not None:
        artifact_path = Path(artifact_path).resolve()
        if progress:
            progress('Serializing geometry for the next stage')
        artifact_arrays = {
            "points": np.asarray(scene.cloud.points, np.float32),
            "normals": np.asarray(scene.cloud.normals, np.float32),
            "curvature": np.asarray(scene.cloud.curvature, np.float32),
            "up": np.asarray(up, np.float32),
            "ground": np.asarray(ground, np.float32),
        }
        if getattr(scene.cloud, "confidence", None) is not None:
            artifact_arrays["confidence"] = np.asarray(
                scene.cloud.confidence, np.float32,
            )
        if getattr(scene.cloud, "colors", None) is not None:
            artifact_arrays["colors"] = np.asarray(
                scene.cloud.colors, np.float32,
            )
        _atomic_npz(artifact_path, **artifact_arrays)
        artifacts["geometry"] = {
            "path": str(artifact_path),
            "format": "npz",
            "points": int(scene.cloud.size),
            "rgb_colors": rgb_surface_available,
        }
        scene_state_path = artifact_path.with_name(
            artifact_path.name.replace(".geometry.npz", ".scene.pkl")
        )
        if progress:
            progress('Serializing volume for descriptors')
        _atomic_pickle(scene_state_path, {"scene": scene})
        artifacts["scene_state"] = {
            "path": str(scene_state_path),
            "format": "pickle",
            "points": int(scene.cloud.size),
            "rgb_colors": rgb_surface_available,
        }

    aligned_volume = getattr(scene, "volume", None)
    base_volume = getattr(aligned_volume, "volume", aligned_volume)
    visibility = np.asarray(
        getattr(base_volume, "visibility", np.zeros((0,), np.uint8)),
        np.uint8,
    )
    volume_limit = int(parameters.get("volume_preview_points", 12000))
    sparse_visibility = (
        base_volume.debug_visibility_points(volume_limit)
        if not visibility.size and hasattr(
            base_volume, "debug_visibility_points",
        ) else None
    )

    diagnostic_surfaces = {}
    if base_volume is not None and hasattr(
            base_volume, "zero_crossing_surface"):
        for field_mode in ("raw", "smoothed"):
            points, _ = base_volume.zero_crossing_surface(field_mode)
            diagnostic_surfaces[field_mode] = np.asarray(points, float)

    def volume_layer(layer_id, label, mask, role, seed,
                     sparse_label=None):
        if sparse_visibility is not None:
            item = sparse_visibility.get(sparse_label or role, {})
            points = np.asarray(
                item.get("points", []), float,
            ).reshape(-1, 3)
            return {
                "id": layer_id, "label": label, "kind": "points",
                "role": role, "total": int(item.get("total", len(points))),
                "sampled": int(len(points)),
                "points": _rounded(points, 3),
                "source_index": list(range(len(points))),
            }
        flat_indices = np.flatnonzero(np.asarray(mask, bool).reshape(-1))
        selection = _sample_rows(
            len(flat_indices), volume_limit, seed=seed,
        )
        selected = flat_indices[selection]
        if len(selected):
            coordinates = np.column_stack(
                np.unravel_index(selected, visibility.shape),
            )
            points = (
                np.asarray(base_volume.origin, float)
                + coordinates * float(base_volume.voxel_size)
            )
        else:
            points = np.zeros((0, 3), float)
        return {
            "id": layer_id, "label": label, "kind": "points", "role": role,
            "total": int(len(flat_indices)), "sampled": int(len(selected)),
            "points": _rounded(points, 3),
            "source_index": selected.astype(int).tolist(),
        }

    if visibility.size:
        occupied_mask = (visibility & VIS_OCCUPIED) != 0
        free_mask = ((visibility & VIS_FREE) != 0) & ~occupied_mask
        unknown_mask = visibility == VIS_UNKNOWN
    else:
        occupied_mask = free_mask = unknown_mask = np.zeros((0,), bool)

    normal_limit = int(parameters.get("normal_preview_points", 1200))
    normal_selection = _sample_rows(
        scene.cloud.size, normal_limit, seed=37,
    )
    normal_origins = aligned_surface[normal_selection]
    normal_targets = (
        normal_origins
        + np.asarray(scene.cloud.normals, float)[normal_selection]
        * max(float(cfg.sdf.voxel_size) * 3.0, 0.04)
    )
    normal_segments = np.stack([
        _to_display(normal_origins, up, ground),
        _to_display(normal_targets, up, ground),
    ], axis=1) if len(normal_origins) else np.zeros((0, 2, 3), float)

    smoothed_surface = diagnostic_surfaces.get(
        "smoothed", np.zeros((0, 3), float),
    )
    smoothed_selection = _sample_rows(
        len(smoothed_surface), surface_limit, seed=41,
    )
    smoothed_aligned = (
        smoothed_surface[smoothed_selection] @ rotation.T
        if len(smoothed_selection) else np.zeros((0, 3), float)
    )
    smoothed_display = _to_display(
        smoothed_aligned, up, ground,
    )

    layers = [
        {
            "id": "fusion_measurements_raw",
            "label": 'Projected RGB-D measurements',
            "kind": "points",
            "role": "fusion",
            "total": int(len(measurement_points_raw)),
            "sampled": int(len(measurement_points_raw)),
            "points": _rounded(measurement_points_raw, 3),
            "colors": _rounded(measurement_colors, 3),
            "frame_index": measurement_frame_indices,
        },
        {
            "id": "camera_path_raw",
            "label": 'Raw camera trajectory',
            "kind": "lines",
            "role": "vector",
            "segments": _rounded(camera_segments_raw, 3),
        },
        volume_layer(
            "volume_unknown", "Voxels inconnus", unknown_mask,
            "unknown", seed=13,
        ),
        volume_layer(
            "volume_free", "Voxels libres", free_mask,
            "free", seed=17,
        ),
        volume_layer(
            "volume_occupied", 'Occupied voxels', occupied_mask,
            "occupied", seed=19,
        ),
        {
            "id": "surface_raw",
            "label": "Raw iso-surface",
            "kind": "surface",
            "role": "surface",
            "total": int(scene.cloud.size),
            "sampled": int(len(surface_selection)),
            "points": _rounded(raw_surface_preview),
            "curvature": _rounded(
                scene.cloud.curvature[surface_selection], 6,
            ),
            "rgb_colors": _rounded(
                surface_rgb_colors[surface_selection], 3,
            ) if rgb_surface_available else None,
            "source_index": surface_selection.tolist(),
        },
        {
            "id": "surface_field_smoothed",
            "label": 'Isosurface from smoothed TSDF',
            "kind": "surface",
            "role": "comparison",
            "total": int(len(smoothed_surface)),
            "sampled": int(len(smoothed_selection)),
            "points": _rounded(smoothed_display),
            "source_index": smoothed_selection.tolist(),
        },
        {
            "id": "surface_gravity",
            "label": 'Gravity-aligned surface',
            "kind": "surface",
            "role": "surface",
            "total": int(scene.cloud.size),
            "sampled": int(len(surface_selection)),
            "points": _rounded(gravity_surface_preview),
            "curvature": _rounded(
                scene.cloud.curvature[surface_selection], 6,
            ),
            "rgb_colors": _rounded(
                surface_rgb_colors[surface_selection], 3,
            ) if rgb_surface_available else None,
            "source_index": surface_selection.tolist(),
        },
        {
            "id": "surface",
            "label": 'Final ground-aligned scan',
            "kind": "surface",
            "role": "surface",
            "total": int(scene.cloud.size),
            "sampled": int(len(surface_selection)),
            "points": _rounded(ground_surface_preview),
            "curvature": _rounded(
                scene.cloud.curvature[surface_selection], 6,
            ),
            "rgb_colors": _rounded(
                surface_rgb_colors[surface_selection], 3,
            ) if rgb_surface_available else None,
            "source_index": surface_selection.tolist(),
        },
        {
            "id": "surface_normals",
            "label": 'SDF gradient normals',
            "kind": "lines",
            "role": "normal",
            "total": int(len(normal_selection)),
            "segments": _rounded(normal_segments, 4),
        },
        {
            "id": "surface_curvature",
            "label": 'Surface curvature',
            "kind": "surface",
            "role": "surface",
            "color_mode": "curvature",
            "total": int(scene.cloud.size),
            "sampled": int(len(surface_selection)),
            "points": _rounded(ground_surface_preview),
            "curvature": _rounded(
                scene.cloud.curvature[surface_selection], 6,
            ),
            "rgb_colors": _rounded(
                surface_rgb_colors[surface_selection], 3,
            ) if rgb_surface_available else None,
            "source_index": surface_selection.tolist(),
        },
    ]

    source_frames = int(trace.get("source_total_frames", len(frames)))
    requested_frames = int(trace.get("total_frames", len(frames)))
    integrated_frames = len(frames)
    effective_stride = int(trace.get("frame_stride", cfg.sdf.frame_stride))
    requested_stride = int(
        trace.get("requested_frame_stride", cfg.sdf.frame_stride)
    )
    volume_summary = trace.get("volume", {})
    min_frames = int(trace.get("min_frames", getattr(cfg.sdf, "min_frames", 0)))
    severe_limit = min(20, max(3, int(np.ceil(source_frames * 0.1))))
    preview_only = integrated_frames < severe_limit
    selection_description = (
        f"{integrated_frames} frame(s) integrated out of {source_frames} available, effective stride {effective_stride} (requested maximum {requested_stride}, minimum {min_frames or 'disabled'}), limit {cfg.sdf.max_frames or 'none'}."
    )
    if preview_only:
        selection_description += (
            ' Very partial preview: the isosurface cannot represent the full scene.'
        )

    steps = []
    steps.append({
        "id": "frame-selection",
        "label": 'Frame selection',
        "description": selection_description,
        "input": source_frames,
        "kept": integrated_frames,
        "rejected": max(0, source_frames - integrated_frames),
        "visible_layers": ["camera_path_raw"],
        "draw_counts": {
            "camera_path_raw": len(camera_segments_raw),
        },
        "warning": "preview_only" if preview_only else None,
    })
    total_frames = max(1, len(frames))
    for index, frame in enumerate(frames):
        camera_count = max(
            0, min(len(camera_segments_raw), index * 2),
        )
        steps.append({
            "id": f"frame-{index + 1:06d}",
            "label": f"Trame {index + 1}",
            "description": (
                f"Projecting and integrating RGB-D frame "
                f"{index + 1}/{total_frames}."
            ),
            "input": index + 1,
            "kept": frame_offsets[index],
            "rejected": 0,
            "visible_layers": [
                "fusion_measurements_raw", "camera_path_raw",
            ],
            "draw_counts": {
                "fusion_measurements_raw": frame_offsets[index],
                "camera_path_raw": camera_count,
            },
            "frame_id": frame.get("frame_id"),
        })
    steps.append({
        "id": "volume",
        "label": 'Free / occupied / unknown volume',
        "description": (
            'Show final volume visibility before surface extraction.'
        ),
        "input": int(volume_summary.get(
            "voxel_count", visibility.size,
        )),
        "kept": int(
            volume_summary.get("free_voxels", np.count_nonzero(free_mask))
            + volume_summary.get(
                "occupied_voxels", np.count_nonzero(occupied_mask),
            )
        ),
        "rejected": int(volume_summary.get(
            "unknown_voxels", np.count_nonzero(unknown_mask),
        )),
        "visible_layers": [
            "fusion_measurements_raw", "camera_path_raw",
            "volume_unknown", "volume_free", "volume_occupied",
        ],
        "draw_counts": {
            "fusion_measurements_raw": len(measurement_points_raw),
            "camera_path_raw": len(camera_segments_raw),
        },
    })
    steps.extend([
        {
            "id": "surface-raw",
            "label": 'Surface field comparison',
            "description": (
                'Compare the selected surface, extracted from raw TSDF by default, with the legacy smoothed field that can erase structures one or two voxels wide.'
            ),
            "input": int(volume_summary.get(
                "active_voxels",
                np.count_nonzero(
                    np.asarray(getattr(base_volume, "weight", [])) > 0,
                ),
            )),
            "kept": int(scene.cloud.size), "rejected": 0,
            "visible_layers": ["surface_raw", "surface_field_smoothed"],
        },
        {
            "id": "gravity",
            "label": 'Gravity alignment',
            "description": (
                'Apply the estimated rotation to align the vertical axis with ShapeNet models.'
            ),
            "input": int(scene.cloud.size),
            "kept": int(scene.cloud.size), "rejected": 0,
            "visible_layers": ["surface_gravity"],
        },
        {
            "id": "ground",
            "label": 'Ground placement',
            "description": (
                f'Subtract estimated ground height ({ground:.3f} m) to place the ground at Y=0.'
            ),
            "input": int(scene.cloud.size),
            "kept": int(scene.cloud.size), "rejected": 0,
            "visible_layers": ["surface"],
        },
        {
            "id": "normals",
            "label": 'SDF normals',
            "description": (
                'Display sampled normals from signed distance field gradients.'
            ),
            "input": int(scene.cloud.size),
            "kept": int(len(normal_selection)), "rejected": 0,
            "visible_layers": ["surface", "surface_normals"],
        },
        {
            "id": "curvature",
            "label": 'Curvature',
            "description": (
                'Color the surface by the local PCA curvature used for keypoint detection.'
            ),
            "input": int(scene.cloud.size),
            "kept": int(scene.cloud.size), "rejected": 0,
            "visible_layers": ["surface_curvature"],
        },
    ])

    display_all = np.vstack([
        raw_surface,
        _to_display(aligned_surface, up, 0.0),
        _to_display(aligned_surface, up, ground),
    ])
    duration = time.perf_counter() - started
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "fusion",
        "scene": {
            **source, "points": int(scene.cloud.size),
            "frames": len(frames), "cache_hit": False,
        },
        "duration_seconds": round(duration, 3),
        "ground": round(ground, 5),
        "bounds": {
            "min": _rounded(display_all.min(axis=0), 4),
            "max": _rounded(display_all.max(axis=0), 4),
        },
        "steps": steps,
        "layers": layers,
        "artifacts": artifacts,
        "metrics": {
            "frames": integrated_frames,
            "source_frames": source_frames,
            "requested_frames": requested_frames,
            "integrated_frames": integrated_frames,
            "temporal_coverage": round(
                integrated_frames / max(1, source_frames), 4,
            ),
            "preview_only": preview_only,
            "frame_stride": effective_stride,
            "requested_frame_stride": requested_stride,
            "min_frames": min_frames,
            "max_frames": int(cfg.sdf.max_frames),
            "sampled_measurements": int(len(measurement_points_raw)),
            "rgb_surface_available": rgb_surface_available,
            "rgb_surface_valid_points": int(np.count_nonzero(
                surface_rgb_valid,
            )),
            "rgb_projection_distance": round(rgb_projection_distance, 4),
            "surface_points": int(scene.cloud.size),
            "surface_field": str(cfg.sdf.surface_field),
            "surface_min_support": float(cfg.sdf.surface_min_support),
            "surface_extraction": str(cfg.sdf.surface_extraction),
            "surface_band_factor": float(cfg.sdf.surface_band_factor),
            "surface_rescue_distance_factor": float(
                cfg.sdf.surface_rescue_distance_factor
            ),
            "surface_rescue_min_weight": float(
                cfg.sdf.surface_rescue_min_weight
            ),
            "surface_extraction_stats": trace.get(
                "surface_extraction", {},
            ),
            "gradient_smoothing_sigma": float(
                cfg.sdf.gradient_smoothing_sigma
            ),
            "raw_zero_crossings": int(len(
                diagnostic_surfaces.get("raw", []),
            )),
            "smoothed_zero_crossings": int(len(smoothed_surface)),
            "backend": trace.get("backend", cfg.sdf.backend),
            "volume_layout": trace.get(
                "volume_layout", cfg.sdf.volume_layout,
            ),
            "voxel_size": float(cfg.sdf.voxel_size),
            "volume": trace.get("volume", {}),
            "ground": round(ground, 5),
            "alignment_rotation": _rounded(rotation, 6),
        },
        "histograms": {},
    }


def _keypoint_scene_result(
        source, parameters, progress=None, artifact_path=None):
    started = time.perf_counter()
    cloud, up, ground, cache_hit = _load_cloud(source, progress=progress)
    up = np.asarray(up, float)
    up /= np.clip(np.linalg.norm(up), 1e-12, None)

    cfg = PipelineConfig()
    for key in asdict(cfg.keypoint):
        if key in parameters:
            setattr(cfg.keypoint, key, parameters[key])
    for key in (
        "floor_height", "max_scan_keypoints", "spatial_keypoint_balance",
        "spatial_keypoint_cell", "min_2d_corner_fraction",
        "descriptor_keypoint_filter_enabled", "descriptor_min_occupied",
        "descriptor_max_hole_fraction", "descriptor_keypoint_min_features",
        "descriptor_keypoint_min_absolute", "descriptor_keypoint_min_ratio",
        "descriptor_keypoint_max_reintroduced_fraction",
        "quality_nms_radius", "quality_min_score_ratio",
    ):
        if key in parameters:
            setattr(cfg.matching, key, parameters[key])

    if progress:
        progress('Detecting and tracing keypoints')
    trace = {}
    scene_state = _load_scene_state(source["path"])
    visibility_sampler = None
    if scene_state is not None:
        if hasattr(scene_state, "sample_visibility_details"):
            visibility_sampler = scene_state.sample_visibility_details
        elif getattr(scene_state, "volume", None) is not None:
            volume = scene_state.volume
            rotation = np.asarray(
                getattr(scene_state, "R", np.eye(3)), float,
            )

            def visibility_sampler(positions):
                aligned = np.asarray(positions, float) @ rotation
                if hasattr(volume, "sample_visibility_details"):
                    return volume.sample_visibility_details(aligned)
                labels = volume.sample_visibility(aligned)
                return labels, np.zeros(len(aligned), bool)
    detected_keypoints = detect_keypoints(
        cloud, cfg.keypoint, trace=trace,
        visibility_sampler=visibility_sampler,
    )
    if progress:
        progress('Diagnostic detection of large vertical planes')
    wall_filter = apply_wall_keypoint_filter(
        cloud, detected_keypoints, cfg.keypoint, up, force_analysis=True,
    )
    wall_analysis = wall_filter.analysis
    wall_keypoints = wall_filter.keypoints
    quality_scene = (
        scene_state if scene_state is not None
        else SimpleNamespace(cloud=cloud, volume=None)
    )
    # Le cache géométrique est la référence exacte de cette itération; le
    # volume du stage précédent ne sert qu'aux sondes de visibilité.
    quality_scene.cloud = cloud
    quality_filter = filter_scene_keypoints(
        quality_scene, wall_keypoints, cfg.keypoint,
    )
    keypoints = quality_filter.keypoints
    response_by_source = np.full(cloud.size, np.nan, dtype=float)
    curvature_candidates = np.asarray(
        trace.get("curvature_candidates", []), int,
    )
    candidates = np.asarray(
        trace.get("pre_harris_visibility_kept", curvature_candidates), int,
    )
    visibility_rejected_sources = np.asarray(
        trace.get("pre_harris_visibility_rejected", []), int,
    )
    responses = np.asarray(trace.get("harris_responses", []), float)
    response_by_source[candidates] = responses

    heights = keypoints.positions @ up - ground
    above_mask = heights >= cfg.matching.floor_height
    above_local = np.flatnonzero(above_mask)
    above_sources = keypoints.source_index[above_local]
    floor_sources = keypoints.source_index[np.flatnonzero(~above_mask)]

    class LightweightFeature:
        def __init__(self, position, response, source_index, local_index):
            self.position = position
            self.response = float(response)
            self.source_index = int(source_index)
            self.selection_score = float(keypoints.selection_scores[local_index])
            self.wall_score = float(keypoints.wall_scores[local_index])
            self.object_score = float(keypoints.object_scores[local_index])
            self.wall_affinity = float(keypoints.wall_affinities[local_index])
            self.wall_proximity = float(keypoints.wall_proximities[local_index])
            self.wall_plane_index = int(keypoints.wall_plane_indices[local_index])
            self.local_index = int(local_index)

    features = [
        LightweightFeature(
            keypoints.positions[index], keypoints.responses[index],
            keypoints.source_index[index], index,
        )
        for index in above_local
    ]
    selection_trace = {}
    retained = _cap_by_response(
        features, cfg.matching.max_scan_keypoints,
        spatial_balance=cfg.matching.spatial_keypoint_balance,
        cell_size=cfg.matching.spatial_keypoint_cell,
        harris_threshold=cfg.keypoint.harris_threshold,
        min_2d_fraction=cfg.matching.min_2d_corner_fraction,
        wall_budget_enabled=cfg.keypoint.wall_budget_enabled,
        wall_budget_affinity_threshold=(
            cfg.keypoint.wall_budget_affinity_threshold
        ),
        wall_budget_proximity_threshold=(
            cfg.keypoint.wall_budget_proximity_threshold
        ),
        wall_budget_max_object_score=(
            cfg.keypoint.wall_reject_max_object_score
        ),
        wall_budget_max_fraction=cfg.keypoint.wall_budget_max_fraction,
        wall_budget_cell_size=cfg.keypoint.wall_budget_cell_size,
        wall_budget_max_per_cell=cfg.keypoint.wall_budget_max_per_cell,
        wall_budget_min_features=cfg.keypoint.wall_budget_min_features,
        component_budget_enabled=cfg.keypoint.component_budget_enabled,
        component_budget_radius=cfg.keypoint.component_budget_radius,
        component_budget_max_per_component=(
            cfg.keypoint.component_budget_max_per_component
        ),
        component_budget_min_features=(
            cfg.keypoint.component_budget_min_features
        ),
        trace=selection_trace,
    )
    retained_sources = np.asarray([item.source_index for item in retained], int)
    retained_local = np.asarray([item.local_index for item in retained], int)
    retained_set = set(retained_sources.tolist())
    wall_budget_rejected_sources = np.asarray([
        features[index].source_index
        for index in selection_trace.get("wall_budget_rejected", [])
    ], int)
    wall_budget_rejected_set = set(wall_budget_rejected_sources.tolist())
    component_budget_rejected_sources = np.asarray([
        features[index].source_index
        for index in selection_trace.get("component_budget_rejected", [])
    ], int)
    component_budget_rejected_set = set(
        component_budget_rejected_sources.tolist()
    )
    cap_rejected_sources = np.asarray([
        index for index in above_sources
        if int(index) not in retained_set
        and int(index) not in wall_budget_rejected_set
        and int(index) not in component_budget_rejected_set
    ], int)
    quality_reintroduced = {
        str(stage): np.asarray(values, int)
        for stage, values in (
            quality_filter.reintroduced_sources or {}
        ).items()
    }
    quality_reintroduced_total = int(sum(
        len(values) for values in quality_reintroduced.values()
    ))

    preview_limit = int(parameters["layer_preview_points"])
    surface_limit = int(parameters["surface_preview_points"])
    all_indices = np.arange(cloud.size, dtype=int)
    surface_selection = _sample_rows(cloud.size, surface_limit, seed=7)
    surface_display = _to_display(cloud.points[surface_selection], up, ground)
    layers = [{
        "id": "surface", "label": 'Fused surface', "kind": "surface",
        "role": "surface", "total": int(cloud.size),
        "sampled": int(len(surface_selection)),
        "points": _rounded(surface_display),
        "curvature": _rounded(cloud.curvature[surface_selection], 6),
        "rgb_colors": _rounded(
            cloud.colors[surface_selection], 3,
        ) if getattr(cloud, "colors", None) is not None else None,
        "source_index": all_indices[surface_selection].tolist(),
    }]
    if getattr(cloud, "confidence", None) is not None:
        layers[0]["confidence"] = _rounded(
            cloud.confidence[surface_selection], 4,
        )
    wall_surface_sources = np.flatnonzero(
        wall_analysis.surface_plane_index >= 0,
    )
    layers.append(_index_layer(
        "wall_surface", 'Vertical plane support',
        wall_surface_sources, cloud, response_by_source, up, ground,
        preview_limit, "wall", attributes={
            "plane_index": wall_analysis.surface_plane_index[
                wall_surface_sources
            ],
        },
    ))
    wall_keypoint_local = np.flatnonzero(
        wall_analysis.keypoint_plane_index >= 0,
    )
    layers.append(_point_layer(
        "wall_keypoint_affinity", 'Keypoints near vertical planes',
        detected_keypoints.positions[wall_keypoint_local],
        detected_keypoints.source_index[wall_keypoint_local], cloud,
        response_by_source, up, ground, preview_limit, "wall",
        attributes={
            "wall_proximity": wall_analysis.plane_proximity[
                wall_keypoint_local
            ],
            "wall_affinity": wall_analysis.plane_affinity[
                wall_keypoint_local
            ],
            "wall_score": wall_analysis.wall_score[wall_keypoint_local],
            "object_score": wall_analysis.object_score[wall_keypoint_local],
            "small_support": wall_analysis.small_support[wall_keypoint_local],
            "large_support": wall_analysis.large_support[wall_keypoint_local],
            "protrusion_score": wall_analysis.protrusion_score[
                wall_keypoint_local
            ],
            "discontinuity_score": wall_analysis.discontinuity_score[
                wall_keypoint_local
            ],
            "object_anchor_score": wall_analysis.object_anchor_score[
                wall_keypoint_local
            ],
            "plane_index": wall_analysis.keypoint_plane_index[
                wall_keypoint_local
            ],
        },
    ))
    likely_wall_local = np.flatnonzero(
        wall_analysis.wall_score >= cfg.keypoint.wall_reject_threshold,
    )
    layers.append(_point_layer(
        "wall_score_high", 'High wall score',
        detected_keypoints.positions[likely_wall_local],
        detected_keypoints.source_index[likely_wall_local], cloud,
        response_by_source, up, ground, preview_limit, "wall",
        attributes={
            "wall_affinity": wall_analysis.plane_affinity[likely_wall_local],
            "wall_score": wall_analysis.wall_score[likely_wall_local],
            "object_score": wall_analysis.object_score[likely_wall_local],
            "small_support": wall_analysis.small_support[likely_wall_local],
            "large_support": wall_analysis.large_support[likely_wall_local],
            "protrusion_score": wall_analysis.protrusion_score[
                likely_wall_local
            ],
            "discontinuity_score": wall_analysis.discontinuity_score[
                likely_wall_local
            ],
            "object_anchor_score": wall_analysis.object_anchor_score[
                likely_wall_local
            ],
            "plane_index": wall_analysis.keypoint_plane_index[
                likely_wall_local
            ],
        },
    ))
    protected_local = np.flatnonzero(wall_analysis.object_score >= 0.50)
    layers.append(_point_layer(
        "wall_object_protected", 'Protected by object geometry',
        detected_keypoints.positions[protected_local],
        detected_keypoints.source_index[protected_local],
        cloud, response_by_source, up, ground, preview_limit, "accepted",
        attributes={
            "wall_affinity": wall_analysis.plane_affinity[protected_local],
            "wall_score": wall_analysis.wall_score[protected_local],
            "object_score": wall_analysis.object_score[protected_local],
            "small_support": wall_analysis.small_support[protected_local],
            "large_support": wall_analysis.large_support[protected_local],
            "protrusion_score": wall_analysis.protrusion_score[protected_local],
            "discontinuity_score": wall_analysis.discontinuity_score[
                protected_local
            ],
            "object_anchor_score": wall_analysis.object_anchor_score[
                protected_local
            ],
            "plane_index": wall_analysis.keypoint_plane_index[protected_local],
        },
    ))
    layers.append(_wall_plane_bounds_layer(
        wall_analysis.planes, up, ground,
    ))
    layers.append(_index_layer(
        "curvature_candidates", 'Curvature > threshold', curvature_candidates, cloud,
        response_by_source, up, ground, preview_limit, "candidate",
    ))
    layers.append(_index_layer(
        "pre_harris_visibility_rejected", 'Boundaries rejected before Harris',
        visibility_rejected_sources, cloud, response_by_source, up, ground,
        preview_limit, "rejected", np.asarray([
            'definite free/unknown boundary'
        ] * len(visibility_rejected_sources), object),
    ))
    layers.append(_index_layer(
        "harris_accepted", '3D Harris corners', trace.get("harris_accepted", []),
        cloud, response_by_source, up, ground, preview_limit, "accepted",
    ))
    low_response = candidates[
        np.isfinite(responses) & (responses <= cfg.keypoint.harris_threshold)
    ]
    layers.append(_index_layer(
        "harris_low", 'Weak Harris response', low_response, cloud,
        response_by_source, up, ground, preview_limit, "candidate",
    ))
    layers.append(_index_layer(
        "corner_2d_accepted", 'Accepted 2D corners',
        trace.get("corner_2d_accepted", []), cloud, response_by_source,
        up, ground, preview_limit, "accepted",
    ))
    for layer_id, label, reason in (
        ("corner_2d_prefilter_rejected", 'Removed before 2D test', 'cell prefilter'),
        ("corner_2d_rejected_hull", "2D hull rejection", "enveloppe trop pleine"),
        ("corner_2d_rejected_support", "2D support rejection", 'plane too small or inconsistent'),
    ):
        values = np.asarray(trace.get(layer_id, []), int)
        layers.append(_index_layer(
            layer_id, label, values, cloud, response_by_source, up, ground,
            preview_limit, "rejected", np.asarray([reason] * len(values), object),
        ))
    layers.extend([
        _index_layer(
            "nms_kept", 'Local maxima', trace.get("nms_kept", []), cloud,
            response_by_source, up, ground, preview_limit, "accepted",
        ),
        _index_layer(
            "nms_rejected", 'Removed by NMS', trace.get("nms_rejected", []),
            cloud, response_by_source, up, ground, preview_limit, "rejected",
            np.asarray(["maximum voisin plus fort"] * len(trace.get("nms_rejected", [])), object),
        ),
    ])

    adjusted_sources = np.asarray(trace.get("adjustment_source_index", []), int)
    adjusted_targets = np.asarray(trace.get("adjustment_target", []), float).reshape(-1, 3)
    layers.append(_point_layer(
        "adjustment_kept", 'Stable adjustments', adjusted_targets,
        adjusted_sources, cloud, response_by_source, up, ground, preview_limit,
        "accepted",
    ))
    rejected_indices, rejected_reasons = _combine_rejections([
        (f"ajustement: {reason}", values)
        for reason, values in trace.get("adjustment_rejected", {}).items()
    ])
    layers.append(_index_layer(
        "adjustment_rejected", 'Rejected adjustments', rejected_indices,
        cloud, response_by_source, up, ground, preview_limit, "rejected",
        rejected_reasons,
    ))
    layers.append(_line_layer(
        trace.get("adjustment_origin", []), trace.get("adjustment_target", []),
        up, ground,
    ))

    final_sources = detected_keypoints.source_index
    layers.append(_point_layer(
        "detected_final", 'Detected keypoints', detected_keypoints.positions,
        final_sources, cloud, response_by_source, up, ground, preview_limit,
        "accepted",
    ))
    dedup_rejected = np.asarray(trace.get("dedup_rejected", []), int)
    if len(dedup_rejected):
        target_by_source = {
            int(source_index): target
            for source_index, target in zip(adjusted_sources, adjusted_targets)
        }
        dedup_positions = np.asarray([
            target_by_source[int(index)] for index in dedup_rejected
        ], float)
    else:
        dedup_positions = np.zeros((0, 3), float)
    layers.append(_point_layer(
        "dedup_rejected", 'Merged duplicates', dedup_positions,
        dedup_rejected, cloud, response_by_source, up, ground, preview_limit,
        "rejected", np.asarray(['same converged corner'] * len(dedup_rejected), object),
    ))

    source_to_position = {
        int(source_index): position
        for source_index, position in zip(keypoints.source_index, keypoints.positions)
    }
    def positions_for(indices):
        return np.asarray([source_to_position[int(index)] for index in indices], float).reshape(-1, 3)

    layers.extend([
        _point_layer(
            "wall_filter_kept", 'Retained after wall filter',
            wall_keypoints.positions, wall_keypoints.source_index, cloud,
            response_by_source, up, ground, preview_limit, "accepted",
            attributes={
                "wall_score": wall_keypoints.wall_scores,
                "object_score": wall_keypoints.object_scores,
                "wall_affinity": wall_keypoints.wall_affinities,
                "wall_proximity": wall_keypoints.wall_proximities,
                "plane_index": wall_keypoints.wall_plane_indices,
                "selection_score": wall_keypoints.selection_scores,
            },
        ),
        _point_layer(
            "planar_filter_rejected", 'Rejected planar interiors',
            quality_filter.planar_rejected_positions,
            quality_filter.planar_rejected_sources,
            cloud, response_by_source, up, ground, preview_limit, "rejected",
            np.asarray(['planar surface interior'] * quality_filter.planar_rejected, object),
        ),
        _point_layer(
            "hole_boundary_rejected", 'Rejected hole boundaries',
            quality_filter.hole_rejected_positions,
            quality_filter.hole_rejected_sources,
            cloud, response_by_source, up, ground, preview_limit, "rejected",
            np.asarray(['unconfirmed RGB-D boundary'] * quality_filter.hole_rejected, object),
        ),
        _point_layer(
            "repeatability_rejected", 'Rejected unstable corners',
            quality_filter.repeatability_rejected_positions,
            quality_filter.repeatability_rejected_sources,
            cloud, response_by_source, up, ground, preview_limit, "rejected",
            np.asarray(['unstable Harris response at larger scale'] * quality_filter.repeatability_rejected, object),
        ),
        _point_layer(
            "quality_score_rejected", 'Rejected weak Harris responses',
            quality_filter.score_rejected_positions,
            quality_filter.score_rejected_sources,
            cloud, response_by_source, up, ground, preview_limit, "rejected",
            np.asarray(['Harris response too weak relative to the scene'] * quality_filter.score_rejected, object),
        ),
        _point_layer(
            "geometric_nms_rejected", 'Rejected local duplicates',
            quality_filter.nms_rejected_positions,
            quality_filter.nms_rejected_sources,
            cloud, response_by_source, up, ground, preview_limit, "rejected",
            np.asarray(['a stronger nearby keypoint is already retained'] * quality_filter.nms_rejected, object),
        ),
        _point_layer(
            "wall_filter_rejected", 'Rejected as wall',
            detected_keypoints.positions[wall_filter.rejected_mask],
            detected_keypoints.source_index[wall_filter.rejected_mask],
            cloud, response_by_source, up, ground, preview_limit, "rejected",
            reasons=np.asarray([
                'high wall score without object evidence'
            ] * int(wall_filter.rejected_mask.sum()), object),
            attributes={
                "wall_score": wall_analysis.wall_score[
                    wall_filter.rejected_mask
                ],
                "object_score": wall_analysis.object_score[
                    wall_filter.rejected_mask
                ],
                "selection_score": (
                    detected_keypoints.responses[
                        wall_filter.rejected_mask
                    ] - cfg.keypoint.wall_penalty_weight
                    * wall_analysis.wall_score[wall_filter.rejected_mask]
                ),
            },
        ),
        _point_layer(
            "above_floor", 'Above ground', positions_for(above_sources),
            above_sources, cloud, response_by_source, up, ground, preview_limit,
            "accepted",
        ),
        _point_layer(
            "floor_rejected", 'Rejected as ground', positions_for(floor_sources),
            floor_sources, cloud, response_by_source, up, ground, preview_limit,
            "rejected", np.asarray(['height below the ground margin'] * len(floor_sources), object),
        ),
        _point_layer(
            "wall_budget_rejected", "Outside wall quota",
            positions_for(wall_budget_rejected_sources),
            wall_budget_rejected_sources, cloud, response_by_source,
            up, ground, preview_limit, "rejected",
            np.asarray([
                'wall quota or cell already represented'
            ] * len(wall_budget_rejected_sources), object),
        ),
        _point_layer(
            "component_budget_rejected", 'Outside regional budget',
            positions_for(component_budget_rejected_sources),
            component_budget_rejected_sources, cloud, response_by_source,
            up, ground, preview_limit, "rejected",
            np.asarray([
                'spatial region already sufficiently represented'
            ] * len(component_budget_rejected_sources), object),
        ),
        _point_layer(
            "retained", "Passed to matching", positions_for(retained_sources),
            retained_sources, cloud, response_by_source, up, ground,
            preview_limit, "retained",
        ),
        _point_layer(
            "cap_rejected", "Hors budget", positions_for(cap_rejected_sources),
            cap_rejected_sources, cloud, response_by_source, up, ground,
            preview_limit, "rejected",
            np.asarray(['budget or spatial balancing'] * len(cap_rejected_sources), object),
        ),
    ])

    counts = trace.get("counts", {})
    steps = [
        {"id": "surface", "input": cloud.size, "kept": cloud.size, "rejected": 0,
         "visible_layers": ["surface"]},
        {"id": "wall_planes", "input": cloud.size,
         "kept": len(wall_surface_sources), "rejected": 0,
         "warning": "diagnostic_only",
         "visible_layers": [
             "surface", "wall_surface", "wall_keypoint_affinity",
             "wall_plane_bounds",
         ]},
        {"id": "wall_scores", "input": len(wall_keypoint_local),
         "kept": len(likely_wall_local), "rejected": 0,
         "warning": "diagnostic_only",
         "visible_layers": [
             "surface", "wall_plane_bounds", "wall_score_high",
             "wall_object_protected",
         ]},
        {"id": "curvature", "input": cloud.size,
         "kept": len(curvature_candidates),
         "rejected": cloud.size - len(curvature_candidates),
         "visible_layers": ["surface", "curvature_candidates"]},
        {"id": "pre_harris_visibility", "input": len(curvature_candidates),
         "kept": len(candidates), "rejected": len(visibility_rejected_sources),
         "visible_layers": [
             "surface", "curvature_candidates",
             "pre_harris_visibility_rejected",
         ]},
        {"id": "harris", "input": len(candidates),
         "kept": len(trace.get("harris_accepted", [])),
         "rejected": len(candidates) - len(trace.get("harris_accepted", [])),
         "visible_layers": ["surface", "harris_accepted", "harris_low"]},
        {"id": "corner_2d", "input": len(low_response),
         "kept": len(trace.get("corner_2d_accepted", [])),
         "rejected": len(low_response) - len(trace.get("corner_2d_accepted", [])),
         "visible_layers": [
             "surface", "harris_accepted", "corner_2d_accepted",
             "corner_2d_prefilter_rejected", "corner_2d_rejected_hull",
             "corner_2d_rejected_support",
         ]},
        {"id": "nms", "input": len(trace.get("nms_input", [])),
         "kept": len(trace.get("nms_kept", [])),
         "rejected": len(trace.get("nms_rejected", [])),
         "visible_layers": ["surface", "nms_kept", "nms_rejected"]},
        {"id": "adjustment", "input": len(trace.get("nms_kept", [])),
         "kept": len(adjusted_sources), "rejected": len(rejected_indices),
         "visible_layers": [
             "surface", "adjustment_kept", "adjustment_rejected",
             "adjustment_vectors",
         ]},
        {"id": "dedup", "input": counts.get("adjusted", 0),
         "kept": detected_keypoints.size, "rejected": len(dedup_rejected),
         "visible_layers": ["surface", "detected_final", "dedup_rejected"]},
        {"id": "wall_filter", "input": detected_keypoints.size,
         "kept": wall_keypoints.size,
         "rejected": int(wall_filter.rejected_mask.sum()),
         "visible_layers": [
             "surface", "wall_plane_bounds", "wall_filter_kept",
             "wall_filter_rejected",
         ]},
        {"id": "planar_filter", "input": wall_keypoints.size,
         "kept": wall_keypoints.size - quality_filter.planar_rejected,
         "rejected": quality_filter.planar_rejected,
         "reintroduced": len(quality_reintroduced.get("planar", [])),
         "visible_layers": [
             "surface", "wall_filter_kept", "planar_filter_rejected",
         ]},
        {"id": "hole_boundary_filter",
         "input": wall_keypoints.size - quality_filter.planar_rejected,
         "kept": (
             wall_keypoints.size - quality_filter.planar_rejected
             - quality_filter.hole_rejected
         ),
         "rejected": quality_filter.hole_rejected,
         "reintroduced": len(quality_reintroduced.get("hole", [])),
         "visible_layers": [
             "surface", "planar_filter_rejected", "hole_boundary_rejected",
         ]},
        {"id": "repeatability_filter",
         "input": (
             wall_keypoints.size - quality_filter.planar_rejected
             - quality_filter.hole_rejected
         ),
         "kept": (
             wall_keypoints.size - quality_filter.planar_rejected
             - quality_filter.hole_rejected
             - quality_filter.repeatability_rejected
         ),
         "rejected": quality_filter.repeatability_rejected,
         "reintroduced": len(quality_reintroduced.get("repeatability", [])),
         "visible_layers": [
             "surface", "hole_boundary_rejected", "repeatability_rejected",
         ]},
        {"id": "quality_score",
         "input": (
             wall_keypoints.size - quality_filter.planar_rejected
             - quality_filter.hole_rejected
             - quality_filter.repeatability_rejected
         ),
         "kept": (
             wall_keypoints.size - quality_filter.planar_rejected
             - quality_filter.hole_rejected
             - quality_filter.repeatability_rejected
             - quality_filter.score_rejected
         ),
         "rejected": quality_filter.score_rejected,
         "reintroduced": len(quality_reintroduced.get("score", [])),
         "visible_layers": [
             "surface", "repeatability_rejected", "quality_score_rejected",
         ]},
        {"id": "geometric_nms",
         "input": (
             wall_keypoints.size - quality_filter.planar_rejected
             - quality_filter.hole_rejected
             - quality_filter.repeatability_rejected
             - quality_filter.score_rejected
         ),
         "kept": keypoints.size,
         "rejected": quality_filter.nms_rejected,
         "reintroduced": len(quality_reintroduced.get("nms", [])),
         "visible_layers": [
             "surface", "quality_score_rejected", "geometric_nms_rejected",
             "above_floor",
         ]},
        {"id": "floor", "input": keypoints.size, "kept": len(above_sources),
         "rejected": len(floor_sources),
         "visible_layers": ["surface", "above_floor", "floor_rejected"]},
        {"id": "wall_budget", "input": len(above_sources),
         "kept": len(above_sources) - len(wall_budget_rejected_sources),
         "rejected": len(wall_budget_rejected_sources),
         "visible_layers": [
             "surface", "wall_plane_bounds", "above_floor",
             "wall_budget_rejected",
         ]},
        {"id": "component_budget",
         "input": len(above_sources) - len(wall_budget_rejected_sources),
         "kept": (
             len(above_sources) - len(wall_budget_rejected_sources)
             - len(component_budget_rejected_sources)
         ),
         "rejected": len(component_budget_rejected_sources),
         "visible_layers": [
             "surface", "above_floor", "component_budget_rejected",
         ]},
        {"id": "cap",
         "input": (
             len(above_sources) - len(wall_budget_rejected_sources)
             - len(component_budget_rejected_sources)
         ),
         "kept": len(retained_sources),
         "rejected": len(cap_rejected_sources),
         "visible_layers": ["surface", "retained", "cap_rejected"]},
    ]
    step_copy = []
    metadata = {item[0]: _step_definition(item) for item in KEYPOINT_STEPS}
    for step in steps:
        definition = metadata[step["id"]]
        step_copy.append({**step, **definition})

    finite_curvature = cloud.curvature[np.isfinite(cloud.curvature)]
    curvature_hist, curvature_edges = np.histogram(
        finite_curvature, bins=np.linspace(0.0, max(0.15, float(finite_curvature.max(initial=0.0))), 25),
    )
    finite_response = responses[np.isfinite(responses)]
    response_hist, response_edges = np.histogram(finite_response, bins=24) if len(finite_response) else (np.zeros(24, int), np.linspace(-1, 1, 25))
    affinity_hist, affinity_edges = np.histogram(
        wall_analysis.plane_affinity, bins=np.linspace(0.0, 1.0, 25),
    )
    proximity_hist, proximity_edges = np.histogram(
        wall_analysis.plane_proximity, bins=np.linspace(0.0, 1.0, 25),
    )
    wall_score_hist, wall_score_edges = np.histogram(
        wall_analysis.wall_score, bins=np.linspace(0.0, 1.0, 25),
    )
    object_score_hist, object_score_edges = np.histogram(
        wall_analysis.object_score, bins=np.linspace(0.0, 1.0, 25),
    )
    object_anchor_hist, object_anchor_edges = np.histogram(
        wall_analysis.object_anchor_score, bins=np.linspace(0.0, 1.0, 25),
    )
    confidence_values = np.asarray(
        getattr(cloud, "confidence", np.zeros(0)), float,
    )
    confidence_values = confidence_values[np.isfinite(confidence_values)]
    if len(confidence_values):
        confidence_hist, confidence_edges = np.histogram(
            confidence_values,
            bins=np.linspace(
                0.0, max(1.0, float(confidence_values.max())), 25,
            ),
        )
    else:
        confidence_hist = np.zeros(24, int)
        confidence_edges = np.linspace(0.0, 1.0, 25)
    display_all = _to_display(cloud.points, up, ground)
    duration = time.perf_counter() - started
    artifacts = {}
    object_group_records = []
    if artifact_path is not None:
        scene = _load_scene_state(source["path"])
        if scene is not None:
            artifact_path = Path(artifact_path).resolve()
            if progress:
                progress('Serializing keypoints for descriptors')
            retained_keypoints = _subset_keypoints(keypoints, retained_local)
            retained_scores = (
                retained_keypoints.selection_scores
                if retained_keypoints.selection_scores is not None
                else retained_keypoints.responses
            )
            retained_walls = (
                retained_keypoints.wall_plane_indices >= 0
                if retained_keypoints.wall_plane_indices is not None
                else np.zeros(retained_keypoints.size, bool)
            )
            retained_objects = (
                retained_keypoints.object_scores
                if retained_keypoints.object_scores is not None
                else np.zeros(retained_keypoints.size, float)
            )
            object_groups = build_object_keypoint_groups(
                cloud.points, retained_keypoints.positions,
                surface_wall_mask=(wall_analysis.surface_plane_index >= 0),
                keypoint_wall_mask=retained_walls,
                scores=retained_scores, object_scores=retained_objects,
                ground=ground,
                **{key: parameters.get("group_"+key, item[0]) for key,item in GROUP_CONTROLS.items()},
            )
            object_group_records = _serialize_object_groups(
                object_groups, retained_keypoints,
            )
            _append_group_preview(layers, steps, retained_keypoints, object_group_records, up, ground)
            if progress:
                progress(
                    f"Object groups: {len(object_group_records)} proposition(s)"
                )
            _atomic_pickle(artifact_path, {
                "scene": scene,
                "keypoints": retained_keypoints,
                "keypoint_groups": object_group_records,
                "parameters": dict(parameters),
                "group_parameters": {key: parameters.get("group_"+key, item[0]) for key,item in GROUP_CONTROLS.items()},
            })
            artifacts["keypoints"] = {
                "path": str(artifact_path),
                "format": "pickle",
                "count": int(len(retained_local)),
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "keypoints",
        "scene": {**source, "points": int(cloud.size), "cache_hit": bool(cache_hit)},
        "duration_seconds": round(duration, 3),
        "ground": round(float(ground), 5),
        "bounds": {
            "min": _rounded(display_all.min(axis=0), 4),
            "max": _rounded(display_all.max(axis=0), 4),
        },
        "steps": step_copy,
        "layers": layers,
        "artifacts": artifacts,
        "metrics": {
            "detected": int(detected_keypoints.size),
            "after_wall_filter": int(wall_keypoints.size),
            "after_quality_filter": int(keypoints.size),
            "wall_rejected": int(wall_filter.rejected_mask.sum()),
            "planar_rejected": quality_filter.planar_rejected,
            "hole_boundary_rejected": quality_filter.hole_rejected,
            "pre_harris_visibility_rejected": int(
                len(visibility_rejected_sources)
            ),
            "quality_reintroduced": quality_reintroduced_total,
            "quality_reintroduced_by_stage": {
                stage: int(len(values))
                for stage, values in quality_reintroduced.items()
            },
            "repeatability_rejected": quality_filter.repeatability_rejected,
            "score_rejected": quality_filter.score_rejected,
            "nms_rejected": quality_filter.nms_rejected,
            "retained": int(len(retained_sources)),
            "harris_3d": int(len(trace.get("harris_accepted", []))),
            "corner_2d": int(len(trace.get("corner_2d_accepted", []))),
            "floor_rejected": int(len(floor_sources)),
            "wall_budget_candidates": int(len(
                selection_trace.get("wall_budget_candidates", [])
            )),
            "wall_budget_rejected": int(len(wall_budget_rejected_sources)),
            "wall_budget_reintroduced": int(len(
                selection_trace.get("wall_budget_reintroduced", [])
            )),
            "wall_budget_quota": int(selection_trace.get(
                "wall_budget_quota", 0,
            )),
            "component_budget_rejected": int(
                len(component_budget_rejected_sources)
            ),
            "component_budget_reintroduced": int(len(
                selection_trace.get("component_budget_reintroduced", [])
            )),
            "component_budget_count": int(selection_trace.get(
                "component_budget_count", 0,
            )),
            "component_budget_sizes": list(selection_trace.get(
                "component_budget_sizes", [],
            )),
            "cap_rejected": int(len(cap_rejected_sources)),
            "wall_planes": int(len(wall_analysis.planes)),
            "wall_surface_points": int(len(wall_surface_sources)),
            "wall_keypoints": int(len(wall_keypoint_local)),
            "wall_score_high": int(len(likely_wall_local)),
            "wall_object_protected": int(len(protected_local)),
            "confidence_available": bool(len(confidence_values)),
            "confidence_p10": (
                round(float(np.percentile(confidence_values, 10)), 4)
                if len(confidence_values) else None
            ),
            "confidence_median": (
                round(float(np.median(confidence_values)), 4)
                if len(confidence_values) else None
            ),
            "retention_ratio": round(len(retained_sources) / max(1, keypoints.size), 4),
            "object_groups": int(len(object_group_records)),
        },
        "wall_planes": [
            {
                "normal": _rounded(plane.normal, 6),
                "offset": round(float(plane.offset), 6),
                "width": round(plane.width, 4),
                "height": round(plane.height, 4),
                "area": round(plane.area, 4),
                "support_count": int(plane.support_count),
                "quality": round(float(plane.quality), 6),
            }
            for plane in wall_analysis.planes
        ],
        "histograms": {
            "curvature": {"counts": curvature_hist.tolist(), "edges": _rounded(curvature_edges, 6)},
            "harris": {"counts": response_hist.tolist(), "edges": _rounded(response_edges, 6)},
            "wall_affinity": {"counts": affinity_hist.tolist(), "edges": _rounded(affinity_edges, 6)},
            "wall_proximity": {"counts": proximity_hist.tolist(), "edges": _rounded(proximity_edges, 6)},
            "wall_score": {"counts": wall_score_hist.tolist(), "edges": _rounded(wall_score_edges, 6)},
            "object_score": {"counts": object_score_hist.tolist(), "edges": _rounded(object_score_edges, 6)},
            "object_anchor": {"counts": object_anchor_hist.tolist(), "edges": _rounded(object_anchor_edges, 6)},
            "confidence": {"counts": confidence_hist.tolist(), "edges": _rounded(confidence_edges, 6)},
        },
    }


def _descriptor_scene_result(
        source, parameters, progress=None, artifact_path=None):
    import db_store
    from candidate_index import load_candidate_index, rank_candidates
    from descriptor_utility import (
        AMBIGUOUS, INFORMATIVE, NO_MATCH, profile_descriptor_utility,
    )
    from matching import build_scan_features

    started = time.perf_counter()
    source_path = Path(source["path"])
    with source_path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "scene" not in payload or "keypoints" not in payload:
        raise ValueError(
            'Descriptors require a chained Keypoints artifact. Run and chain the Keypoints stage first.'
        )
    scene = payload["scene"]
    keypoints = payload["keypoints"]
    cloud = scene.cloud
    up = np.asarray(scene.up, float)
    up /= max(np.linalg.norm(up), 1e-12)
    ground = float(scene.ground)
    visibility_volume = scene.volume
    while (
        hasattr(visibility_volume, "volume")
        and visibility_volume.volume is not visibility_volume
    ):
        visibility_volume = visibility_volume.volume
    visibility_volume.visibility_surface_band_factor = float(
        parameters["visibility_surface_band_factor"]
    )

    cfg = PipelineConfig()
    for key in (
        "occ_grid_res", "occ_extent", "distance_unit",
        "distance_exponent", "max_occupied",
    ):
        setattr(cfg.descriptor, key, parameters[key])
    cfg.ransac.desc_inlier = float(parameters["utility_distance_threshold"])
    cfg.matching.descriptor_ratio_threshold = float(
        parameters["descriptor_ratio_threshold"]
    )
    # Les keypoints de l'artefact sont déjà filtrés et plafonnés. Le calcul des
    # descripteurs ne doit pas leur appliquer une seconde fois le budget mural.
    cfg.matching.max_scan_keypoints = 0
    cfg.matching.floor_height = -1e6
    cfg.keypoint.wall_budget_enabled = False

    if progress:
        progress(f"Construction de {keypoints.size} descripteur(s) de scan")
    features = build_scan_features(
        cloud, scene.volume, keypoints, cfg, up, ground,
    )
    feature_groups = _feature_groups_from_keypoints(
        payload.get("keypoint_groups", []), keypoints, features,
    )

    db_dir = Path(RUNTIME_PATHS.database_dir)
    candidate_index_path = db_dir / "candidate_index.npz"
    if not db_store.is_db_dir(str(db_dir)):
        raise FileNotFoundError(f"ShapeNet database not found : {db_dir}")
    if not candidate_index_path.exists():
        raise FileNotFoundError(
            f'Candidate index not found: {candidate_index_path}'
        )
    candidate_index = load_candidate_index(str(candidate_index_path))
    candidate_count = int(parameters["utility_candidate_models"])
    selected = rank_candidates(
        candidate_index, features, cfg, candidate_count,
        diversify=True, diversity_fraction=0.5,
    )
    files = db_store.model_files(str(db_dir))
    models = []
    for index in selected:
        if 0 <= int(index) < len(files):
            models.append(db_store.load_model(files[int(index)]))
    if progress:
        progress(
            f'Diagnostic pool: {len(models)}/{len(files)} model(s)'
        )
    utilities = profile_descriptor_utility(
        features, models, cfg, up,
        max_model_features=int(parameters["utility_model_features"]),
        rotations=int(parameters["utility_rotations"]),
        distance_threshold=float(parameters["utility_distance_threshold"]),
        margin_threshold=float(parameters["utility_margin_threshold"]),
        max_valid_models=int(parameters["utility_max_valid_models"]),
        entropy_threshold=float(parameters["utility_entropy_threshold"]),
        progress=progress,
    )

    positions = np.asarray([feature.position for feature in features], float)
    source_indices = np.asarray(keypoints.source_index[:len(features)], int)
    response_by_source = np.full(cloud.size, np.nan, float)
    response_by_source[source_indices] = np.asarray([
        feature.response for feature in features
    ], float)
    labels = np.asarray([item.label for item in utilities], object)
    best_distance = np.asarray([item.best_distance for item in utilities], float)
    second_distance = np.asarray([item.second_distance for item in utilities], float)
    margins = np.asarray([item.margin for item in utilities], float)
    valid_models = np.asarray([item.valid_model_count for item in utilities], int)
    compatible_features = np.asarray([
        item.compatible_feature_count for item in utilities
    ], int)
    entropies = np.asarray([item.entropy for item in utilities], float)
    best_models = np.asarray([
        item.best_model or 'none' for item in utilities
    ], object)
    best_synsets = np.asarray([
        item.best_synset or 'none' for item in utilities
    ], object)
    occupied = np.asarray([
        feature.descriptor.n_occupied for feature in features
    ], int)
    unknown = np.asarray([
        feature.descriptor.n_unknown for feature in features
    ], int)
    hole_boundary_removed = np.asarray([
        getattr(feature.descriptor, "n_hole_boundary_removed", 0)
        for feature in features
    ], int)
    totals = np.asarray([
        max(1, feature.descriptor.n_total) for feature in features
    ], int)

    preview_limit = int(parameters.get("layer_preview_points", 18000))
    surface_limit = int(parameters.get("surface_preview_points", 45000))
    surface_selection = _sample_rows(cloud.size, surface_limit, seed=7)
    layers = [{
        "id": "surface", "label": 'Fused surface', "kind": "surface",
        "role": "surface", "total": int(cloud.size),
        "sampled": int(len(surface_selection)),
        "points": _rounded(_to_display(
            cloud.points[surface_selection], up, ground,
        )),
        "curvature": _rounded(cloud.curvature[surface_selection], 6),
        "rgb_colors": _rounded(
            cloud.colors[surface_selection], 3,
        ) if getattr(cloud, "colors", None) is not None else None,
        "source_index": surface_selection.astype(int).tolist(),
    }]
    common_attributes = {
        "occupied_cells": occupied,
        "unknown_cells": unknown,
        "occupied_ratio": occupied / totals,
        "unknown_ratio": unknown / totals,
        "hole_boundary_removed": hole_boundary_removed,
        "best_distance": best_distance,
        "second_distance": second_distance,
        "descriptor_margin": margins,
        "valid_model_count": valid_models,
        "compatible_feature_count": compatible_features,
        "descriptor_entropy": entropies,
        "best_model": best_models,
        "best_synset": best_synsets,
        "utility_label": labels,
    }

    def descriptor_layer(layer_id, label, local_indices, role, reasons=None):
        local_indices = np.asarray(local_indices, int)
        return _point_layer(
            layer_id, label, positions[local_indices],
            source_indices[local_indices], cloud, response_by_source,
            up, ground, preview_limit, role=role, reasons=reasons,
            attributes={
                key: np.asarray(values)[local_indices]
                for key, values in common_attributes.items()
            },
        )

    all_local = np.arange(len(features), dtype=int)
    no_match_local = np.flatnonzero(labels == NO_MATCH)
    ambiguous_local = np.flatnonzero(labels == AMBIGUOUS)
    informative_local = np.flatnonzero(labels == INFORMATIVE)
    zero_occupied_local = np.flatnonzero(occupied <= 0)
    layers.extend([
        descriptor_layer(
            "descriptor_all", 'Computed descriptors', all_local,
            "retained",
        ),
        descriptor_layer(
            "descriptor_no_match", 'No correspondence', no_match_local,
            "rejected", np.asarray([
                'no model below the distance threshold'
            ] * len(no_match_local), object),
        ),
        descriptor_layer(
            "descriptor_ambiguous", 'Ambiguous correspondences',
            ambiguous_local, "candidate", np.asarray([
                'several models have similar scores'
            ] * len(ambiguous_local), object),
        ),
        descriptor_layer(
            "descriptor_informative", "Descriptors informatifs",
            informative_local, "accepted",
        ),
    ])

    group_palette = np.asarray([
        [0.35, 0.78, 0.94], [0.96, 0.60, 0.32],
        [0.53, 0.86, 0.55], [0.73, 0.55, 0.96],
        [0.96, 0.82, 0.36], [0.93, 0.45, 0.52],
        [0.39, 0.84, 0.79], [0.91, 0.58, 0.82],
        [0.64, 0.72, 0.94], [0.78, 0.86, 0.42],
        [0.95, 0.69, 0.48], [0.56, 0.66, 0.76],
    ], dtype=float)
    group_positions = []
    group_sources = []
    group_ids = []
    group_sizes = []
    group_scores = []
    group_colors = []
    group_segments = []
    for group_position, group in enumerate(feature_groups):
        indices = np.asarray(group.get("feature_indices", []), dtype=int)
        indices = indices[(indices >= 0) & (indices < len(features))]
        if not len(indices):
            continue
        group_id = int(group.get("id", group_position))
        color = group_palette[group_position % len(group_palette)]
        centroid = np.asarray(group.get(
            "centroid", positions[indices].mean(axis=0),
        ), dtype=float).reshape(3)
        group_positions.extend(positions[indices])
        group_sources.extend(source_indices[indices])
        group_ids.extend([group_id] * len(indices))
        group_sizes.extend([len(indices)] * len(indices))
        group_scores.extend([
            float(group.get("score", 0.0))
        ] * len(indices))
        group_colors.extend([color] * len(indices))
        group_segments.extend([
            [centroid, position] for position in positions[indices]
        ])
    if group_positions:
        group_positions = np.asarray(group_positions, dtype=float)
        group_sources = np.asarray(group_sources, dtype=int)
        group_layer = _point_layer(
            "descriptor_groups", 'Group members',
            group_positions, group_sources, cloud, response_by_source,
            up, ground, preview_limit, role="retained",
            attributes={
                "group_id": np.asarray(group_ids, dtype=int),
                "group_size": np.asarray(group_sizes, dtype=int),
                "group_score": np.asarray(group_scores, dtype=float),
            },
        )
        # The preview limit may sample rows, so mirror the same deterministic
        # selection when attaching per-point colors.
        group_selection = _sample_rows(
            len(group_positions), preview_limit,
            seed=sum(map(ord, "descriptor_groups")),
        )
        group_layer["colors"] = _rounded(
            np.asarray(group_colors, dtype=float)[group_selection], 3,
        )
        layers.append(group_layer)
        segments = np.asarray(group_segments, dtype=float).reshape(-1, 2, 3)
        layers.append({
            "id": "descriptor_group_links",
            "label": 'Centroid connections',
            "kind": "lines", "role": "vector",
            "total": int(len(segments)),
            "segments": _rounded(np.stack([
                _to_display(segments[:, 0], up, ground),
                _to_display(segments[:, 1], up, ground),
            ], axis=1), 4),
        })

    focus_local = None
    for candidates in (informative_local, ambiguous_local, all_local):
        if len(candidates):
            candidate_margins = np.nan_to_num(
                margins[candidates], nan=-1.0, posinf=1.0, neginf=-1.0,
            )
            focus_local = int(candidates[np.argmax(candidate_margins)])
            break
    descriptor_focus = None
    if focus_local is not None:
        grid_layers, descriptor_focus = _descriptor_grid_layers(
            features[focus_local], scene.volume, cloud,
            source_indices[focus_local], up, ground,
        )
        layers.extend(grid_layers)

    finite_distance = best_distance[np.isfinite(best_distance)]
    distance_limit = max(
        float(parameters["utility_distance_threshold"]),
        float(finite_distance.max(initial=0.0)), 1.0,
    )
    distance_hist, distance_edges = np.histogram(
        finite_distance, bins=np.linspace(0.0, distance_limit, 25),
    )
    margin_hist, margin_edges = np.histogram(
        margins, bins=np.linspace(0.0, 1.0, 25),
    )
    entropy_hist, entropy_edges = np.histogram(
        entropies, bins=np.linspace(0.0, 1.0, 25),
    )
    model_hist, model_edges = np.histogram(
        valid_models,
        bins=np.arange(0, max(2, int(valid_models.max(initial=0)) + 2)),
    )
    steps = [
        {
            "id": "occupation", "input": int(keypoints.size),
            "kept": int(len(features)),
            "rejected": int(keypoints.size - len(features)),
            "visible_layers": [
                "descriptor_context", "descriptor_grid_bounds",
                "descriptor_seed", "descriptor_raw_unknown",
                "descriptor_raw_free", "descriptor_raw_occupied",
            ] if descriptor_focus else ["surface", "descriptor_all"],
        },
        {
            "id": "seed_fill", "input": int(len(features)),
            "kept": int(len(features) - len(zero_occupied_local)),
            "rejected": int(len(zero_occupied_local)),
            "visible_layers": [
                "descriptor_context", "descriptor_grid_bounds",
                "descriptor_seed", "descriptor_final_unknown",
                "descriptor_final_free", "descriptor_final_occupied",
                "descriptor_seed_removed",
            ] if descriptor_focus else ["surface", "descriptor_all"],
        },
        {
            "id": "groups", "input": int(len(features)),
            "kept": int(sum(
                len(group.get("feature_indices", []))
                for group in feature_groups
            )),
            "rejected": int(max(0, len(features) - len({
                int(index)
                for group in feature_groups
                for index in group.get("feature_indices", [])
            }))),
            "kept_label": 'Grouped',
            "rejected_label": 'Isolated',
            "visible_layers": [
                "surface", "descriptor_groups", "descriptor_group_links",
            ],
        },
        {
            "id": "candidate_pool", "input": int(len(files)),
            "kept": int(len(models)),
            "rejected": int(max(0, len(files) - len(models))),
            "visible_layers": ["surface", "descriptor_all"],
        },
        {
            "id": "distance", "input": int(len(features)),
            "kept": int(len(features) - len(no_match_local)),
            "rejected": int(len(no_match_local)),
            "kept_label": 'Correspondences',
            "rejected_label": 'No correspondence',
            "warning": "diagnostic_only",
            "visible_layers": [
                "surface", "descriptor_no_match", "descriptor_ambiguous",
                "descriptor_informative",
            ],
        },
        {
            "id": "ambiguity", "input": int(len(features)),
            "kept": int(len(informative_local)),
            "rejected": int(len(ambiguous_local) + len(no_match_local)),
            "kept_label": "Informative",
            "rejected_label": "Others",
            "warning": "diagnostic_only",
            "visible_layers": [
                "surface", "descriptor_no_match", "descriptor_ambiguous",
                "descriptor_informative",
            ],
        },
    ]
    metadata = {item[0]: (item[1], item[2]) for item in DESCRIPTOR_STEPS}
    steps = [{
        **step,
        "label": metadata[step["id"]][0],
        "explanation": (
            metadata[step["id"]][1]
            + ({
                "occupation": (
                    ' Camera focuses on a representative descriptor: orange = occupied, blue = free, grey = unknown.'
                ),
                "seed_fill": (
                    ' Red cells belonged to an occupied component disconnected from the yellow keypoint.'
                ),
                "distance": (
                    ' Diagnostic only: no descriptor is removed from retrieval at this stage.'
                ),
                "ambiguity": (
                    ' Diagnostic only: displayed classes do not filter descriptors passed to retrieval.'
                ),
            }.get(step["id"], ""))
        ),
    } for step in steps]

    artifacts = {}
    if artifact_path is not None:
        artifact_path = Path(artifact_path).resolve()
        _atomic_pickle(artifact_path, {
            "scene": scene, "keypoints": keypoints, "features": features,
            "keypoint_groups": payload.get("keypoint_groups", []),
            "feature_groups": feature_groups,
            "utility": utilities, "candidate_indices": selected,
            "candidate_names": [model.name for model in models],
            "parameters": dict(parameters),
        })
        artifacts["descriptors"] = {
            "path": str(artifact_path), "format": "pickle",
            "count": int(len(features)),
        }

    display_all = _to_display(cloud.points, up, ground)
    duration = time.perf_counter() - started
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "descriptors",
        "scene": {**source, "points": int(cloud.size), "cache_hit": True},
        "duration_seconds": round(duration, 3),
        "ground": round(ground, 5),
        "bounds": {
            "min": _rounded(display_all.min(axis=0), 4),
            "max": _rounded(display_all.max(axis=0), 4),
        },
        "steps": steps,
        "layers": layers,
        "artifacts": artifacts,
        "metrics": {
            "descriptors": int(len(features)),
            "informative": int(len(informative_local)),
            "ambiguous": int(len(ambiguous_local)),
            "no_match": int(len(no_match_local)),
            "hole_boundary_cells_removed": int(
                hole_boundary_removed.sum()
            ),
            "descriptors_affected_by_holes": int(np.count_nonzero(
                hole_boundary_removed
            )),
            "candidate_models": int(len(models)),
            "database_models": int(len(files)),
            "object_groups": int(len(feature_groups)),
            "informative_ratio": round(
                len(informative_local) / max(1, len(features)), 4,
            ),
        },
        "candidate_pool": [
            {"name": model.name, "synset": model.synset}
            for model in models
        ],
        "feature_groups": [{
            "id": int(group["id"]),
            "descriptor_count": int(group["descriptor_count"]),
            "source_indices": group["source_indices"],
            "centroid": group["centroid"],
            "extent": group["extent"],
            "score": round(float(group.get("score", 0.0)), 6),
        } for group in feature_groups],
        "descriptor_focus": descriptor_focus,
        "database_dir": str(db_dir),
        "histograms": {
            "descriptor_distance": {
                "counts": distance_hist.tolist(),
                "edges": _rounded(distance_edges, 6),
            },
            "descriptor_margin": {
                "counts": margin_hist.tolist(),
                "edges": _rounded(margin_edges, 6),
            },
            "descriptor_entropy": {
                "counts": entropy_hist.tolist(),
                "edges": _rounded(entropy_edges, 6),
            },
            "valid_models": {
                "counts": model_hist.tolist(),
                "edges": _rounded(model_edges, 6),
            },
        },
    }


def _query_scene_result(
        source, parameters, progress=None, artifact_path=None):
    import db_store
    from candidate_index import (
        load_candidate_index, rank_candidates,
        rank_candidates_by_feature_groups,
        rerank_candidates_by_local_feature_groups,
        rerank_candidates_by_local_features,
    )

    started = time.perf_counter()
    source_path = Path(source["path"])
    with source_path.open("rb") as stream:
        payload = pickle.load(stream)
    required = {"scene", "keypoints", "features"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(
            'Top-k requires a chained Descriptors artifact. Run and chain the Descriptors stage first.'
        )
    scene = payload["scene"]
    keypoints = payload["keypoints"]
    features = list(payload["features"])
    cloud = scene.cloud
    up = np.asarray(scene.up, float)
    up /= max(np.linalg.norm(up), 1e-12)
    ground = float(scene.ground)

    cfg = PipelineConfig()
    descriptor_parameters = payload.get("parameters") or {}
    for key in (
        "occ_grid_res", "occ_extent", "distance_unit",
        "distance_exponent", "max_occupied",
    ):
        if key in descriptor_parameters:
            setattr(cfg.descriptor, key, descriptor_parameters[key])
    if "descriptor_ratio_threshold" in descriptor_parameters:
        cfg.matching.descriptor_ratio_threshold = float(
            descriptor_parameters["descriptor_ratio_threshold"]
        )
    cfg.matching.descriptor_ratio_threshold = float(parameters.get(
        "candidate_descriptor_ratio_threshold",
        cfg.matching.descriptor_ratio_threshold,
    ))

    db_dir = Path(RUNTIME_PATHS.database_dir)
    candidate_index_path = db_dir / "candidate_index.npz"
    if not db_store.is_db_dir(str(db_dir)):
        raise FileNotFoundError(f"ShapeNet database not found : {db_dir}")
    if not candidate_index_path.exists():
        raise FileNotFoundError(
            f'Candidate index not found: {candidate_index_path}'
        )
    index = load_candidate_index(str(candidate_index_path))
    files = db_store.model_files(str(db_dir))
    database_count = min(len(index.names), len(files))
    top_k = min(database_count, max(1, int(parameters["candidate_top_k"])))
    pool_k = min(
        database_count,
        max(top_k, int(parameters["candidate_pool_k"])),
    )
    feature_group_records = list(payload.get("feature_groups") or [])
    active_group_records = []
    feature_groups = []
    min_group_descriptors = int(parameters.get(
        "candidate_min_group_descriptors", 6,
    ))
    for record in feature_group_records:
        group = [
            features[int(index)]
            for index in record.get("feature_indices", [])
            if 0 <= int(index) < len(features)
        ]
        if len(group) >= min_group_descriptors:
            active_group_records.append(record)
            feature_groups.append(group)
    grouped_query = bool(parameters.get("grouped_query", True) and feature_groups)
    group_rankings = []
    group_details = {}
    local_preselection = {"available": False, "added": 0}
    group_weights = _query_group_weights(
        active_group_records, feature_groups,
        enabled=bool(parameters.get("candidate_group_weighting", True)),
    )
    global_ranking = []
    if grouped_query:
        if progress:
            progress(
                f'Grouped preselection: {len(feature_groups)} query/queries, merged pool {pool_k}/{database_count}'
            )
        pool, group_rankings = rank_candidates_by_feature_groups(
            index, feature_groups, cfg, pool_k,
            diversify=bool(parameters["candidate_diverse"]),
            diversity_fraction=float(parameters["candidate_diversity"]),
            group_weights=group_weights,
            min_candidates_per_group=int(parameters.get(
                "candidate_pool_min_per_group", 20,
            )),
        )
        global_fraction = float(np.clip(parameters.get(
            "candidate_global_pool_fraction", 0.70,
        ), 0.0, 1.0))
        global_reserve = min(pool_k, int(round(pool_k * global_fraction)))
        if global_reserve:
            global_ranking = rank_candidates(
                index, features, cfg, pool_k,
                diversify=bool(parameters["candidate_diverse"]),
                diversity_fraction=float(parameters["candidate_diversity"]),
            )
            from candidate_index import merge_candidate_pools
            pool = merge_candidate_pools(
                pool, group_rankings, global_ranking, pool_k, global_fraction,
                int(parameters.get("candidate_pool_min_per_group", 20)))
        from local_preselection import supplement_from_database
        pool, group_rankings, local_preselection = supplement_from_database(
            db_dir, index.names, feature_groups, cfg, pool, group_rankings,
            extra_limit=int(parameters.get("candidate_local_extra_k", 64)),
            progress=progress,
        )
        if bool(parameters["local_rerank"]) and len(pool) > top_k:
            selected, group_details = (
                rerank_candidates_by_local_feature_groups(
                    str(db_dir), pool, feature_groups, cfg, top_k,
                    up=up, progress=progress,
                    group_rankings=group_rankings,
                    global_ranking=global_ranking,
                    max_groups_per_candidate=int(
                        parameters.get("candidate_max_groups", 1)
                    ),
                    group_weights=group_weights,
                    min_candidates_per_group=int(parameters.get(
                        "candidate_top_min_per_group", 6,
                    )),
                    local_weight=float(parameters.get(
                        "candidate_local_weight", 0.65,
                    )),
                    target_matches=int(parameters.get(
                        "candidate_target_matches", 4,
                    )),
                    target_coverage=float(parameters.get(
                        "candidate_target_coverage", 0.10,
                    )),
                    extent_weight=float(parameters.get(
                        "candidate_extent_weight", 0.30,
                    )),
                    extent_quota_fraction=float(parameters.get(
                        "candidate_extent_quota_fraction", 0.0,
                    )),
                )
            )
            strategy = "grouped_two_level"
        else:
            selected = list(pool[:top_k])
            strategy = "grouped_global"
    else:
        if progress:
            progress(
                f"Global preselection: pool {pool_k}/{database_count} model(s)"
            )
        pool = rank_candidates(
            index, features, cfg, pool_k,
            diversify=bool(parameters["candidate_diverse"]),
            diversity_fraction=float(parameters["candidate_diversity"]),
        )
        if bool(parameters["local_rerank"]) and len(pool) > top_k:
            if progress:
                progress(
                    f'Local reranking: {len(pool)} candidate(s) to Top-{top_k}'
                )
            selected = rerank_candidates_by_local_features(
                str(db_dir), pool, features, cfg, top_k,
                up=up, progress=progress,
                local_weight=float(parameters.get(
                    "candidate_local_weight", 0.65,
                )),
                target_matches=int(parameters.get(
                    "candidate_target_matches", 4,
                )),
                target_coverage=float(parameters.get(
                    "candidate_target_coverage", 0.10,
                )),
            )
            strategy = "two_level"
        else:
            selected = list(pool[:top_k])
            strategy = "global"
    pool_rank = {int(value): rank + 1 for rank, value in enumerate(pool)}

    group_hits = {}
    for group_position, ranking in enumerate(group_rankings):
        group_id = int(active_group_records[group_position]["id"])
        for rank, model_index in enumerate(ranking, 1):
            group_hits.setdefault(int(model_index), []).append({
                "group_id": group_id, "rank": int(rank),
            })

    def candidate_record(index_value, rank, include_breakdown=True):
        index_value = int(index_value)
        detail = group_details.get(index_value, {})
        best_group_position = int(detail.get("best_group", -1))
        best_group_id = (
            int(active_group_records[best_group_position]["id"])
            if 0 <= best_group_position < len(active_group_records) else None
        )
        def mapped_group_ids(key):
            return [
                int(active_group_records[int(position)]["id"])
                for position in detail.get(key, [])
                if 0 <= int(position) < len(active_group_records)
            ]

        record = {
            "rank": int(rank),
            "index": index_value,
            "name": index.names[index_value],
            "synset": index.synsets[index_value],
            "pool_rank": int(pool_rank.get(index_value, rank)),
            "best_group_id": best_group_id,
            "group_score": json_compatible(detail.get("best_score")),
            "local_group_score": (
                json_compatible(detail.get("group_scores", [])[best_group_position])
                if 0 <= best_group_position < len(detail.get("group_scores", []))
                else None
            ),
            "preselection_group_score": (
                json_compatible(detail.get(
                    "preselection_scores", [],
                )[best_group_position])
                if 0 <= best_group_position < len(detail.get(
                    "preselection_scores", [],
                )) else None
            ),
            "group_weight": (
                round(float(group_weights[best_group_position]), 6)
                if 0 <= best_group_position < len(group_weights) else None
            ),
            "group_evidence": (
                json_compatible(detail.get("group_evidence", [])[best_group_position])
                if (
                    0 <= best_group_position
                    < len(detail.get("group_evidence", []))
                ) else None
            ),
            "origin_group_ids": mapped_group_ids("origin_groups"),
            "selected_for_group_ids": mapped_group_ids(
                "selected_for_groups"
            ),
            "group_hits": group_hits.get(index_value, []),
        }
        if include_breakdown and group_details:
            group_scores = detail.get("group_scores", [])
            preselection_scores = detail.get("preselection_scores", [])
            preselection_ranks = detail.get("preselection_ranks", [])
            combined_scores = detail.get("combined_group_scores", [])
            evidences = detail.get("group_evidence", [])
            evaluated = set(detail.get("evaluated_groups", []))
            origins = set(detail.get("origin_groups", []))
            protected = set(detail.get("selected_for_groups", []))
            record["group_breakdown"] = [{
                "group_id": int(group_record["id"]),
                "evaluated": group_position in evaluated,
                "origin": group_position in origins,
                "protected": group_position in protected,
                "preselection_rank": (
                    preselection_ranks[group_position]
                    if group_position < len(preselection_ranks) else None
                ),
                "preselection_score": json_compatible(
                    preselection_scores[group_position]
                    if group_position < len(preselection_scores) else None
                ),
                "local_score": json_compatible(
                    group_scores[group_position]
                    if group_position < len(group_scores) else None
                ),
                "combined_score": json_compatible(
                    combined_scores[group_position]
                    if group_position < len(combined_scores) else None
                ),
                "evidence": json_compatible(
                    evidences[group_position]
                    if group_position < len(evidences) else None
                ),
            } for group_position, group_record in enumerate(
                active_group_records
            )]
        return record

    pool_records = [
        candidate_record(value, rank + 1)
        for rank, value in enumerate(pool)
    ]
    top_records = [
        candidate_record(value, rank + 1)
        for rank, value in enumerate(selected)
    ]
    group_results = []
    for group_position, record in enumerate(active_group_records):
        if group_details:
            def group_ranking_score(model_index):
                detail = group_details.get(int(model_index), {})
                combined = detail.get("combined_group_scores", [])
                local = detail.get("group_scores", [])
                values = combined if group_position < len(combined) else local
                return (
                    float(values[group_position])
                    if group_position < len(values) else float("-inf")
                )

            ranked_for_group = sorted(
                pool,
                key=lambda model_index: -group_ranking_score(model_index),
            )[:top_k]
        else:
            ranked_for_group = group_rankings[group_position][:top_k]
        group_results.append({
            "id": int(record["id"]),
            "descriptor_count": int(record.get("descriptor_count", 0)),
            "source_indices": record.get("source_indices", []),
            "centroid": record.get("centroid", []),
            "extent": record.get("extent", []),
            "query_weight": round(float(group_weights[group_position]), 6),
            "top_k": [
                candidate_record(model_index, rank + 1, include_breakdown=False)
                for rank, model_index in enumerate(ranked_for_group)
            ],
        })
    if progress:
        progress(
            f'Final Top-k: {len(top_records)}/{database_count} model(s)'
        )

    surface_limit = int(parameters.get("surface_preview_points", 45000))
    surface_selection = _sample_rows(cloud.size, surface_limit, seed=7)
    feature_positions = np.asarray(
        [feature.position for feature in features], float,
    ).reshape(-1, 3)
    if len(feature_positions):
        _, feature_sources = cloud.tree.query(feature_positions, k=1)
        feature_sources = np.asarray(feature_sources, int)
    else:
        feature_sources = np.zeros(0, int)
    layers = [
        {
            "id": "surface", "label": 'Fused surface',
            "kind": "surface", "role": "surface",
            "total": int(cloud.size), "sampled": int(len(surface_selection)),
            "points": _rounded(_to_display(
                cloud.points[surface_selection], up, ground,
            )),
            "curvature": _rounded(cloud.curvature[surface_selection], 6),
            "rgb_colors": _rounded(
                cloud.colors[surface_selection], 3,
            ) if getattr(cloud, "colors", None) is not None else None,
            "source_index": surface_selection.astype(int).tolist(),
        },
        {
            "id": "query_keypoints", "label": 'Scene query',
            "kind": "points", "role": "retained",
            "total": int(len(feature_positions)),
            "sampled": int(len(feature_positions)),
            "points": _rounded(_to_display(feature_positions, up, ground)),
            "source_index": feature_sources.tolist(),
            "response": _rounded(np.asarray([
                feature.response for feature in features
            ], float), 6),
        },
    ]
    if active_group_records:
        grouped_positions = []
        grouped_sources = []
        grouped_ids = []
        for record in active_group_records:
            for feature_index in record.get("feature_indices", []):
                if 0 <= int(feature_index) < len(features):
                    grouped_positions.append(features[int(feature_index)].position)
                    grouped_sources.append(int(feature_sources[int(feature_index)]))
                    grouped_ids.append(int(record["id"]))
        layers.append({
            "id": "query_object_groups", "label": 'Query groups',
            "kind": "points", "role": "candidate",
            "total": int(len(grouped_positions)),
            "sampled": int(len(grouped_positions)),
            "points": _rounded(_to_display(
                np.asarray(grouped_positions, float).reshape(-1, 3), up, ground,
            )),
            "source_index": grouped_sources,
            "group_id": grouped_ids,
        })
    steps = [
        {
            "id": "database", "input": database_count,
            "kept": database_count, "rejected": 0,
            "visible_layers": ["surface"],
        },
        {
            "id": "object_groups", "input": len(features),
            "kept": len(active_group_records),
            "rejected": max(0, len(features) - sum(
                len(group) for group in feature_groups
            )),
            "visible_layers": [
                "surface", "query_keypoints", "query_object_groups",
            ] if active_group_records else ["surface", "query_keypoints"],
            "warning": None if grouped_query else "disabled",
        },
        {
            "id": "global_pool", "input": database_count,
            "kept": len(pool_records),
            "rejected": max(0, database_count - len(pool_records)),
            "visible_layers": [
                "surface", "query_keypoints", "query_object_groups",
            ] if active_group_records else ["surface", "query_keypoints"],
        },
        {
            "id": "local_rerank", "input": len(pool_records),
            "kept": len(top_records),
            "rejected": max(0, len(pool_records) - len(top_records)),
            "visible_layers": [
                "surface", "query_keypoints", "query_object_groups",
            ] if active_group_records else ["surface", "query_keypoints"],
            "warning": None if strategy.endswith("two_level") else "disabled",
        },
        {
            "id": "top_k", "input": len(pool_records),
            "kept": len(top_records),
            "rejected": max(0, len(pool_records) - len(top_records)),
            "visible_layers": [
                "surface", "query_keypoints", "query_object_groups",
            ] if active_group_records else ["surface", "query_keypoints"],
        },
    ]
    metadata = {item[0]: (item[1], item[2]) for item in QUERY_STEPS}
    steps = [{
        **step, "label": metadata[step["id"]][0],
        "explanation": metadata[step["id"]][1],
    } for step in steps]

    artifacts = {}
    if artifact_path is not None:
        artifact_path = Path(artifact_path).resolve()
        _atomic_pickle(artifact_path, {
            **payload,
            "candidate_indices": [int(value) for value in selected],
            "candidate_names": [item["name"] for item in top_records],
            "candidate_pool_indices": [int(value) for value in pool],
            "feature_groups": active_group_records,
            "candidate_group_assignments": {
                int(model_index): int(detail["best_group"])
                for model_index, detail in group_details.items()
                if int(detail.get("best_group", -1)) >= 0
            },
            "candidate_group_details": {
                int(model_index): {
                    "best_group": int(detail.get("best_group", -1)),
                    "origin_groups": [
                        int(value) for value in detail.get(
                            "origin_groups", []
                        )
                    ],
                    "selected_for_groups": [
                        int(value) for value in detail.get(
                            "selected_for_groups", []
                        )
                    ],
                    "group_scores": list(detail.get("group_scores", [])),
                    "preselection_scores": list(detail.get(
                        "preselection_scores", []
                    )),
                    "combined_group_scores": list(detail.get(
                        "combined_group_scores", []
                    )),
                    "group_evidence": list(detail.get(
                        "group_evidence", []
                    )),
                }
                for model_index, detail in group_details.items()
            },
            "query_parameters": dict(parameters),
            "database_dir": str(db_dir),
        })
        artifacts["query"] = {
            "path": str(artifact_path), "format": "pickle",
            "count": int(len(top_records)),
        }

    display_all = _to_display(cloud.points, up, ground)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "query",
        "scene": {**source, "points": int(cloud.size), "cache_hit": True},
        "duration_seconds": round(time.perf_counter() - started, 3),
        "ground": round(ground, 5),
        "bounds": {
            "min": _rounded(display_all.min(axis=0), 4),
            "max": _rounded(display_all.max(axis=0), 4),
        },
        "steps": steps,
        "layers": layers,
        "artifacts": artifacts,
        "metrics": {
            "database_models": database_count,
            "local_preselection": local_preselection,
            "descriptors": len(features),
            "pool_models": len(pool_records),
            "top_k_models": len(top_records),
            "retained": len(top_records),
            "strategy": strategy,
            "object_groups": len(active_group_records),
            "group_weights": [round(float(value), 6) for value in group_weights],
            "pool_min_per_group": int(parameters.get(
                "candidate_pool_min_per_group", 20,
            )),
            "global_pool_fraction": round(float(parameters.get(
                "candidate_global_pool_fraction", 0.70,
            )), 6),
            "top_min_per_group": int(parameters.get(
                "candidate_top_min_per_group", 6,
            )),
            "descriptor_ratio_threshold": round(float(
                cfg.matching.descriptor_ratio_threshold
            ), 6),
            "local_weight": round(float(parameters.get(
                "candidate_local_weight", 0.65,
            )), 6),
            "extent_weight": round(float(parameters.get(
                "candidate_extent_weight", 0.30,
            )), 6),
            "extent_quota_fraction": round(float(parameters.get(
                "candidate_extent_quota_fraction", 0.0,
            )), 6),
            "target_matches": int(parameters.get(
                "candidate_target_matches", 4,
            )),
            "target_coverage": round(float(parameters.get(
                "candidate_target_coverage", 0.10,
            )), 6),
            "max_groups_per_candidate": int(parameters.get(
                "candidate_max_groups", 1,
            )),
        },
        "candidate_pool": pool_records,
        "top_k": top_records,
        "query_groups": group_results,
        "database_dir": str(db_dir),
        "histograms": {},
    }


def _query_candidate_detail(
        artifact_path, candidate_index, db_dir=None, max_model_points=1800,
        group_id=None):
    'Compute Top-k correspondences and a provisional pose on demand.'
    import db_store
    from candidate_index import (
        _extent_compatibility,
        _extent_signature,
        _local_descriptor_score,
    )
    from constellations import one_point_ransac
    from matching import match_features

    artifact_path = Path(artifact_path).resolve()
    with artifact_path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not {
            "scene", "features", "candidate_pool_indices",
    }.issubset(payload):
        raise ValueError('Incomplete or incompatible Top-k artifact')
    candidate_index = int(candidate_index)
    allowed = {
        int(value) for value in payload.get("candidate_pool_indices", [])
    }
    if candidate_index not in allowed:
        raise ValueError('This model is not in the pool for this run')

    db_dir = Path(db_dir or RUNTIME_PATHS.database_dir).resolve()
    files = db_store.model_files(str(db_dir))
    if candidate_index < 0 or candidate_index >= len(files):
        raise ValueError('Model index outside the database')
    model = db_store.load_model(files[candidate_index])
    scene = payload["scene"]
    scan_features = list(payload["features"])
    group_position = int(
        payload.get("candidate_group_assignments", {}).get(
            candidate_index, -1,
        )
    )
    feature_groups = list(payload.get("feature_groups") or [])
    if group_id is not None:
        requested_position = next((
            position for position, record in enumerate(feature_groups)
            if int(record.get("id", -1)) == int(group_id)
        ), None)
        if requested_position is None:
            raise ValueError(f"Unknown query group: {group_id}")
        group_position = int(requested_position)
    active_group = (
        feature_groups[group_position]
        if 0 <= group_position < len(feature_groups) else None
    )
    if active_group is not None:
        grouped_features = [
            scan_features[int(index)]
            for index in active_group.get("feature_indices", [])
            if 0 <= int(index) < len(scan_features)
        ]
        if len(grouped_features) >= 2:
            scan_features = grouped_features
    up = np.asarray(scene.up, float)
    up /= max(np.linalg.norm(up), 1e-12)
    ground = float(scene.ground)

    cfg = PipelineConfig()
    descriptor_parameters = payload.get("parameters") or {}
    for key in (
        "occ_grid_res", "occ_extent", "distance_unit",
        "distance_exponent", "max_occupied",
    ):
        if key in descriptor_parameters:
            setattr(cfg.descriptor, key, descriptor_parameters[key])
    if "utility_distance_threshold" in descriptor_parameters:
        cfg.ransac.desc_inlier = float(
            descriptor_parameters["utility_distance_threshold"]
        )
    if "descriptor_ratio_threshold" in descriptor_parameters:
        cfg.matching.descriptor_ratio_threshold = float(
            descriptor_parameters["descriptor_ratio_threshold"]
        )
    query_parameters = payload.get("query_parameters") or {}
    cfg.matching.descriptor_ratio_threshold = float(query_parameters.get(
        "candidate_descriptor_ratio_threshold",
        cfg.matching.descriptor_ratio_threshold,
    ))

    started = time.perf_counter()
    descriptor_score = _local_descriptor_score(
        model.features, scan_features, cfg, up,
        target_matches=int(query_parameters.get(
            "candidate_target_matches", 4,
        )),
        target_coverage=float(query_parameters.get(
            "candidate_target_coverage", 0.10,
        )),
    )
    model_points_all = np.asarray(model.cloud.points, float)
    model_extent = _extent_signature(model_points_all, up)
    scan_extent = _extent_signature([
        feature.position for feature in scan_features
    ], up)
    extent_similarity = _extent_compatibility(
        model_extent, scan_extent, cfg,
    )
    extent_weight = float(np.clip(query_parameters.get(
        "candidate_extent_weight", 0.30,
    ), 0.0, 1.0))
    local_score = (
        (1.0 - extent_weight) * float(descriptor_score)
        + extent_weight * float(extent_similarity)
        if np.isfinite(descriptor_score) else float("-inf")
    )
    correspondences = match_features(
        model.features, scan_features, cfg, up,
    )
    constellations = one_point_ransac(
        correspondences, model.features, scan_features, cfg, up,
    ) if correspondences else []
    pose = (
        constellations[0].transform if constellations
        else (min(correspondences, key=lambda item: item.desc_dist).transform
              if correspondences else None)
    )

    model_points = model_points_all
    if len(model_points):
        model_height = model_points @ up
        model_points = model_points[
            model_height > float(model.ground) + 1e-5
        ]
    selection = _sample_rows(
        len(model_points), int(max_model_points), seed=candidate_index + 31,
    )
    model_points = model_points[selection]
    placed_points = (
        pose.apply(model_points) if pose is not None else model_points
    )

    inliers = (
        list(constellations[0].inliers) if constellations
        else sorted(correspondences, key=lambda item: item.desc_dist)[:12]
    )
    match_records = []
    correspondence_segments = []
    for correspondence in sorted(
            inliers, key=lambda item: float(item.desc_dist))[:32]:
        model_position = np.asarray(
            model.features[correspondence.model_idx].position, float,
        )
        scene_position = np.asarray(
            scan_features[correspondence.scan_idx].position, float,
        )
        placed_model_position = (
            pose.apply(model_position[None, :])[0]
            if pose is not None else model_position
        )
        display_pair = _to_display(
            np.vstack([placed_model_position, scene_position]), up, ground,
        )
        correspondence_segments.append(_rounded(display_pair, 4))
        match_records.append({
            "model_keypoint": int(correspondence.model_idx),
            "scan_keypoint": int(correspondence.scan_idx),
            "descriptor_distance": round(
                float(correspondence.desc_dist), 4,
            ),
            "theta_deg": round(
                float(correspondence.theta) * 57.2957795, 2,
            ),
            "scale": round(float(correspondence.scale), 4),
            "model_position": _rounded(display_pair[0], 4),
            "scene_position": _rounded(display_pair[1], 4),
        })

    pose_record = None
    if pose is not None:
        pose_record = {
            "source": "constellation" if constellations else "correspondence",
            "theta_deg": round(float(pose.theta) * 57.2957795, 2),
            "scale": round(float(pose.scale), 4),
            "translation": _rounded(np.asarray(pose.t, float), 4),
            "quality": (
                round(float(constellations[0].quality), 4)
                if constellations else None
            ),
            "inliers": len(inliers) if constellations else 0,
            "support_count": len(inliers),
            "support_kind": (
                "inliers" if constellations else "best_correspondences"
            ),
        }
    return {
        "candidate_index": candidate_index,
        "name": str(model.name),
        "synset": str(model.synset),
        "group_id": (
            int(active_group["id"]) if active_group is not None else None
        ),
        "group_descriptor_count": (
            len(scan_features) if active_group is not None else None
        ),
        "local_score": (
            round(float(local_score), 6)
            if np.isfinite(local_score) else None
        ),
        "descriptor_score": (
            round(float(descriptor_score), 6)
            if np.isfinite(descriptor_score) else None
        ),
        "extent_similarity": round(float(extent_similarity), 6),
        "extent_weight": round(float(extent_weight), 6),
        "model_keypoints": len(model.features),
        "scan_keypoints": len(scan_features),
        "correspondences": len(correspondences),
        "constellations": len(constellations),
        "pose": pose_record,
        "placed": pose is not None,
        "model_points": _rounded(
            _to_display(placed_points, up, ground), 4,
        ),
        "correspondence_segments": correspondence_segments,
        "matches": match_records,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _scene_result(
        source, parameters, progress=None, stage="keypoints",
        artifact_path=None):
    if stage in ("matching", "verification", "selection"):
        from debug_retrieval import scene_result
        return scene_result(source, parameters, stage, progress, artifact_path)
    if stage == "fusion":
        return _fusion_scene_result(
            source, parameters, progress=progress,
            artifact_path=artifact_path,
        )
    if stage == "keypoints":
        return _keypoint_scene_result(
            source, parameters, progress=progress,
            artifact_path=artifact_path,
        )
    if stage == "descriptors":
        return _descriptor_scene_result(
            source, parameters, progress=progress,
            artifact_path=artifact_path,
        )
    if stage == "query":
        return _query_scene_result(
            source, parameters, progress=progress,
            artifact_path=artifact_path,
        )
    raise ValueError(f"Debug stage not implemented: {stage}")


def _write_worker_result(payload_path, output_path):
    from sdf_fusion import DenseVolumeLimitError

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    source = payload["source"]
    output = Path(output_path)
    stage = payload.get("stage", "keypoints")
    artifact_path = {
        "fusion": output.with_name(f"{output.stem}.geometry.npz"),
        "keypoints": output.with_name(f"{output.stem}.keypoints.pkl"),
        "descriptors": output.with_name(f"{output.stem}.descriptors.pkl"),
        "query": output.with_name(f"{output.stem}.query.pkl"),
        "matching": output.with_name(f"{output.stem}.matching.pkl"),
        "verification": output.with_name(f"{output.stem}.verification.pkl"),
        "selection": output.with_name(f"{output.stem}.selection.pkl"),
    }.get(stage)
    def progress(message):
        print(message, flush=True)
    try:
        result = _scene_result(
            source, payload["parameters"], progress=progress,
            stage=stage, artifact_path=artifact_path,
        )
    except DenseVolumeLimitError as error:
        print(f"Configuration rejected: {error}", flush=True)
        raise SystemExit(2)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            json_compatible(result), ensure_ascii=False, allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(output))


def _read_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_settings(settings):
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _debug_run_folder(run_id):
    run_id = str(run_id or "")
    if not run_id or Path(run_id).name != run_id:
        raise FileNotFoundError("Identifiant de run de debug invalide")
    folder = RUN_ROOT / run_id
    if not (folder / "manifest.json").exists():
        raise FileNotFoundError(f"Run de debug not found : {run_id}")
    return folder


def _debug_run_metadata(run_id):
    path = _debug_run_folder(run_id) / "run_meta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _write_debug_run_metadata(run_id, updates):
    folder = _debug_run_folder(run_id)
    path = folder / "run_meta.json"
    payload = _debug_run_metadata(run_id)
    payload.update(updates)
    payload["updated_at"] = time.time()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    os.replace(str(temporary), str(path))
    return payload


def _enrich_debug_manifest(manifest):
    metadata = _debug_run_metadata(manifest["id"])
    return {
        **manifest,
        "label": str(metadata.get("label") or "").strip() or manifest["id"],
        "favorite": bool(metadata.get("favorite", False)),
    }


class DebugVisualizerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.events = []
        self.next_event_id = 1
        self.log_lines = []
        self.worker = None
        self.processes = []
        self.stop_requested = False
        self.last_payload = None
        self.pending_watch = False
        self.query_candidate_cache = {}
        settings = _read_settings()
        self.watch_enabled = bool(settings.get("watch_enabled", False))
        self.baseline_run_id = settings.get("baseline_run_id")
        self.current_run_id = self._latest_run_id()
        self.state = {
            "status": "idle", "run_id": self.current_run_id,
            "stage": "keypoints", "trigger": None,
            "started_at": None, "finished_at": None,
            "message": 'Ready', "scenes": {},
        }
        self._watch_signature = self._code_signature()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()

    def _latest_run_id(self):
        manifests = list(RUN_ROOT.glob("*/manifest.json")) if RUN_ROOT.exists() else []
        if not manifests:
            return None
        return max(manifests, key=lambda path: path.stat().st_mtime).parent.name

    def _code_signature(self):
        watched = (
            "sdf_fusion.py", "pipeline.py", "keypoints.py",
            "keypoint_quality.py", "wall_keypoints.py",
            "geometry.py", "matching.py", "descriptors.py",
            "descriptor_utility.py", "candidate_index.py", "local_preselection.py", "config.py",
            "debug_retrieval.py", "constellations.py", "verify.py", "progressive.py",
        )
        return tuple(
            (name, (ROOT / name).stat().st_mtime_ns if (ROOT / name).exists() else 0)
            for name in watched
        )

    def _emit(self, kind, data=None):
        with self.lock:
            event = {
                "id": self.next_event_id, "type": kind,
                "timestamp": time.time(), "data": data or {},
            }
            self.next_event_id += 1
            self.events.append(event)
            self.events = self.events[-500:]
            return event

    def _record_log(self, run_id, stage, message, scene_id=None, log_path=None):
        message = str(message or "").strip()
        if not message:
            return None
        timestamp = time.strftime("%H:%M:%S")
        scope = f"[{scene_id}]" if scene_id else "[debug]"
        line = f"[{timestamp}] {scope} {message}"
        with self.lock:
            self.log_lines.append(line)
            self.log_lines = self.log_lines[-1000:]
            if log_path is not None:
                with Path(log_path).open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
        self._emit("debug_log", {
            "run_id": run_id, "stage": stage, "scene_id": scene_id,
            "message": line,
        })
        return line

    def _stream_worker_output(
            self, process, run_id, stage, scene_id, logs, log_path):
        if process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            rendered = self._record_log(
                run_id, stage, line, scene_id=scene_id, log_path=log_path,
            )
            if rendered:
                logs.append(rendered)
                if len(logs) > 1000:
                    del logs[:-1000]
            with self.lock:
                scene = self.state.get("scenes", {}).get(scene_id)
                if scene is not None:
                    scene["message"] = line[-140:]

    def defaults(self):
        from debug_retrieval import STEPS
        return {
            "schema_version": SCHEMA_VERSION,
            "pipeline_stages": list(PIPELINE_STAGES),
            "stage_steps": [
                _step_definition(item)
                for item in KEYPOINT_STEPS
            ],
            "stage_steps_by_stage": {
                **{stage: [_step_definition(item) for item in items]
                   for stage, items in STEPS.items()},
                "fusion": [
                    {
                        "id": item[0], "label": item[1],
                        "description": item[2],
                    }
                    for item in FUSION_STEPS
                ],
                "keypoints": [
                    _step_definition(item)
                    for item in KEYPOINT_STEPS
                ],
                "descriptors": [
                    {
                        "id": item[0], "label": item[1],
                        "description": item[2],
                    }
                    for item in DESCRIPTOR_STEPS
                ],
                "query": [
                    {
                        "id": item[0], "label": item[1],
                        "description": item[2],
                    }
                    for item in QUERY_STEPS
                ],
            },
            "parameters": default_parameters(),
            "parameter_schema": parameter_schema(),
            "parameters_by_stage": {
                stage: default_parameters(stage)
                for stage in (item["id"] for item in PIPELINE_STAGES)
            },
            "parameter_schemas": {
                stage: parameter_schema(stage)
                for stage in (item["id"] for item in PIPELINE_STAGES)
            },
            "sources": default_sources(),
            "watch_enabled": self.watch_enabled,
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "recent_runs": self.recent_runs(),
        }

    def snapshot(self):
        with self.lock:
            return {
                **self.state,
                "watch_enabled": self.watch_enabled,
                "baseline_run_id": self.baseline_run_id,
                "current_run_id": self.current_run_id,
                "last_event_id": self.next_event_id - 1,
                "logs": list(self.log_lines[-500:]),
            }

    def events_after(self, event_id):
        with self.lock:
            return [event for event in self.events if event["id"] > event_id]

    def recent_runs(self):
        records = []
        for manifest in RUN_ROOT.glob("*/manifest.json") if RUN_ROOT.exists() else []:
            try:
                payload = _enrich_debug_manifest(
                    json.loads(manifest.read_text(encoding="utf-8")),
                )
                scene_records = payload.get("scenes", {})
                stage = payload.get("stage", "keypoints")
                artifact_names = {
                    "query": ("query",),
                    "matching": ("matching",),
                    "verification": ("verification",),
                    "fusion": ("scene_state", "geometry"),
                    "keypoints": ("keypoints",),
                    "descriptors": ("descriptors",),
                }.get(stage, ())
                chainable_scenes = sum(
                    1 for meta in scene_records.values()
                    if any(
                        (meta.get("artifacts") or {}).get(name)
                        for name in artifact_names
                    )
                )
                next_stage = {
                    "query": "matching",
                    "matching": "verification",
                    "verification": "selection",
                    "fusion": "keypoints",
                    "keypoints": "descriptors",
                    "descriptors": "query",
                }.get(stage)
                records.append({
                    "id": payload["id"], "label": payload["label"],
                    "favorite": payload["favorite"],
                    "stage": stage,
                    "created_at": payload.get("created_at"),
                    "status": payload.get("status", "unknown"),
                    "scene_count": len(scene_records),
                    "chainable": bool(next_stage and chainable_scenes > 0),
                    "next_stage": next_stage,
                    "chainable_scene_count": chainable_scenes,
                })
            except Exception:
                continue
        return sorted(
            records,
            key=lambda item: (
                item.get("favorite", False), item.get("created_at") or 0,
            ),
            reverse=True,
        )[:20]

    def load_result(self, run_id):
        run_id = run_id or self.current_run_id
        if not run_id:
            raise FileNotFoundError('No debug result available')
        folder = _debug_run_folder(run_id)
        manifest = _enrich_debug_manifest(
            json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
        )
        scenes = {}
        for scene_id, meta in manifest.get("scenes", {}).items():
            result_path = folder / meta.get("file", f"{scene_id}.json")
            if result_path.exists():
                scenes[scene_id] = json.loads(result_path.read_text(encoding="utf-8"))
        log_path = folder / "debug.log"
        logs = (
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if log_path.exists() else []
        )
        return {"manifest": manifest, "scenes": scenes, "logs": logs[-1000:]}

    def query_candidate_detail(
            self, run_id, scene_id, candidate_index, group_id=None):
        'Load and place a candidate from a Top-k run on demand.'
        run_id = str(run_id or "")
        scene_id = str(scene_id or "")
        candidate_index = int(candidate_index)
        group_id = None if group_id is None else int(group_id)
        cache_key = (run_id, scene_id, candidate_index, group_id)
        with self.lock:
            cached = self.query_candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        payload = self.load_result(run_id)
        if payload["manifest"].get("stage") in ("matching", "verification", "selection"):
            result = payload.get("scenes", {}).get(scene_id) or {}
            detail = (result.get("candidate_details") or {}).get(str(candidate_index))
            if detail is None:
                raise FileNotFoundError('Hypothesis missing from this result')
            return detail
        if payload["manifest"].get("stage") != "query":
            raise ValueError('This run is not a Top-k run')
        result = payload.get("scenes", {}).get(scene_id)
        if result is None:
            raise FileNotFoundError(f"Scene missing from run: {scene_id}")
        artifact = (result.get("artifacts") or {}).get("query") or {}
        raw_path = artifact.get("path") if isinstance(artifact, dict) else artifact
        if not raw_path:
            raise FileNotFoundError('Top-k artifact missing for this scene')

        folder = _debug_run_folder(run_id).resolve()
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute():
            artifact_path = folder / artifact_path
        artifact_path = artifact_path.resolve()
        if not artifact_path.exists() or not path_is_within(
                artifact_path, [folder]):
            raise FileNotFoundError('Top-k artifact invalid or inaccessible')

        database_dir = Path(
            result.get("database_dir") or RUNTIME_PATHS.database_dir
        ).resolve()
        detail = _query_candidate_detail(
            artifact_path, candidate_index, db_dir=database_dir,
            group_id=group_id,
        )
        with self.lock:
            if len(self.query_candidate_cache) >= 48:
                self.query_candidate_cache.pop(next(iter(
                    self.query_candidate_cache
                )))
            self.query_candidate_cache[cache_key] = detail
        return detail

    def chain_sources(self, run_id, target_stage="keypoints"):
        payload = self.load_result(run_id)
        manifest = payload["manifest"]
        source_stage = manifest.get("stage")
        expected_source = {
            "matching": "query",
            "verification": "matching",
            "selection": "verification",
            "keypoints": "fusion",
            "descriptors": "keypoints",
            "query": "descriptors",
        }.get(target_stage)
        if expected_source is None:
            raise ValueError(
                f"Debug chaining not implemented toward: {target_stage}"
            )
        if source_stage != expected_source:
            raise ValueError(
                f"The upstream run must have stage {expected_source}"
            )

        artifact_names = {
            "matching": ("query",),
            "verification": ("matching",),
            "selection": ("verification",),
            "keypoints": ("scene_state", "geometry"),
            "descriptors": ("keypoints",),
            "query": ("descriptors",),
        }[target_stage]

        folder = _debug_run_folder(run_id)
        sources = []
        for scene_id, result in payload.get("scenes", {}).items():
            artifacts = result.get("artifacts") or {}
            artifact_name = next((
                name for name in artifact_names if artifacts.get(name)
            ), None)
            artifact = artifacts.get(artifact_name) if artifact_name else {}
            raw_path = artifact.get("path") if isinstance(artifact, dict) else artifact
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = folder / path
            path = path.resolve()
            if not path.exists() or not path_is_within(path, [folder]):
                continue
            scene = result.get("scene") or {}
            sources.append({
                "id": scene_id,
                "label": str(scene.get("label") or scene_id),
                "path": str(path),
                "enabled": True,
                "upstream": {
                    "run_id": run_id,
                    "stage": source_stage,
                    "scene_id": scene_id,
                    "artifact": artifact_name,
                },
            })
        if not sources:
            labels = {
                "fusion": ("Fusion", "la "),
                "keypoints": ("Keypoints", 'the '),
                "descriptors": ("Descriptors", 'the '),
            }
            label, article = labels.get(source_stage, (source_stage, "le stage "))
            raise ValueError(
                f'This {label} run contains no chainable artifact. Rerun {label} with the current Debug version.'
            )
        return {
            "source_run": {
                "id": run_id,
                "label": manifest.get("label") or run_id,
                "stage": source_stage,
            },
            "target_stage": target_stage,
            "sources": sources,
        }

    def rename_run(self, run_id, label):
        label = str(label or "").strip()
        if not label:
            raise ValueError('Name cannot be empty')
        if len(label) > 80:
            raise ValueError('Name is limited to 80 characters')
        if any(ord(character) < 32 for character in label):
            raise ValueError('Name contains a control character')
        _write_debug_run_metadata(run_id, {"label": label})
        self._emit("run_metadata_changed", {
            "run_id": run_id, "label": label,
        })
        return self.load_result(run_id)

    def set_favorite(self, run_id, favorite):
        favorite = bool(favorite)
        _write_debug_run_metadata(run_id, {"favorite": favorite})
        self._emit("run_metadata_changed", {
            "run_id": run_id, "favorite": favorite,
        })
        return self.load_result(run_id)

    def set_baseline(self, run_id):
        self.load_result(run_id)
        with self.lock:
            self.baseline_run_id = run_id
            settings = _read_settings()
            settings["baseline_run_id"] = run_id
            settings["watch_enabled"] = self.watch_enabled
            _write_settings(settings)
        self._emit("baseline_changed", {"run_id": run_id})
        return self.snapshot()

    def set_watch(self, enabled):
        with self.lock:
            self.watch_enabled = bool(enabled)
            settings = _read_settings()
            settings["watch_enabled"] = self.watch_enabled
            settings["baseline_run_id"] = self.baseline_run_id
            _write_settings(settings)
        self._emit("watch_changed", {"enabled": self.watch_enabled})
        return self.snapshot()

    def start(self, raw=None, trigger="manual"):
        payload = normalized_payload(raw)
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise RuntimeError('A debug computation is already running')
            identity = hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()[:8]
            run_id = time.strftime(
                f"{payload['stage']}-%Y%m%d-%H%M%S-"
            ) + identity
            folder = RUN_ROOT / run_id
            suffix = 0
            while folder.exists():
                suffix += 1
                folder = RUN_ROOT / f"{run_id}-{suffix}"
            run_id = folder.name
            folder.mkdir(parents=True)
            self.last_payload = payload
            self.stop_requested = False
            self.processes = []
            self.log_lines = []
            now = time.time()
            self.state = {
                "status": "running", "run_id": run_id,
                "stage": payload["stage"],
                "trigger": trigger, "started_at": now, "finished_at": None,
                "message": 'Multi-scene computation', "scenes": {
                    source["id"]: {"label": source["label"], "status": "queued", "message": 'Waiting'}
                    for source in payload["sources"]
                },
            }
            self.current_run_id = run_id
            self.worker = threading.Thread(
                target=self._run_batch, args=(run_id, payload), daemon=True,
            )
            self._emit("run_started", {
                "run_id": run_id, "trigger": trigger,
                "stage": payload["stage"],
            })
            self.worker.start()
        return self.snapshot()

    def stop(self):
        with self.lock:
            self.stop_requested = True
            processes = list(self.processes)
        for process in processes:
            try:
                process.terminate()
            except Exception:
                pass
        self._emit("run_stopping")

    def close(self):
        self.stop()

    def _manifest(self, run_id, payload, status, scene_records):
        return {
            "schema_version": SCHEMA_VERSION, "id": run_id,
            "stage": payload["stage"], "created_at": self.state.get("started_at"),
            "finished_at": None if status == "running" else time.time(),
            "status": status,
            "parameters": payload["parameters"], "sources": payload["sources"],
            "scenes": scene_records,
        }

    def _run_batch(self, run_id, payload):
        folder = RUN_ROOT / run_id
        manifest_path = folder / "manifest.json"
        log_path = folder / "debug.log"
        parallel = int(payload["parameters"].get("parallel_scenes", 1))
        cpu_slots = max(1, (os.cpu_count() or 2) // max(1, parallel))
        pending = list(payload["sources"])
        active = {}
        scene_records = {}
        failed = False
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        manifest_path.write_text(json.dumps(
            self._manifest(run_id, payload, "running", scene_records),
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")
        self._record_log(
            run_id, payload["stage"],
            f"Execution {payload['stage']} started · "
            f"{len(payload['sources'])} scene(s) · {parallel} worker(s)",
            log_path=log_path,
        )

        while pending or active:
            with self.lock:
                should_stop = self.stop_requested
            if should_stop:
                for item in active.values():
                    item["process"].terminate()
                break

            while pending and len(active) < parallel:
                source = pending.pop(0)
                scene_id = source["id"]
                job_path = folder / f"{scene_id}.job.json"
                output_path = folder / f"{scene_id}.json"
                job_path.write_text(json.dumps({
                    "stage": payload["stage"], "source": source,
                    "parameters": payload["parameters"],
                }, ensure_ascii=False), encoding="utf-8")
                env = os.environ.copy()
                env["OBJECTSENSING_KDTREE_WORKERS"] = str(cpu_slots)
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker",
                     str(job_path), str(output_path)],
                    cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", env=env,
                    bufsize=1,
                    creationflags=creation_flags,
                )
                logs = []
                reader = threading.Thread(
                    target=self._stream_worker_output,
                    args=(
                        process, run_id, payload["stage"], scene_id, logs,
                        log_path,
                    ),
                    daemon=True,
                )
                active[scene_id] = {
                    "process": process, "source": source, "output": output_path,
                    "logs": logs, "reader": reader, "started": time.time(),
                }
                with self.lock:
                    self.processes.append(process)
                    self.state["scenes"][scene_id].update({
                        "status": "running", "message": 'Preparation',
                    })
                self._emit("scene_started", {
                    "run_id": run_id, "stage": payload["stage"],
                    "scene": source,
                })
                self._record_log(
                    run_id, payload["stage"], 'Worker started',
                    scene_id=scene_id, log_path=log_path,
                )
                reader.start()

            for scene_id, item in list(active.items()):
                process = item["process"]
                if process.poll() is None:
                    continue
                item["reader"].join(timeout=2.0)
                logs = item["logs"]
                duration = time.time() - item["started"]
                if process.returncode == 0 and item["output"].exists():
                    result = json.loads(item["output"].read_text(encoding="utf-8"))
                    scene_records[scene_id] = {
                        "file": item["output"].name, "status": "completed",
                        "duration_seconds": result.get("duration_seconds", duration),
                        "artifacts": result.get("artifacts", {}),
                    }
                    with self.lock:
                        self.state["scenes"][scene_id].update({
                            "status": "completed", "message": 'Result available',
                            "duration_seconds": round(duration, 2),
                        })
                    self._emit("scene_completed", {
                        "run_id": run_id, "stage": payload["stage"],
                        "scene_id": scene_id, "result": result,
                    })
                    self._record_log(
                        run_id, payload["stage"],
                        f"Completed in {duration:.1f} s",
                        scene_id=scene_id, log_path=log_path,
                    )
                else:
                    failed = True
                    message = logs[-1] if logs else f"worker stopped with code {process.returncode}"
                    scene_records[scene_id] = {
                        "file": item["output"].name, "status": "failed",
                        "error": message,
                    }
                    with self.lock:
                        self.state["scenes"][scene_id].update({
                            "status": "failed", "message": message,
                        })
                    self._emit("scene_failed", {
                        "run_id": run_id, "stage": payload["stage"],
                        "scene_id": scene_id,
                        "message": message, "logs": logs[-20:],
                    })
                    self._record_log(
                        run_id, payload["stage"], f"Failed: {message}",
                        scene_id=scene_id, log_path=log_path,
                    )
                with self.lock:
                    if process in self.processes:
                        self.processes.remove(process)
                del active[scene_id]
                manifest_path.write_text(json.dumps(
                    self._manifest(run_id, payload, "running", scene_records),
                    ensure_ascii=False, indent=2,
                ), encoding="utf-8")
            time.sleep(0.1)

        stopped = self.stop_requested
        status = "stopped" if stopped else ("failed" if failed else "completed")
        manifest = self._manifest(run_id, payload, status, scene_records)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        with self.lock:
            self.state.update({
                "status": status, "finished_at": time.time(),
                "message": {
                    "completed": 'Comparison completed',
                    "failed": 'Some scenes failed',
                    "stopped": 'Computation stopped',
                }[status],
            })
            self.processes = []
            rerun = self.pending_watch and self.watch_enabled and self.last_payload
            self.pending_watch = False
        self._record_log(
            run_id, payload["stage"],
            {
                "completed": 'Run completed',
                "failed": 'Run completed with errors',
                "stopped": 'Run stopped',
            }[status],
            log_path=log_path,
        )
        self._emit("run_completed", {
            "run_id": run_id, "stage": payload["stage"], "status": status,
        })
        if rerun:
            time.sleep(0.2)
            try:
                self.start(self.last_payload, trigger="code-watch")
            except RuntimeError:
                pass

    def _watch_loop(self):
        changed_at = None
        while True:
            time.sleep(0.5)
            signature = self._code_signature()
            if signature != self._watch_signature:
                self._watch_signature = signature
                changed_at = time.time()
            if not changed_at or time.time() - changed_at < 0.8:
                continue
            changed_at = None
            with self.lock:
                enabled = self.watch_enabled
                payload = self.last_payload
                running = self.worker is not None and self.worker.is_alive()
                if enabled and payload and running:
                    self.pending_watch = True
            if enabled and payload and not running:
                self._emit("code_changed", {"message": 'Code changed, recomputing'})
                try:
                    self.start(payload, trigger="code-watch")
                except RuntimeError:
                    pass


MANAGER = None if __name__ == "__main__" else DebugVisualizerManager()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("PAYLOAD", "OUTPUT"))
    args = parser.parse_args()
    if args.worker:
        _write_worker_result(args.worker[0], args.worker[1])
        return
    parser.error("This module is served by run_console.py")


if __name__ == "__main__":
    main()
