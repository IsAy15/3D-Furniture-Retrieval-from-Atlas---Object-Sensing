'Local ObjectSensing control server and live visualization.'
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from annotation_store import (
    accepted_from_quality,
    merge_legacy,
    migrate_store,
    normalize_accepted,
    normalize_quality,
    write_store,
)
from debug_visualizer import (
    MANAGER as DEBUG_MANAGER,
    json_compatible,
    normalized_parameters as normalized_debug_parameters,
)
from runtime_paths import RUNTIME_PATHS, path_is_within
from web_assets import PAGE_ASSETS, STATIC_ASSETS, STATIC_ASSET_PATHS


ROOT = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"
STATE_ROOT = RUNTIME_PATHS.state_dir
DEFAULT_RUN_ROOT = RUNTIME_PATHS.run_dir
SCENE_ROOT = RUNTIME_PATHS.scene_dir
SCENE_CATALOG_PATH = STATE_ROOT / "scene_catalog.json"
EXECUTION_PARAMETERS_PATH = STATE_ROOT / "execution_parameters.json"
TARGET_ANNOTATIONS_PATH = STATE_ROOT / "target_annotations.json"
TARGET_ANNOTATION_QUALITY_PATH = STATE_ROOT / "target_annotation_quality.json"
ANNOTATION_TARGETS = {
    "chairs": ("left-wood-chair", "center-folding-chair", "right-wood-seat"),
    "office": ("desk", "cabinet", "chair"),
    "ikea-table": ("table", "chair"),
    "single-chair": ("chair",),
}
CONSOLE_UI = None
BROWSE_ROOTS = RUNTIME_PATHS.browse_roots()
DEBUG_EXECUTION_PARAMETER_MAP = {
    "neighbor_radius": "neighbor_radius",
    "curvature_threshold": "curvature_threshold",
    "harris_k": "harris_k",
    "harris_threshold": "harris_threshold",
    "harris_reference_neighbors": "harris_reference_neighbors",
    "convex_hull_ratio": "convex_hull_ratio",
    "corner_plane_radius_factor": "corner_plane_radius_factor",
    "corner_plane_min_area": "corner_plane_min_area",
    "corner_plane_normal_cos": "corner_plane_normal_cos",
    "nms_radius": "nms_radius",
    "adjust_iterations": "adjust_iterations",
    "jitter_reject": "jitter_reject",
    "dedup_radius": "dedup_radius",
    "wall_vertical_normal_cos": "wall_vertical_normal_cos",
    "wall_plane_normal_cos": "wall_plane_normal_cos",
    "wall_plane_angle_degrees": "wall_plane_angle_degrees",
    "wall_plane_distance": "wall_plane_distance",
    "wall_plane_min_points": "wall_plane_min_points",
    "wall_plane_min_width": "wall_plane_min_width",
    "wall_plane_min_height": "wall_plane_min_height",
    "wall_plane_min_area": "wall_plane_min_area",
    "wall_plane_max_count": "wall_plane_max_count",
    "wall_plane_sample_points": "wall_plane_sample_points",
    "wall_small_radius": "wall_small_radius",
    "wall_large_radius": "wall_large_radius",
    "wall_protrusion_radius": "wall_protrusion_radius",
    "wall_protrusion_distance": "wall_protrusion_distance",
    "wall_object_protection": "wall_object_protection",
    "wall_filter_enabled": "wall_filter_enabled",
    "wall_penalty_weight": "wall_penalty_weight",
    "wall_hard_reject": "wall_hard_reject",
    "wall_reject_threshold": "wall_reject_threshold",
    "wall_reject_max_object_score": "wall_reject_max_object_score",
    "wall_budget_enabled": "wall_budget_enabled",
    "wall_budget_affinity_threshold": "wall_budget_affinity_threshold",
    "wall_budget_proximity_threshold": "wall_budget_proximity_threshold",
    "wall_budget_max_fraction": "wall_budget_max_fraction",
    "wall_budget_cell_size": "wall_budget_cell_size",
    "wall_budget_max_per_cell": "wall_budget_max_per_cell",
    "wall_budget_min_features": "wall_budget_min_features",
    "component_budget_enabled": "component_budget_enabled",
    "component_budget_radius": "component_budget_radius",
    "component_budget_max_per_component": "component_budget_max_per_component",
    "component_budget_min_features": "component_budget_min_features",
    "floor_height": "floor_height",
    "max_scan_keypoints": "max_scan_keypoints",
    "spatial_keypoint_balance": "spatial_keypoints",
    "spatial_keypoint_cell": "spatial_keypoint_cell",
    "min_2d_corner_fraction": "min_2d_corner_fraction",
}
DEBUG_FUSION_EXECUTION_PARAMETER_MAP = {
    "voxel_size": "voxel",
    "truncation": "truncation",
    "frame_stride": "frame_stride",
    "min_frames": "min_frames",
    "max_frames": "max_frames",
    "backend": "fusion_backend",
    "volume_layout": "volume_layout",
    "depth_trunc": "depth_trunc",
    "iso_sampling": "iso_sampling",
    "max_surface_points": "max_surface_points",
    "surface_field": "surface_field",
    "surface_min_support": "surface_min_support",
    "surface_extraction": "surface_extraction",
    "surface_band_factor": "surface_band_factor",
    "surface_rescue_distance_factor": "surface_rescue_distance_factor",
    "surface_rescue_min_weight": "surface_rescue_min_weight",
    "gradient_smoothing_sigma": "gradient_smoothing_sigma",
    "depth_edge_threshold": "depth_edge_threshold",
    "depth_edge_background_weight": "depth_edge_background_weight",
    "depth_edge_radius": "depth_edge_radius",
    "sparse_block_resolution": "sparse_block_resolution",
    "sparse_block_count": "sparse_block_count",
    "visibility_depth_stride": "visibility_depth_stride",
}
DEBUG_DESCRIPTOR_EXECUTION_PARAMETER_MAP = {
    "utility_distance_threshold": "descriptor_distance_threshold",
    "descriptor_ratio_threshold": "descriptor_ratio_threshold",
}
DEBUG_ONLY_PARAMETERS = {
    "surface_preview_points": '3D preview density',
    "volume_preview_points": 'volume preview density',
    "normal_preview_points": 'normal preview density',
    "layer_preview_points": 'Debug layer density',
    "parallel_scenes": 'Debug multi-scene parallelism',
}
SCIENCE_STEPS = (
    "fusion", "keypoints", "query", "matching", "verification", "selection",
)
STEPS = (
    ("validate", 'Validate inputs'),
    ("tests", 'Focused tests'),
    ("index", 'Candidate index'),
    ("fusion", "RGB-D fusion"),
    ("keypoints", 'Keypoints + descriptors'),
    ("query", 'Top-k query'),
    ("matching", "Constellation matching"),
    ("verification", 'Geometric verification'),
    ("selection", 'Global selection'),
)
STEP_META = {
    "validate": {"group": "setup", "subtitle": 'Paths, scene and database'},
    "tests": {"group": "setup", "subtitle": 'Optional quick tests'},
    "index": {"group": "setup", "subtitle": 'Compact ShapeNet index'},
    "fusion": {"group": "pipeline", "subtitle": "RGB-D → TSDF volume"},
    "keypoints": {"group": "pipeline", "subtitle": '3D Harris, filters, descriptors'},
    "query": {"group": "pipeline", "subtitle": 'Pool then Top-k ranking'},
    "matching": {"group": "pipeline", "subtitle": 'Correspondences, poses, constellations'},
    "verification": {"group": "pipeline", "subtitle": 'Coverage and final consistency'},
    "selection": {"group": "pipeline", "subtitle": 'Duplicates, overlap and limits'},
}


def resolve_step_ids(step_ids):
    """Expand a scientific target to its ordered prerequisites exactly once."""
    valid = {item[0] for item in STEPS}
    requested = []
    for step in step_ids or []:
        if step in valid and step not in requested:
            requested.append(step)
    selected_science = [step for step in requested if step in SCIENCE_STEPS]
    if not selected_science:
        return requested
    target_index = max(SCIENCE_STEPS.index(step) for step in selected_science)
    dependencies = list(SCIENCE_STEPS[:target_index + 1])
    insertion = min(index for index, step in enumerate(requested) if step in SCIENCE_STEPS)
    resolved = []
    for index, step in enumerate(requested):
        if index == insertion:
            resolved.extend(dependencies)
        if step not in SCIENCE_STEPS:
            resolved.append(step)
    return list(dict.fromkeys(resolved))


def _existing(*paths):
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return str(candidate.resolve())
    return str(Path(paths[0]).resolve())


def default_config():
    db = _existing(RUNTIME_PATHS.database_dir)
    return {
        "scene_zip": _existing(
            SCENE_ROOT / "office.zip", ROOT / "office.zip",
        ),
        "db_dir": db,
        "candidate_index": str((Path(db) / "candidate_index.npz").resolve()),
        "run_root": str(DEFAULT_RUN_ROOT.resolve()),
        "name_prefix": "",
        "name_suffix": "",
        "fusion_backend": "auto",
        "volume_layout": "auto",
        "depth_trunc": 4.5,
        "iso_sampling": 0.01,
        "max_surface_points": 175000,
        "surface_field": "raw",
        "surface_min_support": 0.20,
        "surface_extraction": "hybrid",
        "surface_band_factor": 1.5,
        "surface_rescue_distance_factor": 1.0,
        "surface_rescue_min_weight": 5.0,
        "gradient_smoothing_sigma": 0.8,
        "depth_edge_threshold": 0.05,
        "depth_edge_background_weight": 0.10,
        "depth_edge_radius": 1,
        "sparse_block_resolution": 16,
        "sparse_block_count": 50000,
        "visibility_depth_stride": 2,
        # Sparse profile measured on Office, IKEA Table and Single Chair:
        # 15 mm, a 60 mm band and at least 50 views on short sequences.
        "voxel": 0.015,
        "truncation": 0.060,
        "frame_stride": 6,
        "min_frames": 50,
        "max_frames": 0,
        "jobs": 6,
        "coverage": 0.45,
        "paper_verification": True,
        "surface_distance_weight": 0.05,
        "neighbor_radius": 0.06,
        "curvature_threshold": 0.105,
        "harris_k": 0.04,
        "harris_threshold": 0.008,
        "harris_reference_neighbors": 6.0,
        "convex_hull_ratio": 1.0471975511965976,
        "corner_plane_radius_factor": 3.0,
        "corner_plane_min_area": 0.02,
        "corner_plane_normal_cos": 0.9,
        "nms_radius": 0.05,
        "adjust_iterations": 5,
        "jitter_reject": None,
        "dedup_radius": 0.01,
        "wall_vertical_normal_cos": 0.30,
        "wall_plane_normal_cos": 0.94,
        "wall_plane_angle_degrees": 10.0,
        "wall_plane_distance": 0.06,
        "wall_plane_min_points": 120,
        "wall_plane_min_width": 0.80,
        "wall_plane_min_height": 0.80,
        "wall_plane_min_area": 0.70,
        "wall_plane_max_count": 6,
        "wall_plane_sample_points": 30000,
        "wall_small_radius": 0.08,
        "wall_large_radius": 0.28,
        "wall_protrusion_radius": 0.25,
        "wall_protrusion_distance": 0.08,
        "wall_object_protection": 0.70,
        "wall_filter_enabled": True,
        "wall_penalty_weight": 0.12,
        "wall_hard_reject": True,
        "wall_reject_threshold": 0.20,
        "wall_reject_max_object_score": 0.35,
        "wall_budget_enabled": True,
        "wall_budget_affinity_threshold": 0.05,
        "wall_budget_proximity_threshold": 0.05,
        "wall_budget_max_fraction": 0.05,
        "wall_budget_cell_size": 0.40,
        "wall_budget_max_per_cell": 1,
        "wall_budget_min_features": 40,
        "component_budget_enabled": True,
        "component_budget_radius": 0.20,
        "component_budget_max_per_component": 64,
        "component_budget_min_features": 40,
        "planar_filter_enabled": True,
        "planar_filter_radius": 0.12,
        "planar_filter_max_residual": 0.012,
        "planar_filter_normal_cos": 0.94,
        "planar_filter_angular_coverage": 0.70,
        "planar_filter_min_neighbors": 24,
        "hole_boundary_filter_enabled": True,
        "hole_boundary_probe_radius": 0.03,
        "hole_boundary_max_fraction": 0.20,
        "repeatability_filter_enabled": True,
        "repeatability_radius_factor": 1.6,
        "repeatability_response_ratio": 0.50,
        "quality_response_ratio": 0.35,
        "quality_score_radius": 0.45,
        "geometric_nms_radius": 0.10,
        "quality_filter_min_features": 80,
        "floor_height": 0.0,
        "max_scan_keypoints": 350,
        "descriptor_distance_threshold": 128.0,
        "descriptor_ratio_threshold": 0.0,
        "min_model_inliers": 3,
        "min_scan_inliers": 3,
        "min_scan_cells": 3,
        "min_model_cells": 3,
        "max_model_keypoints": None,
        "spatial_keypoints": True,
        "spatial_keypoint_cell": 0.5,
        "min_2d_corner_fraction": 0.25,
        "descriptor_keypoint_filter_enabled": True,
        "descriptor_min_occupied": 8,
        "descriptor_max_hole_fraction": 0.35,
        "descriptor_keypoint_min_features": 80,
        "quality_nms_radius": 0.08,
        "quality_min_score_ratio": 0.25,
        "candidate_top_k": 100,
        "candidate_pool_k": 500,
        "candidate_local_extra_k": 64,
        "candidate_top_per_group": 50,
        "candidate_descriptor_distance_threshold": 128.0,
        "group_pose_seed_enabled": True,
        "group_pose_seed_count": 6,
        "candidate_diverse": True,
        "candidate_diversity": 0.5,
        "exhaustive": False,
        "verification_top_constellations": 50,
        "max_registrations_per_model": 1,
        "registration_top_constellations": 5,
        "registration_min_center_distance": 0.45,
        "multi_registration_synsets": ["04379243"],
        "query_start_frames": 20,
        "query_interval_frames": 60,
        "query_mode": "single",
        "cache_registrations_per_model": 3,
        "cache_top_constellations": 50,
        "resume_batch_size": 10,
        "final_reverse_gate": 0.40,
        "final_global_overlap": 0.50,
        "final_min_symmetric_score": 0.55,
        "final_cross_category_center_distance": 0.55,
        "final_cross_category_overlap": 0.25,
        "final_max_per_category": ["chair=2", "table=2", "couch=1"],
        "reuse_identical_run": True,
    }


def execution_default_config():
    config = default_config()
    stored = _read_json(EXECUTION_PARAMETERS_PATH, {}) or {}
    execution_keys = {
        *DEBUG_EXECUTION_PARAMETER_MAP.values(),
        *DEBUG_FUSION_EXECUTION_PARAMETER_MAP.values(),
        *DEBUG_DESCRIPTOR_EXECUTION_PARAMETER_MAP.values(),
    }
    for key, value in (stored.get("parameters") or {}).items():
        if key in execution_keys:
            config[key] = value
    return config


def apply_debug_parameters_to_execution(raw_parameters, stage="keypoints"):
    stage = str(stage or "keypoints")
    parameters = normalized_debug_parameters(raw_parameters, stage=stage)
    parameter_map = {
        "fusion": DEBUG_FUSION_EXECUTION_PARAMETER_MAP,
        "descriptors": DEBUG_DESCRIPTOR_EXECUTION_PARAMETER_MAP,
    }.get(stage, DEBUG_EXECUTION_PARAMETER_MAP)
    applied = {
        execution_key: parameters[debug_key]
        for debug_key, execution_key in parameter_map.items()
    }
    ignored = [
        {"parameter": key, "reason": reason}
        for key, reason in DEBUG_ONLY_PARAMETERS.items()
        if key in parameters
    ]
    payload = {
        "schema_version": 1,
        "updated_at": time.time(),
        "source": "debug_visualizer",
        "parameters": applied,
    }
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = EXECUTION_PARAMETERS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    os.replace(str(temporary), str(EXECUTION_PARAMETERS_PATH))
    return {
        "applied": applied,
        "ignored": ignored,
        "config": execution_default_config(),
    }


def _scene_label(path):
    stem = Path(path).stem
    known = {
        "chairs": "Chairs",
        "office": "Office",
        "ikea-table": "IKEA Table",
        "single-chair": "Single Chair",
    }
    return known.get(
        stem.lower(),
        " ".join(word.capitalize() for word in re.split(r"[-_]+", stem)),
    )


def _scene_catalog_record(path, label=None, built_in=False):
    path = Path(path).expanduser().resolve()
    return {
        "id": hashlib.sha256(
            os.path.normcase(str(path)).encode("utf-8"),
        ).hexdigest()[:12],
        "label": str(label or "").strip() or _scene_label(path),
        "path": str(path),
        "available": path.is_file(),
        "built_in": bool(built_in),
    }


def _stored_scene_catalog():
    payload = _read_json(SCENE_CATALOG_PATH, {}) or {}
    scenes = payload.get("scenes", [])
    return scenes if isinstance(scenes, list) else []


def scene_catalog():
    records = {}
    if SCENE_ROOT.exists():
        for path in sorted(SCENE_ROOT.glob("*.zip")):
            record = _scene_catalog_record(path, built_in=True)
            records[os.path.normcase(record["path"])] = record
    default_scene = Path(_existing(
        ROOT / "scenes" / "office.zip", ROOT / "office.zip",
    ))
    default_record = _scene_catalog_record(default_scene, built_in=True)
    records.setdefault(os.path.normcase(default_record["path"]), default_record)
    for item in _stored_scene_catalog():
        if not isinstance(item, dict) or not item.get("path"):
            continue
        record = _scene_catalog_record(item["path"], item.get("label"))
        records[os.path.normcase(record["path"])] = record
    return sorted(
        records.values(),
        key=lambda item: (
            not item["available"], item["label"].casefold(), item["path"].casefold(),
        ),
    )


def add_scene_catalog_entry(label, path):
    label = str(label or "").strip()
    if not label:
        raise ValueError('Scene name cannot be empty')
    if len(label) > 80:
        raise ValueError('Scene name is limited to 80 characters')
    if any(ord(character) < 32 for character in label):
        raise ValueError('Scene name contains a control character')
    path = Path(str(path or "")).expanduser().resolve()
    if path.suffix.lower() != ".zip":
        raise ValueError('The RGB-D scene must be a .zip file')
    if not path.is_file():
        raise ValueError(f"Scene file not found: {path}")
    stored = [
        item for item in _stored_scene_catalog()
        if isinstance(item, dict) and item.get("path")
        and os.path.normcase(str(Path(item["path"]).expanduser().resolve()))
        != os.path.normcase(str(path))
    ]
    stored.append({"label": label, "path": str(path)})
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = SCENE_CATALOG_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"scenes": stored}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(SCENE_CATALOG_PATH))
    selected = _scene_catalog_record(path, label)
    return {"scene": selected, "scene_options": scene_catalog()}


PATH_PURPOSES = {
    "scene_zip": {"kind": "file", "extensions": {".zip"}},
    "db_dir": {"kind": "folder", "extensions": set()},
    "candidate_index": {"kind": "file", "extensions": {".npz"}},
    "run_root": {"kind": "folder", "extensions": set()},
}


def _validate_path_request(kind, purpose):
    if kind not in {"file", "folder"}:
        raise ValueError('Invalid picker type')
    if purpose not in PATH_PURPOSES:
        raise ValueError('Invalid path field')
    if PATH_PURPOSES[purpose]["kind"] != kind:
        raise ValueError('Path type incompatible with this field')


def _purpose_start_path(purpose):
    return {
        "scene_zip": SCENE_ROOT,
        "db_dir": RUNTIME_PATHS.database_dir,
        "candidate_index": RUNTIME_PATHS.database_dir,
        "run_root": RUNTIME_PATHS.run_dir,
    }[purpose]


def _allowed_browse_path(path):
    return path_is_within(path, BROWSE_ROOTS)


def _existing_browse_directory(path, fallback):
    candidate = Path(str(path or fallback)).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir() or not _allowed_browse_path(candidate):
        candidate = Path(fallback)
    return candidate.resolve()


def browse_local_path(kind, purpose=None, current_path=None):
    """List server-visible files without opening a host GUI dialog."""
    _validate_path_request(kind, purpose)
    fallback = _purpose_start_path(purpose)
    directory = _existing_browse_directory(current_path, fallback)
    extensions = PATH_PURPOSES[purpose]["extensions"]
    entries = []
    try:
        children = sorted(
            directory.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.casefold()),
        )
    except OSError as exc:
        raise ValueError(f"Cannot read directory: {directory}") from exc
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append({
                "name": child.name, "path": str(child.resolve()),
                "type": "folder", "selectable": kind == "folder",
            })
        elif kind == "file" and (
            not extensions or child.suffix.lower() in extensions
        ):
            entries.append({
                "name": child.name, "path": str(child.resolve()),
                "type": "file", "selectable": True,
            })
    parent = directory.parent.resolve()
    return {
        "kind": kind,
        "purpose": purpose,
        "current": str(directory),
        "parent": (
            str(parent)
            if parent != directory and _allowed_browse_path(parent) else None
        ),
        "select_current": kind == "folder",
        "entries": entries,
        "roots": [
            {"name": root.name or str(root), "path": str(root)}
            for root in BROWSE_ROOTS if root.exists()
        ],
        "headless": RUNTIME_PATHS.headless,
    }


def pick_local_path(kind, purpose=None, selected_path=None):
    """Validate a path selected by the browser-based path navigator."""
    _validate_path_request(kind, purpose)
    if not selected_path:
        return {"cancelled": True, "path": None}
    path = Path(selected_path).expanduser().resolve()
    if not _allowed_browse_path(path):
        raise ValueError('Path is outside allowed directories')
    if kind == "folder" and not path.is_dir():
        raise ValueError(f'Directory not found: {path}')
    if kind == "file" and not path.is_file():
        raise ValueError(f'File not found : {path}')
    if purpose == "scene_zip" and path.suffix.lower() != ".zip":
        raise ValueError('The RGB-D scene must be a .zip file')
    if purpose == "candidate_index" and path.suffix.lower() != ".npz":
        raise ValueError('The candidate index must be an .npz file')
    return {"cancelled": False, "path": str(path)}


def _token(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value).strip()).strip("-_")
    return value.lower() or "run"


def normalized_config(raw=None):
    config = default_config()
    config.update(raw or {})
    for name in ("scene_zip", "db_dir", "candidate_index", "run_root"):
        config[name] = str(Path(config[name]).expanduser().resolve())
    if config.get("name_prefix") and config.get("name_suffix"):
        raise ValueError('Use either a name prefix or suffix, not both')
    positive = (
        "voxel", "frame_stride", "jobs", "candidate_top_k",
        "query_start_frames", "query_interval_frames", "resume_batch_size",
        "depth_trunc", "iso_sampling", "sparse_block_resolution",
        "sparse_block_count", "visibility_depth_stride",
        "max_surface_points",
    )
    for name in positive:
        if config.get(name) is None or float(config[name]) <= 0:
            raise ValueError(f"{name} must be strictly positive")
    if float(config["descriptor_distance_threshold"]) <= 0:
        raise ValueError('descriptor_distance_threshold must be strictly positive')
    ratio = float(config["descriptor_ratio_threshold"])
    if ratio < 0 or ratio > 1:
        raise ValueError('descriptor_ratio_threshold must be between 0 and 1')
    if config.get("max_frames") is None:
        config["max_frames"] = 0
    if config.get("min_frames") is None:
        config["min_frames"] = 0
    if int(config["min_frames"]) < 0:
        raise ValueError('min_frames must be nonnegative')
    if config["volume_layout"] not in {"auto", "dense", "sparse"}:
        raise ValueError("volume_layout invalide")
    if config["fusion_backend"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("fusion_backend invalide")
    if config.get("query_mode") not in {"single", "progressive"}:
        raise ValueError("query_mode invalide")
    if config["surface_field"] not in {"raw", "smoothed"}:
        raise ValueError("surface_field invalide")
    if config["surface_extraction"] not in {
        "zero_crossing", "hybrid", "near_zero",
    }:
        raise ValueError("surface_extraction invalide")
    for name in (
        "surface_min_support", "gradient_smoothing_sigma",
        "depth_edge_threshold", "depth_edge_background_weight",
        "depth_edge_radius",
        "surface_band_factor", "surface_rescue_distance_factor",
        "surface_rescue_min_weight",
    ):
        if float(config[name]) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in (
        "surface_min_support", "depth_edge_background_weight",
    ):
        if float(config[name]) > 1:
            raise ValueError(f"{name} must be at most 1")
    return config


def generated_run_name(config):
    scene = _token(Path(config["scene_zip"]).stem)
    frames = (
        f"fs{config['frame_stride']}-minf{config['min_frames']}"
        f"-mf{config['max_frames']}"
    )
    readable = (
        f"{scene}_local_prog_k{config['candidate_top_k']}"
        f"-pool{config['candidate_pool_k']}_v{int(round(config['voxel'] * 1000))}mm_"
        f"vl{_token(config['volume_layout'])}_"
        f"fb{_token(config['fusion_backend'])}_{frames}_"
        f"cov{int(round(config['coverage'] * 100))}_kp{config['max_scan_keypoints']}"
    )
    identity = {
        key: value for key, value in config.items()
        if key not in {"run_root", "name_prefix", "name_suffix", "reuse_identical_run"}
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    readable = f"{readable}_{digest}"
    if config.get("name_prefix"):
        readable = f"{_token(config['name_prefix'])}_{readable}"
    if config.get("name_suffix"):
        readable = f"{readable}_{_token(config['name_suffix'])}"
    return readable


def _same_config(path, config):
    try:
        return json.loads(path.read_text(encoding="utf-8")) == config
    except Exception:
        return False


def execution_paths(config):
    root = Path(config["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    base = generated_run_name(config)
    folder = root / base
    if folder.exists() and not (
        config.get("reuse_identical_run") and _same_config(folder / "run_config.json", config)
    ):
        index = 0
        while (root / f"{base}_{index}").exists():
            index += 1
        folder = root / f"{base}_{index}"
    folder.mkdir(parents=True, exist_ok=True)
    return _paths_for_folder(folder)


def _paths_for_folder(folder):
    folder = Path(folder).resolve()
    name = folder.name
    stem = folder / name
    return {
        "id": name,
        "folder": folder,
        "config": folder / "run_config.json",
        "aln": stem.with_suffix(".aln"),
        "json": stem.with_suffix(".json"),
        "progress": folder / f"{name}_progress.json",
        "scan": folder / f"{name}_scan.ply",
        "viz": folder / f"{name}_viz.ply",
        "html": folder / f"{name}_timeline.html",
        "resume": folder / "resume",
        "stage_cache": folder / "stage_cache",
        "events": folder / "live_events.jsonl",
        "log": folder / "run.log",
    }


def _load_existing_run(run_id):
    record = find_run(run_id)
    config_path = record.get("config")
    folder = record["folder"]
    if folder.name != run_id or not config_path or not config_path.exists():
        raise ValueError(
            'This legacy run has no standalone configuration and cannot be continued.'
        )
    config = normalized_config(json.loads(config_path.read_text(encoding="utf-8")))
    config["run_root"] = str(folder.parent.resolve())
    return config, _paths_for_folder(folder), record


def _optional(args, flag, value):
    if value is not None and value != "":
        args.extend([flag, str(value)])


def retrieval_command(config, paths, stop_after="selection"):
    args = [
        sys.executable, str(ROOT / "cli.py"), "run-progressive",
        "--zip", config["scene_zip"], "--db", config["db_dir"],
        "--out", str(paths["aln"]), "--coverage", str(config["coverage"]),
        "--jobs", str(config["jobs"]), "--voxel", str(config["voxel"]),
        "--fusion-backend", config["fusion_backend"],
        "--volume-layout", config["volume_layout"],
        "--depth-trunc", str(config["depth_trunc"]),
        "--iso-sampling", str(config["iso_sampling"]),
        "--max-surface-points", str(config["max_surface_points"]),
        "--surface-field", config["surface_field"],
        "--surface-min-support", str(config["surface_min_support"]),
        "--surface-extraction", config["surface_extraction"],
        "--surface-band-factor", str(config["surface_band_factor"]),
        "--surface-rescue-distance-factor",
        str(config["surface_rescue_distance_factor"]),
        "--surface-rescue-min-weight",
        str(config["surface_rescue_min_weight"]),
        "--gradient-smoothing-sigma",
        str(config["gradient_smoothing_sigma"]),
        "--depth-edge-threshold", str(config["depth_edge_threshold"]),
        "--depth-edge-background-weight",
        str(config["depth_edge_background_weight"]),
        "--depth-edge-radius", str(config["depth_edge_radius"]),
        "--sparse-block-resolution",
        str(config["sparse_block_resolution"]),
        "--sparse-block-count", str(config["sparse_block_count"]),
        "--visibility-depth-stride",
        str(config["visibility_depth_stride"]),
        "--frame-stride", str(config["frame_stride"]),
        "--min-frames", str(config["min_frames"]),
        "--max-frames", str(config["max_frames"]),
        "--candidate-top-k", str(config["candidate_top_k"]),
        "--candidate-pool-k", str(config["candidate_pool_k"]),
        "--candidate-local-extra-k", str(config.get("candidate_local_extra_k", 64)),
        "--candidate-top-per-group",
        str(config.get("candidate_top_per_group", 50)),
        "--candidate-descriptor-distance-threshold",
        str(config.get("candidate_descriptor_distance_threshold", 128.0)),
        "--group-pose-seed-count",
        str(config.get("group_pose_seed_count", 6)),
        "--candidate-diversity", str(config["candidate_diversity"]),
        "--query-start-frames", str(config["query_start_frames"]),
        "--query-interval-frames", str(config["query_interval_frames"]),
        "--query-mode", str(config.get("query_mode", "single")),
        "--cache-registrations-per-model", str(config["cache_registrations_per_model"]),
        "--cache-top-constellations", str(config["cache_top_constellations"]),
        "--resume-dir", str(paths["resume"]),
        "--resume-batch-size", str(config["resume_batch_size"]),
        "--stage-cache-dir", str(paths["stage_cache"]),
        "--final-global-overlap", str(config["final_global_overlap"]),
        "--final-min-symmetric-score", str(config["final_min_symmetric_score"]),
        "--final-cross-category-center-distance", str(config["final_cross_category_center_distance"]),
        "--final-cross-category-overlap", str(config["final_cross_category_overlap"]),
        "--live-events", str(paths["events"]),
        "--stop-after", str(stop_after),
    ]
    args.append(
        "--group-pose-seed" if config.get("group_pose_seed_enabled", True)
        else "--no-group-pose-seed"
    )
    _optional(args, "--candidate-index", config.get("candidate_index"))
    _optional(args, "--truncation", config.get("truncation"))
    _optional(args, "--surface-distance-weight", config.get("surface_distance_weight"))
    for key in (
        "min_model_inliers", "min_scan_inliers", "min_scan_cells",
        "min_model_cells",
    ):
        _optional(args, f"--{key.replace('_', '-')}", config.get(key))
    for key in (
        "neighbor_radius", "curvature_threshold", "harris_k",
        "harris_threshold", "harris_reference_neighbors",
        "convex_hull_ratio", "corner_plane_radius_factor",
        "corner_plane_min_area", "corner_plane_normal_cos",
        "nms_radius", "adjust_iterations", "dedup_radius",
        "wall_vertical_normal_cos", "wall_plane_normal_cos",
        "wall_plane_angle_degrees", "wall_plane_distance",
        "wall_plane_min_points", "wall_plane_min_width",
        "wall_plane_min_height", "wall_plane_min_area",
        "wall_plane_max_count", "wall_plane_sample_points",
        "wall_small_radius", "wall_large_radius",
        "wall_protrusion_radius", "wall_protrusion_distance",
        "wall_object_protection", "wall_penalty_weight",
        "wall_reject_threshold", "wall_reject_max_object_score",
        "wall_budget_affinity_threshold", "wall_budget_proximity_threshold",
        "wall_budget_max_fraction",
        "wall_budget_cell_size", "wall_budget_max_per_cell",
        "wall_budget_min_features",
        "component_budget_radius", "component_budget_max_per_component",
        "component_budget_min_features",
        "planar_filter_radius", "planar_filter_max_residual",
        "planar_filter_normal_cos", "planar_filter_angular_coverage",
        "planar_filter_min_neighbors", "hole_boundary_probe_radius",
        "hole_boundary_max_fraction", "repeatability_radius_factor",
        "repeatability_response_ratio", "quality_response_ratio",
        "quality_score_radius",
        "geometric_nms_radius", "quality_filter_min_features",
        "descriptor_min_occupied", "descriptor_max_hole_fraction",
        "descriptor_keypoint_min_features", "quality_nms_radius",
        "quality_min_score_ratio",
        "floor_height", "min_2d_corner_fraction",
    ):
        _optional(args, f"--{key.replace('_', '-')}", config.get(key))
    args.extend([
        "--jitter-reject",
        "auto" if config.get("jitter_reject") is None
        else str(config["jitter_reject"]),
    ])
    _optional(args, "--max-scan-keypoints", config.get("max_scan_keypoints"))
    _optional(args, "--max-model-keypoints", config.get("max_model_keypoints"))
    _optional(args, "--descriptor-distance-threshold", config.get("descriptor_distance_threshold"))
    _optional(args, "--descriptor-ratio-threshold", config.get("descriptor_ratio_threshold"))
    _optional(args, "--spatial-keypoint-cell", config.get("spatial_keypoint_cell"))
    _optional(args, "--verification-top-constellations", config.get("verification_top_constellations"))
    _optional(args, "--max-registrations-per-model", config.get("max_registrations_per_model"))
    _optional(args, "--registration-top-constellations", config.get("registration_top_constellations"))
    _optional(args, "--registration-min-center-distance", config.get("registration_min_center_distance"))
    _optional(args, "--final-reverse-gate", config.get("final_reverse_gate"))
    if config.get("paper_verification"):
        args.append("--paper-verification")
    if config.get("spatial_keypoints"):
        args.append("--spatial-keypoints")
    else:
        args.append("--no-spatial-keypoints")
    args.append(
        "--wall-filter" if config.get("wall_filter_enabled")
        else "--no-wall-filter"
    )
    args.append(
        "--wall-hard-reject" if config.get("wall_hard_reject")
        else "--no-wall-hard-reject"
    )
    args.append(
        "--wall-budget" if config.get("wall_budget_enabled")
        else "--no-wall-budget"
    )
    args.append(
        "--component-budget" if config.get("component_budget_enabled")
        else "--no-component-budget"
    )
    for enabled, option in (
        (config.get("planar_filter_enabled"), "planar-filter"),
        (config.get("hole_boundary_filter_enabled"), "hole-boundary-filter"),
        (config.get("repeatability_filter_enabled"), "repeatability-filter"),
        (config.get("descriptor_keypoint_filter_enabled"), "descriptor-keypoint-filter"),
    ):
        args.append(f"--{'' if enabled else 'no-'}{option}")
    if config.get("candidate_diverse"):
        args.append("--candidate-diverse")
    if config.get("exhaustive"):
        args.append("--exhaustive")
    if config.get("multi_registration_synsets"):
        args.append("--multi-registration-synsets")
        args.extend(map(str, config["multi_registration_synsets"]))
    if config.get("final_max_per_category"):
        args.append("--final-max-per-category")
        args.extend(map(str, config["final_max_per_category"]))
    return args


def visualization_command(paths):
    return [
        sys.executable, str(ROOT / "cli.py"), "visualize",
        "--json", str(paths["json"]), "--scan", str(paths["scan"]),
        "--out", str(paths["viz"]), "--progress", str(paths["progress"]),
        "--html", str(paths["html"]),
    ]


class RunManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.events = []
        self.next_event_id = 1
        self.process = None
        self.worker = None
        self.stop_requested = False
        self.cached_stages = set()
        self.state = {
            "status": "idle", "run_id": None, "current_step": None,
            "started_at": None, "finished_at": None, "error": None,
            "steps": [], "artifacts": {}, "command": [], "logs": [],
        }
        self.live = {
            "fusion_frames": [], "surface": None, "keypoints": None,
            "preselection": None, "candidates": [], "selection": None,
        }

    def _publish(self, kind, data=None):
        with self.lock:
            event = {
                "id": self.next_event_id, "type": kind,
                "timestamp": time.time(), "data": data or {},
            }
            self.next_event_id += 1
            self.events.append(event)
            if len(self.events) > 5000:
                self.events = self.events[-4000:]
            self._reduce_live(event)
        return event

    def _reduce_live(self, event):
        kind, data = event["type"], event["data"]
        if kind == "fusion_started":
            self.live["fusion_frames"] = []
            self.live["surface"] = None
            self.live["keypoints"] = None
            self.live["candidates"] = []
        elif kind == "fusion_frame":
            self.live["fusion_frames"].append(data)
            self.live["fusion_frames"] = self.live["fusion_frames"][-300:]
        elif kind == "scene_surface":
            self.live["surface"] = data
        elif kind == "keypoints":
            self.live["keypoints"] = data
        elif kind == "preselection":
            self.live["preselection"] = data
        elif kind == "matching_result":
            self.live["candidates"].append(data)
            self.live["candidates"] = self.live["candidates"][-500:]
        elif kind in {"checkpoint_selection", "final_selection"}:
            self.live["selection"] = data

    def snapshot(self, include_live=False):
        with self.lock:
            snapshot = deepcopy(self.state)
            snapshot["running"] = self.process is not None or (
                self.worker is not None and self.worker.is_alive()
            )
            snapshot["last_event_id"] = self.next_event_id - 1
            if include_live:
                snapshot["live"] = deepcopy(self.live)
            return snapshot

    def events_after(self, event_id):
        with self.lock:
            return [deepcopy(event) for event in self.events if event["id"] > event_id]

    def start(self, raw_config, step_ids, existing_run_id=None):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise RuntimeError('A run is already active')
            if existing_run_id:
                config, paths, record = _load_existing_run(existing_run_id)
                run_label = record["public"].get("label", paths["id"])
            else:
                config = normalized_config(raw_config)
                paths = execution_paths(config)
                run_label = paths["id"]
            paths["config"].write_text(
                json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            step_ids = resolve_step_ids(step_ids)
            if not step_ids:
                raise ValueError('The playlist contains no stages')
            self.stop_requested = False
            self.cached_stages = set()
            self.live = {
                "fusion_frames": [], "surface": None, "keypoints": None,
                "preselection": None, "candidates": [], "selection": None,
            }
            self.state = {
                "status": "running", "run_id": paths["id"],
                "run_label": run_label,
                "run_folder": str(paths["folder"]), "current_step": None,
                "started_at": time.time(), "finished_at": None, "error": None,
                "steps": [
                    {"id": sid, "label": dict(STEPS)[sid], "status": "pending", "message": ""}
                    for sid in step_ids
                ],
                "artifacts": self._artifacts(paths), "command": [], "logs": [],
                "config": config,
            }
            self.worker = threading.Thread(
                target=self._run_playlist, args=(config, paths, step_ids), daemon=True,
            )
            self.worker.start()
            self._publish("run_started", {
                "run_id": paths["id"], "run_label": run_label,
                "steps": step_ids, "resumed": bool(existing_run_id),
            })
            return self.snapshot()

    def stop(self):
        with self.lock:
            self.stop_requested = True
            process = self.process
        if process and process.poll() is None:
            self._terminate_process_tree(process)
        self._publish("stop_requested", {})

    @staticmethod
    def _terminate_process_tree(process, grace_seconds=3):
        try:
            if IS_WINDOWS:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, check=False,
                )
                if process.poll() is None:
                    process.terminate()
                return
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            if process.poll() is None:
                process.terminate()

    def _set_step(self, step_id, status, message=""):
        with self.lock:
            for step in self.state["steps"]:
                if step["id"] == step_id:
                    step.update(status=status, message=message)
                    break
            self.state["current_step"] = step_id if status == "running" else self.state["current_step"]
        self._publish("step_status", {"id": step_id, "status": status, "message": message})

    def _run_playlist(self, config, paths, step_ids):
        try:
            index = 0
            while index < len(step_ids):
                step_id = step_ids[index]
                if self.stop_requested:
                    raise InterruptedError('Run stopped')
                if step_id in SCIENCE_STEPS:
                    block = []
                    while index < len(step_ids) and step_ids[index] in SCIENCE_STEPS:
                        block.append(step_ids[index])
                        index += 1
                    target = block[-1]
                    for dependency in block:
                        self._set_step(
                            dependency, "running",
                            'Prerequisite for ' + dict(STEPS)[target]
                            if dependency != target else 'Running',
                        )
                    message = self._run_step(target, config, paths)
                    for dependency in block:
                        if dependency in self.cached_stages:
                            final_message = 'Intermediate cache reused'
                        elif dependency == target:
                            final_message = message
                        else:
                            final_message = 'Prerequisite executed'
                        self._set_step(
                            dependency, "completed",
                            final_message,
                        )
                    continue
                self._set_step(step_id, "running")
                message = self._run_step(step_id, config, paths)
                self._set_step(step_id, "completed", message or 'Completed')
                index += 1
            if "selection" in step_ids:
                self._publish("visualization_finalizing", {
                    "message": 'Generating final artifacts automatically',
                })
                for required in (paths["json"], paths["scan"]):
                    if not required.exists():
                        raise FileNotFoundError(
                            f"Artefact final requis absent : {required.name}"
                        )
                self._run_process(
                    visualization_command(paths), paths, "selection",
                )
                self._publish("viewer_ready", {
                    "url": self._artifacts(paths)["viewer_url"],
                    "automatic": True,
                })
            with self.lock:
                self.state.update(status="completed", current_step=None, finished_at=time.time())
            self._publish("run_completed", {
                "viewer_url": self._artifacts(paths).get("viewer_url"),
                "steps": list(step_ids),
            })
        except InterruptedError as exc:
            with self.lock:
                self.state.update(status="stopped", current_step=None, finished_at=time.time(), error=str(exc))
            self._publish("run_stopped", {"message": str(exc)})
        except Exception as exc:
            current = self.state.get("current_step")
            if current:
                self._set_step(current, "failed", str(exc))
            with self.lock:
                self.state.update(status="failed", current_step=None, finished_at=time.time(), error=str(exc))
            self._publish("run_failed", {"message": str(exc)})
        finally:
            with self.lock:
                self.process = None

    def _run_step(self, step_id, config, paths):
        if step_id == "validate":
            missing = [name for name in ("scene_zip", "db_dir") if not Path(config[name]).exists()]
            if missing:
                raise FileNotFoundError('Missing input: ' + ", ".join(missing))
            if not (Path(config["db_dir"]) / "index.json").exists():
                raise FileNotFoundError("The database directory contains no index.json")
            return 'Scene and database accessible'
        if step_id == "tests":
            return self._run_process([
                sys.executable, "-m", "pytest", "-m", "smoke", "-q",
            ], paths, step_id)
        if step_id == "index":
            index = Path(config["candidate_index"])
            if index.exists():
                return f"Existing index reused ({index.name})"
            return self._run_process([
                sys.executable, str(ROOT / "cli.py"), "build-candidate-index",
                "--db", config["db_dir"], "--out", str(index),
            ], paths, step_id)
        if step_id in SCIENCE_STEPS:
            return self._run_process(
                retrieval_command(config, paths, stop_after=step_id),
                paths, step_id, live=True,
            )
        raise ValueError(f"Unknown stage: {step_id}")

    def _run_process(self, command, paths, step_id, live=False):
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if IS_WINDOWS else 0
        )
        if live:
            paths["events"].touch(exist_ok=True)
            event_offset = paths["events"].stat().st_size
        else:
            event_offset = 0
        with self.lock:
            self.state["command"] = command
        self._publish("command_started", {"step": step_id, "command": command})
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=env, creationflags=creationflags,
            start_new_session=not IS_WINDOWS,
        )
        with self.lock:
            self.process = process
        tail_stop = threading.Event()
        tail_thread = None
        if live:
            tail_thread = threading.Thread(
                target=self._tail_events, args=(paths["events"], event_offset, tail_stop),
                daemon=True,
            )
            tail_thread.start()
        with paths["log"].open("a", encoding="utf-8") as log:
            log.write("\n> " + subprocess.list2cmdline(command) + "\n")
            for raw_line in iter(process.stdout.readline, ""):
                line = raw_line.rstrip("\r\n")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                with self.lock:
                    self.state["logs"] = (self.state["logs"] + [line])[-500:]
                self._publish("process_log", {"step": step_id, "message": line})
        return_code = process.wait()
        tail_stop.set()
        if tail_thread:
            tail_thread.join(timeout=2)
            self._drain_events(paths["events"], event_offset)
        with self.lock:
            self.process = None
            self.state["artifacts"] = self._artifacts(paths)
        if self.stop_requested:
            raise InterruptedError('Run stopped; resumption will use saved batches')
        if return_code != 0:
            raise RuntimeError(f"Command completed with code {return_code}")
        return 'Command completed'

    def _tail_events(self, path, offset, stop_event):
        with path.open("r", encoding="utf-8") as stream:
            stream.seek(offset)
            while not stop_event.is_set():
                line = stream.readline()
                if not line:
                    time.sleep(0.15)
                    continue
                self._accept_live_line(line)
            for line in stream:
                self._accept_live_line(line)

    def _drain_events(self, path, offset):
        # Le thread principal ne rejoue pas the événements déjà consommés.
        return None

    def _accept_live_line(self, line):
        try:
            event = json.loads(line)
            kind = event.get("type", "pipeline_event")
            data = event.get("data", {})
            stage = {
                "scene_surface": "fusion",
                "keypoints": "keypoints",
                "preselection": "query",
                "matching_completed": "matching",
                "verification_completed": "verification",
                "final_selection": "selection",
            }.get(kind)
            if kind == "stage_completed":
                stage = data.get("stage")
            if kind == "stage_cache_hit" and data.get("stage") in SCIENCE_STEPS:
                stage = data["stage"]
                with self.lock:
                    self.cached_stages.add(stage)
            if stage in SCIENCE_STEPS:
                with self.lock:
                    for step in self.state.get("steps", []):
                        if step["id"] == stage and step["status"] == "running":
                            message = (
                                f"Cache reused · {data.get('frame_count', 0)} trames"
                                if kind == "stage_cache_hit" else 'Completed'
                            )
                            step.update(status="completed", message=message)
                            break
            self._publish(kind, data)
        except Exception:
            return

    @staticmethod
    def _artifacts(paths):
        result = {}
        for name in ("aln", "json", "progress", "scan", "viz", "html", "log", "events"):
            path = paths[name]
            result[name] = {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
        cache_files = (
            list(paths["stage_cache"].glob("scene_*.pkl"))
            if paths["stage_cache"].exists() else []
        )
        result["stage_cache"] = {
            "path": str(paths["stage_cache"]),
            "exists": bool(cache_files),
            "count": len(cache_files),
            "size": sum(path.stat().st_size for path in cache_files),
        }
        result["viewer_url"] = f"/artifact/{paths['id']}/{paths['html'].name}"
        return result


MANAGER = RunManager()


class MultiSceneRunCoordinator:
    'Chain independent runs without multiplying CPU and memory budgets.'

    def __init__(self, manager):
        self.manager = manager
        self.lock = threading.RLock()
        self.worker = None
        self.stop_requested = False
        self.batch = None

    def running(self):
        return bool(self.worker and self.worker.is_alive())

    def start(self, raw_config, scene_zips, step_ids):
        with self.lock:
            if self.running() or self.manager.snapshot().get("running"):
                raise RuntimeError('A run is already active')
            scenes = []
            for raw_path in scene_zips or []:
                path = Path(raw_path).expanduser().resolve()
                if path not in scenes:
                    scenes.append(path)
            if len(scenes) < 2:
                raise ValueError('Select at least two scenes for a multi-scene run')
            missing = [str(path) for path in scenes if not path.is_file()]
            if missing:
                raise FileNotFoundError('Scene not found: ' + missing[0])
            if any(path.suffix.lower() != ".zip" for path in scenes):
                raise ValueError('All scenes must be .zip files')
            base = normalized_config(raw_config)
            steps = resolve_step_ids(step_ids)
            self.stop_requested = False
            self.batch = {
                "status": "running", "strategy": "sequential",
                "total": len(scenes), "completed": 0, "failed": 0,
                "current": None,
                "items": [
                    {"scene_zip": str(path), "label": path.stem, "status": "pending"}
                    for path in scenes
                ],
            }
            self.worker = threading.Thread(
                target=self._run, args=(base, scenes, steps), daemon=True,
            )
            self.worker.start()
            return self.snapshot()

    def snapshot(self):
        state = self.manager.snapshot(include_live=True)
        with self.lock:
            state["batch"] = deepcopy(self.batch)
            state["running"] = state.get("running", False) or self.running()
        return state

    def stop(self):
        with self.lock:
            self.stop_requested = True
        self.manager.stop()

    def clear(self):
        with self.lock:
            if not self.running():
                self.batch = None

    def _run(self, base, scenes, steps):
        try:
            for index, scene in enumerate(scenes):
                with self.lock:
                    if self.stop_requested:
                        break
                    item = self.batch["items"][index]
                    item["status"] = "running"
                    self.batch["current"] = index
                self.manager.start(dict(base, scene_zip=str(scene)), steps)
                with self.lock:
                    self.manager.state["batch"] = deepcopy(self.batch)
                self.manager.worker.join()
                child_status = self.manager.snapshot().get("status")
                with self.lock:
                    item["status"] = child_status
                    item["run_id"] = self.manager.state.get("run_id")
                    if child_status == "completed":
                        self.batch["completed"] += 1
                    else:
                        self.batch["failed"] += 1
                    self.manager.state["batch"] = deepcopy(self.batch)
            with self.lock:
                stopped = self.stop_requested
                self.batch["status"] = "stopped" if stopped else (
                    "completed" if not self.batch["failed"] else "completed_with_errors"
                )
                self.batch["current"] = None
                self.manager.state["batch"] = deepcopy(self.batch)
            self.manager._publish("batch_completed", deepcopy(self.batch))
        finally:
            with self.lock:
                self.worker = None


BATCH_COORDINATOR = MultiSceneRunCoordinator(MANAGER)


def _read_json(path, default=None):
    if not path or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _completed_stage(progress):
    if not progress or not progress.exists():
        return None
    try:
        with progress.open("r", encoding="utf-8") as stream:
            header = stream.read(131072)
    except OSError:
        return None
    match = re.search(r'"completed_stage"\s*:\s*"([a-z]+)"', header)
    if match and match.group(1) in SCIENCE_STEPS:
        return match.group(1)
    return None


def _run_meta_path(folder, run_id):
    folder = Path(folder)
    if folder.name == run_id:
        return folder / "run_meta.json"
    return folder / f".{run_id}_run_meta.json"


def _run_metadata(folder, run_id):
    return _read_json(_run_meta_path(folder, run_id), {}) or {}


def _run_label(folder, run_id):
    metadata = _run_metadata(folder, run_id)
    label = str(metadata.get("label") or "").strip()
    return label or run_id


def _write_run_metadata(folder, run_id, updates):
    path = _run_meta_path(folder, run_id)
    payload = _run_metadata(folder, run_id)
    payload.update(updates)
    payload["updated_at"] = time.time()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    os.replace(str(temporary), str(path))
    return payload


def run_details(run_id):
    record = find_run(run_id)
    config = _read_json(record.get("config"), None)
    return {"run": record["public"], "config": config}


def rename_run(run_id, label):
    record = find_run(run_id)
    label = str(label or "").strip()
    if not label:
        raise ValueError('Name cannot be empty')
    if len(label) > 80:
        raise ValueError('Name is limited to 80 characters')
    if any(ord(character) < 32 for character in label):
        raise ValueError('Name contains a control character')
    _write_run_metadata(record["folder"], run_id, {"label": label})
    with MANAGER.lock:
        if MANAGER.state.get("run_id") == run_id:
            MANAGER.state["run_label"] = label
    return run_details(run_id)


def set_run_favorite(run_id, favorite):
    record = find_run(run_id)
    _write_run_metadata(
        record["folder"], run_id, {"favorite": bool(favorite)},
    )
    return run_details(run_id)


def recent_runs(limit=100):
    return [record["public"] for record in _run_records()[:limit]]


def _run_roots():
    roots = {DEFAULT_RUN_ROOT.resolve()}
    config = MANAGER.state.get("config") or {}
    if config.get("run_root"):
        roots.add(Path(config["run_root"]).resolve())
    return roots


def _record_from_folder(folder):
    progress = next(iter(sorted(folder.glob("*_progress.json"))), None)
    html = next(iter(sorted(folder.glob("*_timeline.html"))), None)
    viz = next(iter(sorted(folder.glob("*_viz.ply"))), None)
    events = folder / "live_events.jsonl"
    config = folder / "run_config.json"
    meta = _run_meta_path(folder, folder.name)
    if not any(path and path.exists() for path in (progress, html, viz, events, config, meta)):
        return None
    modified = max(
        path.stat().st_mtime for path in (progress, html, viz, events, config, meta)
        if path and path.exists()
    )
    return _run_record(
        folder.name, folder, progress, html, viz, events, modified,
        config=config if config.exists() else None,
    )


def _run_record(run_id, folder, progress, html, viz, events, modified,
                config=None):
    has_progress = bool(progress and progress.exists())
    has_events = bool(events and events.exists())
    metadata = _run_metadata(folder, run_id)
    scene_cache_dir = Path(folder) / "stage_cache"
    scene_cache_files = (
        sorted(scene_cache_dir.glob("scene_*.pkl"))
        if scene_cache_dir.exists() else []
    )
    cached_frame_counts = []
    for path in scene_cache_files:
        match = re.fullmatch(r"scene_(\d+)\.pkl", path.name)
        if match:
            cached_frame_counts.append(int(match.group(1)))
    public = {
        "id": run_id,
        "label": str(metadata.get("label") or "").strip() or run_id,
        "favorite": bool(metadata.get("favorite", False)),
        "path": str(folder),
        "modified": modified,
        "viewer_url": f"/replay-artifact/{run_id}/viewer" if html else None,
        "viz_url": f"/replay-artifact/{run_id}/viz" if viz else None,
        "replayable": has_progress or has_events,
        "detailed": has_progress,
        "complete": bool(html or viz),
        "completed_stage": _completed_stage(progress) or (
            "selection" if html or viz else None
        ),
        "resumable": bool(
            config and config.exists() and Path(folder).name == run_id
        ),
        # A completed stage is only historical evidence.  Reusing fusion also
        # requires the serialized volume; a PLY surface cannot restore its
        # free/unknown visibility state.
        "stage_cache": {
            "fusion_available": bool(scene_cache_files),
            "fusion_count": len(scene_cache_files),
            "fusion_frame_counts": cached_frame_counts,
        },
    }
    completed = public["completed_stage"]
    public["next_stage"] = (
        SCIENCE_STEPS[SCIENCE_STEPS.index(completed) + 1]
        if completed in SCIENCE_STEPS[:-1] else None
    )
    return {
        "public": public, "folder": folder, "progress": progress,
        "html": html, "viz": viz, "events": events if has_events else None,
        "config": config,
    }


def _run_records():
    records = {}
    for root in _run_roots():
        if not root.exists():
            continue
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            record = _record_from_folder(folder)
            if record:
                records[record["public"]["id"]] = record
        flat_ids = set()
        for pattern, suffix in (
            ("*_progress.json", "_progress.json"),
            ("*_timeline.html", "_timeline.html"),
            ("*_viz.ply", "_viz.ply"),
        ):
            flat_ids.update(path.name[:-len(suffix)] for path in root.glob(pattern))
        for run_id in flat_ids:
            if run_id in records:
                continue
            progress = root / f"{run_id}_progress.json"
            html = root / f"{run_id}_timeline.html"
            viz = root / f"{run_id}_viz.ply"
            events = root / f"{run_id}_live_events.jsonl"
            record = _run_record(
                run_id, root, progress if progress.exists() else None, html if html.exists() else None,
                viz if viz.exists() else None, events,
                max(path.stat().st_mtime for path in (progress, html, viz) if path and path.exists()),
                config=None,
            )
            records[run_id] = record
    return sorted(
        records.values(),
        key=lambda item: (
            item["public"].get("favorite", False),
            item["public"]["modified"],
        ),
        reverse=True,
    )


def find_run(run_id):
    if not run_id or Path(run_id).name != run_id:
        raise FileNotFoundError("Identifiant de run invalide")
    for record in _run_records():
        if record["public"]["id"] == run_id:
            return record
    raise FileNotFoundError(f"Run not found : {run_id}")


def _compact_candidate(candidate, preview_limit=240):
    verification = candidate.get("verification") or {}
    preview = candidate.get("model_preview") or []
    if len(preview) > preview_limit:
        step = max(1, len(preview) // preview_limit)
        preview = preview[::step][:preview_limit]
    return {
        "model": candidate.get("model"),
        "category": candidate.get("category"),
        "synset": candidate.get("synset"),
        "model_keypoints": candidate.get("model_keypoints", 0),
        "model_preview": preview,
        "correspondences": candidate.get("correspondences", 0),
        "constellations": candidate.get("constellations", 0),
        "elapsed_ms": candidate.get("elapsed_ms", 0),
        "status": candidate.get("status", "pending"),
        "coverage_threshold": candidate.get("coverage_threshold"),
        "registration": candidate.get("registration"),
        "verification": {
            key: verification.get(key) for key in (
                "constellations_available", "constellations_tested", "reverse_gate",
                "min_structure", "min_thickness", "rejection_counts", "status", "best_score",
            ) if key in verification
        },
    }


def _estimated_trace_ground(frames):
    """Estimate the dominant low horizontal layer in legacy visual traces."""
    heights = sorted(
        float(point[1])
        for frame in (frames or [])
        for point in (frame.get("points") or [])
        if isinstance(point, list) and len(point) == 3
    )
    if not heights:
        return 0.0
    low_limit = heights[min(len(heights) - 1, int(len(heights) * 0.35))]
    bin_size = 0.02
    bins = {}
    for height in heights:
        if height > low_limit:
            break
        index = round(height / bin_size)
        bins[index] = bins.get(index, 0) + 1
    dominant = max(bins, key=bins.get)
    return round(dominant * bin_size, 4)


def build_replay(run_id):
    record = find_run(run_id)
    progress_path = record.get("progress")
    if not progress_path or not progress_path.exists():
        live_path = record.get("events")
        if live_path and live_path.exists():
            events = []
            ground = None
            chapter_for = {
                "fusion_frame": "fusion", "keypoints": "keypoints",
                "preselection": "query", "matching_result": "matching",
                "checkpoint_selection": "verification", "final_selection": "selection",
            }
            for line in live_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind, data = event.get("type"), event.get("data") or {}
                if kind == "scene_surface":
                    ground = data.get("ground")
                    continue
                chapter = chapter_for.get(kind)
                if not chapter:
                    continue
                if kind == "preselection":
                    events.append({
                        "chapter": chapter, "type": "query_reset", "checkpoint": None,
                        "data": {
                            "strategy": data.get("strategy"),
                            "database_count": data.get("database_count", 0),
                            "pool_count": len(data.get("pool") or []),
                            "top_k_count": len(data.get("top_k") or []),
                        },
                    })
                    for rank, model in enumerate(data.get("top_k") or [], 1):
                        events.append({
                            "chapter": chapter, "type": "query_candidate", "checkpoint": None,
                            "data": {"model": model, "rank": rank, "top_k_count": len(data.get("top_k") or [])},
                        })
                    continue
                if kind == "matching_result" and data.get("candidate"):
                    data = dict(data, candidate=_compact_candidate(data["candidate"]))
                events.append({"chapter": chapter, "type": kind, "data": data, "checkpoint": None})
            chapters = []
            for chapter in ("fusion", "keypoints", "query", "matching", "verification", "selection"):
                indices = [index for index, event in enumerate(events) if event["chapter"] == chapter]
                if indices:
                    chapters.append({"id": chapter, "start": indices[0], "end": indices[-1], "count": len(indices)})
            return {
                "run": record["public"], "detailed": bool(events),
                "events": events, "chapters": chapters, "database_summary": {},
                "ground": (
                    float(ground) if ground is not None
                    else _estimated_trace_ground([
                        event["data"] for event in events
                        if event["type"] == "fusion_frame"
                    ])
                ),
            }
        return {
            "run": record["public"], "detailed": False, "events": [],
            "chapters": [], "message": 'This run has no detailed trace.',
        }
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    fusion = progress.get("fusion_trace") or {}
    traces = progress.get("execution_trace") or []
    if not traces or not fusion.get("frames"):
        return {
            "run": record["public"], "detailed": False, "events": [],
            "chapters": [], "database_summary": progress.get("database_summary") or {},
            "message": 'Scientific trace missing; displaying the final result.',
        }
    events = []

    def add(chapter, kind, data=None, checkpoint=None):
        events.append({
            "chapter": chapter, "type": kind, "data": data or {},
            "checkpoint": checkpoint,
        })

    for frame in fusion.get("frames") or []:
        add("fusion", "fusion_frame", frame)

    for checkpoint_index, checkpoint in enumerate(traces):
        frame_count = checkpoint.get("frame_count")
        full = checkpoint.get("full") or {}
        scan = full.get("scan") or {}
        add("keypoints", "keypoints", {
            "detected": scan.get("detected_keypoints", 0),
            "retained": scan.get("retained_keypoints", 0),
            "detected_points": scan.get("detected_keypoint_points") or [],
            "retained_points": (
                scan.get("retained_keypoint_points")
                or scan.get("keypoints")
                or []
            ),
            "points": scan.get("keypoints") or [],
            "frame_count": frame_count,
        }, checkpoint_index)
        preselection = full.get("preselection") or {}
        add("query", "query_reset", {
            "strategy": preselection.get("strategy", "unknown"),
            "database_count": preselection.get("database_count", 0),
            "pool_count": len(preselection.get("pool") or []),
            "top_k_count": len(preselection.get("top_k") or []),
            "frame_count": frame_count,
        }, checkpoint_index)
        for rank, model in enumerate(preselection.get("top_k") or [], 1):
            add("query", "query_candidate", {
                "model": model, "rank": rank, "frame_count": frame_count,
                "top_k_count": len(preselection.get("top_k") or []),
                "database_count": preselection.get("database_count", 0),
            }, checkpoint_index)
        matching = full.get("matching") or []
        for rank, candidate in enumerate(matching, 1):
            add("matching", "matching_result", {
                "completed": rank, "total": len(matching),
                "candidate": _compact_candidate(candidate),
                "frame_count": frame_count,
            }, checkpoint_index)
        add("verification", "checkpoint_selection", {
            "candidates": len(full.get("checkpoint_selection") or []),
            "selected": full.get("checkpoint_selected") or [],
            "diagnostics": full.get("checkpoint_selection") or [],
            "frame_count": frame_count,
        }, checkpoint_index)

    final_selection = progress.get("selected") or []
    add("selection", "final_selection", {
        "selected": final_selection,
        "diagnostics": progress.get("final_selection") or [],
        "frame_count": fusion.get("total_frames", 0),
        "viz_url": record["public"].get("viz_url"),
    })
    chapter_order = ("fusion", "keypoints", "query", "matching", "verification", "selection")
    chapters = []
    for chapter in chapter_order:
        indices = [index for index, event in enumerate(events) if event["chapter"] == chapter]
        if indices:
            chapters.append({
                "id": chapter, "start": indices[0], "end": indices[-1],
                "count": len(indices),
            })
    return {
        "run": record["public"], "detailed": bool(traces and fusion),
        "events": events, "chapters": chapters,
        "database_summary": progress.get("database_summary") or {},
        "ground": float(
            fusion.get("ground", _estimated_trace_ground(fusion.get("frames")))
        ),
    }


def runtime_health(require_ready=False):
    checks = {
        "scene_dir": RUNTIME_PATHS.scene_dir.is_dir(),
        "database_index": (RUNTIME_PATHS.database_dir / "index.json").is_file(),
        "run_dir": RUNTIME_PATHS.run_dir.is_dir(),
        "state_dir": RUNTIME_PATHS.state_dir.is_dir(),
    }
    ready = all(checks.values())
    return {
        "status": "ready" if ready else "degraded",
        "ok": ready if require_ready else True,
        "checks": checks,
        "runtime": RUNTIME_PATHS.public_summary(),
    }


def _shapenet_db(raw_path=None):
    import db_store

    path = Path(raw_path or RUNTIME_PATHS.database_dir).expanduser().resolve()
    if not path_is_within(path, RUNTIME_PATHS.browse_roots()):
        raise ValueError('ShapeNet database outside allowed directories')
    if not db_store.is_db_dir(str(path)):
        raise FileNotFoundError(f"db_dir ShapeNet not found : {path}")
    return path


def shapenet_model_page(
        raw_path=None, query="", synset="", offset=0, limit=60,
        include_names=None, exclude_names=None):
    import db_store

    db_dir = _shapenet_db(raw_path)
    index = db_store.load_index(str(db_dir))
    query = str(query or "").strip().lower()
    synset = str(synset or "").strip()
    restrict_names = include_names is not None
    include_names = {
        str(value) for value in (include_names or []) if str(value)
    }
    exclude_names = {
        str(value) for value in (exclude_names or []) if str(value)
    }
    matches = []
    for model_index, (name, category) in enumerate(zip(
            index.get("names", []), index.get("synsets", []))):
        name = str(name)
        category = str(category)
        if synset and category != synset:
            continue
        if query and query not in name.lower():
            continue
        if restrict_names and name not in include_names:
            continue
        if name in exclude_names:
            continue
        matches.append({
            "index": model_index, "name": name, "synset": category,
        })
    offset = max(0, int(offset))
    limit = min(120, max(1, int(limit)))
    return {
        "database_dir": str(db_dir),
        "total_database": int(index.get("n", len(index.get("names", [])))),
        "total_filtered": len(matches),
        "offset": offset, "limit": limit,
        "models": matches[offset:offset + limit],
        "categories": sorted(set(str(value) for value in index.get("synsets", []))),
    }


def shapenet_model_detail(raw_path=None, model_index=0, max_points=5000):
    import db_store
    import numpy as np

    db_dir = _shapenet_db(raw_path)
    files = db_store.model_files(str(db_dir))
    model_index = int(model_index)
    if model_index < 0 or model_index >= len(files):
        raise ValueError('ShapeNet model index out of range')
    model = db_store.load_model(files[model_index])
    points = np.asarray(model.cloud.points, float)
    max_points = min(12000, max(100, int(max_points)))
    if len(points) > max_points:
        selection = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[selection]
    return {
        "index": model_index,
        "name": str(model.name),
        "synset": str(model.synset),
        "mesh_path": str(getattr(model, "mesh_path", "") or ""),
        "point_count": int(model.cloud.size),
        "keypoint_count": len(getattr(model, "features", []) or []),
        "points": np.round(points, 5).tolist(),
    }


def load_target_annotations():
    return ensure_annotation_targets(
        accepted_from_quality(load_target_annotation_quality()),
        factory=list,
    )


def ensure_annotation_targets(payload, factory=dict):
    normalized = deepcopy(payload or {})
    for scene, targets in ANNOTATION_TARGETS.items():
        groups = normalized.setdefault(scene, {})
        for target in targets:
            groups.setdefault(target, factory())
    return normalized


def save_target_annotations(payload):
    accepted = normalize_accepted(payload)
    current = load_target_annotation_quality()
    quality = {
        scene: {
            group: {
                name: current.get(scene, {}).get(group, {}).get(name, 1)
                for name in names
            }
            for group, names in groups.items()
        }
        for scene, groups in accepted.items()
    }
    for scene, groups in current.items():
        for group, models in groups.items():
            for name, value in models.items():
                if int(value) == 0:
                    quality.setdefault(scene, {}).setdefault(group, {})[name] = 0
    write_store(TARGET_ANNOTATIONS_PATH, quality)
    return {
        "annotations": accepted_from_quality(quality),
        "quality": normalize_quality(quality),
        "path": str(TARGET_ANNOTATIONS_PATH.resolve()),
    }


def load_target_annotation_quality():
    quality, _ = migrate_store(
        TARGET_ANNOTATIONS_PATH, TARGET_ANNOTATION_QUALITY_PATH,
    )
    return quality


def save_target_annotation_quality(payload):
    normalized = write_store(
        TARGET_ANNOTATIONS_PATH,
        ensure_annotation_targets(payload, factory=dict),
    )
    return {
        "annotations": ensure_annotation_targets(
            accepted_from_quality(normalized), factory=list,
        ),
        "quality": normalized,
        "path": str(TARGET_ANNOTATIONS_PATH.resolve()),
    }


def save_target_annotation_store(annotations, quality):
    normalized = merge_legacy(annotations or {}, quality or {})
    normalized = ensure_annotation_targets(normalized, factory=dict)
    normalized = write_store(TARGET_ANNOTATIONS_PATH, normalized)
    return {
        "annotations": ensure_annotation_targets(
            accepted_from_quality(normalized), factory=list,
        ),
        "quality": normalized,
        "path": str(TARGET_ANNOTATIONS_PATH.resolve()),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "ObjectSensingConsole/1.0"

    def log_message(self, format, *args):
        if CONSOLE_UI is not None:
            CONSOLE_UI.request(self.command, self.path, args)

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(
            json_compatible(payload), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if parsed.path == "/api/health":
            self._json(runtime_health(require_ready=False))
            return
        if parsed.path == "/api/ready":
            payload = runtime_health(require_ready=True)
            self._json(
                payload,
                HTTPStatus.OK if payload["ok"] else HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        if parsed.path == "/api/debug/defaults":
            self._json(DEBUG_MANAGER.defaults())
            return
        if parsed.path == "/api/debug/state":
            self._json(DEBUG_MANAGER.snapshot())
            return
        if parsed.path == "/api/debug/events":
            after = int(parse_qs(parsed.query).get("after", [0])[0])
            self._json({
                "events": DEBUG_MANAGER.events_after(after),
                "state": DEBUG_MANAGER.snapshot(),
            })
            return
        if parsed.path == "/api/debug/result":
            run_id = parse_qs(parsed.query).get("id", [""])[0]
            try:
                self._json(DEBUG_MANAGER.load_result(run_id))
            except FileNotFoundError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/debug/query-candidate":
            query = parse_qs(parsed.query)
            try:
                self._json(DEBUG_MANAGER.query_candidate_detail(
                    query.get("run_id", [""])[0],
                    query.get("scene_id", [""])[0],
                    int(query.get("candidate_index", [""])[0]),
                    query.get("group_id", [None])[0],
                ))
            except FileNotFoundError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except (ValueError, TypeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/shapenet/models":
            query = parse_qs(parsed.query)
            try:
                annotations = load_target_annotations()
                quality = load_target_annotation_quality()
                annotation_scene = query.get("annotation_scene", [""])[0]
                annotation_target = query.get("annotation_target", [""])[0]
                membership = query.get("membership", [""])[0]
                annotated = list(
                    annotations.get(annotation_scene, {}).get(
                        annotation_target, [],
                    )
                )
                include_names = None
                exclude_names = None
                if membership == "added":
                    include_names = annotated
                elif membership == "unrated":
                    rated = quality.get(annotation_scene, {}).get(
                        annotation_target, {},
                    )
                    include_names = [
                        name for name in annotated if name not in rated
                    ]
                elif membership == "outside":
                    exclude_names = annotated
                elif membership == "pending":
                    rated = quality.get(annotation_scene, {}).get(
                        annotation_target, {},
                    )
                    rejected = [
                        name for name, value in rated.items()
                        if int(value) == 0
                    ]
                    exclude_names = list(dict.fromkeys([
                        *annotated, *rejected,
                    ]))
                self._json(shapenet_model_page(
                    query.get("db", [None])[0],
                    query.get("q", [""])[0],
                    query.get("synset", [""])[0],
                    int(query.get("offset", [0])[0]),
                    int(query.get("limit", [60])[0]),
                    include_names,
                    exclude_names,
                ))
            except (ValueError, FileNotFoundError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/shapenet/model":
            query = parse_qs(parsed.query)
            try:
                self._json(shapenet_model_detail(
                    query.get("db", [None])[0],
                    int(query.get("index", [0])[0]),
                    int(query.get("max_points", [5000])[0]),
                ))
            except (ValueError, FileNotFoundError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/shapenet/annotations":
            self.send_error(404, "Annotation interface is not included in this source export")
            return
        if parsed.path == "/api/defaults":
            steps = []
            for step_id, label in STEPS:
                meta = dict(STEP_META.get(step_id, {}))
                if step_id in SCIENCE_STEPS:
                    position = SCIENCE_STEPS.index(step_id)
                    meta["depends_on"] = list(SCIENCE_STEPS[:position])
                steps.append({"id": step_id, "label": label, **meta})
            self._json({
                "config": execution_default_config(),
                "steps": steps,
                "recent_runs": recent_runs(),
                "scene_options": scene_catalog(),
                "runtime": RUNTIME_PATHS.public_summary(),
            })
            return
        if parsed.path == "/api/state":
            self._json(BATCH_COORDINATOR.snapshot())
            return
        if parsed.path == "/api/events":
            after = int(parse_qs(parsed.query).get("after", [0])[0])
            state = BATCH_COORDINATOR.snapshot()
            state.pop("live", None)
            self._json({"events": MANAGER.events_after(after), "state": state})
            return
        if parsed.path == "/api/replay":
            run_id = parse_qs(parsed.query).get("id", [""])[0]
            try:
                self._json(build_replay(run_id))
            except FileNotFoundError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/run":
            run_id = parse_qs(parsed.query).get("id", [""])[0]
            try:
                self._json(run_details(run_id))
            except FileNotFoundError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if parsed.path.startswith("/replay-artifact/"):
            self._serve_replay_artifact(parsed.path)
            return
        if parsed.path.startswith("/artifact/"):
            self._serve_artifact(parsed.path)
            return
        if parsed.path in PAGE_ASSETS:
            self._serve_file(PAGE_ASSETS[parsed.path])
            return
        if parsed.path in STATIC_ASSETS:
            self._serve_file(STATIC_ASSET_PATHS[parsed.path])
            return
        if parsed.path.startswith("/vendor/"):
            candidate = (ROOT / parsed.path.lstrip("/")).resolve()
            if path_is_within(candidate, [ROOT / "vendor"]):
                self._serve_file(candidate)
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        try:
            payload = self._body()
            if self.path == "/api/debug/run":
                self._json(
                    DEBUG_MANAGER.start(payload, trigger="manual"),
                    HTTPStatus.ACCEPTED,
                )
                return
            if self.path == "/api/shapenet/annotations":
                self.send_error(404, "Annotation interface is not included in this source export")
                return
            if self.path == "/api/debug/baseline":
                self._json(DEBUG_MANAGER.set_baseline(payload.get("run_id")))
                return
            if self.path == "/api/debug/chain":
                self._json(DEBUG_MANAGER.chain_sources(
                    payload.get("run_id"),
                    target_stage=payload.get("target_stage", "keypoints"),
                ))
                return
            if self.path == "/api/debug/watch":
                self._json(DEBUG_MANAGER.set_watch(payload.get("enabled")))
                return
            if self.path == "/api/debug/apply-execution-parameters":
                self._json(apply_debug_parameters_to_execution(
                    payload.get("parameters"),
                    stage=payload.get("stage", "keypoints"),
                ))
                return
            if self.path == "/api/debug/run/rename":
                self._json(DEBUG_MANAGER.rename_run(
                    payload.get("run_id"), payload.get("label"),
                ))
                return
            if self.path == "/api/debug/run/favorite":
                self._json(DEBUG_MANAGER.set_favorite(
                    payload.get("run_id"), payload.get("favorite"),
                ))
                return
            if self.path == "/api/debug/stop":
                DEBUG_MANAGER.stop()
                self._json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/run":
                if BATCH_COORDINATOR.running():
                    raise RuntimeError('A multi-scene batch is already active')
                BATCH_COORDINATOR.clear()
                state = MANAGER.start(
                    payload.get("config"),
                    payload.get("steps") or [i for i, _ in STEPS],
                    existing_run_id=payload.get("run_id"),
                )
                self._json(state, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/run/batch":
                state = BATCH_COORDINATOR.start(
                    payload.get("config"), payload.get("scene_zips"),
                    payload.get("steps") or [i for i, _ in STEPS],
                )
                self._json(state, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/step":
                if BATCH_COORDINATOR.running():
                    raise RuntimeError('A multi-scene batch is already active')
                BATCH_COORDINATOR.clear()
                state = MANAGER.start(
                    payload.get("config"), [payload["step"]],
                    existing_run_id=payload.get("run_id"),
                )
                self._json(state, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/run/rename":
                self._json(rename_run(payload.get("run_id"), payload.get("label")))
                return
            if self.path == "/api/run/favorite":
                self._json(set_run_favorite(
                    payload.get("run_id"), payload.get("favorite"),
                ))
                return
            if self.path == "/api/scenes":
                self._json(add_scene_catalog_entry(
                    payload.get("label"), payload.get("path"),
                ), HTTPStatus.CREATED)
                return
            if self.path == "/api/browse-path":
                self._json(browse_local_path(
                    payload.get("kind"), payload.get("purpose"),
                    payload.get("current_path"),
                ))
                return
            if self.path == "/api/pick-path":
                self._json(pick_local_path(
                    payload.get("kind"), payload.get("purpose"),
                    payload.get("selected_path"),
                ))
                return
            if self.path == "/api/stop":
                BATCH_COORDINATOR.stop()
                self._json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _serve_artifact(self, request_path):
        parts = [unquote(part) for part in request_path.split("/") if part]
        if len(parts) != 3:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, run_id, filename = parts
        candidates = []
        for root in {DEFAULT_RUN_ROOT.resolve(), Path((MANAGER.state.get("config") or {}).get("run_root", DEFAULT_RUN_ROOT)).resolve()}:
            candidates.append((root / run_id / filename).resolve())
        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and candidate.parent.name == run_id:
                self._serve_file(candidate)
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_replay_artifact(self, request_path):
        parts = [unquote(part) for part in request_path.split("/") if part]
        if len(parts) != 3 or parts[0] != "replay-artifact":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, run_id, kind = parts
        try:
            record = find_run(run_id)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = {"viewer": record.get("html"), "viz": record.get("viz")}.get(kind)
        if not path or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._serve_file(path)

    def _serve_file(self, path):
        path = Path(path)
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        cache_control = (
            "no-store"
            if path.suffix.lower() in {".html", ".js", ".css"}
            else "no-cache"
        )
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    'Prevent multiple Run Consoles from sharing one address. On Windows, address reuse can otherwise mix API/frontend versions from different processes.\n    '

    allow_reuse_address = False
    allow_reuse_port = False
    daemon_threads = True

    def server_bind(self):
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()


class TerminalConsole:
    """Small dependency-free supervisor for the local HTTP server."""

    HEADER_ROWS = 3
    FOOTER_ROWS = 2
    POLLING_PATHS = {
        "/api/events", "/api/debug/events", "/api/state",
        "/api/debug/state", "/api/health", "/api/ready",
    }

    def __init__(self, host, port, debug=False, interactive=True):
        self.host = host
        self.port = int(port)
        self.debug = bool(debug)
        self.interactive = bool(interactive and sys.stdin.isatty() and IS_WINDOWS)
        self.started_at = time.monotonic()
        self.requests = 0
        self.last_request = 'none'
        self._last_poll_log = 0.0
        self._server = None
        self._restart = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._terminal_size = None
        self._tui = self.interactive and self._enable_virtual_terminal()

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/"

    def attach(self, server):
        self._server = server

    @staticmethod
    def _enable_virtual_terminal():
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except (AttributeError, OSError):
            return False

    def _dimensions(self):
        size = shutil.get_terminal_size((120, 30))
        return max(50, size.columns), max(10, size.lines)

    @staticmethod
    def _plain(text):
        return str(text).replace("\r", " ").replace("\n", " ")

    def _line(self, text, width):
        text = self._plain(text)
        return text if len(text) <= width else text[:max(1, width - 1)] + "…"

    def _uptime(self):
        elapsed = int(time.monotonic() - self.started_at)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _configure_layout_locked(self, force=False):
        columns, rows = self._dimensions()
        dimensions = (columns, rows)
        if not force and dimensions == self._terminal_size:
            return columns, rows
        self._terminal_size = dimensions
        log_start = self.HEADER_ROWS + 1
        log_end = rows - self.FOOTER_ROWS
        sys.stdout.write("\033[r\033[2J\033[H")
        sys.stdout.write(f"\033[{log_start};{log_end}r")
        sys.stdout.write(f"\033[{log_start};1H")
        self._draw_fixed_locked(columns, rows)
        sys.stdout.flush()
        return columns, rows

    def _write_fixed_line(self, row, text, columns, style=""):
        reset = "\033[0m" if style else ""
        sys.stdout.write(
            f"\033[{row};1H\033[2K{style}{self._line(text, columns)}{reset}"
        )

    def _draw_fixed_locked(self, columns=None, rows=None):
        if columns is None or rows is None:
            columns, rows = self._dimensions()
            if (columns, rows) != self._terminal_size:
                self._configure_layout_locked(force=True)
                return
        mode = "DEBUG" if self.debug else "NORMAL"
        status = (
            f'PID {os.getpid()}  | active {self._uptime()}  |  {self.requests} request(s)  |  logs {mode}'
        )
        separator = "─" * columns
        sys.stdout.write("\0337")
        self._write_fixed_line(
            1, f"ObjectSensing Run Console  |  {self.url}", columns,
            "\033[1;96m",
        )
        self._write_fixed_line(2, status, columns, "\033[1m")
        self._write_fixed_line(
            3, f"Last request: {self.last_request}", columns, "\033[90m",
        )
        self._write_fixed_line(rows - 1, separator, columns, "\033[90m")
        self._write_fixed_line(
            rows,
            '[R] Restart   [D] Debug   [S] Snapshot   [C] Clear logs   [Q] Quit',
            columns,
            "\033[1;97m",
        )
        sys.stdout.write("\0338")
        sys.stdout.flush()

    def _refresh_fixed(self):
        if not self._tui:
            return
        with self._lock:
            columns, rows = self._dimensions()
            if (columns, rows) != self._terminal_size:
                self._configure_layout_locked(force=True)
            else:
                self._draw_fixed_locked(columns, rows)

    def _status_loop(self):
        while not self._stop.wait(1.0):
            self._refresh_fixed()

    def log(self, message, level="INFO"):
        with self._lock:
            stamp = time.strftime("%H:%M:%S")
            line = f"[{stamp}] {level:<6} {self._plain(message)}"
            if self._tui:
                columns, rows = self._configure_layout_locked()
                sys.stdout.write(f"{self._line(line, columns)}\r\n")
                self._draw_fixed_locked(columns, rows)
                sys.stdout.flush()
            else:
                print(line, flush=True)

    def request(self, method, path, args):
        status = str(args[1]) if len(args) > 1 else "?"
        parsed_path = urlparse(path).path
        self.requests += 1
        self.last_request = f"{method} {parsed_path} -> {status}"
        is_error = status.startswith(("4", "5"))
        is_mutation = method not in {"GET", "HEAD"}
        should_log = is_error or is_mutation
        if self.debug:
            should_log = True
            if parsed_path in self.POLLING_PATHS:
                now = time.monotonic()
                should_log = now - self._last_poll_log >= 5.0
                if should_log:
                    self._last_poll_log = now
        if should_log:
            self.log(self.last_request, "ERROR" if is_error else "HTTP")

    def status(self):
        mode = "DEBUG" if self.debug else "NORMAL"
        self.log(
            f'PID {os.getpid()} | active {self._uptime()} | {self.requests} request(s) | logs {mode} | last: {self.last_request}',
            "ETAT",
        )

    def guide(self):
        if self._tui:
            self._refresh_fixed()
        elif self.interactive:
            self.log('Keys: [R] restart  [D] debug  [S] status  [C] clear  [Q] quit', "AIDE")
        else:
            self.log('Non-interactive mode: press Ctrl+C to stop', "AIDE")

    def start_keyboard_listener(self):
        if not self.interactive:
            return
        if self._tui:
            threading.Thread(
                target=self._status_loop,
                name="terminal-status",
                daemon=True,
            ).start()
        threading.Thread(
            target=self._keyboard_loop,
            name="terminal-controls",
            daemon=True,
        ).start()

    def _keyboard_loop(self):
        import msvcrt
        while not self._stop.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.08)
                continue
            key = msvcrt.getwch().lower()
            if key == "r":
                self.log('Server restart requested', "ACTION")
                self._restart.set()
                if self._server is not None:
                    self._server.shutdown()
            elif key == "d":
                self.debug = not self.debug
                self.log(
                    f"Detailed logs {'enabled' if self.debug else 'disabled'}",
                    "ACTION",
                )
                self.guide()
            elif key == "s":
                self.status()
                self.guide()
            elif key == "c":
                self.banner()
            elif key == "q":
                self.log('Stop requested', "ACTION")
                self._stop.set()
                if self._server is not None:
                    self._server.shutdown()

    def banner(self):
        if self._tui:
            with self._lock:
                self._configure_layout_locked(force=True)
            self.log('HTTP server ready', 'READY')
        else:
            self.log(f"ObjectSensing Run Console | {self.url}", 'READY')
            self.status()
            self.guide()

    def consume_restart(self):
        requested = self._restart.is_set()
        self._restart.clear()
        return requested

    def stop_requested(self):
        return self._stop.is_set()

    def close(self):
        self._stop.set()
        if self._tui:
            with self._lock:
                _, rows = self._dimensions()
                sys.stdout.write(f"\033[r\033[{rows};1H\033[2K")
                sys.stdout.flush()
            self._tui = False


def main():
    global CONSOLE_UI
    parser = argparse.ArgumentParser(description='Local ObjectSensing console')
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "OBJECTSENSING_HOST",
            "0.0.0.0" if RUNTIME_PATHS.headless else "127.0.0.1",
        ),
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("OBJECTSENSING_PORT", "8770")),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help='show HTTP requests and additional diagnostics',
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help='disable terminal keyboard shortcuts',
    )
    args = parser.parse_args()
    RUNTIME_PATHS.ensure_writable_directories()
    CONSOLE_UI = TerminalConsole(
        args.host, args.port,
        debug=args.debug,
        interactive=not args.no_interactive,
    )
    CONSOLE_UI.start_keyboard_listener()
    try:
        while not CONSOLE_UI.stop_requested():
            try:
                server = ExclusiveThreadingHTTPServer(
                    (args.host, args.port), Handler,
                )
            except OSError as exc:
                raise SystemExit(
                    f'Cannot start Run Console at {args.host}:{args.port}: port already in use. Close the previous console or choose another port with --port.'
                ) from exc
            CONSOLE_UI.attach(server)
            CONSOLE_UI.banner()
            server.serve_forever()
            server.server_close()
            if CONSOLE_UI.consume_restart() and not CONSOLE_UI.stop_requested():
                CONSOLE_UI.log('Socket closed, restarting...', "ACTION")
                continue
            break
    except KeyboardInterrupt:
        CONSOLE_UI.log('Keyboard interrupt received', "ACTION")
    finally:
        CONSOLE_UI.close()
        BATCH_COORDINATOR.stop()
        DEBUG_MANAGER.close()
        if "server" in locals():
            server.server_close()
        CONSOLE_UI.log('Server stopped cleanly', 'STOPPED')


if __name__ == "__main__":
    main()
