'\nCommand-line interface for the ObjectSensing reproduction. Build a database with build-db-dir, retrieve CAD models from an RGB-D ZIP with run or run-progressive, and create a final HTML viewer with visualize. Outputs include MeshLab ALN transformations and JSON diagnostics. See docs/SETUP.md and docs/USAGE.md.\n'

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from annotation_store import accepted_annotations_from_payload
from live_events import LiveEventWriter


_LIVE_EVENTS = LiveEventWriter()


def _log(msg):
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()
    _LIVE_EVENTS.emit("log", {"message": str(msg)})


def _emit_live_event(kind, data=None):
    _LIVE_EVENTS.emit(kind, data)


_log.event = _emit_live_event


def cmd_build_db(args):
    from config import PipelineConfig
    from database import build_database, cluster_descriptors
    cfg = PipelineConfig()
    synsets = tuple(args.synsets) if args.synsets else None
    _log(f"Preprocessing ShapeNet from {args.shapenet} …")
    db = build_database(args.shapenet, cfg, synsets=synsets,
                        max_per_synset=args.max_per_synset,
                        n_points=args.n_points, progress=_log)
    _log(f"{len(db.models)} models preprocessed.")
    if args.cluster and db.models:
        _log('Clustering descriptors …')
        try:
            cluster_descriptors(db)
        except Exception as exc:
            _log(f"(clustering skipped: {exc})")
    with open(args.out, "wb") as f:
        pickle.dump(db, f)
    _log(f"Database saved -> {args.out}")


def _limit_from_arg(value):
    return None if value is None or value <= 0 else value


def _build_db_dir_worker(task):
    src, syn, n_points, cfg = task
    from database import CATEGORY_NAME, preprocess_mesh

    try:
        name = f"{CATEGORY_NAME.get(syn, syn)}_{src.model_id}"
        ref = src.path if src.kind == "file" else f"{src.path}!{src.member}"
        model = preprocess_mesh(
            src.load(), name, syn, cfg,
            n_points=n_points,
            mesh_path=ref,
        )
        if model is None or model.keypoints.size == 0:
            return None, None, None
        model.cloud._tree = None
        return model, None, None
    except Exception as exc:
        return None, src.model_id, str(exc)


def _write_db_dir_model(out, meta, model_index, model):
    import pickle

    dst = out / f"model_{model_index:05d}.pkl"
    with open(dst, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta["names"].append(model.name)
    meta["synsets"].append(model.synset)
    meta["mesh_paths"].append(model.mesh_path)
    meta["n"] = model_index + 1
    return model_index + 1


def cmd_build_db_dir(args):
    from config import PipelineConfig
    from database import CATEGORY_NAME
    from shapenet_loader import find_model_sources
    import db_store

    cfg = PipelineConfig()
    synsets = tuple(args.synsets) if args.synsets else cfg.shapenet_synsets
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and db_store.model_files(str(out)):
        raise SystemExit(f"{out} already contains models; use --resume or a new directory")
    db_store.write_cfg(str(out), cfg)

    meta = db_store.load_resume_index(str(out)) if args.resume else {
        "n": 0, "names": [], "synsets": [], "mesh_paths": []
    }
    done_names = set(meta["names"])
    model_index = int(meta["n"])
    limit = _limit_from_arg(args.max_per_synset)
    jobs = max(1, int(args.jobs or 1))

    checkpoint_every = max(1, int(args.checkpoint_every))
    _log(f"Construction db_dir -> {out} ({'all models' if limit is None else str(limit) + ' par synset'}, {jobs} job(s))")
    if done_names:
        _log(f"Resuming: {len(done_names)} modeles deja presents")

    for syn in synsets:
        sources = find_model_sources(args.shapenet, syn, limit=limit)
        _log(f"[{syn}] {len(sources)} source(s) a parcourir")
        pending = []
        for src in sources:
            name = f"{CATEGORY_NAME.get(syn, syn)}_{src.model_id}"
            if name not in done_names:
                pending.append((src, syn, args.n_points, cfg))
        if len(pending) != len(sources):
            _log(f"[{syn}] {len(sources) - len(pending)} deja present(s), {len(pending)} remaining")

        if jobs == 1:
            results = map(_build_db_dir_worker, pending)
        else:
            from concurrent.futures import ProcessPoolExecutor
            from parallelism import configure_process_worker

            pool = ProcessPoolExecutor(
                max_workers=jobs,
                initializer=configure_process_worker,
            )
            results = pool.map(_build_db_dir_worker, pending, chunksize=1)

        try:
            for i, (model, skipped_id, error) in enumerate(results, 1):
                if error:
                    _log(f"[skip] {skipped_id}: {error}")
                elif model is not None:
                    model_index = _write_db_dir_model(out, meta, model_index, model)
                    done_names.add(model.name)
                if i % checkpoint_every == 0 or i == len(pending):
                    db_store.write_index(str(out), meta, partial=True)
                    _log(f"[{syn}] {i}/{len(pending)} traites, {model_index} modeles ecrits")
        finally:
            if jobs != 1:
                pool.shutdown()

    db_store.write_index(str(out), meta, partial=False)
    _log(f"OK: {model_index} models written to {out}")


def _apply_overrides(cfg, args):
    if getattr(args, "fusion_backend", None):
        cfg.sdf.backend = args.fusion_backend
    if getattr(args, "volume_layout", None):
        cfg.sdf.volume_layout = args.volume_layout
    for argument, field in (
        ("depth_trunc", "depth_trunc"),
        ("iso_sampling", "iso_sampling"),
        ("max_surface_points", "max_surface_points"),
        ("surface_field", "surface_field"),
        ("surface_min_support", "surface_min_support"),
        ("surface_extraction", "surface_extraction"),
        ("surface_band_factor", "surface_band_factor"),
        ("surface_rescue_distance_factor", "surface_rescue_distance_factor"),
        ("surface_rescue_min_weight", "surface_rescue_min_weight"),
        ("gradient_smoothing_sigma", "gradient_smoothing_sigma"),
        ("depth_edge_threshold", "depth_edge_threshold"),
        ("depth_edge_background_weight", "depth_edge_background_weight"),
        ("depth_edge_radius", "depth_edge_radius"),
        ("sparse_block_resolution", "sparse_block_resolution"),
        ("sparse_block_count", "sparse_block_count"),
        ("visibility_depth_stride", "visibility_depth_stride"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(cfg.sdf, field, value)
    if args.voxel:
        cfg.sdf.voxel_size = args.voxel
        cfg.sdf.truncation = 3.0 * args.voxel
    if getattr(args, "truncation", None):
        cfg.sdf.truncation = args.truncation
    if getattr(args, "all_frames", False):
        cfg.sdf.frame_stride = 1
        cfg.sdf.min_frames = 0
        cfg.sdf.max_frames = 0
    if args.frame_stride:
        cfg.sdf.frame_stride = args.frame_stride
    if getattr(args, "min_frames", None) is not None:
        cfg.sdf.min_frames = args.min_frames
    if args.max_frames is not None:
        cfg.sdf.max_frames = args.max_frames
    if getattr(args, "max_scan_keypoints", None) is not None:
        cfg.matching.max_scan_keypoints = args.max_scan_keypoints
    if getattr(args, "max_model_keypoints", None) is not None:
        cfg.matching.max_model_keypoints = args.max_model_keypoints
    if getattr(args, "no_spatial_keypoints", False):
        cfg.matching.spatial_keypoint_balance = False
    elif getattr(args, "spatial_keypoints", False):
        cfg.matching.spatial_keypoint_balance = True
    if getattr(args, "spatial_keypoint_cell", None) is not None:
        cfg.matching.spatial_keypoint_cell = args.spatial_keypoint_cell
    if getattr(args, "descriptor_distance_threshold", None) is not None:
        cfg.ransac.desc_inlier = args.descriptor_distance_threshold
        if float(getattr(
                cfg.matching, "candidate_descriptor_distance_threshold", 0.0
        )) <= 0.0:
            cfg.matching.candidate_descriptor_distance_threshold = 0.0
    if getattr(args, "descriptor_ratio_threshold", None) is not None:
        cfg.matching.descriptor_ratio_threshold = args.descriptor_ratio_threshold
    if getattr(args, "candidate_descriptor_distance_threshold", None) is not None:
        cfg.matching.candidate_descriptor_distance_threshold = (
            args.candidate_descriptor_distance_threshold
        )
    if getattr(args, "candidate_top_per_group", None) is not None:
        cfg.matching.candidate_top_per_group = args.candidate_top_per_group
    if getattr(args, "candidate_local_extra_k", None) is not None:
        cfg.matching.candidate_local_extra_k = max(0, args.candidate_local_extra_k)
    if getattr(args, "group_pose_seed_enabled", None) is not None:
        cfg.matching.group_pose_seed_enabled = args.group_pose_seed_enabled
    if getattr(args, "group_pose_seed_count", None) is not None:
        cfg.matching.group_pose_seed_count = args.group_pose_seed_count
    for name in (
        "min_model_inliers", "min_scan_inliers", "min_scan_cells",
        "min_model_cells",
    ):
        value = getattr(args, name, None)
        if value is not None:
            target = "min_inliers" if name == "min_model_inliers" else name
            setattr(cfg.ransac, target, value)
    keypoint_overrides = (
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
    )
    for name in keypoint_overrides:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg.keypoint, name, value)
    jitter_reject = getattr(args, "jitter_reject", None)
    if jitter_reject is not None:
        cfg.keypoint.jitter_reject = (
            None if str(jitter_reject).lower() == "auto"
            else float(jitter_reject)
        )
    if getattr(args, "floor_height", None) is not None:
        cfg.matching.floor_height = args.floor_height
    if getattr(args, "wall_filter_enabled", None) is not None:
        cfg.keypoint.wall_filter_enabled = args.wall_filter_enabled
    if getattr(args, "wall_hard_reject", None) is not None:
        cfg.keypoint.wall_hard_reject = args.wall_hard_reject
    if getattr(args, "wall_budget_enabled", None) is not None:
        cfg.keypoint.wall_budget_enabled = args.wall_budget_enabled
    if getattr(args, "component_budget_enabled", None) is not None:
        cfg.keypoint.component_budget_enabled = args.component_budget_enabled
    for name in (
        "planar_filter_enabled", "hole_boundary_filter_enabled",
        "repeatability_filter_enabled",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg.keypoint, name, value)
    for name in (
        "descriptor_min_occupied", "descriptor_max_hole_fraction",
        "descriptor_keypoint_min_features", "quality_nms_radius",
        "quality_min_score_ratio",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg.matching, name, value)
    if getattr(args, "descriptor_keypoint_filter_enabled", None) is not None:
        cfg.matching.descriptor_keypoint_filter_enabled = (
            args.descriptor_keypoint_filter_enabled
        )
    if getattr(args, "min_2d_corner_fraction", None) is not None:
        cfg.matching.min_2d_corner_fraction = args.min_2d_corner_fraction
    if getattr(args, "paper_verification", False):
        cfg.matching.reverse_gate = 0.0
        cfg.matching.min_structure_height = 0.0
        cfg.matching.min_thickness = 0.0
    if getattr(args, "reverse_gate", None) is not None:
        cfg.matching.reverse_gate = args.reverse_gate
    if getattr(args, "surface_distance_weight", None) is not None:
        cfg.matching.surface_distance_weight = args.surface_distance_weight
    if getattr(args, "verification_top_constellations", None):
        cfg.matching.verification_top_constellations = args.verification_top_constellations
    if getattr(args, "max_registrations_per_model", None):
        cfg.matching.max_registrations_per_model = args.max_registrations_per_model
    if getattr(args, "registration_min_center_distance", None):
        cfg.matching.registration_min_center_distance = args.registration_min_center_distance
    if getattr(args, "registration_top_constellations", None):
        cfg.matching.registration_top_constellations = args.registration_top_constellations
    if getattr(args, "multi_registration_synsets", None):
        cfg.matching.multi_registration_synsets = tuple(args.multi_registration_synsets)
    return cfg


def _add_keypoint_detector_arguments(parser):
    parser.add_argument("--neighbor-radius", type=float, default=None)
    parser.add_argument("--curvature-threshold", type=float, default=None)
    parser.add_argument("--harris-k", type=float, default=None)
    parser.add_argument("--harris-threshold", type=float, default=None)
    parser.add_argument("--harris-reference-neighbors", type=float, default=None)
    parser.add_argument("--convex-hull-ratio", type=float, default=None)
    parser.add_argument("--corner-plane-radius-factor", type=float, default=None)
    parser.add_argument("--corner-plane-min-area", type=float, default=None)
    parser.add_argument("--corner-plane-normal-cos", type=float, default=None)
    parser.add_argument("--nms-radius", type=float, default=None)
    parser.add_argument("--adjust-iterations", type=int, default=None)
    parser.add_argument(
        "--jitter-reject", default=None,
        help='Maximum adjustment displacement in meters, or auto',
    )
    parser.add_argument("--dedup-radius", type=float, default=None)
    parser.add_argument("--wall-vertical-normal-cos", type=float, default=None)
    parser.add_argument("--wall-plane-normal-cos", type=float, default=None)
    parser.add_argument("--wall-plane-angle-degrees", type=float, default=None)
    parser.add_argument("--wall-plane-distance", type=float, default=None)
    parser.add_argument("--wall-plane-min-points", type=int, default=None)
    parser.add_argument("--wall-plane-min-width", type=float, default=None)
    parser.add_argument("--wall-plane-min-height", type=float, default=None)
    parser.add_argument("--wall-plane-min-area", type=float, default=None)
    parser.add_argument("--wall-plane-max-count", type=int, default=None)
    parser.add_argument("--wall-plane-sample-points", type=int, default=None)
    parser.add_argument("--wall-small-radius", type=float, default=None)
    parser.add_argument("--wall-large-radius", type=float, default=None)
    parser.add_argument("--wall-protrusion-radius", type=float, default=None)
    parser.add_argument("--wall-protrusion-distance", type=float, default=None)
    parser.add_argument("--wall-object-protection", type=float, default=None)
    parser.add_argument("--wall-penalty-weight", type=float, default=None)
    parser.add_argument("--wall-reject-threshold", type=float, default=None)
    parser.add_argument(
        "--wall-reject-max-object-score", type=float, default=None,
    )
    parser.add_argument("--wall-budget-affinity-threshold", type=float, default=None)
    parser.add_argument("--wall-budget-proximity-threshold", type=float, default=None)
    parser.add_argument("--wall-budget-max-fraction", type=float, default=None)
    parser.add_argument("--wall-budget-cell-size", type=float, default=None)
    parser.add_argument("--wall-budget-max-per-cell", type=int, default=None)
    parser.add_argument("--wall-budget-min-features", type=int, default=None)
    parser.add_argument("--component-budget-radius", type=float, default=None)
    parser.add_argument(
        "--component-budget-max-per-component", type=int, default=None,
    )
    parser.add_argument("--component-budget-min-features", type=int, default=None)
    parser.add_argument("--planar-filter-radius", type=float, default=None)
    parser.add_argument("--planar-filter-max-residual", type=float, default=None)
    parser.add_argument("--planar-filter-normal-cos", type=float, default=None)
    parser.add_argument("--planar-filter-angular-coverage", type=float, default=None)
    parser.add_argument("--planar-filter-min-neighbors", type=int, default=None)
    parser.add_argument("--hole-boundary-probe-radius", type=float, default=None)
    parser.add_argument("--hole-boundary-max-fraction", type=float, default=None)
    parser.add_argument("--repeatability-radius-factor", type=float, default=None)
    parser.add_argument("--repeatability-response-ratio", type=float, default=None)
    parser.add_argument("--quality-response-ratio", type=float, default=None)
    parser.add_argument("--quality-score-radius", type=float, default=None)
    parser.add_argument("--geometric-nms-radius", type=float, default=None)
    parser.add_argument("--quality-filter-min-features", type=int, default=None)


def _add_descriptor_matching_arguments(parser):
    parser.add_argument(
        "--descriptor-distance-threshold", type=float, default=None,
        help='Maximum UDF distance for a local correspondence',
    )
    parser.add_argument(
        "--descriptor-ratio-threshold", type=float, default=None,
        help='Best/second descriptor ratio (0 disables)',
    )
    parser.add_argument("--min-model-inliers", type=int, default=None)
    parser.add_argument("--min-scan-inliers", type=int, default=None)
    parser.add_argument("--min-scan-cells", type=int, default=None)
    parser.add_argument("--min-model-cells", type=int, default=None)
    parser.add_argument(
        "--wall-filter", dest="wall_filter_enabled", action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-wall-filter", dest="wall_filter_enabled", action="store_false",
    )
    parser.add_argument(
        "--wall-hard-reject", dest="wall_hard_reject", action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-wall-hard-reject", dest="wall_hard_reject",
        action="store_false",
    )
    parser.add_argument(
        "--wall-budget", dest="wall_budget_enabled", action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-wall-budget", dest="wall_budget_enabled", action="store_false",
    )
    parser.add_argument(
        "--component-budget", dest="component_budget_enabled",
        action="store_true", default=None,
    )
    parser.add_argument(
        "--no-component-budget", dest="component_budget_enabled",
        action="store_false",
    )
    for option, destination in (
        ("planar-filter", "planar_filter_enabled"),
        ("hole-boundary-filter", "hole_boundary_filter_enabled"),
        ("repeatability-filter", "repeatability_filter_enabled"),
        ("descriptor-keypoint-filter", "descriptor_keypoint_filter_enabled"),
    ):
        parser.add_argument(
            f"--{option}", dest=destination, action="store_true", default=None,
        )
        parser.add_argument(
            f"--no-{option}", dest=destination, action="store_false",
        )
    parser.add_argument("--descriptor-min-occupied", type=int, default=None)
    parser.add_argument("--descriptor-max-hole-fraction", type=float, default=None)
    parser.add_argument("--descriptor-keypoint-min-features", type=int, default=None)
    parser.add_argument("--quality-nms-radius", type=float, default=None)
    parser.add_argument("--quality-min-score-ratio", type=float, default=None)
    parser.add_argument("--floor-height", type=float, default=None)
    parser.add_argument("--min-2d-corner-fraction", type=float, default=None)
    parser.add_argument(
        "--no-spatial-keypoints", action="store_true",
        help='Explicitly disable spatial keypoint balancing',
    )


def _write_outputs(args, regs, mesh_of):
    from io_aln import write_aln
    entries, summary = [], []
    for r in regs:
        mesh = mesh_of(r)
        ref = Path(mesh).name if mesh else (r.model_name + ".obj")
        entries.append((ref, r.transform.matrix()))
        summary.append({
            "model": r.model_name,
            "mesh": mesh,
            "coverage": round(r.coverage, 3),
            "cov_forward": round(float(getattr(r, "cov_forward", 0.0)), 3),
            "cov_reverse": round(float(getattr(r, "cov_reverse", 0.0)), 3),
            "geometric_cov_reverse": round(float(getattr(
                r, "_geometric_cov_reverse", getattr(r, "cov_reverse", 0.0),
            )), 3),
            "visibility_support": (
                round(float(r._visibility_support), 3)
                if hasattr(r, "_visibility_support") else None
            ),
            "visibility_known_fraction": (
                round(float(r._visibility_known_fraction), 3)
                if hasattr(r, "_visibility_known_fraction") else None
            ),
            "visibility_free_fraction": (
                round(float(r._visibility_free_fraction), 3)
                if hasattr(r, "_visibility_free_fraction") else None
            ),
            "visibility_used": bool(getattr(r, "_visibility_used", False)),
            "quality": round(float(r.quality), 1),
            "mean_surface_dist_m": round(r.mean_surface_dist, 4),
            "theta_deg": round(float(r.transform.theta) * 57.2958, 2),
            "scale": round(float(r.transform.scale), 3),
            "translation": [round(float(x), 3) for x in r.transform.t],
        })
    write_aln(args.out, entries)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"Written: {args.out}  (+ summary {Path(args.out).with_suffix('.json').name})")
    for s in summary:
        _log(f"  - {s['model']}: couv {s['coverage']} (av {s['cov_forward']}/ar {s['cov_reverse']}), scale {s['scale']}, theta {s['theta_deg']}°")


def _registration_manifest_entry(reg):
    from verify import symmetric_coverage_score
    from pipeline import _ranking_reverse_coverage

    entry = {
        "model": reg.model_name,
        "category": reg.model_name.split("_", 1)[0],
        "checkpoint_frames": getattr(reg, "_checkpoint_frames", None),
        "coverage": round(float(reg.coverage), 4),
        "cov_reverse": round(float(getattr(reg, "cov_reverse", 0.0)), 4),
        "mean_surface_dist_m": round(float(reg.mean_surface_dist), 4),
        "quality": round(float(reg.quality), 1),
        "scale": round(float(reg.transform.scale), 4),
        "theta_deg": round(float(reg.transform.theta) * 57.2958, 2),
        "translation": [round(float(x), 4) for x in reg.transform.t],
        "balanced_score": round(symmetric_coverage_score(
            float(reg.coverage), _ranking_reverse_coverage(reg),
        ), 4),
    }
    for attr, key in (
        ("_query_group_reverse", "group_cov_reverse"),
        ("_pose_seed_fallback", "pose_seed_fallback"),
        ("_geometric_cov_reverse", "geometric_cov_reverse"),
        ("_visibility_support", "visibility_support"),
        ("_visibility_known_fraction", "visibility_known_fraction"),
        ("_visibility_free_fraction", "visibility_free_fraction"),
        ("_visibility_known_points", "visibility_known_points"),
        ("_visibility_used", "visibility_used"),
    ):
        if hasattr(reg, attr):
            value = getattr(reg, attr)
            entry[key] = value if isinstance(value, (bool, int)) else round(float(value), 4)
    return entry


def _diagnostic_rank(entry):
    status_rank = {
        "selected": 0,
        "kept": 0,
        "rejected_horizontal_iou": 1,
        "rejected_explained_overlap": 2,
        "rejected_category_limit": 3,
        "rejected_cross_category_collision": 4,
        "rejected_min_symmetric_score": 5,
        "rejected_no_scene_support": 6,
        "rejected_missing_footprint": 7,
        "rejected_reverse": 8,
        "rejected_coverage": 9,
        "missing_model": 10,
    }
    return (
        status_rank.get(entry.get("status"), 9),
        -float(entry.get("balanced_score", 0.0)),
        -float(entry.get("coverage", 0.0)),
    )


def _write_scan_for_visualization(args, scene):
    from viz import write_points_ply
    scan_ply = str(Path(args.out).with_suffix("")) + "_scan.ply"
    write_points_ply(scan_ply, scene.cloud.points)
    _log(f"Scan saved for visualization: {scan_ply}")
    return scan_ply


def _candidate_include_names(args):
    names = []
    for path in getattr(args, "candidate_include_json", None) or []:
        try:
            for entry in json.loads(Path(path).read_text(encoding="utf-8")):
                name = entry.get("model")
                if name:
                    names.append(name)
        except Exception as exc:
            _log(f"[warn] ignore {path}: {exc}")
    return list(dict.fromkeys(names))


def _database_manifest(db_dir):
    from collections import Counter
    import db_store

    metadata = db_store.load_index(db_dir)
    counts = Counter(str(value) for value in metadata.get("synsets", []))
    labels = {
        "03001627": "chair",
        "04379243": "table",
        "04256520": "couch",
    }
    return {
        "models": int(metadata.get("n", len(metadata.get("names", [])))),
        "categories": {
            labels.get(synset, synset): int(count)
            for synset, count in sorted(counts.items())
        },
    }


def _parse_category_limits(values):
    limits = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected category limit name=count; received {value!r}")
        name, raw_limit = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty category in {value!r}")
        limit = int(raw_limit)
        if limit < 0:
            raise ValueError(f"Negative limit in {value!r}")
        limits[name] = limit
    return limits


def cmd_run(args):
    import os as _os
    from config import PipelineConfig
    import db_store
    import pipeline

    if db_store.is_db_dir(args.db):
        # Base un-fichier-par-modèle -> matching parallèle à mémoire constante.
        cfg = _apply_overrides(db_store.load_cfg(args.db) or PipelineConfig(), args)
        jobs = args.jobs if (args.jobs and args.jobs > 0) else max(1, (_os.cpu_count() or 1) - 1)
        _log(f"Scene: {args.zip}  (directory database, matching with {jobs} worker(s))")
        scene = pipeline.scan_from_zip(args.zip, cfg, progress=_log)
        _write_scan_for_visualization(args, scene)
        regs = pipeline.retrieve_dir(scene, args.db, cfg, coverage_threshold=args.coverage,
                                     n_jobs=jobs, progress=_log,
                                     candidate_index_path=args.candidate_index,
                                     candidate_top_k=args.candidate_top_k,
                                     candidate_pool_k=args.candidate_pool_k,
                                     candidate_diverse=args.candidate_diverse,
                                     candidate_diversity=args.candidate_diversity,
                                     candidate_include_names=_candidate_include_names(args),
                                     exhaustive=args.exhaustive)
        _log(f"{len(regs)} object(s) retrieved.")
        _write_outputs(args, regs, lambda r: getattr(r, "_mesh_path", ""))
    else:
        # db.pkl monolithique : séquentiel par défaut (le parallèle dupliquerait
        # la base en mémoire). Convertis en dossier avec `split-db` pour le multi-cœurs.
        with open(args.db, "rb") as f:
            db = pickle.load(f)
        cfg = _apply_overrides(db.cfg or PipelineConfig(), args)
        jobs = args.jobs if (args.jobs and args.jobs > 0) else 1
        _log(f'Scene: {args.zip}  (monolithic db.pkl, matching with {jobs} worker(s); use split-db for memory-efficient parallelism)')
        scene = pipeline.scan_from_zip(args.zip, cfg, progress=_log)
        _write_scan_for_visualization(args, scene)
        regs = pipeline.retrieve(scene, db, cfg, coverage_threshold=args.coverage,
                                 progress=_log, n_jobs=jobs, db_path=args.db)
        _log(f"{len(regs)} object(s) retrieved.")
        _write_outputs(args, regs, lambda r: db.models[getattr(r, "_model_index", 0)].mesh_path)


def cmd_run_progressive(args):
    import os as _os
    from config import PipelineConfig
    import db_store
    import progressive

    if not db_store.is_db_dir(args.db):
        raise ValueError('run-progressive requires a directory database (db_dir)')

    cfg = _apply_overrides(db_store.load_cfg(args.db) or PipelineConfig(), args)
    jobs = args.jobs if (args.jobs and args.jobs > 0) else max(1, (_os.cpu_count() or 1) - 1)
    initial_cache = _candidate_include_names(args)
    if args.query_mode == "single":
        _log(f"Single-query scene: {args.zip} ({jobs} worker(s))")
    else:
        _log(
            f"Progressive scene: {args.zip} "
            f"(start {args.query_start_frames}, intervalle "
            f"{args.query_interval_frames}, {jobs} worker(s))"
        )
    if args.resume_dir:
        _log(
            f"batch resumption: {args.resume_dir} "
            f"({args.resume_batch_size} model(s)/save)"
        )
    result = progressive.progressive_retrieve_dir(
        args.zip, args.db, cfg,
        coverage_threshold=args.coverage,
        n_jobs=jobs,
        progress=_log,
        candidate_index_path=args.candidate_index,
        candidate_top_k=args.candidate_top_k,
        candidate_pool_k=args.candidate_pool_k,
        candidate_diverse=args.candidate_diverse,
        candidate_diversity=args.candidate_diversity,
        exhaustive=args.exhaustive,
        initial_cache_names=initial_cache,
        start_frames=args.query_start_frames,
        interval_frames=args.query_interval_frames,
        query_mode=args.query_mode,
        cache_registrations_per_model=args.cache_registrations_per_model,
        cache_top_constellations=args.cache_top_constellations,
        final_reverse_gate=args.final_reverse_gate,
        final_max_per_category=_parse_category_limits(args.final_max_per_category),
        final_global_overlap=args.final_global_overlap,
        final_min_symmetric_score=args.final_min_symmetric_score,
        final_cross_category_center_distance=args.final_cross_category_center_distance,
        final_cross_category_overlap=args.final_cross_category_overlap,
        resume_dir=args.resume_dir,
        resume_batch_size=args.resume_batch_size,
        stop_after=args.stop_after,
        stage_cache_dir=args.stage_cache_dir,
    )
    _write_scan_for_visualization(args, result.scene)
    if result.completed_stage == "selection":
        suffix = (
            'on the final query' if args.query_mode == "single"
            else 'across all checkpoints'
        )
        _log(f"{len(result.registrations)} object(s) merged {suffix}.")
        _write_outputs(
            args, result.registrations,
            lambda r: getattr(r, "_mesh_path", ""),
        )
    else:
        _log(
            f'Stop requested after stage {result.completed_stage} ; intermediate artifacts and checkpoints are preserved.'
        )
    manifest = {
        "trace_version": 1,
        "completed_stage": result.completed_stage,
        "run_config": {
            "scene_zip": str(args.zip),
            "database": str(args.db),
            "coverage_threshold": args.coverage,
            "jobs": jobs,
            "voxel": args.voxel,
            "fusion_backend": args.fusion_backend,
            "volume_layout": args.volume_layout,
            "surface_field": args.surface_field,
            "surface_min_support": args.surface_min_support,
            "surface_extraction": args.surface_extraction,
            "surface_band_factor": args.surface_band_factor,
            "surface_rescue_distance_factor":
                args.surface_rescue_distance_factor,
            "surface_rescue_min_weight": args.surface_rescue_min_weight,
            "gradient_smoothing_sigma": args.gradient_smoothing_sigma,
            "depth_edge_threshold": args.depth_edge_threshold,
            "depth_edge_background_weight":
                args.depth_edge_background_weight,
            "depth_edge_radius": args.depth_edge_radius,
            "sparse_block_resolution": args.sparse_block_resolution,
            "sparse_block_count": args.sparse_block_count,
            "visibility_depth_stride": args.visibility_depth_stride,
            "frame_stride": args.frame_stride,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "max_scan_keypoints": args.max_scan_keypoints,
            "descriptor_distance_threshold": cfg.ransac.desc_inlier,
            "descriptor_ratio_threshold": cfg.matching.descriptor_ratio_threshold,
            "candidate_top_k": args.candidate_top_k,
            "candidate_pool_k": args.candidate_pool_k,
            "candidate_local_extra_k": cfg.matching.candidate_local_extra_k,
            "candidate_diverse": args.candidate_diverse,
            "candidate_diversity": args.candidate_diversity,
            "query_mode": args.query_mode,
            "coverage": args.coverage,
        },
        "database_summary": _database_manifest(args.db),
        "fusion_trace": getattr(result.scene, "fusion_trace", None),
        "query_start_frames": args.query_start_frames,
        "query_interval_frames": args.query_interval_frames,
        "query_mode": args.query_mode,
        "final_min_symmetric_score": args.final_min_symmetric_score,
        "final_cross_category_center_distance": args.final_cross_category_center_distance,
        "final_cross_category_overlap": args.final_cross_category_overlap,
        "steps": [
            {
                "frame_count": step.frame_count,
                "models": [reg.model_name for reg in step.registrations],
                "cache_names": step.cache_names,
            }
            for step in result.steps
        ],
        "execution_trace": [
            {
                "frame_count": step.frame_count,
                "cache_names": step.cache_names,
                "cache": (step.trace or {}).get("cache"),
                "full": (step.trace or {}).get("full"),
            }
            for step in result.steps
        ],
        "selected": [_registration_manifest_entry(reg) for reg in result.registrations],
        "final_candidates": [
            _registration_manifest_entry(reg)
            for reg in sorted(
                result.final_candidates,
                key=lambda r: (
                    -min(float(r.coverage), float(getattr(r, "cov_reverse", 0.0))),
                    -float(r.coverage),
                ),
            )
        ],
        "final_revalidation": sorted(result.final_diagnostics, key=_diagnostic_rank),
        "final_revalidation_trace": [
            {**entry, "event_index": index}
            for index, entry in enumerate(result.final_diagnostics)
        ],
        "final_selection": sorted(
            result.final_selection_diagnostics,
            key=_diagnostic_rank,
        ),
        "final_selection_trace": [
            {**entry, "event_index": index}
            for index, entry in enumerate(result.final_selection_diagnostics)
        ],
    }
    manifest_path = Path(args.out).with_suffix("").with_name(
        Path(args.out).stem + "_progress"
    ).with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"Historique progressif : {manifest_path}")


def cmd_split_db(args):
    import db_store
    _log(f"Conversion {args.db} -> directory {args.out} (one file per model) …")
    n = db_store.convert_pkl_to_dir(args.db, args.out, progress=_log)
    _log(f"OK : {n} models. Next run: python cli.py run --zip <scene>.zip --db {args.out} --out scene.aln")


def cmd_build_candidate_index(args):
    from candidate_index import build_candidate_index, default_index_path, save_candidate_index

    out = args.out or default_index_path(args.db)
    _log(f'Building candidate index from {args.db}')
    index = build_candidate_index(args.db, progress=_log)
    save_candidate_index(index, out)
    _log(f"Candidate index saved -> {out} ({len(index.names)} modeles)")


def cmd_sweep_keypoints(args):
    from debug_visualizer import default_parameters
    from keypoint_sweep import (
        parse_assignment, parse_grid, parse_source, run_keypoint_sweep,
        settings_from_json,
    )

    defaults = default_parameters("keypoints")
    sources = [parse_source(value) for value in args.source]
    settings = (
        settings_from_json(args.settings_json, defaults)
        if args.settings_json else {}
    )
    settings.update(
        dict(parse_assignment(value, defaults) for value in args.settings)
    )
    grid = dict(parse_grid(value, defaults) for value in args.grid)
    if not grid:
        grid = {
            "component_budget_radius": [0.20, 0.25, 0.35],
            "component_budget_max_per_component": [32, 48, 64],
        }
    descriptor_evaluator = None
    if args.descriptor_db:
        target_annotations = None
        if args.target_annotations:
            target_annotations = accepted_annotations_from_payload(json.loads(
                Path(args.target_annotations).read_text(encoding="utf-8")
            ))
        from descriptor_utility import DescriptorSweepEvaluator
        descriptor_evaluator = DescriptorSweepEvaluator(
            args.descriptor_db,
            candidate_index_path=args.descriptor_candidate_index,
            candidate_models=args.utility_candidate_models,
            model_features=args.utility_model_features,
            scan_features=args.utility_scan_features,
            rotations=args.utility_rotations,
            target_annotations=target_annotations,
            candidate_pool_size=args.utility_candidate_pool_size,
        )
        _log(
            f'Descriptor objective enabled: {args.utility_candidate_models} model(s), {args.utility_model_features} feature(s)/model, {args.utility_scan_features} sampled keypoint(s)'
        )
    payload = run_keypoint_sweep(
        sources, settings, grid, args.out,
        coverage_cell=args.coverage_cell,
        resume=args.resume, progress=_log,
        descriptor_evaluator=descriptor_evaluator,
    )
    recommended = payload.get("recommended_trial")
    if recommended:
        trial = next(
            item for item in payload["trials"] if item["id"] == recommended
        )
        _log(
            'Recommended setting: '
            + ", ".join(
                f"{key}={value}" for key, value in trial["parameters"].items()
            )
            + f" ({trial['aggregate']['total_retained']} points)"
        )
    elif payload.get("best_effort_trial"):
        _log(
            'No setting meets all recall thresholds; inspect best_effort_trial without treating it as a recommendation.'
        )
    if payload.get("pareto_profiles"):
        _log("Profils Pareto:")
        by_id = {trial["id"]: trial for trial in payload["trials"]}
        for profile in payload["pareto_profiles"]:
            trial = by_id[profile["trial"]]
            detail = ", ".join(
                f"{key}={value}"
                for key, value in trial["parameters"].items()
            )
            _log(f"  {profile['role']}: {detail}")
    _log(f"Sweep saved -> {args.out}")


def cmd_visualize(args):
    from viz import build_visualization
    from timeline_viewer import build_timeline_html

    out = build_visualization(args.json, args.out, scan_ply=args.scan, progress=_log)
    _log(f"Visualization written: {out}")
    html_out = args.html or str(Path(args.out).with_suffix(".html"))
    progress_json = args.progress
    if progress_json is None:
        summary_path = Path(args.json)
        progress_json = str(summary_path.with_name(summary_path.stem + "_progress.json"))
    build_timeline_html(args.json, out, html_out, progress_json=progress_json)
    _log(f'Timeline viewer: {html_out}')
    _log('Open the HTML to inspect the scene, Top-k candidates and decisions.')


def main():
    p = argparse.ArgumentParser(description='ObjectSensing pipeline (reimplementation).')
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-db", help='Preprocess ShapeNetCore into a model database.')
    b.add_argument("--shapenet", required=True, help='ShapeNetCore.v2 root')
    b.add_argument("--out", default="db.pkl")
    b.add_argument("--synsets", nargs="*", help='Synset IDs (default: chair/table/couch)')
    b.add_argument("--max-per-synset", type=int, default=200)
    b.add_argument("--n-points", type=int, default=6000)
    b.add_argument("--cluster", dest="cluster", action="store_true", help='K-means descriptor clustering')
    b.add_argument("--no-cluster", dest="cluster", action="store_false", help='Disable descriptor clustering')
    b.set_defaults(cluster=True)
    b.set_defaults(func=cmd_build_db)

    bd = sub.add_parser("build-db-dir", help='Preprocess ShapeNetCore into a resumable directory database.')
    bd.add_argument("--shapenet", required=True, help='ShapeNetCore.v2 root or directory of synset ZIP archives')
    bd.add_argument("--out", required=True, help='Output db_dir directory')
    bd.add_argument("--synsets", nargs="*", help='Synset IDs (default: chair/table/couch)')
    bd.add_argument("--max-per-synset", type=int, default=0, help='0 = all available models')
    bd.add_argument("--n-points", type=int, default=3000)
    bd.add_argument("--jobs", type=int, default=1, help='Parallel preprocessing workers')
    bd.add_argument("--checkpoint-every", type=int, default=10)
    bd.add_argument("--resume", dest="resume", action="store_true", help='Resume from an existing index')
    bd.add_argument("--no-resume", dest="resume", action="store_false", help='Ignore existing index')
    bd.set_defaults(resume=True)
    bd.set_defaults(func=cmd_build_db_dir)

    r = sub.add_parser("run", help='Retrieve and place models in an RGB-D scene.')
    r.add_argument("--zip", required=True, help='ObjectSensing RGB-D sequence (.zip)')
    r.add_argument("--db", required=True, help='Preprocessed database (build-db)')
    r.add_argument("--out", default="scene.aln")
    r.add_argument("--coverage", type=float, default=0.4, help='Coverage threshold')
    r.add_argument("--jobs", type=int, default=0, help='Matching workers (0 = automatic)')
    r.add_argument("--voxel", type=float, default=None, help='TSDF voxel size (m)')
    r.add_argument("--truncation", type=float, default=None, help='TSDF truncation band (m; defaults to 3*voxel when --voxel is set)')
    r.add_argument("--fusion-backend", choices=("auto", "cpu", "cuda"), default=None,
                   help='TSDF fusion engine (cuda requires CuPy/CUDA)')
    r.add_argument(
        "--volume-layout", choices=("auto", "dense", "sparse"),
        default=None,
        help='TSDF volume layout (auto favors sparse on CPU)',
    )
    r.add_argument("--depth-trunc", type=float, default=None)
    r.add_argument("--iso-sampling", type=float, default=None)
    r.add_argument("--max-surface-points", type=int, default=None)
    r.add_argument(
        "--surface-field", choices=("raw", "smoothed"), default=None,
    )
    r.add_argument("--surface-min-support", type=float, default=None)
    r.add_argument(
        "--surface-extraction",
        choices=("zero_crossing", "hybrid", "near_zero"), default=None,
    )
    r.add_argument("--surface-band-factor", type=float, default=None)
    r.add_argument(
        "--surface-rescue-distance-factor", type=float, default=None,
    )
    r.add_argument("--surface-rescue-min-weight", type=float, default=None)
    r.add_argument("--gradient-smoothing-sigma", type=float, default=None)
    r.add_argument("--depth-edge-threshold", type=float, default=None)
    r.add_argument(
        "--depth-edge-background-weight", type=float, default=None,
    )
    r.add_argument("--depth-edge-radius", type=int, default=None)
    r.add_argument("--sparse-block-resolution", type=int, default=None)
    r.add_argument("--sparse-block-count", type=int, default=None)
    r.add_argument("--visibility-depth-stride", type=int, default=None)
    r.add_argument("--frame-stride", type=int, default=None)
    r.add_argument(
        "--min-frames", type=int, default=None,
        help='Reduce stride automatically to retain at least N frames (0 disables)',
    )
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--all-frames", action="store_true", help='Force frame_stride=1 and remove the frame limit')
    r.add_argument("--max-scan-keypoints", type=int, default=None, help='Maximum scene keypoints retained for matching')
    r.add_argument("--max-model-keypoints", type=int, default=None, help='Maximum model keypoints retained for matching')
    r.add_argument("--spatial-keypoints", action="store_true", help='Distribute the keypoint budget across 3D cells')
    r.add_argument("--spatial-keypoint-cell", type=float, default=None, help='Spatial balancing cell size (m)')
    _add_keypoint_detector_arguments(r)
    _add_descriptor_matching_arguments(r)
    r.add_argument("--reverse-gate", type=float, default=None, help='Minimum reverse coverage for pose acceptance')
    r.add_argument("--paper-verification", action="store_true", help='Disable added safeguards and use coverage plus surface distance')
    r.add_argument("--surface-distance-weight", type=float, default=None, help='Surface distance weight in the score')
    r.add_argument("--verification-top-constellations", type=int, default=None, help='Constellations verified per model in a standard query')
    r.add_argument("--max-registrations-per-model", type=int, default=None, help='Maximum distinct poses retained per model')
    r.add_argument("--registration-min-center-distance", type=float, default=None, help='Minimum horizontal distance between poses of the same model')
    r.add_argument("--registration-top-constellations", type=int, default=None, help='Constellations verified per model in multi-pose mode')
    r.add_argument("--multi-registration-synsets", nargs="*", default=None, help='Synsets allowed to produce multiple poses')
    r.add_argument("--candidate-index", default=None, help='Compact preselection index for db_dir')
    r.add_argument("--candidate-top-k", type=int, default=0, help='Candidate model count before full matching')
    r.add_argument("--candidate-pool-k", type=int, default=0, help='Broad preselection pool reranked locally before final Top-k')
    r.add_argument("--candidate-local-extra-k", type=int, default=None, help='Local UDF supplement to the original pool (configuration: 64; 0 disables)')
    r.add_argument("--exhaustive", action="store_true", help='Explicitly test the entire database, bypassing preselection')
    r.add_argument("--candidate-diverse", action="store_true", help='Supplement Top-k with diverse sampling by category')
    r.add_argument("--candidate-diversity", type=float, default=0.5, help='Fraction of Top-k reserved for diversity')
    r.add_argument("--candidate-include-json", action="append", default=[], help='JSON summary whose model names must be included in candidates')
    r.add_argument("--live-events", default=None, help='JSONL event stream for a live interface')
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser(
        "run-progressive",
        help='Query the database on temporal prefixes with caching.',
    )
    rp.add_argument("--zip", required=True, help='ObjectSensing RGB-D sequence (.zip)')
    rp.add_argument("--db", required=True, help='Preprocessed directory database')
    rp.add_argument("--out", default="scene_progressive.aln")
    rp.add_argument("--coverage", type=float, default=0.4, help='Coverage threshold')
    rp.add_argument("--jobs", type=int, default=0, help='Matching workers (0 = automatic)')
    rp.add_argument("--voxel", type=float, default=None, help='TSDF voxel size (m)')
    rp.add_argument("--truncation", type=float, default=None, help="Bande de troncature TSDF")
    rp.add_argument("--fusion-backend", choices=("auto", "cpu", "cuda"), default=None,
                    help='TSDF fusion engine (cuda requires CuPy/CUDA)')
    rp.add_argument(
        "--volume-layout", choices=("auto", "dense", "sparse"),
        default=None,
        help='TSDF volume layout (auto favors sparse on CPU)',
    )
    rp.add_argument("--depth-trunc", type=float, default=None)
    rp.add_argument("--iso-sampling", type=float, default=None)
    rp.add_argument("--max-surface-points", type=int, default=None)
    rp.add_argument(
        "--surface-field", choices=("raw", "smoothed"), default=None,
    )
    rp.add_argument("--surface-min-support", type=float, default=None)
    rp.add_argument(
        "--surface-extraction",
        choices=("zero_crossing", "hybrid", "near_zero"), default=None,
    )
    rp.add_argument("--surface-band-factor", type=float, default=None)
    rp.add_argument(
        "--surface-rescue-distance-factor", type=float, default=None,
    )
    rp.add_argument("--surface-rescue-min-weight", type=float, default=None)
    rp.add_argument("--gradient-smoothing-sigma", type=float, default=None)
    rp.add_argument("--depth-edge-threshold", type=float, default=None)
    rp.add_argument(
        "--depth-edge-background-weight", type=float, default=None,
    )
    rp.add_argument("--depth-edge-radius", type=int, default=None)
    rp.add_argument("--sparse-block-resolution", type=int, default=None)
    rp.add_argument("--sparse-block-count", type=int, default=None)
    rp.add_argument("--visibility-depth-stride", type=int, default=None)
    rp.add_argument("--frame-stride", type=int, default=None)
    rp.add_argument(
        "--min-frames", type=int, default=None,
        help='Reduce stride automatically to retain at least N frames (0 disables)',
    )
    rp.add_argument("--max-frames", type=int, default=None)
    rp.add_argument("--all-frames", action="store_true")
    rp.add_argument("--max-scan-keypoints", type=int, default=None)
    rp.add_argument("--max-model-keypoints", type=int, default=None)
    rp.add_argument("--spatial-keypoints", action="store_true")
    rp.add_argument("--spatial-keypoint-cell", type=float, default=None)
    _add_keypoint_detector_arguments(rp)
    _add_descriptor_matching_arguments(rp)
    rp.add_argument("--reverse-gate", type=float, default=None)
    rp.add_argument("--paper-verification", action="store_true")
    rp.add_argument("--surface-distance-weight", type=float, default=None)
    rp.add_argument("--verification-top-constellations", type=int, default=None)
    rp.add_argument("--max-registrations-per-model", type=int, default=None)
    rp.add_argument("--registration-min-center-distance", type=float, default=None)
    rp.add_argument("--registration-top-constellations", type=int, default=None)
    rp.add_argument("--multi-registration-synsets", nargs="*", default=None)
    rp.add_argument("--candidate-index", default=None)
    rp.add_argument("--candidate-top-k", type=int, default=0)
    rp.add_argument("--candidate-pool-k", type=int, default=0)
    rp.add_argument("--candidate-local-extra-k", type=int, default=None)
    rp.add_argument(
        "--candidate-descriptor-distance-threshold", type=float, default=None,
        help='Local reranking threshold; 0 reuses the RANSAC threshold',
    )
    rp.add_argument(
        "--candidate-top-per-group", type=int, default=None,
        help=(
            'Independent final budget per object group; 0 uses the legacy global Top-k'
        ),
    )
    rp.add_argument(
        "--group-pose-seed", dest="group_pose_seed_enabled",
        action="store_true", default=None,
    )
    rp.add_argument(
        "--no-group-pose-seed", dest="group_pose_seed_enabled",
        action="store_false",
    )
    rp.add_argument("--group-pose-seed-count", type=int, default=None)
    rp.add_argument("--exhaustive", action="store_true", help='Test the entire database at each general query')
    rp.add_argument("--candidate-diverse", action="store_true")
    rp.add_argument("--candidate-diversity", type=float, default=0.5)
    rp.add_argument("--candidate-include-json", action="append", default=[])
    rp.add_argument("--live-events", default=None, help='JSONL event stream for a live interface')
    rp.add_argument("--query-start-frames", type=int, default=20)
    rp.add_argument("--query-interval-frames", type=int, default=30)
    rp.add_argument(
        "--query-mode", choices=("single", "progressive"), default="single",
        help=(
            'single queries the final reconstruction once; progressive uses temporal prefixes'
        ),
    )
    rp.add_argument(
        "--stop-after",
        choices=("fusion", "keypoints", "query", "matching", "verification", "selection"),
        default="selection",
        help='Stop cleanly after this scientific stage (default: selection)',
    )
    rp.add_argument("--cache-registrations-per-model", type=int, default=3)
    rp.add_argument("--cache-top-constellations", type=int, default=50)
    rp.add_argument(
        "--resume-dir",
        default=None,
        help='Persistent matching-batch directory; restarting resumes completed models',
    )
    rp.add_argument(
        "--resume-batch-size",
        type=int,
        default=20,
        help='Model count between atomic resume saves',
    )
    rp.add_argument(
        "--stage-cache-dir",
        default=None,
        help='Versioned cache of fused SceneScan objects per checkpoint',
    )
    rp.add_argument(
        "--final-reverse-gate",
        type=float,
        default=None,
        help='Revalidate progressive hypotheses on the final scan with minimum reverse coverage',
    )
    rp.add_argument(
        "--final-max-per-category",
        nargs="*",
        default=[],
        help='Final selection limits by category, e.g. chair=2 table=2 couch=1',
    )
    rp.add_argument(
        "--final-global-overlap",
        type=float,
        default=0.58,
        help='Maximum explained-point overlap between categories at final selection',
    )
    rp.add_argument(
        "--final-min-symmetric-score",
        type=float,
        default=0.0,
        help='Minimum forward/reverse harmonic score at final selection',
    )
    rp.add_argument(
        "--final-cross-category-center-distance",
        type=float,
        default=0.0,
        help='Horizontal distance below which categories conflict when sharing the same support',
    )
    rp.add_argument(
        "--final-cross-category-overlap",
        type=float,
        default=1.0,
        help='Minimum global overlap triggering cross-category collision rejection',
    )
    rp.set_defaults(func=cmd_run_progressive)

    s = sub.add_parser("split-db", help='Convert db.pkl to a directory (one file per model) for parallel matching.')
    s.add_argument("--db", required=True, help="db.pkl monolithique existant")
    s.add_argument("--out", required=True, help='Output database directory')
    s.set_defaults(func=cmd_split_db)

    ci = sub.add_parser("build-candidate-index", help='Build the compact preselection index of a db_dir database.')
    ci.add_argument("--db", required=True, help='db_dir directory')
    ci.add_argument("--out", default=None, help='Output .npz file (default: <db>/candidate_index.npz)')
    ci.set_defaults(func=cmd_build_candidate_index)

    sk = sub.add_parser(
        "sweep-keypoints",
        help='Compare keypoint settings on cached scenes.',
    )
    sk.add_argument(
        "--source", action="append", required=True,
        help='Scene as name=cache.pkl (repeatable)',
    )
    sk.add_argument(
        "--grid", action="append", default=[],
        help='Grid axis as parameter=v1,v2,... (repeatable)',
    )
    sk.add_argument(
        "--set", dest="settings", action="append", default=[],
        help='Fixed setting as parameter=value (repeatable)',
    )
    sk.add_argument(
        "--settings-json", default=None,
        help='Debug or Run Console JSON settings used as a baseline',
    )
    sk.add_argument("--out", required=True, help="Rapport JSON reprenable")
    sk.add_argument(
        "--coverage-cell", type=float, default=0.35,
        help='Cell size for spatial coverage measurement (m)',
    )
    sk.add_argument(
        "--descriptor-db", default=None,
        help='db_dir used to measure descriptor utility',
    )
    sk.add_argument(
        "--descriptor-candidate-index", default=None,
        help='Diagnostic candidate index (default: <descriptor-db>/candidate_index.npz)',
    )
    sk.add_argument("--utility-candidate-models", type=int, default=24)
    sk.add_argument("--utility-model-features", type=int, default=24)
    sk.add_argument("--utility-scan-features", type=int, default=120)
    sk.add_argument("--utility-rotations", type=int, default=8)
    sk.add_argument(
        "--utility-candidate-pool-size", type=int, default=500,
        help='Pool size for annotated recall measurement',
    )
    sk.add_argument(
        "--target-annotations", default=None,
        help=(
            'JSON {scene: {object: [acceptable models]}} for Recall@pool; no ground truth is inferred when absent'
        ),
    )
    sk.add_argument("--resume", dest="resume", action="store_true")
    sk.add_argument("--no-resume", dest="resume", action="store_false")
    sk.set_defaults(resume=True, func=cmd_sweep_keypoints)

    v = sub.add_parser(
        "visualize",
        help='Build a colored PLY and HTML timeline viewer.',
    )
    v.add_argument("--json", required=True, help='JSON summary produced by run')
    v.add_argument("--scan", default=None, help='<out>_scan.ply saved by run (optional)')
    v.add_argument("--out", default="scene_viz.ply")
    v.add_argument(
        "--progress",
        default=None,
        help='*_progress.json manifest (inferred from the summary by default)',
    )
    v.add_argument(
        "--html",
        default=None,
        help='Output HTML viewer (default: same stem as --out)',
    )
    v.set_defaults(func=cmd_visualize)

    args = p.parse_args()
    global _LIVE_EVENTS
    _LIVE_EVENTS = LiveEventWriter(getattr(args, "live_events", None))
    if _LIVE_EVENTS.enabled:
        _LIVE_EVENTS.emit("process_started", {"command": args.cmd})
    args.func(args)


if __name__ == "__main__":
    main()
