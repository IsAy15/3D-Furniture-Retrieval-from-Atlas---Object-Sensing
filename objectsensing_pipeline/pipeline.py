'\nObjectSensing orchestration: RGB-D fusion, surface geometry, keypoints, occupancy descriptors and primitives; candidate retrieval from a prepared CAD database; matching, 1-Point RANSAC, geometric verification, global selection and ALN/JSON export.\n'

from __future__ import annotations

import hashlib
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from config import PipelineConfig
from constellations import Constellation, one_point_ransac
from database import ModelDatabase, object_surface_cloud
from descriptors import descriptor_distance
from geometry import PointCloud, detect_gravity_rotation, estimate_ground_plane
from keypoints import detect_keypoints
from keypoint_quality import filter_scene_keypoints
from matching import Correspondence, build_scan_features, match_features
from transforms import GroundTransform
from sdf_fusion import fuse_sequence
from verify import (
    Registration,
    symmetric_coverage_score,
    verify_model,
    verify_model_candidates,
)
import db_store
from live_events import emit_progress_event, has_progress_events
from wall_keypoints import apply_wall_keypoint_filter


@dataclass
class SceneScan:
    cloud: PointCloud
    volume: object
    ground: float
    up: np.ndarray
    alignment_rotation: Optional[np.ndarray] = None
    fusion_trace: Optional[dict] = None


class RetrievalStageComplete(Exception):
    """Return a usable partial result at an explicit scientific boundary."""

    def __init__(self, stage, trace=None, registrations=None):
        super().__init__(stage)
        self.stage = str(stage)
        self.trace = trace or {}
        self.registrations = list(registrations or [])


def _scene_visibility_sampler(scene):
    volume = getattr(scene, "volume", None)
    if volume is None:
        return None
    sampler = getattr(volume, "sample_visibility_details", None)
    if sampler is not None:
        return sampler
    legacy = getattr(volume, "sample_visibility", None)
    if legacy is None:
        return None

    def sample_details(positions):
        positions = np.asarray(positions, float)
        return legacy(positions), np.zeros(len(positions), bool)

    return sample_details


class _AlignedVolume:
    'Wrap a TSDFVolume for sampling in a rotated frame. Map aligned coordinates back to the original volume using q_original = q_aligned @ R.'

    def __init__(self, volume, R):
        self.volume = volume
        self.R = np.asarray(R, float)

    def sample(self, positions):
        return self.volume.sample(np.asarray(positions, float) @ self.R)

    def sample_gradient(self, positions):
        grad = self.volume.sample_gradient(np.asarray(positions, float) @ self.R)
        return grad @ self.R.T

    def sample_visibility(self, positions):
        return self.volume.sample_visibility(np.asarray(positions, float) @ self.R)

    def sample_visibility_details(self, positions):
        aligned = np.asarray(positions, float) @ self.R
        if hasattr(self.volume, "sample_visibility_details"):
            return self.volume.sample_visibility_details(aligned)
        labels = self.volume.sample_visibility(aligned)
        return labels, np.zeros(len(aligned), bool)


def scan_from_zip(zip_path, cfg: PipelineConfig, progress=None,
                  alignment_rotation=None, fusion_trace=None) -> SceneScan:
    'Fuse an RGB-D sequence, align to Y-up and prepare SDF normals and curvature. A shared alignment_rotation keeps temporal prefixes in a common coordinate system.\n    '
    cloud, vol = fuse_sequence(
        zip_path, cfg.sdf, progress=progress, trace=fusion_trace,
    )
    # Gravity alignment : sans ça, le scan est dans le repère des poses (souvent
    # incliné/Y-bas) alors que the models sont Y-haut -> meubles mal posés.
    gravity_hint = getattr(vol, "gravity_up_hint", None)
    R = (detect_gravity_rotation(
             cloud.points, up_hint=gravity_hint,
         ) if alignment_rotation is None
         else np.asarray(alignment_rotation, float))
    pts = cloud.points @ R.T
    nrm = cloud.normals @ R.T
    curv = cloud.curvature
    confidence = getattr(cloud, "confidence", None)
    cloud = PointCloud(pts, nrm, curv, confidence)
    vol = _AlignedVolume(vol, R)
    if fusion_trace is not None:
        fusion_trace["alignment_rotation"] = np.round(R, 6).tolist()
        for frame in fusion_trace.get("frames", []):
            frame_points = np.asarray(frame.get("points", []), float)
            if len(frame_points):
                frame["points"] = np.round(frame_points @ R.T, 3).tolist()
            camera_position = np.asarray(frame.get("camera_position", []), float)
            if camera_position.shape == (3,):
                frame["camera_position"] = np.round(camera_position @ R.T, 4).tolist()
    up = np.array([0.0, 1.0, 0.0])
    _, ground = estimate_ground_plane(cloud.points, up_hint=up)
    if fusion_trace is not None:
        fusion_trace["ground"] = round(float(ground), 4)
    if progress:
        progress('gravity-aligned scan (Y-up)')
    if has_progress_events(progress):
        preview = cloud.points
        if len(preview) > 24000:
            preview = preview[np.linspace(0, len(preview) - 1, 24000).astype(int)]
        emit_progress_event(progress, "scene_surface", {
            "points": np.round(preview, 3).tolist(),
            "point_count": int(cloud.size),
            "ground": round(float(ground), 4),
            "alignment_rotation": np.round(R, 6).tolist(),
        })
    return SceneScan(cloud, vol, ground, up, R, fusion_trace)


def _trace_registration(reg):
    if reg is None:
        return None
    trace = {
        "model": reg.model_name,
        "coverage": round(float(reg.coverage), 4),
        "cov_reverse": round(float(getattr(reg, "cov_reverse", 0.0)), 4),
        "mean_surface_dist_m": round(float(reg.mean_surface_dist), 4),
        "quality": round(float(reg.quality), 2),
        "theta_deg": round(float(reg.transform.theta) * 57.2958, 2),
        "scale": round(float(reg.transform.scale), 4),
        "translation": [round(float(x), 4) for x in reg.transform.t],
        "balanced_score": round(_symmetric_score(
            float(reg.coverage), _ranking_reverse_coverage(reg),
        ), 4),
    }
    for attr, key in (
        ("_pose_seed_fallback", "pose_seed_fallback"),
        ("_geometric_cov_reverse", "geometric_cov_reverse"),
        ("_normal_compatible_coverage", "normal_compatible_coverage"),
        ("_query_group_reverse", "group_cov_reverse"),
        ("_query_group_surface_points", "group_surface_points"),
        ("_visibility_support", "visibility_support"),
        ("_visibility_known_fraction", "visibility_known_fraction"),
        ("_visibility_free_fraction", "visibility_free_fraction"),
        ("_visibility_known_points", "visibility_known_points"),
        ("_visibility_used", "visibility_used"),
        ("_structure_height", "structure_height"),
        ("_thickness", "thickness"),
        ("_model_zone_fraction", "model_zone_fraction"),
        ("_supported_model_zones", "supported_model_zones"),
        ("_model_zone_count", "model_zone_count"),
        ("_support_extent_ratio", "support_extent_ratio"),
    ):
        if getattr(reg, attr, None) is not None:
            value = getattr(reg, attr)
            trace[key] = value if isinstance(value, (bool, int)) else round(float(value), 4)
    return trace


def _trace_matching_geometry(corres, constellations, model_features, scan_features):
    correspondence_preview = []
    for corr in sorted(corres, key=lambda item: float(item.desc_dist))[:12]:
        correspondence_preview.append({
            "model_keypoint": int(corr.model_idx),
            "scan_keypoint": int(corr.scan_idx),
            "descriptor_distance": round(float(corr.desc_dist), 4),
            "theta_deg": round(float(corr.theta) * 57.2958, 2),
            "scale": round(float(corr.scale), 4),
            "translation": [round(float(x), 4) for x in corr.transform.t],
            "model_position": [round(float(x), 4) for x in model_features[corr.model_idx].position],
            "scan_position": [round(float(x), 4) for x in scan_features[corr.scan_idx].position],
        })
    constellation_preview = []
    for constellation in constellations[:8]:
        constellation_preview.append({
            "quality": round(float(constellation.quality), 2),
            "inliers": len(constellation.inliers),
            "theta_deg": round(float(constellation.transform.theta) * 57.2958, 2),
            "scale": round(float(constellation.transform.scale), 4),
            "translation": [round(float(x), 4) for x in constellation.transform.t],
            "inlier_pairs": [
                {
                    "model_keypoint": int(corr.model_idx),
                    "scan_keypoint": int(corr.scan_idx),
                    "descriptor_distance": round(float(corr.desc_dist), 4),
                    "model_position": [
                        round(float(x), 4)
                        for x in model_features[corr.model_idx].position
                    ],
                    "scan_position": [
                        round(float(x), 4)
                        for x in scan_features[corr.scan_idx].position
                    ],
                }
                for corr in constellation.inliers[:16]
            ],
        })
    constellation_scene_support = []
    for constellation in constellations:
        scan_keypoints = list(dict.fromkeys(
            int(corr.scan_idx) for corr in constellation.inliers
        ))
        constellation_scene_support.append({
            "quality": round(float(constellation.quality), 2),
            "scan_keypoints": scan_keypoints,
        })
    return (correspondence_preview, constellation_preview,
            constellation_scene_support)


def _trace_model_preview(model, max_points=240):
    points = np.asarray(model.cloud.points, float)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).astype(int)]
    return np.round(points, 3).tolist()


def _group_pose_constellations(model_features, scan_features, cfg, up):
    """Generate coarse object-group poses when local triplets are unavailable."""
    model_points = np.asarray([
        feature.position for feature in model_features
    ], dtype=float)
    scan_points = np.asarray([
        feature.position for feature in scan_features
    ], dtype=float)
    if len(model_points) < 3 or len(scan_points) < 3:
        return []
    up = np.asarray(up, dtype=float)
    up /= max(np.linalg.norm(up), 1e-9)
    model_height = float(np.ptp(model_points @ up))
    scan_height = float(np.ptp(scan_points @ up))
    if model_height <= 1e-6 or scan_height <= 1e-6:
        return []
    scale = float(np.clip(
        scan_height / model_height,
        cfg.matching.scale_min, cfg.matching.scale_max,
    ))
    model_center = np.median(model_points, axis=0)
    scan_center = np.median(scan_points, axis=0)
    model_floor = float(np.min(model_points @ up))
    scan_floor = float(np.min(scan_points @ up))
    seeds = []
    scan_tree = cKDTree(scan_points)
    rotations = max(
        int(getattr(cfg.matching, "n_uniform_rotations", 36)),
        int(getattr(cfg.matching, "group_pose_seed_count", 6)),
    )
    for theta in np.linspace(0.0, 2.0 * np.pi, rotations, endpoint=False):
        transform = GroundTransform(theta, scale, np.zeros(3), up)
        rotated_center = transform.apply(model_center[None, :])[0]
        translation = scan_center - rotated_center
        translated_floor = scale * model_floor + float(translation @ up)
        translation += (scan_floor - translated_floor) * up
        transform.t = translation
        transformed = transform.apply(model_points)
        tree = cKDTree(transformed)
        forward = float(np.mean(np.minimum(
            tree.query(scan_points, k=1)[0], 0.5,
        )))
        reverse = float(np.mean(np.minimum(
            scan_tree.query(transformed, k=1)[0], 0.5,
        )))
        shape_quality = float(np.exp(
            -(0.72 * forward + 0.28 * reverse) / 0.12
        ))

        # Une pose de groupe doit exposer son support local au lieu de rester
        # une constellation vide. L'association gloutonne est bijective : un
        # keypoint de chaque côté ne peut expliquer qu'une seule paire.
        distances = np.linalg.norm(
            transformed[:, None, :] - scan_points[None, :, :], axis=2,
        )
        used_model = set()
        used_scan = set()
        inliers = []
        support_radius = float(cfg.ransac.geom_inlier)
        for flat_index in np.argsort(distances, axis=None):
            model_index, scan_index = np.unravel_index(
                int(flat_index), distances.shape,
            )
            distance = float(distances[model_index, scan_index])
            if distance >= support_radius:
                break
            if model_index in used_model or scan_index in used_scan:
                continue
            model_descriptor = getattr(
                model_features[model_index], "descriptor", None,
            )
            scan_descriptor = getattr(
                scan_features[scan_index], "descriptor", None,
            )
            descriptor_dist = (
                descriptor_distance(
                    model_descriptor, scan_descriptor, cfg.descriptor,
                    -theta, up,
                )
                if model_descriptor is not None and scan_descriptor is not None
                else float("inf")
            )
            inliers.append(Correspondence(
                int(model_index), int(scan_index), float(theta), scale,
                float(descriptor_dist), transform,
            ))
            used_model.add(int(model_index))
            used_scan.add(int(scan_index))

        support_ratio = len(inliers) / max(1, min(
            len(model_points), len(scan_points),
        ))
        quality = shape_quality * (0.65 + 0.35 * support_ratio)
        seeds.append(Constellation(transform, inliers, quality))
    seeds.sort(key=lambda constellation: -constellation.quality)
    return seeds[:max(1, int(getattr(
        cfg.matching, "group_pose_seed_count", 6,
    )))]


def prepare_model_matching(model, sfeats, cfg, up,
                 trace=None, allow_group_pose_seed=False):
    """Compute correspondences and pose seeds without running ICP."""
    if trace is not None:
        trace.update({
            "model": model.name,
            "category": model.name.split("_", 1)[0],
            "synset": str(getattr(model, "synset", "")),
            "model_keypoints": len(model.features),
            "model_keypoint_preview": [np.asarray(feature.position, float).tolist() for feature in model.features],
            "model_preview": _trace_model_preview(model),
        })
    started = time.monotonic()
    matching_trace = {} if trace is not None else None
    if matching_trace is None:
        corres = match_features(model.features, sfeats, cfg, up)
    else:
        corres = match_features(
            model.features, sfeats, cfg, up, diagnostics=matching_trace,
        )
    if trace is not None:
        trace["correspondence_filter"] = matching_trace
        trace["correspondences"] = len(corres)
        trace["correspondence_model_keypoints"] = len({item.model_idx for item in corres})
        trace["correspondence_scan_keypoints"] = len({item.scan_idx for item in corres})
    if not corres:
        if trace is not None:
            trace.update(status="rejected_no_correspondence", elapsed_ms=round((time.monotonic() - started) * 1000))
        return []
    ransac_trace = {} if trace is not None else None
    cons = one_point_ransac(
        corres, model.features, sfeats, cfg, up,
        diagnostics=ransac_trace,
    )
    pose_seed_fallback = False
    if (
        not cons and allow_group_pose_seed
        and getattr(cfg.matching, "group_pose_seed_enabled", True)
    ):
        cons = _group_pose_constellations(
            model.features, sfeats, cfg, up,
        )
        pose_seed_fallback = bool(cons)
    if pose_seed_fallback:
        for constellation in cons:
            constellation._pose_seed_fallback = True
    if trace is not None:
        corr_preview, cons_preview, cons_scene_support = _trace_matching_geometry(
            corres, cons, model.features, sfeats,
        )
        trace.update({
            "constellations": len(cons),
            "correspondence_preview": corr_preview,
            "constellation_preview": cons_preview,
            "constellation_scene_support": cons_scene_support,
            "ransac": ransac_trace,
            "group_pose_seed_fallback": pose_seed_fallback,
        })
    if not cons:
        if trace is not None:
            trace.update(status="rejected_no_constellation", elapsed_ms=round((time.monotonic() - started) * 1000))
        return []
    if trace is not None:
        trace["status"] = "matched"
    return cons


def _match_model(model, sfeats, scene_cloud, cfg, up, coverage_threshold,
                 trace=None, allow_group_pose_seed=False, prepared_constellations=None):
    started = time.monotonic()
    cons = (prepare_model_matching(model, sfeats, cfg, up, trace,
            allow_group_pose_seed) if prepared_constellations is None
            else prepared_constellations)
    if not cons:
        return None
    verification_trace = {} if trace is not None else None
    verify_kwargs = dict(
        threshold=cfg.keypoint.neighbor_radius,
        coverage_threshold=coverage_threshold,
        group_points=([feature.position for feature in sfeats] if allow_group_pose_seed else None),
        up=up,
        top=cfg.matching.verification_top_constellations,
        reverse_gate=cfg.matching.reverse_gate,
        min_structure=cfg.matching.min_structure_height,
        min_thickness=cfg.matching.min_thickness,
        model_zone_grid=getattr(cfg.matching, "model_zone_grid", 2),
        model_zone_min_support=getattr(cfg.matching, "model_zone_min_support", 0.15),
        min_model_zone_fraction=getattr(cfg.matching, "min_model_zone_fraction", 0.35),
        min_support_extent_ratio=getattr(cfg.matching, "min_support_extent_ratio", 0.20),
        support_locality_weight=getattr(cfg.matching, "support_locality_weight", 0.10),
        surface_distance_weight=cfg.matching.surface_distance_weight,
        scale_bounds=(cfg.matching.scale_min, cfg.matching.scale_max),
    )
    if verification_trace is not None:
        verify_kwargs["diagnostics"] = verification_trace
    reg = verify_model(model.name, object_surface_cloud(model), scene_cloud, cons, **verify_kwargs)
    if reg is not None:
        reg._pose_seed_fallback = bool(getattr(cons[0], "_pose_seed_fallback", False))
    if trace is not None:
        trace["verification"] = verification_trace
        trace["registration"] = _trace_registration(reg)
        trace["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    if reg is not None and reg.coverage >= coverage_threshold:
        if trace is not None:
            trace["status"] = "candidate"
        return reg
    if trace is not None:
        trace["status"] = "rejected_coverage" if reg is not None else "rejected_verification"
        trace["coverage_threshold"] = float(coverage_threshold)
    return None


def _match_model_candidates(model, sfeats, scene_cloud, cfg, up, coverage_threshold,
                            trace=None, allow_group_pose_seed=False, prepared_constellations=None):
    multi_synsets = tuple(getattr(cfg.matching, "multi_registration_synsets", ()) or ())
    if cfg.matching.max_registrations_per_model <= 1:
        reg = _match_model(
            model, sfeats, scene_cloud, cfg, up, coverage_threshold,
            trace=trace, allow_group_pose_seed=allow_group_pose_seed,
            prepared_constellations=prepared_constellations,
        )
        return [] if reg is None else [reg]
    if multi_synsets and model.synset not in multi_synsets:
        reg = _match_model(
            model, sfeats, scene_cloud, cfg, up, coverage_threshold,
            trace=trace, allow_group_pose_seed=allow_group_pose_seed,
            prepared_constellations=prepared_constellations,
        )
        return [] if reg is None else [reg]

    started = time.monotonic()
    cons = (prepare_model_matching(model, sfeats, cfg, up, trace,
            allow_group_pose_seed) if prepared_constellations is None
            else prepared_constellations)
    if not cons:
        return []
    verification_trace = {} if trace is not None else None
    verify_kwargs = dict(
        threshold=cfg.keypoint.neighbor_radius,
        coverage_threshold=coverage_threshold,
        group_points=([feature.position for feature in sfeats] if allow_group_pose_seed else None),
        up=up,
        reverse_gate=cfg.matching.reverse_gate,
        min_structure=cfg.matching.min_structure_height,
        min_thickness=cfg.matching.min_thickness,
        model_zone_grid=getattr(cfg.matching, "model_zone_grid", 2),
        model_zone_min_support=getattr(cfg.matching, "model_zone_min_support", 0.15),
        min_model_zone_fraction=getattr(cfg.matching, "min_model_zone_fraction", 0.35),
        min_support_extent_ratio=getattr(cfg.matching, "min_support_extent_ratio", 0.20),
        support_locality_weight=getattr(cfg.matching, "support_locality_weight", 0.10),
        surface_distance_weight=cfg.matching.surface_distance_weight,
        top=cfg.matching.registration_top_constellations,
        max_results=cfg.matching.max_registrations_per_model,
        min_center_distance=cfg.matching.registration_min_center_distance,
        scale_bounds=(cfg.matching.scale_min, cfg.matching.scale_max),
    )
    if verification_trace is not None:
        verify_kwargs["diagnostics"] = verification_trace
    regs = verify_model_candidates(
        model.name, object_surface_cloud(model), scene_cloud, cons, **verify_kwargs,
    )
    for reg in regs:
        reg._pose_seed_fallback = bool(getattr(cons[0], "_pose_seed_fallback", False))
    kept = [r for r in regs if r.coverage >= coverage_threshold]
    if trace is not None:
        trace.update({
            "verification": verification_trace,
            "registrations": [_trace_registration(reg) for reg in regs],
            "status": "candidate" if kept else ("rejected_coverage" if regs else "rejected_verification"),
            "coverage_threshold": float(coverage_threshold),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        })
    return kept


# --- état partagé pour the workers multiprocessing (Windows = spawn) ---
_WORKER: dict = {}


def _init_worker(db_path, sfeats, scene_pts, cfg, up, thr):
    import pickle
    from parallelism import configure_process_worker

    configure_process_worker()
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    _WORKER.update(db=db, sfeats=sfeats, scene=PointCloud(np.asarray(scene_pts)),
                   cfg=cfg, up=np.asarray(up, float), thr=thr)


def _match_index(mi):
    w = _WORKER
    reg = _match_model(w["db"].models[mi], w["sfeats"], w["scene"], w["cfg"], w["up"], w["thr"])
    if reg is not None:
        reg._model_index = mi
    return reg


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _progress_eta(done: int, total: int, start_time: float) -> str:
    if done <= 0 or total <= 0:
        return "ETA inconnue"
    elapsed = max(0.0, time.monotonic() - start_time)
    rate = done / elapsed if elapsed > 0 else 0.0
    if rate <= 0:
        return "ETA inconnue"
    remaining = max(0, total - done) / rate
    return (
        f"reste env. {_format_duration(remaining)}, "
        f"{rate:.2f} modele/s"
    )


def _matching_code_fingerprint() -> str:
    """Invalidate registrations cached before a scientific implementation change."""
    digest = hashlib.sha256()
    for name in (
        "pipeline.py", "matching.py", "descriptors.py", "constellations.py",
        "transforms.py", "geometry.py", "verify.py", "database.py",
        "candidate_index.py", "local_preselection.py", "progressive.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _resume_signature(tasks, scene, cfg, coverage_threshold) -> str:
    points = np.asarray(scene.cloud.points)
    if len(points):
        scene_summary = (
            len(points),
            np.round(points.min(0), 5).tolist(),
            np.round(points.max(0), 5).tolist(),
            np.round(points[::max(1, len(points) // 1024)].sum(0), 5).tolist(),
        )
    else:
        scene_summary = (0, [], [], [])
    payload = (
        2,
        _matching_code_fingerprint(),
        [(int(index), os.path.basename(path)) for index, path in tasks],
        scene_summary,
        float(coverage_threshold),
        cfg,
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=4)).hexdigest()


def _resume_path(resume_dir, resume_key) -> Optional[Path]:
    if not resume_dir or not resume_key:
        return None
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(resume_key)).strip("._")
    if not safe_key:
        raise ValueError('resume_key is empty after normalization')
    path = Path(resume_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{safe_key}.pkl"


def _load_resume_state(path: Optional[Path], signature: str, tasks):
    if path is None or not path.exists():
        return {"version": 1, "signature": signature, "results": {}}
    with path.open("rb") as handle:
        state = pickle.load(handle)
    if state.get("version") != 1 or state.get("signature") != signature:
        raise ValueError(
            f'Checkpoint incompatible: {path}. Use a new resume directory or remove this checkpoint before changing parameters.'
        )
    expected = {int(index) for index, _ in tasks}
    results = {
        int(index): result for index, result in state.get("results", {}).items()
        if int(index) in expected
    }
    return {"version": 1, "signature": signature, "results": results}


def _save_resume_state(path: Optional[Path], state) -> None:
    if path is None:
        return
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _extend_registration_results(candidates, result):
    if isinstance(result, dict) and "registrations" in result:
        result = result["registrations"]
    if isinstance(result, list):
        candidates.extend(result)
    elif result is not None:
        candidates.append(result)


def _worker_traces(result):
    if isinstance(result, dict) and "registrations" in result:
        traces = result.get("traces")
        if traces is not None:
            return [trace for trace in traces if trace is not None]
        trace = result.get("trace")
        return [] if trace is None else [trace]
    return []


def _worker_trace(result):
    traces = _worker_traces(result)
    return traces[0] if traces else None


def _tag_group_result(registrations, group_id, scan_features=None):
    values = registrations if isinstance(registrations, list) else [registrations]
    for registration in values:
        if registration is not None:
            registration._query_group_id = int(group_id)
            points = [feature.position for feature in (scan_features or [])
                      if hasattr(feature, "position")]
            if len(points) >= 3:
                registration._query_group_points = np.asarray(points, float)


def _trace_group(model_trace, group_id):
    if model_trace is not None:
        model_trace["query_group_id"] = int(group_id)


def _worker_feature_groups(model_index):
    groups = _WORKER.get("candidate_feature_groups", {}).get(int(model_index))
    return groups or [(None, _WORKER["sfeats"])]


def retrieve(scene: SceneScan, db: ModelDatabase, cfg: PipelineConfig,
             coverage_threshold: float = 0.4, progress=None,
             n_jobs: int = 1, db_path=None) -> List[Registration]:
    'Retrieve and register database models, then apply non-maximum object selection. Parallel workers load model data from disk to avoid copying the whole database.'
    detected_kps = detect_keypoints(
        scene.cloud, cfg.keypoint,
        visibility_sampler=_scene_visibility_sampler(scene),
    )
    wall_filter = apply_wall_keypoint_filter(
        scene.cloud, detected_kps, cfg.keypoint, scene.up,
    )
    wall_kps = wall_filter.keypoints
    quality_filter = filter_scene_keypoints(scene, wall_kps, cfg.keypoint)
    kps = quality_filter.keypoints
    feature_trace = {}
    sfeats = build_scan_features(
        scene.cloud, scene.volume, kps, cfg, scene.up, scene.ground,
        trace=feature_trace,
    )
    if progress:
        progress(
            f'scan: {scene.cloud.size} pts, {detected_kps.size} keypoints (walls rejected: {int(wall_filter.rejected_mask.sum())}, plans: {quality_filter.planar_rejected}, trous: {quality_filter.hole_rejected}, instables: {quality_filter.repeatability_rejected}, faibles: {quality_filter.score_rejected}, doublons: {quality_filter.nms_rejected}, retained for matching: {len(sfeats)})'
        )

    candidates: List[Registration] = []
    n = len(db.models)
    matching_start = time.monotonic()

    if n_jobs and n_jobs > 1 and db_path:
        try:
            from concurrent.futures import ProcessPoolExecutor
            done = 0
            with ProcessPoolExecutor(
                max_workers=n_jobs, initializer=_init_worker,
                initargs=(str(db_path), sfeats, scene.cloud.points, cfg,
                          list(map(float, scene.up)), coverage_threshold),
            ) as ex:
                for reg in ex.map(_match_index, range(n), chunksize=4):
                    done += 1
                    if reg is not None:
                        candidates.append(reg)
                    if progress and done % 20 == 0:
                        progress(
                            f'modeles testes: {done}/{n} (candidates: {len(candidates)}, {_progress_eta(done, n, matching_start)})'
                        )
            return _non_max_select(candidates, db, scene, cfg)
        except Exception as exc:                # repli séquentiel si le pool échoue
            if progress:
                progress(f"matching parallele indisponible ({exc}) -> sequentiel")
            candidates = []

    for mi, model in enumerate(db.models):
        reg = _match_model(model, sfeats, scene.cloud, cfg, scene.up, coverage_threshold)
        if reg is not None:
            reg._model_index = mi
            candidates.append(reg)
        if progress and (mi + 1) % 20 == 0:
            done = mi + 1
            progress(
                f'modeles testes: {done}/{n} (candidates: {len(candidates)}, {_progress_eta(done, n, matching_start)})'
            )

    return _non_max_select(candidates, db, scene, cfg)


def _scene_footprint(reg: Registration, db: ModelDatabase, scene: SceneScan):
    model = db.models[getattr(reg, "_model_index", 0)]
    return reg.transform.apply(object_surface_cloud(model).points)


def _symmetric_score(coverage: float, cov_reverse: float) -> float:
    'Harmonic mean of forward and reverse geometric coverage.'
    return symmetric_coverage_score(coverage, cov_reverse)


def _ranking_reverse_coverage(reg: Registration) -> float:
    """Use scan-to-model geometry for ranking; visibility remains a gate.

    Occupied/known model points measure compatibility with a volume, not the
    fraction of scan surface explained. Preserve legacy behavior when the
    separate geometric measurement is unavailable.
    """
    geometry = float(getattr(reg, "_geometric_cov_reverse",
                             getattr(reg, "cov_reverse", 0.0)))
    return min(geometry, float(getattr(reg, "_query_group_reverse", geometry)))


def _registration_rank(reg: Registration):
    coverage = float(reg.coverage)
    cov_reverse = _ranking_reverse_coverage(reg)
    return (
        -_symmetric_score(coverage, cov_reverse),
        -coverage,
        -cov_reverse,
        float(getattr(reg, "mean_surface_dist", 0.0)),
    )


def _non_max_select(cands: List[Registration], db: ModelDatabase, scene: SceneScan,
                    cfg: PipelineConfig, overlap_thresh: float = 0.5) -> List[Registration]:
    'Greedy selection: rank quality, mark explained scan regions and reject excessive overlap.'
    from scipy.spatial import cKDTree
    cands = sorted(cands, key=_registration_rank)
    selected: List[Registration] = []
    explained = np.zeros(scene.cloud.size, bool)
    tree = scene.cloud.tree
    for reg in cands:
        fp = _scene_footprint(reg, db, scene)
        lists = tree.query_ball_point(fp[::5], cfg.keypoint.neighbor_radius)
        flat = [j for sub in lists for j in sub]
        if not flat:
            continue
        idx = np.unique(np.asarray(flat, int))
        overlap = explained[idx].mean() if len(idx) else 1.0
        if overlap > overlap_thresh:
            continue
        explained[idx] = True
        selected.append(reg)
    return selected


# ---------------------------------------------------------------------------
# Matching parallèle à mémoire constante : base = un fichier par modèle.
# Chaque worker charge UN modèle à la fois (et non toute la base) -> pas de
# duplication mémoire. État partagé (scan, cfg) injecté une fois par worker.
# ---------------------------------------------------------------------------
def _init_files(sfeats, scene_pts, cfg, up, thr, capture_trace=False,
                candidate_feature_groups=None):
    from parallelism import configure_process_worker

    configure_process_worker()
    _WORKER.update(sfeats=sfeats, scene=PointCloud(np.asarray(scene_pts)),
                   cfg=cfg, up=np.asarray(up, float), thr=thr,
                   capture_trace=bool(capture_trace),
                   candidate_feature_groups=(candidate_feature_groups or {}))


def _footprint(reg, model, max_pts=2000):
    fp = reg.transform.apply(object_surface_cloud(model).points)
    if len(fp) > max_pts:
        fp = fp[np.linspace(0, len(fp) - 1, max_pts).astype(int)]
    return fp


def _work_file(task):
    mi, path = task
    model = db_store.load_model(path)
    w = _WORKER
    registrations = []
    traces = []
    for group_id, scan_features in _worker_feature_groups(mi):
        model_trace = {} if w.get("capture_trace") else None
        reg = _match_model(
            model, scan_features, w["scene"], w["cfg"], w["up"], w["thr"],
            trace=model_trace, allow_group_pose_seed=(group_id is not None),
        )
        if group_id is not None:
            _trace_group(model_trace, group_id)
            _tag_group_result(reg, group_id, scan_features)
        if model_trace is not None:
            traces.append(model_trace)
        if reg is not None:
            reg._footprint = _footprint(reg, model)
            reg._mesh_path = model.mesh_path
            reg._model_index = mi
            registrations.append(reg)
    if w.get("capture_trace"):
        return {"registrations": registrations, "traces": traces}
    return registrations


def _work_file_multi(task):
    mi, path = task
    model = db_store.load_model(path)
    w = _WORKER
    out = []
    traces = []
    for group_id, scan_features in _worker_feature_groups(mi):
        model_trace = {} if w.get("capture_trace") else None
        regs = _match_model_candidates(
            model, scan_features, w["scene"], w["cfg"], w["up"], w["thr"],
            trace=model_trace, allow_group_pose_seed=(group_id is not None),
        )
        if group_id is not None:
            _trace_group(model_trace, group_id)
            _tag_group_result(regs, group_id, scan_features)
        if model_trace is not None:
            traces.append(model_trace)
        for reg in regs:
            reg._footprint = _footprint(reg, model)
            reg._mesh_path = model.mesh_path
            reg._model_index = mi
            out.append(reg)
    return {"registrations": out, "traces": traces} if w.get("capture_trace") else out


def _hbbox(fp, up):
    'Bounding box in the horizontal plane perpendicular to up.'
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    u = fp @ e1; v = fp @ e2
    return np.array([u.min(), v.min()]), np.array([u.max(), v.max()])


def _hbbox_iou(b1, b2):
    (lo1, hi1), (lo2, hi2) = b1, b2
    inter = np.maximum(0.0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2)).prod()
    a1 = (hi1 - lo1).prod(); a2 = (hi2 - lo2).prod()
    union = a1 + a2 - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _hbbox_center(box):
    lo, hi = box
    return 0.5 * (lo + hi)


def _category_of(reg: Registration) -> str:
    return str(reg.model_name).split("_", 1)[0]


def _nms_diagnostic(reg, status, reason, **extra):
    entry = {
        "registration": reg,
        "status": status,
        "reason": reason,
        "model": reg.model_name,
        "category": _category_of(reg),
    }
    entry.update(extra)
    return entry


def _non_max_select_fp_with_diagnostics(
        cands, scene, cfg, overlap_thresh=0.5, iou_thresh=0.35,
        max_per_category=None, global_overlap_thresh=0.58,
        min_coverage_by_category=None, min_symmetric_score: float = None,
        cross_category_center_distance: float = 0.0,
        cross_category_overlap_thresh: float = 1.0):
    'Greedy non-maximum selection using explained-scan overlap and same-category horizontal footprint IoU. Allows nearby objects of different categories when their support is compatible.\n    '
    cands = sorted(cands, key=_registration_rank)
    selected = []
    diagnostics = []
    explained_owner_by_category = {}
    global_explained_owner = np.full(scene.cloud.size, -1, int)
    category_counts = {}
    max_per_category = dict(max_per_category or {})
    min_coverage_by_category = dict(min_coverage_by_category or {})
    sel_boxes = []
    tree = scene.cloud.tree
    up = scene.up
    null_score = float(getattr(cfg.matching, "null_hypothesis_score", 0.0))
    requested_score = float(min_symmetric_score or 0.0)
    selection_score = max(null_score, requested_score)
    for reg in cands:
        category = _category_of(reg)
        if (getattr(cfg.matching, "require_constellation_support", True)
                and getattr(reg, "_pose_seed_fallback", False)):
            diagnostics.append(_nms_diagnostic(
                reg, "rejected_unconfirmed_pose_seed", "no_descriptor_constellation",
            ))
            continue
        sym_score = _symmetric_score(
            float(reg.coverage), _ranking_reverse_coverage(reg)
        )
        if sym_score < selection_score:
            explicit_gate = requested_score >= null_score and requested_score > 0.0
            diagnostics.append(_nms_diagnostic(
                reg,
                ("rejected_min_symmetric_score" if explicit_gate
                 else "rejected_null_hypothesis"),
                ("min_symmetric_score" if explicit_gate
                 else "null_hypothesis"),
                symmetric_score=round(float(sym_score), 4),
                min_symmetric_score=(
                    round(requested_score, 4) if explicit_gate else None
                ),
                null_hypothesis_score=round(null_score, 4),
                null_hypothesis_margin=round(float(sym_score - null_score), 4),
            ))
            continue
        category_min_coverage = min_coverage_by_category.get(category)
        if (
            category_min_coverage is not None
            and float(reg.coverage) < float(category_min_coverage)
        ):
            diagnostics.append(_nms_diagnostic(
                reg, "rejected_category_min_coverage",
                "category_min_coverage",
                coverage=round(float(reg.coverage), 4),
                category_min_coverage=round(float(category_min_coverage), 4),
            ))
            continue
        fp = getattr(reg, "_footprint", None)
        if fp is None or len(fp) == 0:
            diagnostics.append(_nms_diagnostic(
                reg, "rejected_missing_footprint", "missing_footprint",
            ))
            continue
        box = _hbbox(fp, up)
        conflicts = [
            (_hbbox_iou(box, selected_box), selected_reg)
            for selected_box, selected_reg in sel_boxes
            if _category_of(selected_reg) == category
        ]
        max_iou, conflict = max(conflicts, default=(0.0, None), key=lambda x: x[0])
        if max_iou > iou_thresh:
            diagnostics.append(_nms_diagnostic(
                reg, "rejected_horizontal_iou", "horizontal_iou",
                horizontal_iou=round(float(max_iou), 4),
                horizontal_iou_threshold=round(float(iou_thresh), 4),
                conflict_model=conflict.model_name if conflict is not None else None,
            ))
            continue
        lists = tree.query_ball_point(fp, cfg.keypoint.neighbor_radius)
        flat = [j for sub in lists for j in sub]
        if not flat:
            diagnostics.append(_nms_diagnostic(
                reg, "rejected_no_scene_support", "no_scene_support",
            ))
            continue
        idx = np.unique(np.asarray(flat, int))
        group_bounds = getattr(reg, "_query_group_bounds", None)
        if group_bounds is not None:
            lo, hi = group_bounds
            points = scene.cloud.points[idx]
            idx = idx[np.all((points >= lo) & (points <= hi), axis=1)]
            if not len(idx):
                diagnostics.append(_nms_diagnostic(
                    reg, "rejected_no_scene_support", "no_group_surface_support"))
                continue
        explained_owner = explained_owner_by_category.setdefault(
            category, np.full(scene.cloud.size, -1, int)
        )
        owners = explained_owner[idx]
        overlap = float((owners >= 0).mean()) if len(idx) else 1.0
        if overlap > overlap_thresh:
            owned = owners[owners >= 0]
            conflict = None
            if len(owned):
                owner_ids, counts = np.unique(owned, return_counts=True)
                conflict = selected[int(owner_ids[int(np.argmax(counts))])]
            diagnostics.append(_nms_diagnostic(
                reg, "rejected_explained_overlap", "explained_overlap",
                explained_overlap=round(overlap, 4),
                explained_overlap_threshold=round(float(overlap_thresh), 4),
                conflict_model=conflict.model_name if conflict is not None else None,
            ))
            continue
        global_owners = global_explained_owner[idx]
        global_overlap = float((global_owners >= 0).mean()) if len(idx) else 1.0
        if global_overlap > global_overlap_thresh:
            owned = global_owners[global_owners >= 0]
            conflict = None
            if len(owned):
                owner_ids, counts = np.unique(owned, return_counts=True)
                conflict = selected[int(owner_ids[int(np.argmax(counts))])]
            diagnostics.append(_nms_diagnostic(
                reg, "rejected_global_explained_overlap",
                "global_explained_overlap",
                global_explained_overlap=round(global_overlap, 4),
                global_explained_overlap_threshold=round(
                    float(global_overlap_thresh), 4
                ),
                conflict_model=conflict.model_name if conflict is not None else None,
            ))
            continue
        cross_conflict = None
        cross_distance = None
        if (
            cross_category_center_distance
            and cross_category_overlap_thresh is not None
            and global_overlap > float(cross_category_overlap_thresh)
        ):
            center = _hbbox_center(box)
            best_distance = float("inf")
            for selected_box, selected_reg in sel_boxes:
                if _category_of(selected_reg) == category:
                    continue
                distance = float(np.linalg.norm(center - _hbbox_center(selected_box)))
                if distance < best_distance:
                    best_distance = distance
                    cross_conflict = selected_reg
            cross_distance = best_distance
            if (
                cross_conflict is not None
                and best_distance < float(cross_category_center_distance)
            ):
                diagnostics.append(_nms_diagnostic(
                    reg, "rejected_cross_category_collision",
                    "cross_category_collision",
                    global_explained_overlap=round(global_overlap, 4),
                    cross_category_overlap_threshold=round(
                        float(cross_category_overlap_thresh), 4
                    ),
                    center_distance=round(float(best_distance), 4),
                    center_distance_threshold=round(
                        float(cross_category_center_distance), 4
                    ),
                    conflict_model=cross_conflict.model_name,
                ))
                continue
        limit = max_per_category.get(category)
        if limit is not None and category_counts.get(category, 0) >= int(limit):
            diagnostics.append(_nms_diagnostic(
                reg, "rejected_category_limit", "category_limit",
                category_count=category_counts.get(category, 0),
                category_limit=int(limit),
            ))
            continue
        explained_owner[idx] = len(selected)
        global_explained_owner[idx] = len(selected)
        sel_boxes.append((box, reg))
        selected.append(reg)
        category_counts[category] = category_counts.get(category, 0) + 1
        diagnostics.append(_nms_diagnostic(
            reg, "selected", "selected",
            explained_overlap=round(overlap, 4),
            global_explained_overlap=round(global_overlap, 4),
            horizontal_iou=round(float(max_iou), 4),
            center_distance=(
                None if cross_distance is None or not np.isfinite(cross_distance)
                else round(float(cross_distance), 4)
            ),
        ))
    return selected, diagnostics


def _non_max_select_fp(cands, scene, cfg, overlap_thresh=0.5, iou_thresh=0.35,
                       max_per_category=None, global_overlap_thresh=0.58,
                       min_coverage_by_category=None,
                       min_symmetric_score: float = None,
                       cross_category_center_distance: float = 0.0,
                       cross_category_overlap_thresh: float = 1.0):
    selected, _ = _non_max_select_fp_with_diagnostics(
        cands, scene, cfg,
        overlap_thresh=overlap_thresh,
        iou_thresh=iou_thresh,
        max_per_category=max_per_category,
        global_overlap_thresh=global_overlap_thresh,
        min_coverage_by_category=min_coverage_by_category,
        min_symmetric_score=min_symmetric_score,
        cross_category_center_distance=cross_category_center_distance,
        cross_category_overlap_thresh=cross_category_overlap_thresh,
    )
    return selected


def _trace_json_value(value):
    if isinstance(value, np.ndarray):
        return [_trace_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return round(number, 4) if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_trace_json_value(item) for item in value]
    return str(value)


def _trace_selection_entry(raw):
    entry = _trace_registration(raw["registration"]) or {}
    entry.update({
        "status": raw.get("status", "unknown"),
        "reason": raw.get("reason", "unknown"),
        "category": raw.get("category", ""),
    })
    for key, value in raw.items():
        if key in {"registration", "model", "category", "status", "reason"}:
            continue
        entry[key] = _trace_json_value(value)
    return entry


def _select_for_trace(candidates, scene, cfg, trace):
    if trace is None:
        return _non_max_select_fp(candidates, scene, cfg)
    selected, diagnostics = _non_max_select_fp_with_diagnostics(
        candidates, scene, cfg,
    )
    trace["checkpoint_selection"] = [
        _trace_selection_entry(item) for item in diagnostics
    ]
    trace["checkpoint_selected"] = [
        _trace_registration(registration) for registration in selected
    ]
    return selected


def _trace_detected_keypoints(keypoints, up, ground, wall_analysis=None,
                              wall_rejected=None):
    """Serialize every detected keypoint for the visual execution trace."""
    count = int(getattr(keypoints, "size", len(keypoints.positions)))
    wall_rejected = (
        np.asarray(wall_rejected, bool)
        if wall_rejected is not None else np.zeros(count, bool)
    )
    records = []
    for index, (position, response) in enumerate(zip(
            keypoints.positions, keypoints.responses)):
        record = {
            "position": [round(float(x), 4) for x in position],
            "response": round(float(response), 6),
            "height": round(float(np.dot(position, up) - ground), 4),
            "wall_rejected": bool(wall_rejected[index]),
        }
        if wall_analysis is not None:
            record.update({
                "wall_score": round(
                    float(wall_analysis.wall_score[index]), 6,
                ),
                "object_score": round(
                    float(wall_analysis.object_score[index]), 6,
                ),
            })
        records.append(record)
    return records


def _trace_retained_keypoints(features):
    return [
        {
            "position": [round(float(x), 4) for x in feature.position],
            "response": round(float(feature.response), 6),
            "selection_score": round(float(
                feature.response
                if getattr(feature, "selection_score", None) is None
                else feature.selection_score
            ), 6),
            "wall_score": round(float(getattr(feature, "wall_score", 0.0)), 6),
            "object_score": round(float(getattr(feature, "object_score", 0.0)), 6),
            "height": round(float(feature.height), 4),
        }
        for feature in features
    ]


def retrieve_dir(scene: SceneScan, db_dir: str, cfg: PipelineConfig,
                 coverage_threshold: float = 0.4, n_jobs: int = 1, progress=None,
                 candidate_index_path=None, candidate_top_k: int = 0,
                 candidate_pool_k: int = 0,
                 candidate_diverse: bool = False, candidate_diversity: float = 0.5,
                 candidate_include_names=None, candidate_only_names=None,
                 exhaustive: bool = False, resume_dir=None, resume_key=None,
                 resume_batch_size: int = 20, trace=None, stop_after=None,
                 return_all_candidates: bool = False):
    'Memory-efficient parallel retrieval using one-file-per-model database storage.'
    detected_kps = detect_keypoints(
        scene.cloud, cfg.keypoint,
        visibility_sampler=_scene_visibility_sampler(scene),
    )
    wall_filter = apply_wall_keypoint_filter(
        scene.cloud, detected_kps, cfg.keypoint, scene.up,
    )
    wall_kps = wall_filter.keypoints
    quality_filter = filter_scene_keypoints(scene, wall_kps, cfg.keypoint)
    kps = quality_filter.keypoints
    feature_trace = {}
    sfeats = build_scan_features(
        scene.cloud, scene.volume, kps, cfg, scene.up, scene.ground,
        trace=feature_trace,
    )
    feature_groups = []
    if cfg.matching.grouped_candidate_query and len(sfeats) and kps.size:
        from keypoint_groups import (
            build_object_keypoint_groups, map_groups_to_features,
        )
        group_scores = (
            getattr(kps, "selection_scores", None)
            if getattr(kps, "selection_scores", None) is not None
            else kps.responses
        )
        group_wall_mask = (
            kps.wall_plane_indices >= 0
            if getattr(kps, "wall_plane_indices", None) is not None
            else np.zeros(kps.size, bool)
        )
        group_object_scores = (
            kps.object_scores
            if getattr(kps, "object_scores", None) is not None
            else np.zeros(kps.size, float)
        )
        object_groups = build_object_keypoint_groups(
            scene.cloud.points, kps.positions,
            surface_wall_mask=(
                wall_filter.analysis.surface_plane_index >= 0
            ),
            keypoint_wall_mask=group_wall_mask,
            scores=group_scores,
            object_scores=group_object_scores,
            ground=scene.ground,
        )
        feature_groups = map_groups_to_features(
            object_groups, kps.positions, sfeats,
            min_features=cfg.matching.candidate_min_group_descriptors,
        )
    if progress:
        progress(
            f'scan: {scene.cloud.size} pts, {detected_kps.size} keypoints (walls rejected: {int(wall_filter.rejected_mask.sum())}, plans: {quality_filter.planar_rejected}, trous: {quality_filter.hole_rejected}, instables: {quality_filter.repeatability_rejected}, faibles: {quality_filter.score_rejected}, doublons: {quality_filter.nms_rejected}, retained: {len(sfeats)})'
        )
    detected_trace = retained_trace = None
    if has_progress_events(progress) or trace is not None:
        detected_trace = _trace_detected_keypoints(
            detected_kps, scene.up, scene.ground,
            wall_analysis=wall_filter.analysis,
            wall_rejected=wall_filter.rejected_mask,
        )
        retained_trace = _trace_retained_keypoints(sfeats)
    if has_progress_events(progress):
        emit_progress_event(progress, "keypoints", {
            "detected": int(detected_kps.size),
            "wall_rejected": int(wall_filter.rejected_mask.sum()),
            "after_wall_filter": int(wall_kps.size),
            "after_quality_filter": int(kps.size),
            "planar_rejected": quality_filter.planar_rejected,
            "hole_rejected": quality_filter.hole_rejected,
            "repeatability_rejected": quality_filter.repeatability_rejected,
            "score_rejected": quality_filter.score_rejected,
            "nms_rejected": quality_filter.nms_rejected,
            "feature_quality": feature_trace,
            "retained": len(sfeats),
            "detected_points": detected_trace,
            "retained_points": retained_trace,
            "points": retained_trace,
        })
    if trace is not None:
        trace.clear()
        trace.update({
            "resume_key": resume_key,
            "mode": "cache" if candidate_only_names else "full",
            "scan": {
                "points": int(scene.cloud.size),
                "detected_keypoints": int(detected_kps.size),
                "wall_rejected_keypoints": int(
                    wall_filter.rejected_mask.sum()
                ),
                "after_wall_filter_keypoints": int(wall_kps.size),
                "after_quality_filter_keypoints": int(kps.size),
                "planar_rejected_keypoints": quality_filter.planar_rejected,
                "hole_rejected_keypoints": quality_filter.hole_rejected,
                "repeatability_rejected_keypoints": quality_filter.repeatability_rejected,
                "score_rejected_keypoints": quality_filter.score_rejected,
                "nms_rejected_keypoints": quality_filter.nms_rejected,
                "feature_quality": feature_trace,
                "retained_keypoints": len(sfeats),
                "detected_keypoint_points": detected_trace,
                "retained_keypoint_points": retained_trace,
                "keypoints": retained_trace,
            },
            "preselection": {},
            "matching": [],
        })
    if stop_after == "keypoints":
        emit_progress_event(progress, "stage_completed", {
            "stage": "keypoints",
            "detected": int(detected_kps.size),
            "wall_rejected": int(wall_filter.rejected_mask.sum()),
            "after_wall_filter": int(wall_kps.size),
            "after_quality_filter": int(kps.size),
            "planar_rejected": quality_filter.planar_rejected,
            "hole_rejected": quality_filter.hole_rejected,
            "repeatability_rejected": quality_filter.repeatability_rejected,
            "score_rejected": quality_filter.score_rejected,
            "nms_rejected": quality_filter.nms_rejected,
            "feature_quality": feature_trace,
            "retained": len(sfeats),
        })
        raise RetrievalStageComplete("keypoints", trace=trace)
    files = db_store.model_files(db_dir)
    if trace is not None:
        trace["database_models"] = len(files)
    selected = None
    candidate_group_details = {}
    if candidate_only_names:
        metadata = db_store.load_index(db_dir)
        by_name = {name: i for i, name in enumerate(metadata.get("names", []))}
        selected = [
            by_name[name] for name in dict.fromkeys(candidate_only_names)
            if name in by_name
        ]
        if progress:
            progress(f"cache: {len(selected)} target model(s)")
        if trace is not None:
            trace["preselection"] = {
                "strategy": "cache",
                "pool": list(dict.fromkeys(candidate_only_names)),
                "top_k": [metadata["names"][index] for index in selected],
                "forced": list(dict.fromkeys(candidate_only_names)),
            }
    elif exhaustive:
        if progress:
            progress(f"recherche exhaustive: {len(files)} modeles")
        if trace is not None:
            trace["preselection"] = {
                "strategy": "exhaustive",
                "pool_count": len(files),
                "top_k_count": len(files),
            }
    elif candidate_index_path and candidate_top_k and candidate_top_k > 0:
        try:
            from candidate_index import (
                load_candidate_index,
                rank_candidates,
                rank_candidates_by_feature_groups,
                rerank_candidates_by_local_features,
                rerank_candidates_by_local_feature_groups,
            )
            cindex = load_candidate_index(candidate_index_path)
            pool_k = max(int(candidate_top_k), int(candidate_pool_k or 0))
            grouped_query = bool(feature_groups)
            group_rankings = []
            global_ranking = []
            if grouped_query:
                pool, group_rankings = rank_candidates_by_feature_groups(
                    cindex, feature_groups, cfg, pool_k,
                    diversify=candidate_diverse,
                    diversity_fraction=candidate_diversity,
                    min_candidates_per_group=(
                        cfg.matching.candidate_pool_min_per_group
                    ),
                )
                global_ranking = rank_candidates(
                    cindex, sfeats, cfg, pool_k,
                    diversify=candidate_diverse,
                    diversity_fraction=candidate_diversity,
                    include_names=candidate_include_names,
                )
                from candidate_index import merge_candidate_pools
                pool = merge_candidate_pools(
                    pool, group_rankings, global_ranking, pool_k,
                    cfg.matching.candidate_global_pool_fraction,
                    cfg.matching.candidate_pool_min_per_group)
                from local_preselection import supplement_from_database
                pool, group_rankings, local_preselection = supplement_from_database(
                    db_dir, cindex.names, feature_groups, cfg, pool, group_rankings,
                    extra_limit=cfg.matching.candidate_local_extra_k,
                    progress=progress,
                )
                if trace is not None:
                    trace["local_preselection"] = local_preselection
            else:
                pool = rank_candidates(
                    cindex, sfeats, cfg, pool_k,
                    diversify=candidate_diverse,
                    diversity_fraction=candidate_diversity,
                    include_names=candidate_include_names,
                )
            if pool_k > int(candidate_top_k) and len(pool) > int(candidate_top_k):
                if progress:
                    progress(
                        f'preselection niveau 1: {len(pool)}/{len(files)} candidate models'
                    )
                if grouped_query:
                    selected, candidate_group_details = rerank_candidates_by_local_feature_groups(
                        db_dir, pool, feature_groups, cfg,
                        int(candidate_top_k), up=scene.up,
                        progress=progress,
                        group_rankings=group_rankings,
                        global_ranking=global_ranking,
                        max_groups_per_candidate=0,
                        min_candidates_per_group=(
                            cfg.matching.candidate_top_min_per_group
                        ),
                        candidates_per_group=(
                            cfg.matching.candidate_top_per_group
                        ),
                        local_weight=(
                            cfg.matching.candidate_group_local_weight
                        ),
                    )
                else:
                    selected = rerank_candidates_by_local_features(
                        db_dir, pool, sfeats, cfg, int(candidate_top_k),
                        include_names=candidate_include_names,
                        up=scene.up,
                        progress=progress,
                    )
            else:
                selected = pool
            if trace is not None:
                per_group_top_k = []
                if grouped_query and candidate_group_details:
                    for group_index in range(len(feature_groups)):
                        per_group_top_k.append([
                            cindex.names[int(index)] for index in selected
                            if group_index in candidate_group_details.get(
                                int(index), {}
                            ).get("selected_for_groups", [])
                        ])
                trace["preselection"] = {
                    "strategy": (
                        "grouped_per_group" if grouped_query
                        and cfg.matching.candidate_top_per_group > 0
                        else "grouped_two_level" if grouped_query
                        and pool_k > int(candidate_top_k)
                        else "two_level" if pool_k > int(candidate_top_k)
                        else "grouped_global" if grouped_query else "global"
                    ),
                    "feature_group_count": len(feature_groups),
                    "database_count": len(files),
                    "pool": [cindex.names[int(index)] for index in pool],
                    "top_k": [cindex.names[int(index)] for index in selected],
                    "top_k_per_group": per_group_top_k,
                    "candidate_top_per_group": int(
                        cfg.matching.candidate_top_per_group
                    ),
                    "forced": list(dict.fromkeys(candidate_include_names or [])),
                    "candidate_top_k": int(candidate_top_k),
                    "candidate_pool_k": int(pool_k),
                    "diverse": bool(candidate_diverse),
                    "diversity_fraction": float(candidate_diversity),
                }
            if progress:
                progress(
                    f'preselection: {len(selected)}/{len(files)} candidate models'
                )
            if has_progress_events(progress):
                emit_progress_event(progress, "preselection", {
                    "strategy": "two_level" if pool_k > int(candidate_top_k) else "global",
                    "database_count": len(files),
                    "pool": [cindex.names[int(index)] for index in pool],
                    "top_k": [cindex.names[int(index)] for index in selected],
                })
        except Exception as exc:
            if progress:
                progress(f"preselection failed: {exc}")
            if trace is not None:
                trace["preselection"] = {
                    "strategy": "error",
                    "error": str(exc),
                }
            raise RuntimeError(
                'Requested preselection failed; exhaustive fallback refused to avoid testing the entire database'
            ) from exc
    tasks = [(i, files[i]) for i in selected] if selected is not None else list(enumerate(files))
    candidate_feature_groups = {}
    if feature_groups and candidate_group_details:
        for model_index in (selected or []):
            detail = candidate_group_details.get(int(model_index), {})
            group_ids = list(detail.get("selected_for_groups") or [])
            best_group = int(detail.get("best_group", -1))
            if best_group >= 0 and best_group not in group_ids:
                group_ids.append(best_group)
            valid = sorted({
                int(group_id) for group_id in group_ids
                if 0 <= int(group_id) < len(feature_groups)
            })
            if valid:
                candidate_feature_groups[int(model_index)] = [
                    (group_id, feature_groups[group_id]) for group_id in valid
                ]
    if stop_after == "query":
        emit_progress_event(progress, "stage_completed", {
            "stage": "query",
            "candidates": len(tasks),
            "database_count": len(files),
        })
        raise RetrievalStageComplete("query", trace=trace)
    up = list(map(float, scene.up))
    cands: List[Registration] = []
    n = len(tasks)
    checkpoint_path = _resume_path(resume_dir, resume_key)
    signature = _resume_signature(tasks, scene, cfg, coverage_threshold)
    state = _load_resume_state(checkpoint_path, signature, tasks)
    completed = state["results"]
    for model_index, _ in tasks:
        if model_index in completed:
            result = completed[model_index]
            _extend_registration_results(cands, result)
            if trace is not None:
                trace["matching"].extend(_worker_traces(result))
    pending = [task for task in tasks if task[0] not in completed]
    if progress and completed:
        progress(
            f"Resuming {resume_key}: {len(completed)}/{n} model(s) already tested, "
            f"{len(pending)} remaining"
        )
    if not pending:
        if trace is not None:
            trace["trace_complete"] = len(state["results"]) == n
        if stop_after == "matching":
            emit_progress_event(progress, "matching_completed", {
                "tested": n,
                "candidates": len(cands),
                "resumed": True,
            })
            raise RetrievalStageComplete(
                "matching", trace=trace, registrations=cands,
            )
        selected_registrations = _select_for_trace(cands, scene, cfg, trace)
        if has_progress_events(progress):
            emit_progress_event(progress, "checkpoint_selection", {
                "candidates": len(cands),
                "selected": [
                    _trace_registration(reg) for reg in selected_registrations
                ],
                "diagnostics": (trace or {}).get("checkpoint_selection", []),
                "resumed": True,
            })
        return cands if return_all_candidates else selected_registrations

    batch_size = max(1, int(resume_batch_size or 1))
    matching_start = time.monotonic()
    newly_done = 0

    def record(task, result):
        nonlocal newly_done
        model_index = int(task[0])
        state["results"][model_index] = result
        _extend_registration_results(cands, result)
        worker_traces = _worker_traces(result)
        if trace is not None:
            trace["matching"].extend(worker_traces)
        if has_progress_events(progress):
            for worker_trace in worker_traces:
                emit_progress_event(progress, "matching_result", {
                    "completed": len(state["results"]),
                    "total": n,
                    "candidate": worker_trace,
                })
        newly_done += 1
        if newly_done % batch_size == 0:
            _save_resume_state(checkpoint_path, state)
            if progress and checkpoint_path is not None:
                progress(
                    f"batch saved: {len(state['results'])}/{n} model(s) -> "
                    f"{checkpoint_path.name}"
                )

    try:
        if n_jobs and n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(
                max_workers=n_jobs, initializer=_init_files,
                initargs=(sfeats, scene.cloud.points, cfg, up, coverage_threshold,
                          trace is not None, candidate_feature_groups),
            ) as ex:
                worker_fn = _work_file_multi if cfg.matching.max_registrations_per_model > 1 else _work_file
                for task, result in zip(pending, ex.map(worker_fn, pending, chunksize=1)):
                    record(task, result)
                    if progress and newly_done % 20 == 0:
                        done = len(state["results"])
                        progress(
                            f'modeles testes: {done}/{n} (candidates: {len(cands)}, {_progress_eta(newly_done, len(pending), matching_start)})'
                        )
        else:
            _init_files(
                sfeats, scene.cloud.points, cfg, up, coverage_threshold,
                trace is not None, candidate_feature_groups,
            )
            worker_fn = _work_file_multi if cfg.matching.max_registrations_per_model > 1 else _work_file
            for task in pending:
                record(task, worker_fn(task))
                if progress and newly_done % 20 == 0:
                    done = len(state["results"])
                    progress(
                        f'modeles testes: {done}/{n} (candidates: {len(cands)}, {_progress_eta(newly_done, len(pending), matching_start)})'
                    )
    finally:
        if newly_done:
            _save_resume_state(checkpoint_path, state)

    if trace is not None:
        trace["trace_complete"] = len(state["results"]) == n
    if stop_after == "matching":
        emit_progress_event(progress, "matching_completed", {
            "tested": n,
            "candidates": len(cands),
            "resumed": False,
        })
        raise RetrievalStageComplete(
            "matching", trace=trace, registrations=cands,
        )
    selected_registrations = _select_for_trace(cands, scene, cfg, trace)
    if has_progress_events(progress):
        emit_progress_event(progress, "checkpoint_selection", {
            "candidates": len(cands),
            "selected": [_trace_registration(reg) for reg in selected_registrations],
            "diagnostics": (trace or {}).get("checkpoint_selection", []),
        })
    return cands if return_all_candidates else selected_registrations


def run(zip_path, db: ModelDatabase, cfg: Optional[PipelineConfig] = None,
        coverage_threshold: float = 0.4, progress=None,
        n_jobs: int = 1, db_path=None) -> List[Registration]:
    cfg = cfg or db.cfg or PipelineConfig()
    scene = scan_from_zip(zip_path, cfg, progress=progress)
    return retrieve(scene, db, cfg, coverage_threshold=coverage_threshold,
                    progress=progress, n_jobs=n_jobs, db_path=db_path)


def run_dir(zip_path, db_dir: str, cfg: PipelineConfig,
            coverage_threshold: float = 0.4, progress=None, n_jobs: int = 1,
            candidate_index_path=None, candidate_top_k: int = 0,
            candidate_pool_k: int = 0,
            candidate_diverse: bool = False, candidate_diversity: float = 0.5,
            candidate_include_names=None, candidate_only_names=None,
            exhaustive: bool = False):
    scene = scan_from_zip(zip_path, cfg, progress=progress)
    return retrieve_dir(scene, db_dir, cfg, coverage_threshold=coverage_threshold,
                        n_jobs=n_jobs, progress=progress,
                        candidate_index_path=candidate_index_path,
                        candidate_top_k=candidate_top_k,
                        candidate_pool_k=candidate_pool_k,
                        candidate_diverse=candidate_diverse,
                        candidate_diversity=candidate_diversity,
                        candidate_include_names=candidate_include_names,
                        candidate_only_names=candidate_only_names,
                        exhaustive=exhaustive)
