'Temporal queries with cached models. Each checkpoint reconstructs a sequence prefix, includes cached models in preselection and merges distinct poses in a shared coordinate system before final revalidation. Also supports a single final query.\n'

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from config import PipelineConfig
from database import object_surface_cloud
import db_store
import pipeline
from live_events import emit_progress_event, has_progress_events
from sdf_fusion import RGBDSequence, _select_frame_ids
from verify import (
    coverage_score, model_support_distribution, visibility_coverage_score,
    group_surface_coverage,
)


@dataclass
class ProgressiveStep:
    frame_count: int
    registrations: list
    cache_names: List[str]
    trace: Optional[dict] = None


@dataclass
class ProgressiveResult:
    scene: pipeline.SceneScan
    registrations: list
    steps: List[ProgressiveStep]
    final_candidates: list
    final_diagnostics: list
    final_selection_diagnostics: list
    completed_stage: str = "selection"


def selected_frame_count(zip_path, cfg: PipelineConfig) -> int:
    seq = RGBDSequence(zip_path, depth_scale=cfg.sdf.depth_scale)
    try:
        # Use exactly the same adaptive sampling as fusion. Counting with the
        # requested stride would truncate short sequences a second time when
        # min_frames makes the effective stride smaller.
        frame_ids, _ = _select_frame_ids(seq.frame_ids, cfg.sdf)
        count = len(frame_ids)
    finally:
        seq.close()
    return count


def progressive_checkpoints(total_frames: int, start_frames: int = 20,
                            interval_frames: int = 30) -> List[int]:
    if total_frames <= 0:
        return []
    start = max(1, min(int(start_frames), total_frames))
    interval = max(1, int(interval_frames))
    checkpoints = list(range(start, total_frames + 1, interval))
    if not checkpoints or checkpoints[-1] != total_frames:
        checkpoints.append(total_frames)
    return checkpoints


SCENE_CACHE_VERSION = 4


def _scene_cache_signature(zip_path, cfg, frame_count, alignment_rotation):
    source = Path(zip_path).resolve()
    stat = source.stat()
    sdf = cfg.sdf
    payload = {
        "version": SCENE_CACHE_VERSION,
        "source": str(source),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "frame_count": int(frame_count),
        "sdf": {
            name: getattr(sdf, name, None)
            for name in (
                "voxel_size", "truncation", "depth_trunc", "depth_scale",
                "frame_stride", "min_frames", "max_frames", "iso_sampling",
                "max_surface_points",
                "surface_field", "surface_min_support",
                "surface_extraction", "surface_band_factor",
                "surface_rescue_distance_factor", "surface_rescue_min_weight",
                "gradient_smoothing_sigma",
                "depth_edge_threshold", "depth_edge_background_weight",
                "depth_edge_radius",
                "volume_layout", "sparse_block_resolution",
                "sparse_block_count", "visibility_depth_stride",
            )
        },
        "alignment_rotation": (
            None if alignment_rotation is None
            else np.round(np.asarray(alignment_rotation, float), 8).tolist()
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scene_cache_path(cache_dir, frame_count):
    if not cache_dir:
        return None
    return Path(cache_dir) / f"scene_{int(frame_count):06d}.pkl"


def _scene_surface_payload(scene, frame_count):
    preview = np.asarray(scene.cloud.points, float)
    if len(preview) > 24000:
        preview = preview[np.linspace(0, len(preview) - 1, 24000).astype(int)]
    return {
        "points": np.round(preview, 3).tolist(),
        "point_count": int(scene.cloud.size),
        "ground": round(float(scene.ground), 4),
        "alignment_rotation": np.round(
            np.asarray(scene.alignment_rotation), 6,
        ).tolist(),
        "frame_count": int(frame_count),
        "cached": True,
    }


def _load_scene_cache(path, signature, frame_count, progress=None):
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("version") != SCENE_CACHE_VERSION:
            raise ValueError('different version')
        if payload.get("signature") != signature:
            raise ValueError('different signature')
        scene = payload["scene"]
    except Exception as exc:
        if progress:
            progress(f"scene cache ignored ({path.name}: {exc})")
        emit_progress_event(progress, "stage_cache_invalid", {
            "stage": "fusion", "frame_count": int(frame_count),
            "path": str(path), "reason": str(exc),
        })
        return None

    if progress:
        progress(
            f"scene cache reused: {frame_count} trames <- {path.name}"
        )
    emit_progress_event(progress, "stage_cache_hit", {
        "stage": "fusion", "frame_count": int(frame_count),
        "path": str(path), "size_bytes": int(path.stat().st_size),
    })
    base_volume = getattr(scene.volume, "volume", scene.volume)
    emit_progress_event(progress, "fusion_started", {
        "total_frames": int(frame_count), "backend": "cache",
        "voxel_size": float(getattr(base_volume, "voxel_size", 0.0)),
        "cached": True,
    })
    emit_progress_event(
        progress, "scene_surface", _scene_surface_payload(scene, frame_count),
    )
    return scene


def _save_scene_cache(path, signature, scene, frame_count, progress=None):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.cloud._tree = None
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump({
                "version": SCENE_CACHE_VERSION,
                "signature": signature,
                "scene": scene,
            }, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    if progress:
        progress(
            f"scene cache saved: {frame_count} trames -> {path.name}"
        )
    emit_progress_event(progress, "stage_cache_saved", {
        "stage": "fusion", "frame_count": int(frame_count),
        "path": str(path), "size_bytes": int(path.stat().st_size),
    })


def _scan_from_cache_or_zip(zip_path, cfg, frame_count, progress=None,
                            alignment_rotation=None, fusion_trace=None,
                            stage_cache_dir=None):
    if not stage_cache_dir:
        return pipeline.scan_from_zip(
            zip_path, cfg, progress=progress,
            alignment_rotation=alignment_rotation, fusion_trace=fusion_trace,
        )
    signature = _scene_cache_signature(
        zip_path, cfg, frame_count, alignment_rotation,
    )
    cache_path = _scene_cache_path(stage_cache_dir, frame_count)
    cached = _load_scene_cache(
        cache_path, signature, frame_count, progress=progress,
    )
    if cached is not None:
        return cached
    scene = pipeline.scan_from_zip(
        zip_path, cfg, progress=progress,
        alignment_rotation=alignment_rotation, fusion_trace=fusion_trace,
    )
    _save_scene_cache(
        cache_path, signature, scene, frame_count, progress=progress,
    )
    return scene


def _cache_names(registrations) -> List[str]:
    return list(dict.fromkeys(reg.model_name for reg in registrations))


def _json_float(value, ndigits: int = 4, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return round(number, ndigits)


def _score_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _diagnostic_entry(reg, status: str, reason: str, coverage=None,
                      cov_reverse=None, mean_surface_dist=None):
    coverage = getattr(reg, "coverage", 0.0) if coverage is None else coverage
    cov_reverse = getattr(reg, "cov_reverse", 0.0) if cov_reverse is None else cov_reverse
    mean_surface_dist = (
        getattr(reg, "mean_surface_dist", None)
        if mean_surface_dist is None else mean_surface_dist
    )
    transform = getattr(reg, "transform", None)
    translation = getattr(transform, "t", None)
    if translation is not None:
        translation = [_json_float(x, 4) for x in translation]
    entry = {
        "status": status,
        "reason": reason,
        "model": reg.model_name,
        "category": reg.model_name.split("_", 1)[0],
        "checkpoint_frames": getattr(reg, "_checkpoint_frames", None),
        "coverage": _json_float(coverage, 4, 0.0),
        "cov_reverse": _json_float(cov_reverse, 4, 0.0),
        "mean_surface_dist_m": _json_float(mean_surface_dist, 4),
        "quality": _json_float(getattr(reg, "quality", None), 1),
        "scale": _json_float(getattr(transform, "scale", None), 4),
        "theta_deg": _json_float(
            _score_float(getattr(transform, "theta", None), math.nan) * 57.2958,
            2,
        ),
        "translation": translation,
        "balanced_score": round(pipeline._symmetric_score(
            _score_float(coverage), pipeline._ranking_reverse_coverage(reg),
        ), 4),
    }
    for attr, key in (
        ("_geometric_cov_reverse", "geometric_cov_reverse"),
        ("_query_group_reverse", "group_cov_reverse"),
        ("_query_group_surface_points", "group_surface_points"),
        ("_visibility_support", "visibility_support"),
        ("_visibility_known_fraction", "visibility_known_fraction"),
        ("_visibility_free_fraction", "visibility_free_fraction"),
        ("_visibility_known_points", "visibility_known_points"),
        ("_visibility_used", "visibility_used"),
        ("_model_zone_fraction", "model_zone_fraction"),
        ("_supported_model_zones", "supported_model_zones"),
        ("_model_zone_count", "model_zone_count"),
        ("_support_extent_ratio", "support_extent_ratio"),
    ):
        if hasattr(reg, attr):
            value = getattr(reg, attr)
            entry[key] = value if isinstance(value, (bool, int)) else _json_float(value, 4)
    return entry


def _selection_diagnostic_entry(raw):
    reg = raw["registration"]
    entry = _diagnostic_entry(
        reg,
        raw.get("status", "unknown"),
        raw.get("reason", "unknown"),
    )
    for key, value in raw.items():
        if key == "registration":
            continue
        if key in entry:
            continue
        entry[key] = value
    return entry


def _final_revalidate(registrations, db_dir: str, scene: pipeline.SceneScan,
                      cfg: PipelineConfig, coverage_threshold: float,
                      reverse_gate: Optional[float], progress=None):
    'Rescore progressive hypotheses on the final scan so that early local fits must remain compatible with the complete observation.\n    '
    if reverse_gate is None:
        return registrations, [
            _diagnostic_entry(reg, "kept", "not_revalidated")
            for reg in registrations
        ]

    meta = db_store.load_index(db_dir)
    files = db_store.model_files(db_dir)
    by_name = {name: i for i, name in enumerate(meta.get("names", []))}
    models = {}
    kept = []
    diagnostics = []
    rejected_coverage = 0
    rejected_reverse = 0
    missing_model = 0
    threshold = cfg.keypoint.neighbor_radius

    for reg in registrations:
        idx = by_name.get(reg.model_name)
        if idx is None or idx >= len(files):
            missing_model += 1
            diagnostics.append(_diagnostic_entry(reg, "missing_model", "model_not_in_db"))
            continue
        model = models.get(idx)
        if model is None:
            model = db_store.load_model(files[idx])
            models[idx] = model
        object_cloud = object_surface_cloud(model)
        group_support = group_surface_coverage(
            object_cloud, scene.cloud, reg.transform,
            getattr(reg, "_query_group_points", None), threshold,
            scene.up, getattr(scene, "ground", None))
        if group_support is not None:
            reg._query_group_reverse, reg._query_group_surface_points, reg._query_group_bounds = group_support
        fwd, rev, msd, struct_h, thickness = coverage_score(
            object_cloud, scene.cloud, reg.transform, threshold, up=scene.up,
        )
        zone_fraction, extent_ratio, supported_zones, zone_count = (
            model_support_distribution(
                object_cloud, scene.cloud, reg.transform, threshold,
                zone_grid=getattr(cfg.matching, "model_zone_grid", 2),
                zone_min_support=getattr(
                    cfg.matching, "model_zone_min_support", 0.15,
                ),
            )
        )
        geometric_rev = rev
        validation_rev = rev
        visibility_used = False
        visibility = None
        if bool(getattr(cfg.matching, "visibility_reverse_enabled", True)):
            visibility = visibility_coverage_score(
                object_cloud, reg.transform, getattr(scene, "volume", None),
                scan_cloud=scene.cloud, threshold=threshold,
            )
        if visibility is not None:
            support, known_fraction, free_fraction, known_points = visibility
            reg._visibility_support = support
            reg._visibility_known_fraction = known_fraction
            reg._visibility_free_fraction = free_fraction
            reg._visibility_known_points = known_points
            enough_visibility = (
                known_points >= int(getattr(
                    cfg.matching, "visibility_min_known_points", 30,
                ))
                and known_fraction >= float(getattr(
                    cfg.matching, "visibility_min_known_fraction", 0.05,
                ))
            )
            if enough_visibility:
                validation_rev = support
                visibility_used = True
        reg.coverage = fwd
        reg.cov_forward = fwd
        reg.cov_reverse = validation_rev
        reg._geometric_cov_reverse = geometric_rev
        reg._visibility_used = visibility_used
        reg._structure_height = struct_h
        reg._thickness = thickness
        reg._model_zone_fraction = zone_fraction
        reg._supported_model_zones = supported_zones
        reg._model_zone_count = zone_count
        reg._support_extent_ratio = extent_ratio
        reg.mean_surface_dist = msd
        reg._footprint = pipeline._footprint(reg, model)
        reg._mesh_path = model.mesh_path
        reg._model_index = idx
        if fwd < coverage_threshold:
            rejected_coverage += 1
            diagnostics.append(_diagnostic_entry(
                reg, "rejected_coverage",
                f"coverage < {coverage_threshold}",
                coverage=fwd, cov_reverse=validation_rev, mean_surface_dist=msd,
            ))
            continue
        from verify import normal_compatible_coverage
        normal_support = normal_compatible_coverage(
            object_cloud, scene.cloud, reg.transform, threshold,
            getattr(cfg.matching, "normal_support_angle_deg", 0.0))
        reg._normal_compatible_coverage = normal_support
        if normal_support is not None and normal_support < coverage_threshold:
            diagnostics.append(_diagnostic_entry(
                reg, "rejected_normal_support",
                f"normal_compatible_coverage {normal_support:.4f} < {coverage_threshold}",
                coverage=fwd, cov_reverse=validation_rev))
            continue
        if validation_rev < reverse_gate:
            rejected_reverse += 1
            metric = "visibility_support" if visibility_used else "cov_reverse"
            diagnostics.append(_diagnostic_entry(
                reg, "rejected_reverse",
                f"{metric} < {reverse_gate}",
                coverage=fwd, cov_reverse=validation_rev, mean_surface_dist=msd,
            ))
            continue
        min_zone_fraction = getattr(
            cfg.matching, "min_model_zone_fraction", 0.35,
        )
        if zone_fraction < min_zone_fraction:
            diagnostics.append(_diagnostic_entry(
                reg, "rejected_model_zones",
                f"model_zone_fraction < {min_zone_fraction}",
                coverage=fwd, cov_reverse=validation_rev,
            ))
            continue
        min_extent_ratio = getattr(
            cfg.matching, "min_support_extent_ratio", 0.20,
        )
        if extent_ratio < min_extent_ratio:
            diagnostics.append(_diagnostic_entry(
                reg, "rejected_support_locality",
                f"support_extent_ratio < {min_extent_ratio}",
                coverage=fwd, cov_reverse=validation_rev,
            ))
            continue
        kept.append(reg)
        diagnostics.append(_diagnostic_entry(
            reg, "kept", "passed_final_revalidation",
            coverage=fwd, cov_reverse=validation_rev, mean_surface_dist=msd,
        ))

    if progress:
        progress(
            f'validation finale: {len(kept)}/{len(registrations)} hypothesis/hypotheses retained ({rejected_coverage} coverage rejection, {rejected_reverse} reverse coverage rejection < {reverse_gate}, {missing_model} missing model)'
        )
    return kept, diagnostics


def progressive_retrieve_dir(
    zip_path,
    db_dir: str,
    cfg: PipelineConfig,
    coverage_threshold: float = 0.4,
    n_jobs: int = 1,
    progress: Optional[Callable[[str], None]] = None,
    candidate_index_path=None,
    candidate_top_k: int = 0,
    candidate_pool_k: int = 0,
    candidate_diverse: bool = False,
    candidate_diversity: float = 0.5,
    exhaustive: bool = False,
    initial_cache_names=None,
    start_frames: int = 20,
    interval_frames: int = 30,
    query_mode: str = "single",
    cache_registrations_per_model: int = 3,
    cache_top_constellations: int = 50,
    final_reverse_gate: Optional[float] = None,
    final_max_per_category=None,
    final_global_overlap: float = 0.58,
    final_min_symmetric_score: float = 0.0,
    final_cross_category_center_distance: float = 0.0,
    final_cross_category_overlap: float = 1.0,
    resume_dir=None,
    resume_batch_size: int = 20,
    stop_after: Optional[str] = None,
    stage_cache_dir=None,
) -> ProgressiveResult:
    allowed_stages = {None, "fusion", "keypoints", "query", "matching", "verification", "selection"}
    if stop_after not in allowed_stages:
        raise ValueError(f"Unknown stop stage: {stop_after}")
    query_mode = str(query_mode or "single").strip().lower()
    if query_mode not in {"single", "progressive"}:
        raise ValueError(f"Unknown query mode: {query_mode}")
    total = selected_frame_count(zip_path, cfg)
    checkpoints = (
        [total] if query_mode == "single" and total > 0
        else progressive_checkpoints(total, start_frames, interval_frames)
    )
    if not checkpoints:
        raise ValueError('No frames available for progressive retrieval')

    final_cfg = deepcopy(cfg)
    final_cfg.sdf.max_frames = total
    if progress:
        progress(f"scan final commun: {total} fused frames")
    fusion_trace = {}
    final_scene = _scan_from_cache_or_zip(
        zip_path, final_cfg, total, progress=progress,
        fusion_trace=fusion_trace, stage_cache_dir=stage_cache_dir,
    )
    if stop_after == "fusion":
        # Le mode progressif historique consomme un volume par préfixe. Le mode
        # local mono-requête ne matérialise que la reconstruction finale.
        for checkpoint in checkpoints if query_mode == "progressive" else []:
            if checkpoint == total:
                continue
            if progress:
                progress(
                    f"preparing progressive fusion: prefix "
                    f"{checkpoint}/{total}"
                )
            step_cfg = deepcopy(cfg)
            step_cfg.sdf.max_frames = checkpoint
            _scan_from_cache_or_zip(
                zip_path, step_cfg, checkpoint, progress=progress,
                alignment_rotation=final_scene.alignment_rotation,
                stage_cache_dir=stage_cache_dir,
            )
        emit_progress_event(progress, "stage_completed", {
            "stage": "fusion",
            "frames": int(total),
            "points": int(final_scene.cloud.size),
            "cached_volumes": len(checkpoints),
        })
        return ProgressiveResult(
            final_scene, [], [], [], [], [], completed_stage="fusion",
        )

    cache = list(dict.fromkeys(initial_cache_names or []))
    accumulated = []
    steps = []
    for checkpoint in checkpoints:
        emit_progress_event(progress, "checkpoint_started", {
            "frame_count": int(checkpoint),
            "total_frames": int(total),
            "query_mode": query_mode,
        })
        step_cfg = deepcopy(cfg)
        step_cfg.sdf.max_frames = checkpoint
        if checkpoint == total:
            scene = final_scene
        else:
            if progress:
                progress(f"progressive query: prefix {checkpoint}/{total}")
            scene = _scan_from_cache_or_zip(
                zip_path, step_cfg, checkpoint, progress=progress,
                alignment_rotation=final_scene.alignment_rotation,
                stage_cache_dir=stage_cache_dir,
            )

        cache_regs = []
        cache_trace = None
        if cache and query_mode == "progressive":
            cache_trace = {}
            cache_cfg = deepcopy(step_cfg)
            cache_cfg.matching.max_registrations_per_model = max(
                1, int(cache_registrations_per_model)
            )
            cache_cfg.matching.registration_top_constellations = max(
                1, int(cache_top_constellations)
            )
            cache_cfg.matching.multi_registration_synsets = ()
            if progress:
                progress(
                    f"passe cache: {len(cache)} model(s), "
                    f"{cache_cfg.matching.max_registrations_per_model} pose(s)/model"
                )
            cache_regs = pipeline.retrieve_dir(
                scene, db_dir, cache_cfg,
                coverage_threshold=coverage_threshold,
                n_jobs=n_jobs,
                progress=progress,
                candidate_only_names=cache,
                resume_dir=resume_dir,
                resume_key=f"frames_{checkpoint:06d}_cache",
                resume_batch_size=resume_batch_size,
                trace=cache_trace,
            )

        full_trace = {}
        try:
            full_regs = pipeline.retrieve_dir(
                scene, db_dir, step_cfg,
                coverage_threshold=coverage_threshold,
                n_jobs=n_jobs,
                progress=progress,
                candidate_index_path=candidate_index_path,
                candidate_top_k=candidate_top_k,
                candidate_pool_k=candidate_pool_k,
                candidate_diverse=candidate_diverse,
                candidate_diversity=candidate_diversity,
                candidate_include_names=cache,
                exhaustive=exhaustive,
                resume_dir=resume_dir,
                resume_key=(
                    "single_query_full" if query_mode == "single"
                    else f"frames_{checkpoint:06d}_full"
                ),
                resume_batch_size=resume_batch_size,
                trace=full_trace,
                stop_after=stop_after,
                return_all_candidates=(query_mode == "single"),
            )
        except pipeline.RetrievalStageComplete as partial:
            partial_trace = partial.trace or full_trace
            partial_step = ProgressiveStep(
                checkpoint, partial.registrations, list(cache),
                {"cache": cache_trace, "full": partial_trace},
            )
            return ProgressiveResult(
                final_scene,
                list(partial.registrations),
                [partial_step],
                list(partial.registrations),
                [],
                [],
                completed_stage=partial.stage,
            )
        raw_regs = cache_regs + full_regs
        regs = pipeline._non_max_select_fp(raw_regs, scene, step_cfg)
        for reg in regs:
            reg._checkpoint_frames = checkpoint
            reg._query_frames = checkpoint
        # En mono-requête, la scene de ce checkpoint est déjà la scene
        # finale. Conserver uniquement la output de la NMS gloutonne perd the
        # alternatives : si son premier choix échoue ensuite au test de
        # visibilité, the candidats qu'il avait masqués ne sont jamais rejoués.
        # La validation finale doit donc précéder l'unique sélection globale.
        accumulated.extend(raw_regs if query_mode == "single" else regs)
        cache = list(dict.fromkeys(cache + _cache_names(regs)))
        steps.append(ProgressiveStep(
            checkpoint, regs, list(cache),
            {"cache": cache_trace, "full": full_trace},
        ))
        if progress:
            if query_mode == "single":
                progress(
                    f"final query: {len(regs)} object(s), "
                    f"{checkpoint} frame(s) analyzed"
                )
            else:
                progress(
                    f"checkpoint {checkpoint}: {len(regs)} object(s), "
                    f"cache {len(cache)} model(s)"
                )
        if has_progress_events(progress):
            emit_progress_event(progress, "checkpoint_completed", {
                "frame_count": int(checkpoint),
                "registrations": [pipeline._trace_registration(reg) for reg in regs],
                "cache_names": list(cache),
                "query_mode": query_mode,
            })

    final_candidates, final_diagnostics = _final_revalidate(
        accumulated, db_dir, final_scene, cfg, coverage_threshold,
        final_reverse_gate, progress=progress,
    )
    emit_progress_event(progress, "verification_completed", {
        "input_candidates": len(accumulated),
        "kept": len(final_candidates),
        "rejected": max(0, len(accumulated) - len(final_candidates)),
    })
    if stop_after == "verification":
        emit_progress_event(progress, "stage_completed", {
            "stage": "verification",
            "candidates": len(final_candidates),
        })
        return ProgressiveResult(
            final_scene, final_candidates, steps, final_candidates,
            final_diagnostics, [], completed_stage="verification",
        )
    merged, selection_raw = pipeline._non_max_select_fp_with_diagnostics(
        final_candidates, final_scene, cfg,
        max_per_category=final_max_per_category,
        global_overlap_thresh=final_global_overlap,
        min_coverage_by_category={"table": 0.6, "couch": 0.7},
        min_symmetric_score=final_min_symmetric_score,
        cross_category_center_distance=final_cross_category_center_distance,
        cross_category_overlap_thresh=final_cross_category_overlap,
    )
    final_selection_diagnostics = [
        _selection_diagnostic_entry(entry) for entry in selection_raw
    ]
    if has_progress_events(progress):
        emit_progress_event(progress, "final_selection", {
            "selected": [pipeline._trace_registration(reg) for reg in merged],
            "diagnostics": final_selection_diagnostics,
        })
    emit_progress_event(progress, "stage_completed", {
        "stage": "selection",
        "selected": len(merged),
    })
    return ProgressiveResult(
        final_scene, merged, steps, final_candidates, final_diagnostics,
        final_selection_diagnostics, completed_stage="selection",
    )
