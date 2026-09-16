"""Independent retrieval debug stages, backed by production matching functions."""
import copy
import pickle
import time
from pathlib import Path

import numpy as np

from config import PipelineConfig


STEPS = {
    "matching": [("poses", "Constellations and poses", 'Local correspondences, RANSAC and group geometry fallback when needed. No ICP at this stage.')],
    "verification": [("icp", "ICP and coverage", 'Refine saved poses, test surfaces and free/unknown visibility. No new matching.')],
    "selection": [("decisions", 'Global decisions', 'Rank verified poses, test the null hypothesis and remove overlaps.')],
}

REASONS = {
    "group_pose_seed_fallback": 'Group geometry fallback: proximity support without descriptor validation of every pair.',
    "ransac_constellation": 'RANSAC constellation: multiple correspondences support the same pose.',
    "rejected_no_correspondence": 'No local correspondence passes the filters.',
    "rejected_no_constellation": 'Correspondences do not support an admissible constellation.',
    "rejected_icp_or_scale": 'Invalid ICP result or scale. Showing the pre-ICP seed, not a refined pose.',
    "rejected_reverse_gate": 'Insufficient geometric reverse coverage after ICP.',
    "rejected_structure": 'Explained structure has insufficient vertical extent.',
    "rejected_thickness": 'Explained support is too thin.',
    "rejected_model_zones": 'Too few model zones have observed support.',
    "rejected_support_locality": 'Support is concentrated on too small a part of the model.',
    "rejected_coverage": 'Forward coverage is below the threshold.',
    "rejected_verification": 'No pose passes the verification safeguards.',
    "accepted_pose": 'Pose admissible after ICP; not yet a final result.',
    "verified": 'Pose revalidated and passed to selection without prior duplicate suppression.',
    "rejected_final_validation": 'Pose rejected during revalidation: inspect final_validation for the failed threshold.',
    "rejected_normal_support": 'Nearby surfaces have insufficient normal agreement.',
    "rejected_null_hypothesis": 'Score too low: the no-object hypothesis wins.',
    "rejected_horizontal_iou": 'Footprint overlaps a retained model of the same category too much.',
    "rejected_explained_overlap": 'This scan region is already explained by the same category.',
    "rejected_global_explained_overlap": 'This scan region is already explained, across all categories.',
    "selected": 'Pose retained after global checks.',
}

# (config section, field, label, minimum, maximum, step, explanation)
PARAMETERS = {
    "matching": [
        ("ransac", "desc_inlier", 'Descriptor threshold', 0, 100000, 16, 'Lower values reduce ambiguous correspondences but may lose a good model. UDF cell units.'),
        ("matching", "descriptor_ratio_threshold", 'Descriptor ratio', 0, 1, .05, '0 disables the ratio. A low positive threshold requires stronger separation between first and second neighbors.'),
        ("ransac", "geom_inlier", 'Geometric distance', .001, 1, .01, 'Maximum aligned-keypoint distance in meters for constellation support.'),
        ("ransac", "min_scan_inliers", 'Minimum scene support', 3, 100, 1, 'Minimum distinct scene points required to accept a RANSAC constellation.'),
        ("ransac", "max_seeds", "Germes RANSAC", 1, 1000, 1, "More seeds explore more poses and increase matching time."),
        ("matching", "group_pose_seed_enabled", "Group fallback", None, None, None, 'Test geometric orientations when correspondences produce no constellation. This support is not descriptor evidence.'),
        ("matching", "group_pose_seed_count", "Poses de secours", 1, 100, 1, 'Group geometry poses passed to verification.'),
    ],
    "verification": [
        ("matching", "normal_support_angle_deg", 'Normal tolerance (°)', 0, 90, 5, 'Experimental: require minimum coverage from compatibly oriented surfaces. 0 disables this check.'),
        ("keypoint", "neighbor_radius", 'Surface tolerance (m)', .001, .5, .005, 'Radius for coverage and distributed support measurement. Does not recompute keypoints.'),
        ("matching", "verification_top_constellations", 'Verified poses', 1, 200, 1, 'Maximum constellations refined to select the best result per model.'),
        ("matching", "max_registrations_per_model", 'Results per model', 1, 20, 1, 'Values above 1 retain multiple sufficiently separated poses.'),
        ("matching", "registration_top_constellations", "Poses en mode multiple", 1, 200, 1, 'ICP budget when multiple results per model are requested.'),
        ("matching", "reverse_gate", "Coverage inverse ICP", 0, 1, .01, 'Geometric safeguard during ICP, before visibility-aware revalidation.'),
        ("matching", "min_model_zone_fraction", "Zones soutenues", 0, 1, .05, 'Minimum fraction of model zones explained by the scan.'),
        ("matching", "min_support_extent_ratio", 'Support extent', 0, 1, .05, 'Prevent validation of an entire object from a small supported fragment.'),
        ("matching", "visibility_reverse_enabled", 'Reverse visibility', None, None, None, 'Revalidation uses free/occupied evidence when enough points are known; otherwise it keeps geometric coverage.'),
    ],
    "selection": [
        ("matching", "null_hypothesis_score", 'Null hypothesis', 0, 1, .01, 'Below this harmonic score, select no object.'),
    ],
}
EXTRA = {
    "verification": {
        "coverage_threshold": (.45, "Coverage minimale", 0, 1, .01, 'Fraction of the model explained by the scene required for selection.'),
        "final_reverse_gate": (.4, "Coverage inverse finale", 0, 1, .01, 'Post-ICP threshold: visibility when usable, otherwise geometric coverage.'),
    },
    "selection": {
        "overlap_thresh": (.5, 'Same-category overlap', 0, 1, .01, 'Fraction of scan points already explained by a model of the same category.'),
        "iou_thresh": (.35, "IoU horizontale", 0, 1, .01, 'Maximum same-category footprint overlap.'),
        "global_overlap_thresh": (.58, "Recouvrement global", 0, 1, .01, 'Fraction already explained across all categories. No implicit category quota.'),
    },
}


def defaults(stage):
    cfg = PipelineConfig()
    values = {key: getattr(getattr(cfg, section), key)
              for section, key, *_ in PARAMETERS[stage]}
    values.update({key: item[0] for key, item in EXTRA.get(stage, {}).items()})
    values["parallel_scenes"] = 1
    return values


def schema(stage):
    values = defaults(stage)
    items = [(key, label, low, high, step, help_text)
             for _, key, label, low, high, step, help_text in PARAMETERS[stage]]
    items += [(key, *item[1:]) for key, item in EXTRA.get(stage, {}).items()]
    items.append(("parallel_scenes", 'Parallel scenes', 1, 4, 1,
                  'Each scene uses one worker. Increasing this uses more memory.'))
    return [{"id": key, "label": label,
             "type": "bool" if isinstance(values[key], bool) else
                     "int" if isinstance(values[key], int) else "float",
             "min": low, "max": high, "step": step, "unit": "",
             "summary": help_text, "detail": help_text, "group": "Retrieval",
             "higher": help_text, "lower": help_text,
             "cost": 'Upstream stages are not recomputed.'}
            for key, label, low, high, step, help_text in items]


def _configuration(payload, stage, parameters):
    cfg = copy.deepcopy(payload.get("retrieval_config") or PipelineConfig())
    if "retrieval_config" not in payload:
        for key, value in (payload.get("parameters") or {}).items():
            if hasattr(cfg.descriptor, key):
                setattr(cfg.descriptor, key, value)
    for section, key, *_ in PARAMETERS[stage]:
        setattr(getattr(cfg, section), key, parameters.get(key, getattr(getattr(cfg, section), key)))
    return cfg


def _retained_model_preview(model, max_points=18000):
    """Render the normalized source mesh, excluding descriptor-only virtual ground."""
    mesh_path = str(getattr(model, "mesh_path", "") or "")
    if mesh_path:
        try:
            from database import _normalize_mesh, _sample_surface
            from viz import _load_mesh

            mesh = _normalize_mesh(_load_mesh(mesh_path), str(model.synset))
            if mesh is None or not len(mesh.faces):
                raise ValueError('Mesh has no triangles')
            points, _ = _sample_surface(mesh, max_points, np.random.default_rng(0))
            return points, {"geometry_source": "normalized-mesh",
                            "geometry_note": 'Source mesh surface without the virtual descriptor ground plane.'}
        except (OSError, ValueError, KeyError, ImportError) as error:
            note = f"Maillage source indisponible ({type(error).__name__}): showing database point cloud."
    else:
        note = 'No source mesh specified: showing the database point cloud.'
    points = np.asarray(model.cloud.points)
    indices = np.linspace(0, len(points)-1, min(len(points), max_points)).astype(int)
    return points[indices], {"geometry_source": "database-cloud", "geometry_note": note}


def scene_result(source, parameters, stage, progress=None, artifact_path=None):
    import db_store
    import pipeline
    from progressive import _final_revalidate
    from debug_visualizer import _atomic_pickle, _rounded, _to_display
    from runtime_paths import RUNTIME_PATHS

    started = time.perf_counter()
    with Path(source["path"]).open("rb") as stream:
        payload = pickle.load(stream)
    prerequisite = {"matching": "candidate_indices", "verification": "matching_records",
                    "selection": "verified_registrations"}[stage]
    if not isinstance(payload, dict) or prerequisite not in payload or "scene" not in payload:
        raise ValueError(f"Incompatible upstream artifact for {stage} : {prerequisite} absent")
    scene = payload["scene"]
    cfg = _configuration(payload, stage, parameters)
    db_dir = str(payload.get("database_dir") or RUNTIME_PATHS.database_dir)
    files = db_store.model_files(db_dir)
    records = []
    output_registrations = []
    layers, steps, candidates, details = [], [], [], {}
    model_previews = {}

    def points_layer(identifier, points, role, label, **presentation):
        points = np.asarray(points, float).reshape(-1, 3)
        display = _to_display(points, scene.up, scene.ground)
        layers.append(dict(id=identifier, label=label, kind="points", role=role,
                           points=_rounded(display), total=len(points), sampled=len(points),
                           **presentation))
        return display

    scan_points = scene.cloud.points
    scan_indices = np.linspace(0, len(scan_points)-1,
                               min(len(scan_points), 80000 if stage == "selection" else 45000)).astype(int)
    points_layer("surface", scan_points[scan_indices], "surface", 'Unchanged upstream RGB-D scan')
    layers[-1]["kind"] = "surface"
    scan_colors = getattr(scene.cloud, "colors", None)
    if scan_colors is not None:
        layers[-1]["rgb_colors"] = _rounded(np.asarray(scan_colors)[scan_indices], 3)
    features = payload.get("features", [])
    points_layer("query_keypoints", [f.position for f in features], "candidate", 'Scene descriptors')
    base = ["surface", "query_keypoints"]
    steps.append(dict(id="input", label='Saved input', explanation='Reuse upstream artifact without fusion or descriptor recomputation.',
                      input=0, kept=0, rejected=0, visible_layers=base))

    def add_pose(model, transform, trace, label, reason, group_id=None, matches=()):
        index = len(candidates)
        identifier = f"pose-{index}"
        geometry = {"geometry_source": "database-cloud"}
        if stage == "selection" and reason == "selected":
            cache_key = (model.name, str(getattr(model, "mesh_path", "")))
            if cache_key not in model_previews:
                model_previews[cache_key] = _retained_model_preview(model)
            points, geometry = model_previews[cache_key]
        else:
            points = np.asarray(model.cloud.points)
            points = points[np.linspace(0, len(points)-1, min(len(points), 360)).astype(int)] if len(points) else points
        placed = transform is not None
        display = points_layer(identifier, transform.apply(points) if placed else [],
                               "rejected" if reason.startswith("rejected") else "accepted", label,
                               display_role="model", **geometry)
        # Both views describe the same immutable geometry; avoid doubling Python
        # float lists for every pose when RANSAC produces a large shortlist.
        display_points = layers[-1]["points"]
        segments = []
        for match in matches:
            pair = np.asarray([match["model_position"], match["scan_position"]], float)
            if placed:
                pair[0] = transform.apply(pair[:1])[0]
            segments.append(_rounded(_to_display(pair, scene.up, scene.ground)))
        line_id = identifier + "-links"
        layers.append(dict(id=line_id, label="Support de pose", kind="lines", role="vector",
                           total=len(segments), segments=segments))
        candidates.append(dict(index=index, model_index=getattr(model, "debug_index", None),
                               name=model.name, synset=model.synset, rank=index+1,
                               status=reason, best_group_id=group_id, pool_rank=None))
        details[str(index)] = dict(candidate_index=index, name=model.name, synset=model.synset,
            group_id=group_id, placed=placed, model_points=display_points, matches=list(matches),
            **geometry,
            correspondence_segments=segments, correspondences=trace.get("correspondences", 0),
            constellations=trace.get("constellations", 0), local_score=None,
            elapsed_seconds=0, diagnostics={key: value for key, value in trace.items()
                if key not in ("model_preview", "model_keypoint_preview", "constellation_preview",
                               "correspondence_preview", "constellation_scene_support")}, status=reason,
            explanation=REASONS.get(reason, reason),
            pose=dict(source="recorded", theta_deg=round(float(transform.theta)*180/np.pi, 2),
                      scale=float(transform.scale)) if placed else None)
        steps.append(dict(id=identifier, label=f"{model.name.split('_', 1)[0]} · hypothesis {index+1}",
                          explanation=label + " : " + REASONS.get(reason, reason), input=1,
                          kept=int(not reason.startswith("rejected")), rejected=int(reason.startswith("rejected")),
                          candidate_index=index, visible_layers=base+[identifier, line_id]))
        return identifier

    if stage == "matching":
        groups = payload.get("feature_groups") or []
        assignments = payload.get("candidate_group_details") or {}
        for number, model_index in enumerate(payload["candidate_indices"]):
            model_index = int(model_index)
            model = db_store.load_model(files[model_index])
            detail = assignments.get(model_index, assignments.get(str(model_index), {}))
            group_indices = list(detail.get("selected_for_groups") or [])
            best = int(detail.get("best_group", -1))
            if best >= 0:
                group_indices.append(best)
            group_indices = sorted({int(g) for g in group_indices if 0 <= int(g) < len(groups)})
            for group_index in group_indices or [None]:
                feature_indices = (groups[group_index]["feature_indices"] if group_index is not None
                                   else list(range(len(features))))
                scan_features = [features[int(i)] for i in feature_indices]
                trace = {}
                constellations = pipeline.prepare_model_matching(model, scan_features, cfg, scene.up,
                    trace=trace, allow_group_pose_seed=group_index is not None)
                group_id = groups[group_index].get("id", group_index) if group_index is not None else None
                records.append(dict(model_index=model_index, group_id=group_id,
                    feature_indices=feature_indices, constellations=constellations, trace=trace))
                for pose_index, constellation in enumerate(constellations):
                    pairs = pipeline._trace_matching_geometry([], [constellation], model.features, scan_features)[1][0]["inlier_pairs"]
                    reason = "group_pose_seed_fallback" if trace.get("group_pose_seed_fallback") else "ransac_constellation"
                    add_pose(model, constellation.transform, trace,
                             f"{model.name} · pose {pose_index+1}", reason, group_id, pairs)
                if not constellations:
                    add_pose(model, None, trace, model.name, trace["status"], group_id)
            if progress:
                elapsed = time.perf_counter()-started
                remaining = elapsed/(number+1)*(len(payload["candidate_indices"])-number-1)
                progress(f"Matching {number+1}/{len(payload['candidate_indices'])} : {model.name} · estimated remaining {remaining/60:.1f} min")
        payload["matching_records"] = records
    elif stage == "verification":
        for number, record in enumerate(payload["matching_records"]):
            model = db_store.load_model(files[record["model_index"]])
            scan_features = [features[int(i)] for i in record["feature_indices"]]
            trace = copy.deepcopy(record["trace"])
            regs = pipeline._match_model_candidates(model, scan_features, scene.cloud, cfg, scene.up,
                parameters["coverage_threshold"], trace=trace,
                allow_group_pose_seed=record["group_id"] is not None,
                prepared_constellations=record["constellations"])
            for reg in regs:
                if record["group_id"] is not None:
                    pipeline._tag_group_result(reg, record["group_id"], scan_features)
            validated, validation = _final_revalidate(regs, db_dir, scene, cfg,
                parameters["coverage_threshold"], parameters["final_reverse_gate"])
            output_registrations.extend(validated)
            trace["final_validation"] = validation
            records.append(dict(**record, verification_trace=trace))
            from transforms import GroundTransform
            for pose in trace.get("verification", {}).get("poses", []):
                transform = (GroundTransform(float(pose["theta_deg"])*np.pi/180,
                             float(pose["scale"]), np.asarray(pose["translation"]), scene.up)
                             if "translation" in pose else None)
                # Failed ICP has no refined transform: show its input seed explicitly.
                if transform is None and record["constellations"]:
                    transform = record["constellations"][int(pose["rank"])-1].transform
                add_pose(model, transform, {**trace, "pose_verification": pose},
                         f"{model.name} · ICP {pose['rank']}", pose["status"], record["group_id"])
            for reg in regs:
                reason = "verified" if any(reg is item for item in validated) else "rejected_final_validation"
                add_pose(model, reg.transform, {**trace, "registration": pipeline._trace_registration(reg)},
                         model.name + " · revalidation", reason, record["group_id"])
            if not regs:
                add_pose(model, None, trace, model.name, trace.get("status", "rejected_verification"), record["group_id"])
            if progress:
                progress(f"Verification {number+1}/{len(payload['matching_records'])} : {len(validated)} validated pose(s) · {model.name}")
        payload["verified_registrations"] = output_registrations
        payload["verification_records"] = records
    else:
        regs = payload["verified_registrations"]
        # Older verification artifacts do not yet carry group support. Compute
        # it from their exact recorded features without changing any pose.
        from database import object_surface_cloud
        from verify import group_surface_coverage
        groups = payload.get("feature_groups") or []
        for reg in regs:
            group_id = getattr(reg, "_query_group_id", None)
            if group_id is None or not 0 <= group_id < len(groups):
                continue
            group_features = [features[int(i)] for i in groups[group_id]["feature_indices"]]
            pipeline._tag_group_result(reg, group_id, group_features)
            model = db_store.load_model(files[reg._model_index])
            support = group_surface_coverage(
                object_surface_cloud(model), scene.cloud, reg.transform,
                getattr(reg, "_query_group_points", None), cfg.keypoint.neighbor_radius,
                scene.up, getattr(scene, "ground", None))
            if support is not None:
                reg._query_group_reverse, reg._query_group_surface_points, reg._query_group_bounds = support
        selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics(regs, scene, cfg,
            **{key: parameters[key] for key in EXTRA["selection"]})
        selected_layers = []
        for reg, diagnostic in zip(sorted(regs, key=pipeline._registration_rank), diagnostics):
            model = db_store.load_model(files[reg._model_index])
            identifier = add_pose(model, reg.transform, {**diagnostic, "registration": pipeline._trace_registration(reg)},
                                 reg.model_name, diagnostic["status"], getattr(reg, "_query_group_id", None))
            if any(reg is item for item in selected):
                selected_layers.append(identifier)
                model_keypoints = np.asarray([feature.position for feature in model.features], float).reshape(-1, 3)
                keypoint_id = identifier + "-keypoints"
                points_layer(keypoint_id, reg.transform.apply(model_keypoints),
                             "retained", 'Retained model keypoints', display_role="model-keypoints")
                selected_layers.append(keypoint_id)
        steps.append(dict(id="final", label='Final result', explanation='Retained models on the RGB-D scan. Switch between Scene, Models and Both; yellow keypoints are optional. An empty result is a valid decision.',
                          input=len(regs), kept=len(selected), rejected=len(regs)-len(selected),
                          visible_layers=["surface"]+selected_layers,
                          presentation="final-selection"))
        payload["selected_registrations"] = selected
        payload["selection_diagnostics"] = diagnostics
        output_registrations = selected
    payload["retrieval_config"] = cfg
    payload["database_dir"] = db_dir
    artifacts = {}
    if artifact_path is not None:
        _atomic_pickle(Path(artifact_path), payload)
        artifacts[stage] = dict(path=str(Path(artifact_path).resolve()), format="pickle")
    display = _to_display(scene.cloud.points, scene.up, scene.ground)
    return dict(schema_version=1, stage=stage, scene={**source, "points": len(scan_points), "cache_hit": True},
        duration_seconds=round(time.perf_counter()-started, 3), ground=float(scene.ground),
        bounds=dict(min=_rounded(display.min(axis=0)) if len(display) else [0,0,0],
                    max=_rounded(display.max(axis=0)) if len(display) else [0,0,0]),
        steps=steps, layers=layers, artifacts=artifacts, top_k=candidates,
        candidate_details=details, histograms={}, database_dir=db_dir,
        metrics=dict(retained=len(output_registrations) if stage != "matching" else
                     sum(bool(r["constellations"]) for r in records), hypotheses=len(candidates)))
