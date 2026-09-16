"""Reproducible parameter sweeps over cached keypoint debug scenes."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


def _parse_scalar(raw, reference=None):
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", "auto"}:
        return None
    if isinstance(reference, bool):
        raise ValueError(f"Invalid boolean value: {raw}")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(text)
    if isinstance(reference, float):
        return float(text)
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def parse_assignment(raw, defaults):
    if "=" not in raw:
        raise ValueError(f"Expected parameter as name=value: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if key not in defaults:
        raise ValueError(f"Unknown keypoint parameter: {key}")
    return key, _parse_scalar(value, defaults[key])


def parse_grid(raw, defaults):
    if "=" not in raw:
        raise ValueError(f"Expected grid as name=v1,v2: {raw}")
    key, raw_values = raw.split("=", 1)
    key = key.strip()
    if key not in defaults:
        raise ValueError(f"Unknown keypoint parameter: {key}")
    values = [
        _parse_scalar(item, defaults[key])
        for item in raw_values.split(",") if item.strip()
    ]
    if not values:
        raise ValueError(f"Empty grid for {key}")
    return key, values


def parse_source(raw):
    if "=" not in raw:
        raise ValueError(f"Expected source as name=path: {raw}")
    label, raw_path = raw.split("=", 1)
    path = Path(raw_path.strip()).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source de sweep not found : {path}")
    source_id = "-".join(label.strip().lower().split()) or path.stem.lower()
    return {
        "id": source_id,
        "label": label.strip() or path.stem,
        "path": str(path),
        "enabled": True,
    }


def settings_from_json(path, defaults):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("parameters", payload)
    if not isinstance(raw, dict):
        raise ValueError('The settings JSON must contain a parameters object')
    aliases = {"spatial_keypoints": "spatial_keypoint_balance"}
    settings = {}
    for raw_key, value in raw.items():
        key = aliases.get(raw_key, raw_key)
        if key in defaults:
            settings[key] = value
    return settings


def parameter_combinations(grid):
    keys = list(grid)
    if not keys:
        return [{}]
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _layer_indices(result, layer_id):
    for layer in result.get("layers", []):
        if layer.get("id") == layer_id:
            return set(int(value) for value in layer.get("source_index", []))
    return set()


def _layer_points(result, layer_id):
    for layer in result.get("layers", []):
        if layer.get("id") == layer_id:
            return np.asarray(layer.get("points", []), float).reshape(-1, 3)
    return np.zeros((0, 3), float)


def summarize_scene(result, coverage_cell=0.35):
    metrics = dict(result.get("metrics", {}))
    retained = _layer_indices(result, "retained")
    harris = _layer_indices(result, "harris_accepted")
    corners = _layer_indices(result, "corner_2d_accepted")
    wall = _layer_indices(result, "wall_keypoint_affinity")
    high_wall = _layer_indices(result, "wall_score_high")
    points = _layer_points(result, "retained")
    if len(points):
        cells = np.floor(points / max(float(coverage_cell), 1e-6)).astype(int)
        spatial_cells = len({tuple(row) for row in cells.tolist()})
    else:
        spatial_cells = 0
    retained_count = int(metrics.get("retained", len(retained)))
    quality_reintroduced = int(metrics.get("quality_reintroduced", 0))
    wall_reintroduced = int(metrics.get("wall_budget_reintroduced", 0))
    component_reintroduced = int(
        metrics.get("component_budget_reintroduced", 0)
    )
    return {
        "detected": int(metrics.get("detected", 0)),
        "after_wall_filter": int(metrics.get("after_wall_filter", 0)),
        "retained": retained_count,
        "retained_harris_3d": int(len(retained & harris)),
        "retained_corner_2d": int(len(retained & corners)),
        "retained_wall_affinity": int(len(retained & wall)),
        "retained_high_wall": int(len(retained & high_wall)),
        "retained_high_wall_ratio": round(
            len(retained & high_wall) / max(1, retained_count), 6,
        ),
        "spatial_cells": int(spatial_cells),
        "component_count": int(metrics.get("component_budget_count", 0)),
        "component_rejected": int(
            metrics.get("component_budget_rejected", 0)
        ),
        "pre_harris_visibility_rejected": int(
            metrics.get("pre_harris_visibility_rejected", 0)
        ),
        "quality_reintroduced": quality_reintroduced,
        "wall_budget_reintroduced": wall_reintroduced,
        "component_budget_reintroduced": component_reintroduced,
        "total_reintroduced": (
            quality_reintroduced + wall_reintroduced + component_reintroduced
        ),
        "duration_seconds": float(result.get("duration_seconds", 0.0)),
    }


def _ratio(value, baseline):
    return 1.0 if baseline <= 0 else float(value) / float(baseline)


def aggregate_trial(scene_metrics, baseline_metrics):
    rows = []
    for scene, baseline in zip(scene_metrics, baseline_metrics):
        rows.append({
            "scene": scene["scene"],
            "retained_ratio": _ratio(scene["retained"], baseline["retained"]),
            "spatial_recall": _ratio(
                scene["spatial_cells"], baseline["spatial_cells"],
            ),
            "harris_recall": _ratio(
                scene["retained_harris_3d"], baseline["retained_harris_3d"],
            ),
            "corner_2d_recall": _ratio(
                scene["retained_corner_2d"], baseline["retained_corner_2d"],
            ),
            "descriptor_informative_recall": _ratio(
                scene.get("descriptor_informative", 0),
                baseline.get("descriptor_informative", 0),
            ),
            "descriptor_nonwall_recall": _ratio(
                scene.get("descriptor_informative_nonwall", 0),
                baseline.get("descriptor_informative_nonwall", 0),
            ),
        })
    total_retained = sum(item["retained"] for item in scene_metrics)
    baseline_retained = sum(item["retained"] for item in baseline_metrics)
    minimum_spatial = min((item["spatial_recall"] for item in rows), default=0.0)
    minimum_harris = min((item["harris_recall"] for item in rows), default=0.0)
    minimum_corner = min((item["corner_2d_recall"] for item in rows), default=0.0)
    descriptor_available = bool(scene_metrics) and all(
        "descriptor_evaluated" in item for item in scene_metrics
    )
    minimum_descriptor = min((
        item["descriptor_informative_recall"] for item in rows
    ), default=0.0)
    minimum_descriptor_nonwall = min((
        item["descriptor_nonwall_recall"] for item in rows
    ), default=0.0)
    feasible = (
        minimum_spatial >= 0.85
        and minimum_harris >= 0.75
        and minimum_corner >= 0.75
        and (
            not descriptor_available
            or (
                minimum_descriptor >= 0.75
                and minimum_descriptor_nonwall >= 0.75
            )
        )
    )
    descriptor_informative = sum(
        item.get("descriptor_informative", 0) for item in scene_metrics
    )
    descriptor_wall = sum(
        item.get("descriptor_informative_wall", 0) for item in scene_metrics
    )
    retained_high_wall = sum(
        item.get("retained_high_wall", 0) for item in scene_metrics
    )
    total_reintroduced = sum(
        item.get("total_reintroduced", 0) for item in scene_metrics
    )
    pre_harris_visibility_rejected = sum(
        item.get("pre_harris_visibility_rejected", 0)
        for item in scene_metrics
    )
    candidate_recall_rows = [
        item for item in scene_metrics
        if item.get("candidate_recall_evaluated")
    ]
    candidate_recall_available = bool(candidate_recall_rows)
    minimum_candidate_recall = min((
        float(item.get("candidate_target_recall", 0.0))
        for item in candidate_recall_rows
    ), default=0.0)
    if candidate_recall_available:
        feasible = feasible and minimum_candidate_recall >= 1.0
    return {
        "total_retained": int(total_retained),
        "retained_ratio": round(_ratio(total_retained, baseline_retained), 6),
        "min_spatial_recall": round(minimum_spatial, 6),
        "min_harris_recall": round(minimum_harris, 6),
        "min_corner_2d_recall": round(minimum_corner, 6),
        "descriptor_utility_available": descriptor_available,
        "min_descriptor_informative_recall": round(
            minimum_descriptor, 6,
        ) if descriptor_available else None,
        "min_descriptor_nonwall_recall": round(
            minimum_descriptor_nonwall, 6,
        ) if descriptor_available else None,
        "descriptor_informative": int(descriptor_informative),
        "descriptor_informative_nonwall": int(sum(
            item.get("descriptor_informative_nonwall", 0)
            for item in scene_metrics
        )),
        "descriptor_informative_wall": int(descriptor_wall),
        "descriptor_informative_wall_ratio": round(
            descriptor_wall / max(1, descriptor_informative), 6,
        ) if descriptor_available else None,
        "candidate_recall_available": candidate_recall_available,
        "min_candidate_target_recall": (
            round(minimum_candidate_recall, 6)
            if candidate_recall_available else None
        ),
        "retained_high_wall": int(retained_high_wall),
        "retained_high_wall_ratio": round(
            retained_high_wall / max(1, total_retained), 6,
        ),
        "pre_harris_visibility_rejected": int(
            pre_harris_visibility_rejected
        ),
        "total_reintroduced": int(total_reintroduced),
        "reintroduced_ratio": round(
            total_reintroduced / max(1, total_retained), 6,
        ),
        "mean_spatial_recall": round(float(np.mean([
            item["spatial_recall"] for item in rows
        ])), 6),
        "duration_seconds": round(sum(
            item["duration_seconds"] for item in scene_metrics
        ), 3),
        "feasible": bool(feasible),
        "per_scene_recall": rows,
    }


def _trial_id(values):
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _signature(
        sources, settings, grid, coverage_cell,
        descriptor_evaluator=None):
    source_state = []
    for source in sources:
        path = Path(source["path"])
        stat = path.stat()
        source_state.append({
            "id": source["id"], "path": str(path.resolve()),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        })
    payload = {
        "sources": source_state, "settings": settings, "grid": grid,
        "coverage_cell": coverage_cell,
        "descriptor_utility": (
            descriptor_evaluator.signature()
            if descriptor_evaluator is not None else None
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _pareto_objectives(trial):
    aggregate = trial["aggregate"]
    descriptor_available = bool(
        aggregate.get("descriptor_utility_available")
    )
    quality_recall = (
        aggregate.get("min_descriptor_nonwall_recall")
        if descriptor_available
        else aggregate.get("min_harris_recall")
    )
    if aggregate.get("candidate_recall_available"):
        quality_recall = min(
            float(quality_recall or 0.0),
            float(aggregate.get("min_candidate_target_recall") or 0.0),
        )
    wall_ratio = (
        aggregate.get("descriptor_informative_wall_ratio")
        if descriptor_available
        else aggregate.get("retained_high_wall_ratio")
    )
    return np.asarray([
        float(aggregate["total_retained"]),
        -float(aggregate["min_spatial_recall"]),
        -float(quality_recall or 0.0),
        float(wall_ratio or 0.0),
        float(aggregate.get("reintroduced_ratio", 0.0)),
    ], float)


def pareto_front_trials(trials, feasible_only=False):
    """Return non-dominated completed trials for minimization objectives."""
    candidates = [
        trial for trial in trials
        if trial.get("status") == "completed"
        and (
            not feasible_only
            or bool(trial.get("aggregate", {}).get("feasible"))
        )
    ]
    front = []
    for index, trial in enumerate(candidates):
        values = _pareto_objectives(trial)
        dominated = False
        for other_index, other in enumerate(candidates):
            if other_index == index:
                continue
            other_values = _pareto_objectives(other)
            if np.all(other_values <= values) and np.any(
                    other_values < values):
                dominated = True
                break
        if not dominated:
            front.append(trial)
    return sorted(front, key=lambda item: tuple(_pareto_objectives(item)))


def select_pareto_profiles(front, limit=3):
    """Pick quality, knee-like balanced, and efficiency representatives."""
    front = list(front or [])
    if not front:
        return []
    objectives = np.vstack([_pareto_objectives(trial) for trial in front])
    lower = objectives.min(axis=0)
    span = np.maximum(objectives.max(axis=0) - lower, 1e-12)
    normalized = (objectives - lower) / span
    descriptor_available = any(
        trial["aggregate"].get("descriptor_utility_available")
        for trial in front
    )
    quality_index = min(
        range(len(front)),
        key=lambda index: (
            objectives[index, 2], objectives[index, 1],
            objectives[index, 3], objectives[index, 0],
        ),
    )
    efficiency_index = min(
        range(len(front)),
        key=lambda index: (
            objectives[index, 0], objectives[index, 3],
            objectives[index, 2],
        ),
    )
    weights = np.asarray(
        [
            0.27, 0.23,
            0.32 if descriptor_available else 0.22,
            0.10, 0.08,
        ],
        float,
    )
    balanced_index = int(np.argmin(np.sqrt(
        np.sum(weights[None, :] * normalized ** 2, axis=1)
    )))
    selected_indices = [quality_index]
    if efficiency_index not in selected_indices:
        selected_indices.append(efficiency_index)
    if balanced_index not in selected_indices:
        selected_indices.append(balanced_index)
    elif len(selected_indices) < min(int(limit), len(front)):
        remaining = [
            index for index in range(len(front))
            if index not in selected_indices
        ]
        if remaining:
            # A duplicate knee is replaced by the most complementary Pareto
            # extreme, rather than by the first trial in sort order.
            balanced_index = max(
                remaining,
                key=lambda index: min(
                    np.linalg.norm(
                        normalized[index] - normalized[selected_index]
                    )
                    for selected_index in selected_indices
                ),
            )
            selected_indices.append(balanced_index)

    role_by_index = {
        quality_index: "quality",
        balanced_index: "balanced",
        efficiency_index: "efficiency",
    }
    ordered = []
    for role, index in (
        ("quality", quality_index),
        ("balanced", balanced_index),
        ("efficiency", efficiency_index),
    ):
        if index in selected_indices and index not in [
                item[1] for item in ordered]:
            ordered.append((role, index))
    for index in selected_indices:
        if index not in [item[1] for item in ordered]:
            ordered.append((role_by_index.get(index, "alternative"), index))
    return [
        {"role": role, "trial": front[index]["id"]}
        for role, index in ordered[:int(limit)]
    ]


def _rank_trials(payload):
    completed = [
        trial for trial in payload["trials"]
        if trial.get("status") == "completed"
    ]
    feasible = [trial for trial in completed if trial["aggregate"]["feasible"]]
    ranked = sorted(
        completed,
        key=lambda trial: (
            not trial["aggregate"]["feasible"],
            -(trial["aggregate"].get(
                "min_candidate_target_recall", 0.0,
            ) or 0.0),
            trial["aggregate"]["total_retained"],
            trial["aggregate"].get(
                "descriptor_informative_wall_ratio", 0.0,
            ) or 0.0,
            -(trial["aggregate"].get(
                "min_descriptor_nonwall_recall", 0.0,
            ) or 0.0),
            -trial["aggregate"]["min_spatial_recall"],
            -trial["aggregate"]["min_harris_recall"],
            -trial["aggregate"]["min_corner_2d_recall"],
        ),
    )
    payload["ranking"] = [trial["id"] for trial in ranked]
    payload["recommended_trial"] = (
        min(
            feasible,
            key=lambda trial: (
                -(trial["aggregate"].get(
                    "min_candidate_target_recall", 0.0,
                ) or 0.0),
                trial["aggregate"]["total_retained"],
                trial["aggregate"].get(
                    "descriptor_informative_wall_ratio", 0.0,
                ) or 0.0,
                -(trial["aggregate"].get(
                    "min_descriptor_nonwall_recall", 0.0,
                ) or 0.0),
                -trial["aggregate"]["min_spatial_recall"],
                -trial["aggregate"]["min_harris_recall"],
                -trial["aggregate"]["min_corner_2d_recall"],
            ),
        )["id"] if feasible else None
    )
    payload["best_effort_trial"] = ranked[0]["id"] if ranked else None
    front = pareto_front_trials(completed)
    feasible_front = pareto_front_trials(completed, feasible_only=True)
    payload["pareto_front"] = [trial["id"] for trial in front]
    payload["pareto_feasible_front"] = [
        trial["id"] for trial in feasible_front
    ]
    payload["pareto_profiles"] = select_pareto_profiles(
        feasible_front or front,
    )


def run_keypoint_sweep(
        sources, settings, grid, out, coverage_cell=0.35, resume=True,
        progress=None, scene_runner=None, descriptor_evaluator=None):
    from debug_visualizer import default_parameters, normalized_parameters
    from debug_visualizer import _keypoint_scene_result

    scene_runner = scene_runner or _keypoint_scene_result
    parameters = default_parameters("keypoints")
    parameters.update(settings)
    parameters = normalized_parameters(parameters, stage="keypoints")
    signature = _signature(
        sources, settings, grid, coverage_cell, descriptor_evaluator,
    )
    out = Path(out)
    payload = None
    if resume and out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError(
                'The existing sweep uses different sources or parameters; choose another --out or use --no-resume.'
            )
    if payload is None:
        payload = {
            "schema_version": 3,
            "kind": "keypoint_parameter_sweep",
            "signature": signature,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sources": sources,
            "settings": settings,
            "grid": grid,
            "coverage_cell": float(coverage_cell),
            "descriptor_utility": (
                descriptor_evaluator.signature()
                if descriptor_evaluator is not None else None
            ),
            "baseline": [],
            "trials": [],
            "ranking": [],
            "recommended_trial": None,
            "pareto_front": [],
            "pareto_feasible_front": [],
            "pareto_profiles": [],
        }

    def evaluate_scene(source, scene_parameters, label):
        if descriptor_evaluator is None:
            result = scene_runner(source, scene_parameters, progress=None)
            return summarize_scene(result, coverage_cell)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix="keypoint-sweep-", dir=str(out.parent)) as folder:
            artifact_path = Path(folder) / f"{source['id']}-{label}.pkl"
            result = scene_runner(
                source, scene_parameters, progress=None,
                artifact_path=artifact_path,
            )
            if not artifact_path.exists():
                raise ValueError(
                    'Descriptor utility requires a .pkl source containing the fused scene and its volume.'
                )
            descriptor_metrics = descriptor_evaluator.evaluate(
                artifact_path, progress=None, source=source,
            )
        return {
            **summarize_scene(result, coverage_cell),
            **descriptor_metrics,
        }

    if not payload["baseline"]:
        if progress:
            progress('baseline without regional budget')
        baseline_parameters = dict(parameters)
        baseline_parameters["component_budget_enabled"] = False
        if "max_scan_keypoints" in grid:
            positive_budgets = [
                int(value) for value in grid["max_scan_keypoints"]
                if int(value) > 0
            ]
            if positive_budgets:
                baseline_parameters["max_scan_keypoints"] = max(
                    positive_budgets
                )
        for source in sources:
            payload["baseline"].append({
                "scene": source["id"],
                **evaluate_scene(source, baseline_parameters, "baseline"),
            })
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _atomic_json(out, payload)

    existing = {trial["id"] for trial in payload["trials"]}
    combinations = parameter_combinations(grid)
    for index, values in enumerate(combinations, start=1):
        trial_id = _trial_id(values)
        if trial_id in existing:
            if progress:
                progress(f"essai {index}/{len(combinations)} already present")
            continue
        if progress:
            detail = ", ".join(f"{key}={value}" for key, value in values.items())
            progress(f"essai {index}/{len(combinations)}: {detail}")
        trial_parameters = dict(parameters)
        trial_parameters.update(values)
        scene_metrics = []
        trial = {
            "id": trial_id, "parameters": values, "status": "running",
            "scenes": scene_metrics,
        }
        payload["trials"].append(trial)
        _atomic_json(out, payload)
        try:
            for source in sources:
                scene_metrics.append({
                    "scene": source["id"],
                    **evaluate_scene(source, trial_parameters, trial_id),
                })
            trial["aggregate"] = aggregate_trial(
                scene_metrics, payload["baseline"],
            )
            trial["status"] = "completed"
        except Exception as exc:
            trial["status"] = "failed"
            trial["error"] = f"{type(exc).__name__}: {exc}"
        _rank_trials(payload)
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _atomic_json(out, payload)
    _rank_trials(payload)
    _atomic_json(out, payload)
    return payload
