"""Calibrate descriptor distance and within-model ratio thresholds.

The calibration reuses descriptor debug artifacts. It does not rebuild RGB-D
fusion, keypoints, or scan descriptors. Labels are category-level proxies:
models from categories expected in a scene are positive, while unexpected
categories and wall-supported keypoints are negative. Full retrieval remains
the final validation because no per-keypoint ground truth is available.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import db_store
from descriptor_utility import (
    _compatible_model_features,
    _limited_thetas,
    _top_model_features,
)
from descriptors import descriptor_distance_batch, effective_distance_unit


DEFAULT_EXPECTED_SYNSETS = {
    "office": {"03001627", "04379243"},
    "ikea-table": {"04379243"},
    "single-chair": {"03001627"},
}
DEFAULT_THRESHOLDS = (
    8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048,
)
DEFAULT_RATIOS = (0.0, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)


@dataclass(frozen=True)
class CalibrationRecord:
    scene: str
    scan_index: int
    model: str
    synset: str
    best_distance: float
    second_distance: float
    wall_score: float
    expected_category: bool
    scan_position: tuple = (0.0, 0.0, 0.0)
    model_feature_index: int = -1


def _finite_number(value, default=float("inf")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _sample_indices(length: int, limit: int):
    if limit <= 0 or length <= limit:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, limit).astype(int))


def _best_feature_distances(model_features, scan_feature, cfg, up, rotations):
    distances = []
    compatible = _compatible_model_features(model_features, scan_feature, cfg)
    for feature_index in compatible:
        model_feature = model_features[feature_index]
        thetas = _limited_thetas(
            model_feature, scan_feature, cfg, up, rotations,
        )
        if not len(thetas):
            continue
        values = descriptor_distance_batch(
            model_feature.descriptor, scan_feature.descriptor,
            cfg.descriptor, -thetas, up,
        )
        if len(values):
            value = float(np.min(values))
            if math.isfinite(value):
                distances.append((value, int(feature_index)))
    distances.sort(key=lambda item: item[0])
    best = distances[0][0] if distances else float("inf")
    second = distances[1][0] if len(distances) > 1 else float("inf")
    best_index = distances[0][1] if distances else -1
    return best, second, best_index


def collect_artifact_records(
        artifact_path, db_dir, scene_id=None, expected_synsets=None, *,
        max_scan_features=120, max_model_features=24, rotations=8,
        progress=None):
    """Recompute distances in the current cell unit from one debug artifact."""
    artifact_path = Path(artifact_path).resolve()
    with artifact_path.open("rb") as stream:
        payload = pickle.load(stream)
    features = list(payload.get("features") or [])
    if not features:
        raise ValueError(f"No descriptors in {artifact_path}")
    scene_id = str(scene_id or artifact_path.stem.split(".", 1)[0])
    expected = set(
        expected_synsets
        or DEFAULT_EXPECTED_SYNSETS.get(scene_id, set())
    )
    if not expected:
        raise ValueError(f"Expected categories missing for {scene_id}")

    cfg = db_store.load_cfg(str(Path(db_dir).resolve()))
    cfg.descriptor.distance_unit = 0.0
    cfg.matching.max_scan_keypoints = 0
    cfg.matching.floor_height = -1e6
    up = np.asarray(payload["scene"].up, float)
    up /= max(np.linalg.norm(up), 1e-12)

    files = db_store.model_files(str(Path(db_dir).resolve()))
    candidate_indices = list(payload.get("candidate_indices") or [])
    if not candidate_indices:
        raise ValueError(
            f"L'artefact {artifact_path} ne contient pas candidate_indices"
        )
    models = [
        db_store.load_model(files[int(index)])
        for index in candidate_indices
        if 0 <= int(index) < len(files)
    ]
    keypoints = payload.get("keypoints")
    wall_scores = np.asarray(
        getattr(keypoints, "wall_scores", np.zeros(len(features))), float,
    )
    selected_scan = _sample_indices(len(features), int(max_scan_features))
    prepared_models = [
        (model, _top_model_features(model.features, max_model_features))
        for model in models
    ]
    records = []
    for order, scan_index in enumerate(selected_scan, 1):
        scan_feature = features[int(scan_index)]
        wall_score = (
            float(wall_scores[int(scan_index)])
            if int(scan_index) < len(wall_scores) else 0.0
        )
        for model, model_features in prepared_models:
            best, second, best_index = _best_feature_distances(
                model_features, scan_feature, cfg, up, rotations,
            )
            records.append(CalibrationRecord(
                scene=scene_id,
                scan_index=int(scan_index),
                model=str(model.name),
                synset=str(model.synset),
                best_distance=best,
                second_distance=second,
                wall_score=wall_score,
                expected_category=str(model.synset) in expected,
                scan_position=tuple(
                    float(value) for value in scan_feature.position
                ),
                model_feature_index=int(best_index),
            ))
        if progress and (order == 1 or order % 10 == 0 or order == len(selected_scan)):
            progress(
                f"[{scene_id}] {order}/{len(selected_scan)} descripteur(s) "
                f"x {len(prepared_models)} model(s)"
            )
    return records, {
        "scene": scene_id,
        "artifact": str(artifact_path),
        "expected_synsets": sorted(expected),
        "scan_features": int(len(selected_scan)),
        "candidate_models": int(len(prepared_models)),
        "distance_unit": float(effective_distance_unit(cfg.descriptor)),
        "distance_exponent": float(cfg.descriptor.distance_exponent),
    }


def _accepted(record: CalibrationRecord, threshold: float, ratio: float):
    if not math.isfinite(record.best_distance):
        return False
    if record.best_distance >= float(threshold):
        return False
    if ratio <= 0.0:
        return True
    # This mirrors matching.match_features: the ratio is only applied when a
    # second feature also survives the absolute descriptor threshold.
    if not math.isfinite(record.second_distance):
        return True
    if record.second_distance >= float(threshold):
        return True
    if record.second_distance <= 1e-12:
        return False
    return record.best_distance / record.second_distance <= float(ratio)


def evaluate_setting(records: Sequence[CalibrationRecord], threshold, ratio,
                     wall_threshold=0.20, min_model_correspondences=4,
                     min_scan_spread=0.12):
    labels = np.asarray([
        item.expected_category and item.wall_score < wall_threshold
        for item in records
    ], bool)
    predicted = np.asarray([
        _accepted(item, threshold, ratio) for item in records
    ], bool)
    tp = int(np.count_nonzero(labels & predicted))
    fp = int(np.count_nonzero(~labels & predicted))
    fn = int(np.count_nonzero(labels & ~predicted))
    tn = int(np.count_nonzero(~labels & ~predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (recall + (1.0 - fpr))

    model_groups = {}
    for index, item in enumerate(records):
        key = (item.scene, item.model, item.synset, item.expected_category)
        group = model_groups.setdefault(key, {})
        if predicted[index]:
            group[(item.scan_index, item.model_feature_index)] = (
                item.scan_position, item.model_feature_index,
            )
    viable_models = []
    for (scene, model, synset, expected), scan_points in model_groups.items():
        values = list(scan_points.values())
        points = np.asarray([item[0] for item in values], float)
        count = len(points)
        distinct_model_features = len({item[1] for item in values if item[1] >= 0})
        if count > 1:
            centered = points - points.mean(axis=0)
            spread = float(2.0 * np.max(np.linalg.norm(centered, axis=1)))
        else:
            spread = 0.0
        if (
                count >= int(min_model_correspondences)
                and distinct_model_features >= int(min_model_correspondences)
                and spread >= float(min_scan_spread)):
            viable_models.append({
                "scene": scene, "model": model, "synset": synset,
                "expected_category": bool(expected),
                "correspondences": count,
                "distinct_model_features": distinct_model_features,
                "scan_spread": round(spread, 6),
            })
    expected_viable = sum(
        item["expected_category"] for item in viable_models
    )
    unexpected_viable = len(viable_models) - expected_viable
    unexpected_model_count = len({
        (item.scene, item.model) for item in records
        if not item.expected_category
    })

    per_scene = []
    for scene in sorted({item.scene for item in records}):
        indices = np.asarray([item.scene == scene for item in records], bool)
        positive = labels & indices
        accepted_positive = positive & predicted
        per_scene.append({
            "scene": scene,
            "positives": int(np.count_nonzero(positive)),
            "accepted_positives": int(np.count_nonzero(accepted_positive)),
            "positive_recall": round(
                np.count_nonzero(accepted_positive)
                / max(1, np.count_nonzero(positive)), 6,
            ),
            "accepted_total": int(np.count_nonzero(predicted & indices)),
            "viable_expected_by_synset": {
                synset: sum(
                    item["scene"] == scene
                    and item["synset"] == synset
                    and item["expected_category"]
                    for item in viable_models
                )
                for synset in sorted({
                    item.synset for item in records
                    if item.scene == scene and item.expected_category
                })
            },
            "viable_unexpected_models": sum(
                item["scene"] == scene and not item["expected_category"]
                for item in viable_models
            ),
        })
    return {
        "threshold": float(threshold),
        "ratio": float(ratio),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "false_positive_rate": round(fpr, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "accepted": int(np.count_nonzero(predicted)),
        "viable_expected_models": int(expected_viable),
        "viable_unexpected_models": int(unexpected_viable),
        "viable_unexpected_rate": round(
            unexpected_viable / max(1, unexpected_model_count), 6,
        ),
        "per_scene": per_scene,
    }


def select_setting(
        records: Sequence[CalibrationRecord],
        thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
        ratios: Iterable[float] = DEFAULT_RATIOS,
        *, wall_threshold=0.20, min_scene_recall=0.02,
        min_scene_positives=2, min_model_correspondences=4,
        min_viable_models_per_category=2, min_scan_spread=0.12):
    rows = []
    for threshold in thresholds:
        for ratio in ratios:
            row = evaluate_setting(
                records, threshold, ratio, wall_threshold=wall_threshold,
                min_model_correspondences=min_model_correspondences,
                min_scan_spread=min_scan_spread,
            )
            positive_feasible = all(
                item["accepted_positives"] >= min(
                    int(min_scene_positives), item["positives"],
                )
                and item["positive_recall"] >= float(min_scene_recall)
                for item in row["per_scene"] if item["positives"] > 0
            )
            constellation_feasible = all(
                count >= min(
                    int(min_viable_models_per_category),
                    len({
                        record.model for record in records
                        if record.scene == item["scene"]
                        and record.synset == synset
                        and record.expected_category
                    }),
                )
                for item in row["per_scene"]
                for synset, count in item["viable_expected_by_synset"].items()
            )
            row["feasible"] = bool(
                positive_feasible and constellation_feasible
            )
            rows.append(row)
    ranked = sorted(rows, key=_setting_sort_key)
    return ranked[0], rows


def _setting_sort_key(item):
    return (
        not item["feasible"],
        item["viable_unexpected_rate"],
        -item["viable_expected_models"],
        item["accepted"],
        item["threshold"],
        item["ratio"] if item["ratio"] > 0 else 2.0,
        -item["balanced_accuracy"],
    )


def _quantiles(values):
    values = np.asarray([
        value for value in values if math.isfinite(float(value))
    ], float)
    if not len(values):
        return {}
    return {
        name: round(float(value), 6)
        for name, value in zip(
            ("q05", "q10", "q25", "q50", "q75", "q90", "q95"),
            np.quantile(values, (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)),
        )
    }


def build_report(records, sources, thresholds=DEFAULT_THRESHOLDS,
                 ratios=DEFAULT_RATIOS, wall_threshold=0.20):
    recommended, rows = select_setting(
        records, thresholds, ratios, wall_threshold=wall_threshold,
    )
    positive = [
        item for item in records
        if item.expected_category and item.wall_score < wall_threshold
    ]
    negative = [
        item for item in records
        if not (item.expected_category and item.wall_score < wall_threshold)
    ]
    ratios_all = [
        item.best_distance / item.second_distance
        for item in records
        if math.isfinite(item.best_distance)
        and math.isfinite(item.second_distance)
        and item.second_distance > 1e-12
    ]
    return {
        "schema_version": 1,
        "kind": "descriptor_threshold_calibration",
        "supervision": (
            "weak category labels plus operational constellation viability; "
            "full retrieval is required for final validation"
        ),
        "sources": sources,
        "record_count": len(records),
        "positive_proxy_count": len(positive),
        "negative_proxy_count": len(negative),
        "distributions": {
            "positive_best_distance": _quantiles(
                item.best_distance for item in positive
            ),
            "negative_best_distance": _quantiles(
                item.best_distance for item in negative
            ),
            "within_model_ratio": _quantiles(ratios_all),
        },
        "recommended": recommended,
        "settings": sorted(rows, key=_setting_sort_key),
    }


def _parse_mapping(values):
    artifacts = []
    for raw in values:
        scene, separator, path = str(raw).partition("=")
        if not separator:
            raise ValueError('Use --artifact scene=path.pkl')
        artifacts.append((scene.strip(), Path(path.strip()).resolve()))
    return artifacts


def _parse_expected(values):
    expected = {key: set(value) for key, value in DEFAULT_EXPECTED_SYNSETS.items()}
    for raw in values or []:
        scene, separator, synsets = str(raw).partition("=")
        if not separator:
            raise ValueError("Utilisez --expected scene=synset1,synset2")
        expected[scene.strip()] = {
            item.strip() for item in synsets.split(",") if item.strip()
        }
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--expected", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--scan-features", type=int, default=120)
    parser.add_argument("--model-features", type=int, default=24)
    parser.add_argument("--rotations", type=int, default=8)
    parser.add_argument("--wall-threshold", type=float, default=0.20)
    args = parser.parse_args(argv)

    expected = _parse_expected(args.expected)
    records = []
    sources = []
    for scene, artifact in _parse_mapping(args.artifact):
        scene_records, source = collect_artifact_records(
            artifact, args.db, scene, expected.get(scene),
            max_scan_features=args.scan_features,
            max_model_features=args.model_features,
            rotations=args.rotations,
            progress=print,
        )
        records.extend(scene_records)
        sources.append(source)
    report = build_report(
        records, sources, wall_threshold=args.wall_threshold,
    )
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    recommended = report["recommended"]
    print(
        f"Recommandation: t_desc={recommended['threshold']:.3f}, "
        f"ratio={recommended['ratio']:.3f}, "
        f"balanced_accuracy={recommended['balanced_accuracy']:.3f}"
    )
    print(f"Rapport: {output}")
    return report


if __name__ == "__main__":
    main()
