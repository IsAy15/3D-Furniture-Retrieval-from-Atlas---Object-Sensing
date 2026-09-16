'Diagnostic measurement of scan descriptor utility on a small model pool. Measures local distinctiveness without changing the descriptors passed to retrieval.\n'

from __future__ import annotations

from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from descriptors import descriptor_distance_batch
from matching import KeyPointFeature, _candidate_thetas


NO_MATCH = "no_match"
AMBIGUOUS = "ambiguous"
INFORMATIVE = "informative"


@dataclass
class DescriptorUtility:
    label: str
    best_distance: float
    second_distance: float
    margin: float
    valid_model_count: int
    compatible_feature_count: int
    entropy: float
    best_model: str | None
    best_synset: str | None


def _limited_thetas(model_feature, scan_feature, cfg, up, limit):
    up = np.asarray(up, float)
    up /= max(np.linalg.norm(up), 1e-12)
    axis = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, axis)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(up, e1)
    values = np.asarray(
        _candidate_thetas(model_feature, scan_feature, cfg, up, e1, e2),
        float,
    )
    if not len(values):
        return values
    # Les primitives peuvent proposer le meme angle plusieurs fois.
    values = np.unique(np.round(np.mod(values, 2.0 * math.pi), 8))
    limit = max(1, int(limit))
    if len(values) > limit:
        selection = np.linspace(0, len(values) - 1, limit).astype(int)
        values = values[selection]
    return values


def _compatible_model_features(
        model_features: Sequence[KeyPointFeature],
        scan_feature: KeyPointFeature, cfg):
    if not model_features:
        return []
    height = np.asarray([feature.height for feature in model_features], float)
    size6d = np.asarray([feature.size6d for feature in model_features], float)
    occupied = np.asarray([
        max(1, int(feature.descriptor.n_occupied))
        for feature in model_features
    ], float)
    scan_height = float(scan_feature.height)
    scales = (
        scan_height / np.maximum(height, 1e-9)
        if scan_height > 1e-3 else np.ones(len(model_features))
    )
    scales = np.where(height > 1e-3, scales, 1.0)
    scale_ok = (
        (scales >= cfg.matching.scale_min)
        & (scales <= cfg.matching.scale_max)
    )
    ratio = np.maximum(size6d, scan_feature.size6d) / np.maximum(
        np.minimum(size6d, scan_feature.size6d), 1e-9,
    )
    size_ok = (ratio <= cfg.matching.size_ratio_reject).all(axis=1)
    confidence = np.minimum(
        1.0,
        (
            scan_feature.descriptor.n_occupied
            + scan_feature.descriptor.n_unknown
        ) / occupied,
    )
    return np.flatnonzero(
        scale_ok & size_ok
        & (confidence >= cfg.matching.confidence_threshold)
    ).tolist()


def _top_model_features(features, limit):
    features = list(features or [])
    if not limit or len(features) <= int(limit):
        return features
    order = sorted(
        range(len(features)),
        key=lambda index: -float(getattr(features[index], "response", 0.0)),
    )[:int(limit)]
    return [features[index] for index in order]


def profile_descriptor_utility(
        scan_features: Sequence[KeyPointFeature],
        models: Iterable,
        cfg,
        up,
        *,
        max_model_features: int = 24,
        rotations: int = 8,
        distance_threshold: float | None = None,
        margin_threshold: float = 0.12,
        max_valid_models: int = 6,
        entropy_threshold: float = 0.80,
        progress=None):
    'Measure the discriminative power of each scan descriptor. Minimize distance per model, then compare the best and second-best models. Diagnostic only; does not filter keypoints.\n    '
    scan_features = list(scan_features or [])
    models = list(models or [])
    threshold = float(
        cfg.ransac.desc_inlier
        if distance_threshold is None else distance_threshold
    )
    per_scan = [[] for _ in scan_features]
    compatible_counts = np.zeros(len(scan_features), dtype=int)

    for model_index, model in enumerate(models, 1):
        model_features = _top_model_features(
            getattr(model, "features", []), max_model_features,
        )
        for scan_index, scan_feature in enumerate(scan_features):
            best = float("inf")
            compatible = _compatible_model_features(
                model_features, scan_feature, cfg,
            )
            compatible_counts[scan_index] += len(compatible)
            for feature_index in compatible:
                model_feature = model_features[feature_index]
                thetas = _limited_thetas(
                    model_feature, scan_feature, cfg, up, rotations,
                )
                if not len(thetas):
                    continue
                distances = descriptor_distance_batch(
                    model_feature.descriptor,
                    scan_feature.descriptor,
                    cfg.descriptor,
                    -thetas,
                    up,
                )
                if len(distances):
                    best = min(best, float(np.min(distances)))
            per_scan[scan_index].append((
                best,
                str(getattr(model, "name", model_index)),
                str(getattr(model, "synset", "")),
            ))
        if progress:
            progress(
                f"utilite descripteurs: {model_index}/{len(models)} "
                "modele(s)"
            )

    output = []
    for scan_index, candidates in enumerate(per_scan):
        candidates.sort(key=lambda item: item[0])
        finite = [item for item in candidates if math.isfinite(item[0])]
        best = finite[0] if finite else (float("inf"), None, None)
        second_distance = finite[1][0] if len(finite) > 1 else float("inf")
        valid = [item for item in finite if item[0] <= threshold]
        if not math.isfinite(best[0]) or best[0] > threshold:
            label = NO_MATCH
            margin = 0.0
            entropy = 1.0 if finite else 0.0
        else:
            margin = (
                1.0 if not math.isfinite(second_distance)
                else max(0.0, (second_distance - best[0])
                         / max(abs(second_distance), 1e-9))
            )
            if len(valid) <= 1:
                entropy = 0.0
            else:
                weights = np.exp(-np.asarray(
                    [item[0] for item in valid], float,
                ) / max(threshold, 1e-9))
                probabilities = weights / max(float(weights.sum()), 1e-12)
                entropy = float(
                    -np.sum(probabilities * np.log(
                        np.maximum(probabilities, 1e-12),
                    )) / math.log(len(probabilities))
                )
            informative = (
                margin >= float(margin_threshold)
                or (
                    len(valid) <= int(max_valid_models)
                    and entropy <= float(entropy_threshold)
                )
            )
            label = INFORMATIVE if informative else AMBIGUOUS
        output.append(DescriptorUtility(
            label=label,
            best_distance=float(best[0]),
            second_distance=float(second_distance),
            margin=float(margin),
            valid_model_count=len(valid),
            compatible_feature_count=int(compatible_counts[scan_index]),
            entropy=float(entropy),
            best_model=best[1],
            best_synset=best[2],
        ))
    return output


def summarize_descriptor_utilities(
        utilities, wall_scores=None, wall_threshold=0.55,
        hole_boundary_removed=None):
    """Aggregate usefulness while exposing wall-supported correspondences."""
    utilities = list(utilities or [])
    count = len(utilities)
    labels = np.asarray([item.label for item in utilities], object)
    informative = labels == INFORMATIVE
    ambiguous = labels == AMBIGUOUS
    no_match = labels == NO_MATCH
    if wall_scores is None:
        wall_scores = np.zeros(count, float)
    wall_scores = np.asarray(wall_scores, float).reshape(-1)[:count]
    if len(wall_scores) < count:
        wall_scores = np.pad(wall_scores, (0, count - len(wall_scores)))
    wall = wall_scores >= float(wall_threshold)
    informative_wall = informative & wall
    informative_nonwall = informative & ~wall
    margins = np.asarray([item.margin for item in utilities], float)
    entropies = np.asarray([item.entropy for item in utilities], float)
    finite_margins = margins[np.isfinite(margins)]
    finite_entropies = entropies[np.isfinite(entropies)]
    removed = np.asarray(
        hole_boundary_removed if hole_boundary_removed is not None else [],
        int,
    )
    informative_count = int(np.count_nonzero(informative))
    return {
        "descriptor_evaluated": int(count),
        "descriptor_informative": informative_count,
        "descriptor_informative_nonwall": int(
            np.count_nonzero(informative_nonwall)
        ),
        "descriptor_informative_wall": int(
            np.count_nonzero(informative_wall)
        ),
        "descriptor_ambiguous": int(np.count_nonzero(ambiguous)),
        "descriptor_no_match": int(np.count_nonzero(no_match)),
        "descriptor_informative_ratio": round(
            informative_count / max(1, count), 6,
        ),
        "descriptor_informative_wall_ratio": round(
            np.count_nonzero(informative_wall) / max(1, informative_count),
            6,
        ),
        "descriptor_mean_margin": round(
            float(np.mean(finite_margins)) if len(finite_margins) else 0.0,
            6,
        ),
        "descriptor_mean_entropy": round(
            float(np.mean(finite_entropies)) if len(finite_entropies) else 0.0,
            6,
        ),
        "descriptor_hole_boundary_cells_removed": int(removed.sum()),
        "descriptor_hole_boundary_affected": int(np.count_nonzero(removed)),
    }


class DescriptorSweepEvaluator:
    """Reusable DB-backed evaluator for multi-trial keypoint sweeps."""

    requires_artifact = True

    def __init__(
            self, db_dir, candidate_index_path=None, candidate_models=24,
            model_features=24, scan_features=120, rotations=8,
            distance_unit=0.0,
            distance_threshold=None, margin_threshold=0.12,
            max_valid_models=6, entropy_threshold=0.80,
            target_annotations=None, candidate_pool_size=500):
        import db_store
        from candidate_index import default_index_path, load_candidate_index
        from config import PipelineConfig

        self.db_dir = str(Path(db_dir).resolve())
        if not db_store.is_db_dir(self.db_dir):
            raise FileNotFoundError(
                f"ShapeNet database not found : {self.db_dir}"
            )
        index_path = candidate_index_path or default_index_path(self.db_dir)
        self.candidate_index_path = str(Path(index_path).resolve())
        if not Path(self.candidate_index_path).exists():
            raise FileNotFoundError(
                f'Candidate index not found: {self.candidate_index_path}'
            )
        self.cfg = db_store.load_cfg(self.db_dir) or PipelineConfig()
        self.cfg.descriptor.distance_unit = float(distance_unit)
        self.cfg.matching.max_scan_keypoints = 0
        self.cfg.matching.floor_height = -1e6
        self.cfg.keypoint.wall_budget_enabled = False
        self.index = load_candidate_index(self.candidate_index_path)
        self.files = db_store.model_files(self.db_dir)
        self._model_cache = {}
        self.candidate_models = max(1, int(candidate_models))
        self.model_features = max(1, int(model_features))
        self.scan_features = max(1, int(scan_features))
        self.rotations = max(1, int(rotations))
        self.distance_threshold = (
            None if distance_threshold is None
            else float(distance_threshold)
        )
        self.margin_threshold = float(margin_threshold)
        self.max_valid_models = max(1, int(max_valid_models))
        self.entropy_threshold = float(entropy_threshold)
        self.target_annotations = dict(target_annotations or {})
        self.candidate_pool_size = max(1, int(candidate_pool_size))

    def signature(self):
        return {
            "db_dir": self.db_dir,
            "candidate_index": self.candidate_index_path,
            "candidate_models": self.candidate_models,
            "model_features": self.model_features,
            "scan_features": self.scan_features,
            "rotations": self.rotations,
            "distance_unit": float(self.cfg.descriptor.distance_unit),
            "distance_threshold": self.distance_threshold,
            "margin_threshold": self.margin_threshold,
            "max_valid_models": self.max_valid_models,
            "entropy_threshold": self.entropy_threshold,
            "target_annotations": self.target_annotations,
            "candidate_pool_size": self.candidate_pool_size,
        }

    def _model(self, index):
        import db_store

        index = int(index)
        if index not in self._model_cache:
            self._model_cache[index] = db_store.load_model(self.files[index])
        return self._model_cache[index]

    def evaluate(self, artifact_path, progress=None, source=None):
        from candidate_index import rank_candidates
        from keypoints import KeyPoints
        from matching import build_scan_features

        with Path(artifact_path).open("rb") as stream:
            payload = pickle.load(stream)
        scene = payload["scene"]
        keypoints = payload["keypoints"]
        if keypoints.size > self.scan_features:
            selection = np.linspace(
                0, keypoints.size - 1, self.scan_features,
            ).astype(int)

            def subset(name):
                values = getattr(keypoints, name, None)
                return None if values is None else np.asarray(values)[selection]

            keypoints = KeyPoints(
                positions=subset("positions"), normals=subset("normals"),
                responses=subset("responses"),
                source_index=subset("source_index"),
                selection_scores=subset("selection_scores"),
                wall_scores=subset("wall_scores"),
                object_scores=subset("object_scores"),
                wall_affinities=subset("wall_affinities"),
                wall_proximities=subset("wall_proximities"),
                wall_plane_indices=subset("wall_plane_indices"),
            )
        up = np.asarray(scene.up, float)
        up /= max(np.linalg.norm(up), 1e-12)
        features = build_scan_features(
            scene.cloud, scene.volume, keypoints, self.cfg, up,
            float(scene.ground),
        )
        selected = rank_candidates(
            self.index, features, self.cfg, self.candidate_models,
            diversify=True, diversity_fraction=0.5,
        )
        pool = rank_candidates(
            self.index, features, self.cfg, self.candidate_pool_size,
            diversify=True, diversity_fraction=0.5,
        )
        models = [self._model(index) for index in selected]
        utilities = profile_descriptor_utility(
            features, models, self.cfg, up,
            max_model_features=self.model_features,
            rotations=self.rotations,
            distance_threshold=self.distance_threshold,
            margin_threshold=self.margin_threshold,
            max_valid_models=self.max_valid_models,
            entropy_threshold=self.entropy_threshold,
            progress=progress,
        )
        wall_scores = getattr(keypoints, "wall_scores", None)
        removed = [
            getattr(feature.descriptor, "n_hole_boundary_removed", 0)
            for feature in features
        ]
        source_id = None if source is None else str(source.get("id", ""))
        recall = summarize_candidate_recall(
            [str(self.index.names[int(index)]) for index in pool],
            self.target_annotations.get(source_id),
        )
        return summarize_descriptor_utilities(
            utilities, wall_scores=wall_scores,
            wall_threshold=float(getattr(
                self.cfg.keypoint, "wall_reject_threshold", 0.55,
            )),
            hole_boundary_removed=removed,
        ) | {
            "descriptor_retained_keypoints": int(
                payload["keypoints"].size
            ),
            "descriptor_sampled_keypoints": int(keypoints.size),
        } | recall


def summarize_candidate_recall(ranked_names, acceptable_names):
    """Summarize annotated pool recall without inventing ground truth."""
    if acceptable_names is None:
        return {
            "candidate_recall_evaluated": False,
            "candidate_pool_size": int(len(ranked_names)),
        }
    if isinstance(acceptable_names, dict):
        target_groups = {
            str(label): tuple(dict.fromkeys(str(name) for name in names))
            for label, names in acceptable_names.items()
        }
    else:
        target_groups = {
            str(name): (str(name),)
            for name in dict.fromkeys(acceptable_names)
        }
    ranks = {str(name): index + 1 for index, name in enumerate(ranked_names)}
    hit_groups = {
        label: min(
            ((ranks[name], name) for name in names if name in ranks),
            default=None,
        )
        for label, names in target_groups.items()
    }
    hits = {label: hit for label, hit in hit_groups.items() if hit is not None}
    missing = [label for label, hit in hit_groups.items() if hit is None]
    return {
        "candidate_recall_evaluated": True,
        "candidate_pool_size": int(len(ranked_names)),
        "candidate_target_count": int(len(target_groups)),
        "candidate_target_hits": int(len(hits)),
        "candidate_target_recall": (
            float(len(hits) / len(target_groups)) if target_groups else 1.0
        ),
        "candidate_target_first_rank": min(
            (hit[0] for hit in hits.values()), default=None,
        ),
        "candidate_target_hit_names": {
            label: {"name": hit[1], "rank": int(hit[0])}
            for label, hit in hits.items()
        },
        "candidate_target_missing_names": missing,
    }
