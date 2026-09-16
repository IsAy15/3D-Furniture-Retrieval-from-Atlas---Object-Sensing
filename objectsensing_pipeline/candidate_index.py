'\nCompact preselection index for db_dir databases. Lightweight height histograms and primitive statistics shortlist models before detailed RANSAC/ICP verification.\n'

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import db_store
from database import object_surface_cloud
from descriptors import descriptor_distance_batch

HEIGHT_MAX = 1.4
HEIGHT_BINS = 32


@dataclass
class CandidateIndex:
    names: List[str]
    synsets: List[str]
    height_hist: np.ndarray
    size_mean: np.ndarray
    occ_mean: np.ndarray
    n_features: np.ndarray
    shape_signature: Optional[np.ndarray] = None


def _shape_signature_from_features(features, up=None) -> np.ndarray:
    """Compact rotation-invariant layout of a model or scene object group."""
    if not features:
        return np.zeros(63, dtype=np.float32)
    points = np.asarray([
        getattr(feature, "position", np.zeros(3)) for feature in features
    ], dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return np.zeros(63, dtype=np.float32)
    up = np.asarray([0.0, 1.0, 0.0] if up is None else up, dtype=float)
    up /= max(np.linalg.norm(up), 1e-9)
    vertical = points @ up
    horizontal = points - vertical[:, None] * up[None, :]
    center = np.median(horizontal, axis=0)
    radial = np.linalg.norm(horizontal - center, axis=1)
    height = vertical - float(vertical.min())
    scale = max(float(np.ptp(vertical)), float(np.percentile(radial, 95)), 1e-6)
    height = np.clip(height / scale, 0.0, 2.0)
    radial = np.clip(radial / scale, 0.0, 2.0)
    height_hist, _ = np.histogram(height, bins=8, range=(0.0, 2.0))
    radial_hist, _ = np.histogram(radial, bins=8, range=(0.0, 2.0))
    sample = points[_top_feature_indices(features, 64)]
    distances = np.linalg.norm(
        sample[:, None, :] - sample[None, :, :], axis=2,
    )
    pairwise = distances[np.triu_indices(len(sample), 1)] / scale
    pair_hist, _ = np.histogram(pairwise, bins=8, range=(0.0, 2.5))
    occupancy, _, _ = np.histogram2d(
        height, radial, bins=(6, 6), range=((0.0, 2.0), (0.0, 2.0)),
    )
    extent = _extent_signature(points, up) / scale
    signature = np.concatenate([
        height_hist, radial_hist, pair_hist, occupancy.reshape(-1), extent,
    ]).astype(np.float32)
    norm = float(np.linalg.norm(signature))
    return signature / norm if norm > 1e-9 else signature


def _top_feature_indices(features, limit):
    order = sorted(
        range(len(features)),
        key=lambda index: -float(getattr(features[index], "response", 0.0)),
    )
    return np.asarray(order[:min(int(limit), len(order))], dtype=int)


def _feature_weights(features):
    w = np.array([max(float(getattr(f, "response", 0.0)), 0.0) for f in features], dtype=np.float32)
    if len(w) == 0:
        return w
    if float(w.max()) <= 1e-9:
        return np.ones(len(w), dtype=np.float32)
    return 1.0 + w / float(w.max())


def _height_hist_from_features(features, bins: int = HEIGHT_BINS) -> np.ndarray:
    if not features:
        return np.zeros(bins, dtype=np.float32)
    heights = np.array([max(0.0, min(HEIGHT_MAX, float(f.height))) for f in features], dtype=np.float32)
    hist, _ = np.histogram(heights, bins=bins, range=(0.0, HEIGHT_MAX), weights=_feature_weights(features))
    hist = hist.astype(np.float32)
    s = float(np.linalg.norm(hist))
    return hist / s if s > 1e-9 else hist


def _size_mean_from_features(features) -> np.ndarray:
    if not features:
        return np.zeros(6, dtype=np.float32)
    sizes = np.array([np.log1p(np.asarray(f.size6d, dtype=np.float32)) for f in features], dtype=np.float32)
    return sizes.mean(axis=0).astype(np.float32)


def _occ_mean_from_features(features) -> float:
    if not features:
        return 0.0
    return float(np.mean([max(1, int(f.descriptor.n_occupied)) for f in features]))


def build_candidate_index(db_dir: str, progress=None) -> CandidateIndex:
    index = db_store.load_index(db_dir)
    files = db_store.model_files(db_dir)
    names, synsets = [], []
    hists, sizes, occs, counts, shapes = [], [], [], [], []
    for i, path in enumerate(files):
        model = db_store.load_model(path)
        names.append(model.name)
        synsets.append(model.synset)
        feats = model.features
        hists.append(_height_hist_from_features(feats))
        sizes.append(_size_mean_from_features(feats))
        occs.append(_occ_mean_from_features(feats))
        counts.append(len(feats))
        shapes.append(_shape_signature_from_features(feats))
        if progress and ((i + 1) % 500 == 0 or i + 1 == len(files)):
            progress(f'candidate index: {i + 1}/{len(files)} modeles')
    if len(names) != int(index.get("n", len(names))):
        raise ValueError('Database index does not match model files')
    return CandidateIndex(
        names=names,
        synsets=synsets,
        height_hist=np.vstack(hists).astype(np.float32) if hists else np.zeros((0, HEIGHT_BINS), dtype=np.float32),
        size_mean=np.vstack(sizes).astype(np.float32) if sizes else np.zeros((0, 6), dtype=np.float32),
        occ_mean=np.asarray(occs, dtype=np.float32),
        n_features=np.asarray(counts, dtype=np.int32),
        shape_signature=np.vstack(shapes).astype(np.float32),
    )


def save_candidate_index(index: CandidateIndex, path: str):
    np.savez_compressed(
        path,
        names=np.asarray(index.names, dtype=object),
        synsets=np.asarray(index.synsets, dtype=object),
        height_hist=index.height_hist,
        size_mean=index.size_mean,
        occ_mean=index.occ_mean,
        n_features=index.n_features,
        shape_signature=(
            index.shape_signature if index.shape_signature is not None
            else np.zeros((len(index.names), 63), dtype=np.float32)
        ),
    )


def load_candidate_index(path: str) -> CandidateIndex:
    data = np.load(path, allow_pickle=True)
    shape_signature = (
        np.asarray(data["shape_signature"], dtype=np.float32)
        if "shape_signature" in data.files else None
    )
    return CandidateIndex(
        names=[str(x) for x in data["names"].tolist()],
        synsets=[str(x) for x in data["synsets"].tolist()],
        height_hist=np.asarray(data["height_hist"], dtype=np.float32),
        size_mean=np.asarray(data["size_mean"], dtype=np.float32),
        occ_mean=np.asarray(data["occ_mean"], dtype=np.float32),
        n_features=np.asarray(data["n_features"], dtype=np.int32),
        shape_signature=shape_signature,
    )


def _rescaled_model_hist_score(model_hist: np.ndarray, scan_hist: np.ndarray, scales: Iterable[float]) -> np.ndarray:
    bins = model_hist.shape[1]
    centers = (np.arange(bins, dtype=np.float32) + 0.5) * (HEIGHT_MAX / bins)
    best = np.zeros(model_hist.shape[0], dtype=np.float32)
    for scale in scales:
        target = np.clip((centers * float(scale)) / HEIGHT_MAX * bins, 0, bins - 1).astype(np.int32)
        remapped = np.zeros_like(model_hist)
        for src_bin, dst_bin in enumerate(target):
            remapped[:, dst_bin] += model_hist[:, src_bin]
        score = remapped @ scan_hist
        best = np.maximum(best, score.astype(np.float32))
    return best


def _take_stratified(ranked: np.ndarray, quota: int, chosen: set) -> List[int]:
    if quota <= 0 or len(ranked) == 0:
        return []
    out = []
    positions = np.linspace(0, len(ranked) - 1, min(len(ranked), quota * 3)).astype(int)
    for pos in positions:
        idx = int(ranked[pos])
        if idx not in chosen:
            out.append(idx)
            chosen.add(idx)
            if len(out) >= quota:
                break
    return out


def rank_candidates(index: CandidateIndex, scan_features, cfg, top_k: int,
                    synsets: Optional[Iterable[str]] = None,
                    diversify: bool = False,
                    diversity_fraction: float = 0.5,
                    include_names: Optional[Iterable[str]] = None) -> List[int]:
    if top_k <= 0 or len(index.names) == 0:
        return []
    scan_hist = _height_hist_from_features(scan_features)
    scan_size = _size_mean_from_features(scan_features)

    scales = np.linspace(cfg.matching.scale_min, cfg.matching.scale_max, 9)
    height_score = _rescaled_model_hist_score(index.height_hist, scan_hist, scales)
    size_dist = np.linalg.norm(index.size_mean - scan_size[None, :], axis=1)
    feat_bonus = np.log1p(np.maximum(index.n_features, 0)).astype(np.float32)
    scores = height_score - 0.08 * size_dist + 0.015 * feat_bonus
    if index.shape_signature is not None and len(index.shape_signature) == len(scores):
        scan_shape = _shape_signature_from_features(scan_features)
        if float(np.linalg.norm(scan_shape)) > 1e-9:
            shape_score = index.shape_signature @ scan_shape
            scores += 0.45 * shape_score.astype(np.float32)

    if synsets:
        allowed = set(synsets)
        mask = np.array([s in allowed for s in index.synsets], dtype=bool)
        scores = np.where(mask, scores, -np.inf)

    k = min(int(top_k), len(scores))
    if k <= 0:
        return []
    finite = np.where(np.isfinite(scores))[0]
    if len(finite) == 0:
        return []
    ranked = finite[np.argsort(-scores[finite])]
    forced = []
    if include_names:
        by_name = {name: i for i, name in enumerate(index.names)}
        forced = [int(by_name[name]) for name in include_names if name in by_name and np.isfinite(scores[by_name[name]])]
    forced = list(dict.fromkeys(forced))
    if not diversify:
        out = forced + [int(i) for i in ranked if int(i) not in set(forced)]
        return out[:k]

    diversity_fraction = float(np.clip(diversity_fraction, 0.0, 0.9))
    global_k = max(1, int(round(k * (1.0 - diversity_fraction))))
    selected = forced + [int(i) for i in ranked[:global_k] if int(i) not in set(forced)]
    selected = selected[:global_k]
    chosen = set(selected)
    remaining = k - len(selected)
    if remaining <= 0:
        return selected[:k]

    synset_values = sorted(set(index.synsets[i] for i in ranked))
    base_quota = max(1, remaining // max(1, len(synset_values)))
    for syn in synset_values:
        syn_ranked = np.array([i for i in ranked if index.synsets[int(i)] == syn], dtype=np.int64)
        selected.extend(_take_stratified(syn_ranked, base_quota, chosen))
    if len(selected) < k:
        selected.extend(_take_stratified(ranked, k - len(selected), chosen))
    return selected[:k]


def rank_candidates_by_feature_groups(
        index: CandidateIndex, feature_groups, cfg, top_k: int,
        diversify: bool = False, diversity_fraction: float = 0.5,
        rrf_constant: float = 60.0, group_weights=None,
        min_candidates_per_group: int = 0):
    """Fuse one lightweight candidate ranking per object proposal.

    Reciprocal-rank fusion keeps a model that is very strong for one group
    without requiring it to explain unrelated furniture elsewhere in the scan.
    """
    groups = [list(group) for group in feature_groups if len(group) >= 2]
    if not groups:
        return [], []
    rankings = [
        rank_candidates(
            index, group, cfg, top_k,
            diversify=diversify,
            diversity_fraction=diversity_fraction,
        )
        for group in groups
    ]
    weights = np.ones(len(groups), dtype=float)
    if group_weights is not None:
        provided = np.asarray(group_weights, dtype=float).reshape(-1)
        if len(provided) == len(groups):
            weights = np.where(np.isfinite(provided), provided, 0.0)
            weights = np.clip(weights, 0.05, 20.0)
            weights /= max(float(weights.mean()), 1e-9)
    fused = {}
    best_rank = {}
    for group_index, ranking in enumerate(rankings):
        for rank, model_index in enumerate(ranking, 1):
            model_index = int(model_index)
            fused[model_index] = fused.get(model_index, 0.0) + (
                float(weights[group_index])
                / (float(rrf_constant) + rank)
            )
            best_rank[model_index] = min(
                rank, best_rank.get(model_index, rank),
            )
    ordered = sorted(
        fused,
        key=lambda model_index: (
            -fused[model_index], best_rank[model_index], model_index,
        ),
    )
    limit = min(int(top_k), len(ordered))
    quota = min(
        max(0, int(min_candidates_per_group)),
        limit // max(1, len(groups)),
    )
    protected = set()
    if quota:
        # Chaque proposition fiable conserve quelques hypothèses avant que the
        # groupes the plus faciles ne remplissent tout le pool fusionné.
        ownership = {}
        for group_index, ranking in enumerate(rankings):
            for rank, model_index in enumerate(ranking, 1):
                affinity = float(weights[group_index]) / (
                    float(rrf_constant) + rank
                )
                current = ownership.get(int(model_index))
                if current is None or affinity > current[0]:
                    ownership[int(model_index)] = (affinity, group_index)
        for group_index in np.argsort(-weights):
            kept = 0
            for model_index in rankings[int(group_index)]:
                model_index = int(model_index)
                if ownership.get(model_index, (None, None))[1] != int(group_index):
                    continue
                if model_index in protected:
                    continue
                protected.add(model_index)
                kept += 1
                if kept >= quota:
                    break
    selected = [value for value in ordered if value in protected]
    selected.extend(
        value for value in ordered
        if value not in protected and len(selected) < limit
    )
    return selected[:limit], rankings


def merge_candidate_pools(group_pool, group_rankings, global_ranking, limit,
                          global_fraction=0.7, min_per_group=20):
    """Keep group reservations when adding whole-scene candidates.

    Round-robin reservations take precedence over the global fraction when
    the budget cannot accommodate both. Inputs and their order stay intact.
    """
    limit = max(0, int(limit))
    quota = min(max(0, int(min_per_group)), limit // max(1, len(group_rankings)))
    selected, seen = [], set()
    def add(value):
        value = int(value)
        if value not in seen and len(selected) < limit:
            seen.add(value)
            selected.append(value)
    for rank in range(quota):
        for ranking in group_rankings:
            if rank < len(ranking):
                add(ranking[rank])
    reserve = int(round(limit * float(np.clip(global_fraction, 0, 1))))
    for value in global_ranking[:reserve]:
        add(value)
    for value in group_pool:
        add(value)
    for value in global_ranking:
        add(value)
    return selected


def _top_features(features, limit: int):
    if not features:
        return []
    if limit and len(features) > int(limit):
        order = sorted(
            range(len(features)),
            key=lambda i: -float(getattr(features[i], "response", 0.0)),
        )[: int(limit)]
        return [features[i] for i in order]
    return list(features)


def _extent_signature(points, up):
    """Return rotation-invariant horizontal extents and vertical height."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return np.zeros(3, dtype=float)
    up = np.asarray(up, dtype=float).reshape(3)
    up /= max(np.linalg.norm(up), 1e-9)
    reference = np.eye(3)[int(np.argmin(np.abs(up)))]
    horizontal_x = np.cross(up, reference)
    horizontal_x /= max(np.linalg.norm(horizontal_x), 1e-9)
    horizontal_z = np.cross(up, horizontal_x)
    horizontal = np.column_stack([
        points @ horizontal_x, points @ horizontal_z,
    ])
    centered = horizontal - horizontal.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(centered))
    _, basis = np.linalg.eigh(covariance)
    horizontal_extent = np.ptp(centered @ basis, axis=0)
    vertical_extent = float(np.ptp(points @ up))
    return np.asarray([
        *np.sort(horizontal_extent), vertical_extent,
    ], dtype=float)


def _extent_compatibility(model_extent, scan_extent, cfg):
    """Softly compare complete model size with a possibly partial scan group."""
    model_extent = np.asarray(model_extent, dtype=float).reshape(3)
    scan_extent = np.asarray(scan_extent, dtype=float).reshape(3)
    if np.any(model_extent <= 1e-5) or np.any(scan_extent <= 1e-5):
        return 0.5
    scale = float(np.clip(
        scan_extent[2] / model_extent[2],
        cfg.matching.scale_min, cfg.matching.scale_max,
    ))
    scaled_model = model_extent * scale
    ratios = np.maximum(scaled_model, scan_extent) / np.maximum(
        np.minimum(scaled_model, scan_extent), 1e-6,
    )
    # Une différence modérée est attendue car the objets RGB-D sont partiels.
    # Les canapés de plusieurs mètres restent néanmoins nettement pénalisés
    # lorsqu'ils tentent d'expliquer une chaise compacte.
    return float(np.exp(-0.9 * np.mean(np.log(ratios))))


def _normalized_group_points(features, up):
    points = np.asarray([
        getattr(feature, "position", np.zeros(3)) for feature in features
    ], dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return np.zeros((0, 3), dtype=float)
    up = np.asarray(up, dtype=float).reshape(3)
    up /= max(np.linalg.norm(up), 1e-9)
    reference = np.eye(3)[int(np.argmin(np.abs(up)))]
    axis_x = np.cross(up, reference)
    axis_x /= max(np.linalg.norm(axis_x), 1e-9)
    axis_z = np.cross(up, axis_x)
    projected = np.column_stack([
        points @ axis_x, points @ up, points @ axis_z,
    ])
    projected[:, (0, 2)] -= np.median(projected[:, (0, 2)], axis=0)
    projected[:, 1] -= float(projected[:, 1].min())
    radial = np.linalg.norm(projected[:, (0, 2)], axis=1)
    scale = max(
        float(np.ptp(projected[:, 1])),
        float(np.percentile(radial, 95)), 1e-6,
    )
    return projected / scale


def _group_shape_compatibility(model_features, scan_features, up,
                               n_rotations=12):
    """Compare whole keypoint layouts while tolerating yaw and partial scans."""
    model = _normalized_group_points(model_features, up)
    scan = _normalized_group_points(scan_features, up)
    if len(model) < 2 or len(scan) < 2:
        return 0.0
    model = model[_top_feature_indices(model_features, 96)]
    scan_tree = cKDTree(scan)
    best = 0.0
    for theta in np.linspace(
            0.0, 2.0 * math.pi, max(1, int(n_rotations)), endpoint=False):
        cosine, sine = math.cos(theta), math.sin(theta)
        rotated = model.copy()
        rotated[:, 0] = cosine * model[:, 0] - sine * model[:, 2]
        rotated[:, 2] = sine * model[:, 0] + cosine * model[:, 2]
        model_tree = cKDTree(rotated)
        scene_to_model = model_tree.query(scan, k=1)[0]
        model_to_scene = scan_tree.query(rotated, k=1)[0]
        # Scene est partielle : sa compatibilité vers le modèle domine, mais
        # le terme inverse empêche une grande table d'expliquer un seul montant.
        distance = (
            0.72 * float(np.mean(np.minimum(scene_to_model, 1.0)))
            + 0.28 * float(np.mean(np.minimum(model_to_scene, 1.0)))
        )
        best = max(best, math.exp(-distance / 0.24))
    return float(best)


def _local_descriptor_score(model_features, scan_features, cfg, up,
                            max_scan_features: int = 80,
                            max_model_features: int = 80,
                            n_rotations: int = 12,
                            target_matches: int = 4,
                            target_coverage: float = 0.10,
                            return_details: bool = False):
    'Lightweight reranking before RANSAC. Reuses local descriptors to rank a broad pool without constructing constellations or performing surface verification.\n    '
    m_feats = _top_features(model_features, max_model_features)
    s_feats = _top_features(scan_features, max_scan_features)
    if not m_feats or not s_feats:
        return float("-inf")

    m_height = np.asarray([mf.height for mf in m_feats], dtype=np.float32)
    m_size6d = np.asarray([mf.size6d for mf in m_feats], dtype=np.float32)
    m_nocc = np.asarray(
        [max(1, int(mf.descriptor.n_occupied)) for mf in m_feats],
        dtype=np.float32,
    )
    thetas = np.linspace(
        0.0, 2.0 * math.pi,
        max(1, min(int(n_rotations), int(cfg.matching.n_uniform_rotations))),
        endpoint=False,
    )
    candidate_threshold = float(getattr(
        cfg.matching, "candidate_descriptor_distance_threshold",
        cfg.ransac.desc_inlier,
    ))
    desc_norm = max(
        1.0,
        candidate_threshold if candidate_threshold > 0.0
        else float(cfg.ransac.desc_inlier),
    )
    distances = np.full((len(s_feats), len(m_feats)), np.inf, dtype=np.float32)
    for scan_index, sf in enumerate(s_feats):
        sh = float(sf.height)
        scales = (sh / np.maximum(m_height, 1e-9)) if sh > 1e-3 else np.ones(len(m_feats))
        scales = np.where(m_height > 1e-3, scales, 1.0)
        scale_ok = (
            (scales >= cfg.matching.scale_min)
            & (scales <= cfg.matching.scale_max)
        )
        ratio = np.maximum(m_size6d, sf.size6d) / np.maximum(
            np.minimum(m_size6d, sf.size6d), 1e-9
        )
        size_ok = (ratio <= cfg.matching.size_ratio_reject).all(axis=1)
        conf = np.minimum(
            1.0,
            (sf.descriptor.n_occupied + sf.descriptor.n_unknown) / m_nocc,
        )
        ok = np.where(scale_ok & size_ok & (conf >= cfg.matching.confidence_threshold))[0]
        if len(ok) == 0:
            continue
        for mi in ok:
            dist = descriptor_distance_batch(
                m_feats[int(mi)].descriptor, sf.descriptor,
                cfg.descriptor, thetas, up,
            )
            if len(dist):
                distances[scan_index, int(mi)] = float(np.min(dist))
    finite_rows = np.isfinite(distances).any(axis=1)
    if not finite_rows.any():
        evidence = {
            "score": float("-inf"), "matches": 0,
            "scan_coverage": 0.0, "model_coverage": 0.0,
            "coverage": 0.0, "spatial_spread": 0.0,
            "mean_similarity": 0.0, "support": 0.0,
            "coverage_support": 0.0, "ratio_rejected": 0,
        }
        return evidence if return_details else evidence["score"]

    ratio_threshold = float(getattr(
        cfg.matching, "descriptor_ratio_threshold", 1.0,
    ))
    ratio_ok = np.ones(len(s_feats), dtype=bool)
    ratio_rejected = 0
    if 0.0 < ratio_threshold < 1.0 and len(m_feats) > 1:
        for scan_index in np.where(finite_rows)[0]:
            values = np.sort(distances[scan_index][
                np.isfinite(distances[scan_index])
            ])
            if len(values) >= 2 and values[1] > 1e-9:
                ratio_ok[scan_index] = (
                    float(values[0]) / float(values[1]) <= ratio_threshold
                )
                ratio_rejected += int(not ratio_ok[scan_index])

    eligible_scan = np.where(finite_rows & ratio_ok)[0]
    if len(eligible_scan) == 0:
        evidence = {
            "score": float("-inf"), "matches": 0,
            "scan_coverage": 0.0, "model_coverage": 0.0,
            "coverage": 0.0, "spatial_spread": 0.0,
            "mean_similarity": 0.0,
            "support": 0.0, "coverage_support": 0.0,
            "ratio_rejected": int(ratio_rejected),
        }
        return evidence if return_details else evidence["score"]
    assignment_cost = np.where(
        np.isfinite(distances[eligible_scan]), distances[eligible_scan], 1e6,
    )
    local_scan_assignment, model_assignment = linear_sum_assignment(
        assignment_cost,
    )
    scan_assignment = eligible_scan[local_scan_assignment]
    assigned_distances = distances[scan_assignment, model_assignment]
    valid = np.isfinite(assigned_distances)
    scan_assignment = scan_assignment[valid]
    model_assignment = model_assignment[valid]
    assigned_distances = assigned_distances[valid]
    similarities = np.exp(-np.minimum(assigned_distances / desc_norm, 8.0))
    strong = similarities >= 0.45
    scan_assignment = scan_assignment[strong]
    model_assignment = model_assignment[strong]
    similarities = similarities[strong]
    if len(similarities) == 0:
        evidence = {
            "score": float("-inf"), "matches": 0,
            "scan_coverage": 0.0, "model_coverage": 0.0,
            "coverage": 0.0, "spatial_spread": 0.0,
            "mean_similarity": 0.0,
            "support": 0.0, "coverage_support": 0.0,
            "ratio_rejected": int(ratio_rejected),
        }
        return evidence if return_details else evidence["score"]

    scan_coverage = len(similarities) / max(1, len(s_feats))
    model_coverage = len(similarities) / max(1, len(m_feats))
    coverage = (
        2.0 * scan_coverage * model_coverage
        / max(scan_coverage + model_coverage, 1e-9)
    )
    spatial_spread = 1.0
    positions = np.asarray([
        getattr(feature, "position", np.zeros(3)) for feature in s_feats
    ], dtype=float).reshape(-1, 3)
    if len(positions) >= 2:
        full_extent = np.linalg.norm(np.ptp(positions, axis=0))
        matched_extent = np.linalg.norm(np.ptp(
            positions[scan_assignment], axis=0,
        )) if len(scan_assignment) >= 2 else 0.0
        spatial_spread = float(np.clip(
            matched_extent / max(full_extent, 1e-9), 0.0, 1.0,
        ))
    mean_similarity = float(np.mean(similarities))
    # Une ou deux ressemblances locales ne doivent pas pouvoir dominer le
    # classement. La similarité reste utile, mais elle est pondérée par le
    # nombre de correspondences distinctes, leur couverture et leur étendue.
    support = float(np.clip(
        len(similarities) / max(1, int(target_matches)), 0.0, 1.0,
    ))
    coverage_support = float(np.clip(
        coverage / max(float(target_coverage), 1e-9), 0.0, 1.0,
    ))
    score = float(
        0.35 * mean_similarity
        + 0.25 * support
        + 0.20 * coverage_support
        + 0.20 * spatial_spread
    )
    evidence = {
        "score": score,
        "matches": int(len(similarities)),
        "scan_coverage": float(scan_coverage),
        "model_coverage": float(model_coverage),
        "coverage": float(coverage),
        "spatial_spread": float(spatial_spread),
        "mean_similarity": mean_similarity,
        "support": support,
        "coverage_support": coverage_support,
        "ratio_rejected": int(ratio_rejected),
    }
    return evidence if return_details else score


def rerank_candidates_by_local_features(
        db_dir: str,
        candidate_indices: Sequence[int],
        scan_features,
        cfg,
        top_k: int,
        include_names: Optional[Iterable[str]] = None,
        up=None,
        progress=None,
        max_scan_features: int = 80,
        max_model_features: int = 80,
        n_rotations: int = 12,
        local_weight: float = 0.65,
        target_matches: int = 4,
        target_coverage: float = 0.10) -> List[int]:
    'Rerank a broad pool using serialized local descriptors. Keep top_k models for full registration, with explicitly requested names included first.\n    '
    if top_k <= 0:
        return []
    candidate_indices = list(dict.fromkeys(int(i) for i in candidate_indices))
    if len(candidate_indices) <= top_k:
        return candidate_indices

    files = db_store.model_files(db_dir)
    meta = db_store.load_index(db_dir)
    by_name = {name: i for i, name in enumerate(meta.get("names", []))}
    forced = []
    if include_names:
        for name in include_names:
            idx = by_name.get(name)
            if idx is not None and idx in candidate_indices:
                forced.append(int(idx))
    forced = list(dict.fromkeys(forced))

    start = time.monotonic()
    scored = []
    total = len(candidate_indices)
    up = np.asarray(cfg.up_axis if up is None else up, dtype=float)
    up /= max(np.linalg.norm(up), 1e-9)
    for pos, model_index in enumerate(candidate_indices, 1):
        try:
            model = db_store.load_model(files[model_index])
            local_score = _local_descriptor_score(
                model.features, scan_features, cfg, up,
                max_scan_features=max_scan_features,
                max_model_features=max_model_features,
                n_rotations=n_rotations,
                target_matches=target_matches,
                target_coverage=target_coverage,
            )
        except Exception:
            local_score = float("-inf")
        preselection_score = 1.0 - (
            (pos - 1) / max(1, total)
        )
        local_value = (
            float(local_score) if math.isfinite(float(local_score)) else 0.0
        )
        blend = float(np.clip(local_weight, 0.0, 1.0))
        score = (
            blend * local_value
            + (1.0 - blend) * preselection_score
        )
        scored.append((float(score), pos, model_index))
        if progress and (pos % 50 == 0 or pos == total):
            elapsed = max(time.monotonic() - start, 1e-9)
            rate = pos / elapsed
            remaining = (total - pos) / rate if rate > 0 else 0.0
            progress(
                "reranking local: "
                f"{pos}/{total} model(s), approximately remaining {int(round(remaining))}s"
            )

    forced_set = set(forced)
    ranked = [
        int(idx) for _, _, idx in sorted(
            (item for item in scored if item[2] not in forced_set),
            key=lambda item: (-item[0], item[1]),
        )
    ]
    return (forced + ranked)[: min(int(top_k), len(candidate_indices))]


def rerank_candidates_by_local_feature_groups(
        db_dir: str, candidate_indices: Sequence[int], feature_groups, cfg,
        top_k: int, up=None, progress=None, max_model_features: int = 80,
        n_rotations: int = 12, group_rankings=None,
        global_ranking=None,
        max_groups_per_candidate: int = 0, group_weights=None,
        min_candidates_per_group: int = 0, local_weight: float = 0.65,
        candidates_per_group: int = 0,
        target_matches: int = 4, target_coverage: float = 0.10,
        extent_weight: float = 0.30,
        extent_quota_fraction: float = 0.0):
    """Rerank a fused pool while preserving group-specific evidence.

    ``max_groups_per_candidate=0`` evaluates every group that proposed the
    candidate. A positive value keeps only the strongest origin groups. A
    candidate injected by an external/global ranking has no origin group, so
    it is evaluated against every group instead of being silently discarded.
    """
    groups = [list(group) for group in feature_groups if len(group) >= 2]
    candidate_indices = list(dict.fromkeys(
        int(index) for index in candidate_indices
    ))
    if not groups or not candidate_indices or top_k <= 0:
        return [], {}
    files = db_store.model_files(db_dir)
    up = np.asarray(cfg.up_axis if up is None else up, dtype=float)
    up /= max(np.linalg.norm(up), 1e-9)
    group_extents = [
        _extent_signature([
            getattr(feature, "position", np.zeros(3)) for feature in group
        ], up)
        for group in groups
    ]
    scored = []
    details = {}
    group_rank_maps = [
        {int(model_index): rank for rank, model_index in enumerate(ranking)}
        for ranking in (group_rankings or [])
    ]
    group_ranking_sizes = [len(ranking) for ranking in (group_rankings or [])]
    global_rank_map = {
        int(model_index): rank
        for rank, model_index in enumerate(global_ranking or [])
    }
    global_ranking_size = len(global_ranking or [])
    started = time.monotonic()
    total = len(candidate_indices)
    weights = np.ones(len(groups), dtype=float)
    if group_weights is not None:
        provided = np.asarray(group_weights, dtype=float).reshape(-1)
        if len(provided) == len(groups):
            weights = np.clip(np.where(
                np.isfinite(provided), provided, 0.0,
            ), 0.05, 20.0)
            weights /= max(float(weights.mean()), 1e-9)
    blend = float(np.clip(local_weight, 0.0, 1.0))
    for position, model_index in enumerate(candidate_indices, 1):
        group_scores = [float("-inf")] * len(groups)
        group_evidence = [None] * len(groups)
        preselection_scores = np.zeros(len(groups), dtype=float)
        preselection_ranks = [None] * len(groups)
        if len(group_rank_maps) == len(groups):
            for group_index, rank_map in enumerate(group_rank_maps):
                rank = rank_map.get(model_index)
                if rank is None:
                    continue
                preselection_ranks[group_index] = int(rank) + 1
                preselection_scores[group_index] = 1.0 - (
                    int(rank) / max(1, group_ranking_sizes[group_index])
                )
        origin_groups = sorted(
            np.flatnonzero(preselection_scores > 0.0).tolist(),
            key=lambda group_index: (
                -preselection_scores[group_index], group_index,
            ),
        )
        global_rank = global_rank_map.get(model_index)
        global_preselection_score = (
            1.0 - (int(global_rank) / max(1, global_ranking_size))
            if global_rank is not None else 0.0
        )
        if not origin_groups and global_preselection_score > 0.0:
            preselection_scores.fill(global_preselection_score)
        eligible_groups = list(range(len(groups)))
        if len(group_rank_maps) == len(groups):
            eligible_groups = origin_groups or eligible_groups
            if int(max_groups_per_candidate) > 0 and origin_groups:
                eligible_groups = origin_groups[:int(max_groups_per_candidate)]
        try:
            model = db_store.load_model(files[model_index])
            model_points = (object_surface_cloud(model).points
                            if getattr(model, "cloud", None) is not None else [
                                getattr(feature, "position", np.zeros(3))
                                for feature in model.features])
            model_extent = _extent_signature(model_points, up)
            for group_index in eligible_groups:
                group = groups[group_index]
                evidence = _local_descriptor_score(
                    model.features, group, cfg, up,
                    max_scan_features=len(group),
                    max_model_features=max_model_features,
                    n_rotations=n_rotations,
                    target_matches=target_matches,
                    target_coverage=target_coverage,
                    return_details=True,
                )
                if isinstance(evidence, dict):
                    local_score = float(evidence["score"])
                    extent_similarity = _extent_compatibility(
                        model_extent, group_extents[group_index], cfg,
                    )
                    shape_similarity = _group_shape_compatibility(
                        model.features, group, up,
                        n_rotations=n_rotations,
                    )
                    local_value = (
                        local_score if math.isfinite(local_score) else 0.0
                    )
                    evidence = {
                        **evidence,
                        "descriptor_score": local_score,
                        "extent_similarity": extent_similarity,
                        "group_shape_similarity": shape_similarity,
                        "score": (
                            0.50 * local_value
                            + 0.35 * shape_similarity
                            + 0.15 * extent_similarity
                        ),
                    }
                    group_scores[group_index] = float(evidence["score"])
                    group_evidence[group_index] = evidence
                else:
                    group_scores[group_index] = float(evidence)
        except Exception:
            group_scores = [float("-inf")] * len(groups)
        values = np.asarray(group_scores, float)
        local_values = np.where(np.isfinite(values), values, 0.0)
        combined_values = (
            blend * local_values
            + (1.0 - blend) * preselection_scores
        )
        # La confiance du groupe ne doit pas écraser la preuve géométrique.
        reliability = 0.85 + 0.15 * weights
        weighted_values = combined_values * reliability
        has_evidence = (
            np.isfinite(values) | (preselection_scores > 0.0)
        )
        weighted_values = np.where(has_evidence, weighted_values, -np.inf)
        best_group = (int(np.argmax(weighted_values))
                      if len(values) and np.isfinite(weighted_values).any()
                      else -1)
        best_score = (
            float(weighted_values[best_group])
            if best_group >= 0 else float("-inf")
        )
        details[model_index] = {
            "best_group": best_group,
            "best_score": best_score,
            "group_scores": group_scores,
            "preselection_scores": preselection_scores.tolist(),
            "preselection_ranks": preselection_ranks,
            "combined_group_scores": weighted_values.tolist(),
            "weighted_group_scores": weighted_values.tolist(),
            "group_evidence": group_evidence,
            "evaluated_groups": eligible_groups,
            "origin_groups": origin_groups,
            "global_preselection_rank": (
                int(global_rank) + 1 if global_rank is not None else None
            ),
            "global_preselection_score": global_preselection_score,
            "selected_for_groups": [],
        }
        scored.append((best_score, position, model_index))
        if progress and (position % 50 == 0 or position == total):
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = position / elapsed
            remaining = (total - position) / rate if rate > 0 else 0.0
            progress(
                "group reranking: "
                f"{position}/{total} model(s), approximately remaining "
                f"{int(round(remaining))}s"
            )
    ranked = [int(model_index) for _, _, model_index in sorted(
        scored, key=lambda item: (-item[0], item[1]),
    )]
    per_group_limit = max(0, int(candidates_per_group))
    if per_group_limit:
        selected_set = set()
        candidate_order = {
            model_index: position
            for position, model_index in enumerate(candidate_indices)
        }
        for group_index in np.argsort(-weights):
            group_ranked = sorted(
                candidate_indices,
                key=lambda model_index: (
                    -float(details[model_index]["combined_group_scores"][
                        int(group_index)
                    ]),
                    candidate_order[model_index],
                ),
            )
            kept = 0
            for model_index in group_ranked:
                value = details[model_index]["combined_group_scores"][
                    int(group_index)
                ]
                if not math.isfinite(float(value)):
                    continue
                details[model_index]["selected_for_groups"].append(
                    int(group_index)
                )
                selected_set.add(int(model_index))
                kept += 1
                if kept >= per_group_limit:
                    break
        return [
            model_index for model_index in ranked
            if model_index in selected_set
        ], details
    limit = min(int(top_k), len(ranked))
    quota = min(
        max(0, int(min_candidates_per_group)),
        limit // max(1, len(groups)),
    )
    protected = []
    protected_set = set()
    extent_quota = min(
        limit,
        max(0, int(round(limit * float(np.clip(
            extent_quota_fraction, 0.0, 1.0,
        ))))),
    )
    extent_quota_per_group = extent_quota // max(1, len(groups))
    if extent_quota_per_group:
        for group_index in np.argsort(-weights):
            extent_candidates = sorted(
                candidate_indices,
                key=lambda model_index: -float(
                    (details[model_index]["group_evidence"][
                        int(group_index)
                    ] or {}).get("extent_similarity", float("-inf"))
                ),
            )
            kept = 0
            for model_index in extent_candidates:
                evidence = details[model_index]["group_evidence"][
                    int(group_index)
                ]
                if not evidence or model_index in protected_set:
                    continue
                protected.append(int(model_index))
                protected_set.add(int(model_index))
                details[model_index].setdefault(
                    "selected_for_extent_groups", [],
                ).append(int(group_index))
                kept += 1
                if kept >= extent_quota_per_group:
                    break
    if quota:
        for group_index in np.argsort(-weights):
            group_candidates = sorted(
                candidate_indices,
                key=lambda model_index: -float(
                    details[model_index]["combined_group_scores"][
                        int(group_index)
                    ]
                ),
            )
            kept = 0
            for model_index in group_candidates:
                value = details[model_index]["combined_group_scores"][
                    int(group_index)
                ]
                if not math.isfinite(float(value)):
                    continue
                if model_index in protected_set:
                    continue
                protected.append(int(model_index))
                protected_set.add(int(model_index))
                details[model_index]["selected_for_groups"].append(
                    int(group_index)
                )
                kept += 1
                if kept >= quota:
                    break
    selected = [
        model_index for model_index in ranked if model_index in protected_set
    ]
    selected.extend(
        model_index for model_index in ranked
        if model_index not in protected_set
    )
    return selected[:limit], details


def default_index_path(db_dir: str) -> str:
    return str(Path(db_dir) / "candidate_index.npz")
