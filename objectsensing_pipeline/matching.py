'\nAssemble keypoint features and match model to scan. Features contain position, height, a local UDF or occupancy descriptor and support primitives. Test primitive-aligned yaw angles or uniform rotations, then apply confidence and primitive-size filters to estimate yaw, scale and translation.\n'

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from config import PipelineConfig
from descriptors import (LocalDescriptor, build_model_descriptor,
                         build_scan_descriptor, descriptor_distance,
                         descriptor_distance_batch, matching_confidence)
from geometry import PointCloud
from keypoints import KeyPoints
from primitives import PrimitiveSet, detect_primitives
from transforms import GroundTransform, estimate_from_correspondence


@dataclass
class KeyPointFeature:
    position: np.ndarray
    normal: np.ndarray
    height: float                      # hauteur au-dessus du plan du sol
    descriptor: LocalDescriptor
    primitives: PrimitiveSet
    size6d: np.ndarray                 # vecteur 6D (Ah,Av1,Av2,Lh,Lv1,Lv2)
    response: float = 0.0              # response de Harris (force du key point)
    selection_score: float | None = None  # priorité après contexte scan-only
    wall_score: float = 0.0
    object_score: float = 0.0
    wall_affinity: float = 0.0
    wall_proximity: float = 0.0
    wall_plane_index: int = -1


def _feature_selection_score(feature) -> float:
    value = getattr(feature, "selection_score", None)
    return float(feature.response if value is None else value)


def _response_order(feats, indices, spatial_balance=False, cell_size=0.5):
    'Rank a subset by response with optional spatial balancing.'
    indices = list(indices)
    if not spatial_balance:
        return sorted(indices, key=lambda i: -_feature_selection_score(feats[i]))

    cell_size = max(float(cell_size), 1e-6)
    cells = {}
    for i in indices:
        key = tuple(np.floor(np.asarray(feats[i].position) / cell_size).astype(int))
        cells.setdefault(key, []).append(i)
    for cell_indices in cells.values():
        cell_indices.sort(key=lambda i: -_feature_selection_score(feats[i]))

    ordered = []
    depth = 0
    ordered_cells = sorted(cells)
    while len(ordered) < len(indices):
        added = False
        for key in ordered_cells:
            cell_indices = cells[key]
            if depth < len(cell_indices):
                ordered.append(cell_indices[depth])
                added = True
        if not added:
            break
        depth += 1
    return ordered


def _wall_budget_indices(feats, indices, k, affinity_threshold,
                         proximity_threshold,
                         max_object_score, max_fraction, cell_size,
                         max_per_cell, min_features):
    'Distribute the small wall quota across planes and spatial cells.'
    wall = []
    non_wall = []
    for index in indices:
        feature = feats[index]
        affinity = float(getattr(feature, "wall_affinity", 0.0))
        proximity = float(getattr(feature, "wall_proximity", 0.0))
        object_score = float(getattr(feature, "object_score", 0.0))
        plane_index = int(getattr(feature, "wall_plane_index", -1))
        if (
            plane_index >= 0
            and (
                affinity >= float(affinity_threshold)
                or proximity >= float(proximity_threshold)
            )
            and object_score <= float(max_object_score)
        ):
            wall.append(index)
        else:
            non_wall.append(index)

    quota = max(0, int(round(float(k) * float(max_fraction))))
    if not wall or quota >= len(wall):
        return list(indices), [], wall, quota, []

    cell_size = max(float(cell_size), 1e-6)
    per_cell = max(1, int(max_per_cell))
    cells = {}
    for index in wall:
        feature = feats[index]
        position_cell = tuple(
            np.floor(np.asarray(feature.position, float) / cell_size).astype(int)
        )
        key = (int(getattr(feature, "wall_plane_index", -1)), *position_cell)
        cells.setdefault(key, []).append(index)
    candidates = []
    for cell_indices in cells.values():
        cell_indices.sort(
            key=lambda i: -_feature_selection_score(feats[i]),
        )
        candidates.extend(cell_indices[:per_cell])
    candidates.sort(key=lambda i: -_feature_selection_score(feats[i]))
    admitted_wall = candidates[:quota]
    admitted = set(non_wall)
    admitted.update(admitted_wall)
    kept = [index for index in indices if index in admitted]
    rejected = [index for index in wall if index not in admitted]
    minimum = min(max(0, int(min_features)), len(indices), int(k))
    reintroduced = []
    if len(kept) < minimum:
        missing = minimum - len(kept)
        reintroduced = sorted(
            rejected, key=lambda i: -_feature_selection_score(feats[i]),
        )[:missing]
        admitted.update(reintroduced)
        kept = [index for index in indices if index in admitted]
        rejected = [index for index in wall if index not in admitted]
    return kept, rejected, wall, quota, reintroduced


def _component_budget_indices(feats, indices, radius, max_per_component,
                              min_features, harris_threshold=None,
                              min_2d_fraction=0.0):
    'Cap each spatial component without imposing a fixed total.'
    indices = list(indices)
    if not indices:
        return [], [], [], [], []
    radius = max(float(radius), 1e-6)
    component_cap = max(1, int(max_per_component))
    positions = np.asarray([feats[index].position for index in indices], float)
    parent = np.arange(len(indices), dtype=int)
    rank = np.zeros(len(indices), dtype=np.int8)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return int(value)

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    from scipy.spatial import cKDTree
    for left, right in cKDTree(positions).query_pairs(radius):
        union(int(left), int(right))

    components = {}
    for local_index, feature_index in enumerate(indices):
        components.setdefault(find(local_index), []).append(feature_index)
    ordered_components = list(components.values())
    admitted = set()
    rejected = []
    for component in ordered_components:
        ordered = _response_order(feats, component)
        selected = []
        if harris_threshold is not None and min_2d_fraction > 0:
            corner_2d = [
                index for index in component
                if feats[index].response <= harris_threshold
            ]
            corner_3d = [
                index for index in component
                if feats[index].response > harris_threshold
            ]
            reserve_2d = min(
                len(corner_2d),
                max(1, int(round(component_cap * min_2d_fraction))),
            )
            reserve_3d = min(len(corner_3d), component_cap - reserve_2d)
            selected.extend(_response_order(feats, corner_3d)[:reserve_3d])
            selected.extend(_response_order(feats, corner_2d)[:reserve_2d])
        if len(selected) < component_cap:
            selected_set = set(selected)
            selected.extend(
                index for index in ordered if index not in selected_set
            )
        selected = selected[:component_cap]
        selected_set = set(selected)
        admitted.update(selected)
        rejected.extend(index for index in ordered if index not in selected_set)

    kept = [index for index in indices if index in admitted]
    minimum = min(max(0, int(min_features)), len(indices))
    reintroduced = []
    if len(kept) < minimum:
        reintroduced = sorted(
            rejected,
            key=lambda index: -_feature_selection_score(feats[index]),
        )[:minimum - len(kept)]
        admitted.update(reintroduced)
        kept = [index for index in indices if index in admitted]
        rejected = [index for index in rejected if index not in admitted]
    component_sizes = [len(component) for component in ordered_components]
    return kept, rejected, ordered_components, component_sizes, reintroduced


def _cap_by_response(feats, k, spatial_balance=False, cell_size=0.5,
                     harris_threshold=None, min_2d_fraction=0.0,
                     wall_budget_enabled=False,
                     wall_budget_affinity_threshold=0.05,
                     wall_budget_proximity_threshold=0.05,
                     wall_budget_max_object_score=0.35,
                     wall_budget_max_fraction=0.05,
                     wall_budget_cell_size=0.40,
                     wall_budget_max_per_cell=1,
                     wall_budget_min_features=40,
                     component_budget_enabled=False,
                     component_budget_radius=0.35,
                     component_budget_max_per_component=24,
                     component_budget_min_features=40,
                     trace=None):
    'Select at most k features. Spatial mode ranks within cells and samples round-robin. When a Harris threshold is supplied, reserve part of the budget for 2D corners. k <= 0 disables capping.\n    '
    if not k or k <= 0 or (
        len(feats) <= k
        and not wall_budget_enabled
        and not component_budget_enabled
    ):
        if trace is not None:
            trace.update({
                "wall_budget_candidates": [],
                "wall_budget_rejected": [],
                "wall_budget_reintroduced": [],
                "wall_budget_quota": 0,
                "component_budget_rejected": [],
                "component_budget_reintroduced": [],
                "component_budget_count": 0,
                "component_budget_sizes": [],
            })
        return feats

    all_indices = list(range(len(feats)))
    if wall_budget_enabled:
        effective_budget = min(int(k), len(all_indices))
        (all_indices, wall_rejected, wall_candidates, wall_quota,
         wall_reintroduced) = (
            _wall_budget_indices(
                feats, all_indices, effective_budget,
                wall_budget_affinity_threshold,
                wall_budget_proximity_threshold,
                wall_budget_max_object_score,
                wall_budget_max_fraction,
                wall_budget_cell_size,
                wall_budget_max_per_cell,
                wall_budget_min_features,
            )
        )
    else:
        wall_rejected, wall_candidates, wall_quota = [], [], 0
        wall_reintroduced = []
    if component_budget_enabled:
        (all_indices, component_rejected, components, component_sizes,
         component_reintroduced) = _component_budget_indices(
            feats, all_indices,
            component_budget_radius,
            component_budget_max_per_component,
            component_budget_min_features,
            harris_threshold,
            min_2d_fraction,
        )
    else:
        component_rejected, components, component_sizes = [], [], []
        component_reintroduced = []
    if trace is not None:
        trace.update({
            "wall_budget_candidates": wall_candidates,
            "wall_budget_rejected": wall_rejected,
            "wall_budget_reintroduced": wall_reintroduced,
            "wall_budget_quota": wall_quota,
            "component_budget_rejected": component_rejected,
            "component_budget_reintroduced": component_reintroduced,
            "component_budget_count": len(components),
            "component_budget_sizes": component_sizes,
        })
    if len(all_indices) <= k:
        return [feats[index] for index in all_indices]
    if harris_threshold is None or min_2d_fraction <= 0:
        order = _response_order(feats, all_indices, spatial_balance, cell_size)
        return [feats[i] for i in order[:k]]

    corner_2d = [i for i in all_indices if feats[i].response <= harris_threshold]
    corner_3d = [i for i in all_indices if feats[i].response > harris_threshold]
    reserve_2d = min(len(corner_2d), int(round(k * min_2d_fraction)))
    reserve_3d = min(len(corner_3d), k - reserve_2d)

    order_3d = _response_order(feats, corner_3d, spatial_balance, cell_size)
    order_2d = _response_order(feats, corner_2d, spatial_balance, cell_size)
    selected = order_3d[:reserve_3d] + order_2d[:reserve_2d]

    if len(selected) < k:
        selected_set = set(selected)
        remainder = _response_order(feats, all_indices, spatial_balance, cell_size)
        selected.extend(i for i in remainder if i not in selected_set)
    return [feats[i] for i in selected[:k]]


@dataclass
class Correspondence:
    model_idx: int
    scan_idx: int
    theta: float
    scale: float
    desc_dist: float
    transform: GroundTransform


def _primitive_radius(cfg: PipelineConfig) -> float:
    return max(3 * cfg.keypoint.neighbor_radius, cfg.descriptor.udf_extent * 1.5)


def build_model_features(cloud: PointCloud, kps: KeyPoints, cfg: PipelineConfig,
                         up, ground: float, surf_tree=None) -> List[KeyPointFeature]:
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    tree = cloud.tree
    # UDF = distance à la surface COMPLÈTE du modèle (eq.3). Idéalement un
    # échantillonnage dense fourni par l'appelant ; sinon repli sur le nuage.
    if surf_tree is None:
        surf_tree = tree
    R = _primitive_radius(cfg)
    feats = []
    for i in range(kps.size):
        p = kps.positions[i]
        desc = build_model_descriptor(surf_tree, p, cfg.descriptor)
        idx = tree.query_ball_point(p, R)
        pset = detect_primitives(cloud.points[idx], cloud.normals[idx], cfg.primitive, up)
        feats.append(KeyPointFeature(p, kps.normals[i], float(p @ up - ground),
                                     desc, pset, pset.size_vector(cfg.primitive),
                                     float(kps.responses[i])))
    return _cap_by_response(
        feats, cfg.matching.max_model_keypoints,
        spatial_balance=cfg.matching.spatial_keypoint_balance,
        cell_size=cfg.matching.spatial_keypoint_cell,
        harris_threshold=cfg.keypoint.harris_threshold,
        min_2d_fraction=cfg.matching.min_2d_corner_fraction,
    )


def _filter_scan_feature_quality(feats, cfg, trace=None):
    'Remove weak descriptors, then apply a second quality-based NMS.'
    feats = list(feats)
    count = len(feats)
    if not count:
        if trace is not None:
            trace.update({"descriptor_input": 0, "descriptor_rejected": 0,
                          "score_rejected": 0, "quality_nms_rejected": 0})
        return feats
    matching_cfg = cfg.matching
    enabled = bool(matching_cfg.descriptor_keypoint_filter_enabled)
    ceiling = max(0, int(matching_cfg.descriptor_keypoint_min_features))
    absolute = max(0, int(getattr(
        matching_cfg, "descriptor_keypoint_min_absolute", ceiling,
    )))
    ratio = max(0.0, float(getattr(
        matching_cfg, "descriptor_keypoint_min_ratio", 1.0,
    )))
    minimum = min(count, max(absolute, int(np.ceil(count * ratio))))
    if ceiling > 0:
        minimum = min(minimum, ceiling)
    keep = np.ones(count, bool)
    descriptor_ok = np.ones(count, bool)
    utility_scores = np.zeros(count, float)
    for index, feature in enumerate(feats):
        descriptor = feature.descriptor
        occupied = int(descriptor.n_occupied)
        removed = int(getattr(descriptor, "n_hole_boundary_removed", 0))
        hole_fraction = removed / max(1, occupied + removed)
        support = min(1.0, occupied / max(1.0, float(matching_cfg.descriptor_min_occupied) * 2.0))
        utility = support * max(0.0, 1.0 - hole_fraction)
        utility_scores[index] = max(0.0, _feature_selection_score(feature)) * utility
        feature.selection_score = utility_scores[index]
        if enabled:
            descriptor_ok[index] = (
                occupied >= int(matching_cfg.descriptor_min_occupied)
                and hole_fraction <= float(matching_cfg.descriptor_max_hole_fraction)
            )
    if enabled:
        keep &= descriptor_ok
    three_d = np.asarray([
        float(feature.response) > float(cfg.keypoint.harris_threshold)
        for feature in feats
    ], bool)
    score_ok = np.ones(count, bool)
    if np.any(three_d):
        peak = float(np.max(utility_scores[three_d]))
        if peak > 0:
            score_ok[three_d] = utility_scores[three_d] >= (
                peak * float(matching_cfg.quality_min_score_ratio)
            )
    if enabled:
        keep &= score_ok

    reintroduced = []

    def restore_floor(mask, eligible=None):
        if int(mask.sum()) >= minimum:
            return mask
        if eligible is None:
            eligible = np.ones(count, bool)
        rejected = np.flatnonzero(~mask & np.asarray(eligible, bool))
        fraction = max(0.0, float(getattr(
            matching_cfg,
            "descriptor_keypoint_max_reintroduced_fraction", 1.0,
        )))
        maximum = int(np.floor(len(rejected) * fraction))
        needed = min(minimum - int(mask.sum()), maximum)
        if needed <= 0:
            return mask
        order = rejected[np.argsort(-utility_scores[rejected])]
        restored = order[:needed]
        mask[restored] = True
        reintroduced.extend(int(value) for value in restored)
        return mask

    # Empty/hole-dominated descriptors are hard rejects. Only features that
    # passed the intrinsic descriptor gate may be restored after soft scoring.
    keep = restore_floor(keep, eligible=descriptor_ok)
    before_nms = keep.copy()
    radius = max(0.0, float(matching_cfg.quality_nms_radius))
    if enabled and radius > 0 and int(keep.sum()) > 1:
        active = np.flatnonzero(keep)
        tree = cKDTree(np.asarray([feats[index].position for index in active]))
        order = np.argsort(-utility_scores[active])
        suppressed = np.zeros(len(active), bool)
        selected = []
        for local in order:
            if suppressed[local]:
                continue
            selected.append(local)
            for neighbor in tree.query_ball_point(tree.data[local], radius):
                if neighbor != local:
                    suppressed[neighbor] = True
        keep[:] = False
        keep[active[np.asarray(selected, int)]] = True
        keep = restore_floor(keep, eligible=descriptor_ok)
    if trace is not None:
        trace.update({
            "descriptor_input": count,
            "descriptor_rejected": int(np.count_nonzero(~descriptor_ok)),
            "score_rejected": int(np.count_nonzero(descriptor_ok & ~score_ok)),
            "quality_nms_rejected": int(np.count_nonzero(before_nms & ~keep)),
            "quality_retained": int(keep.sum()),
            "descriptor_reintroduced": sorted(set(reintroduced)),
            "descriptor_adaptive_floor": int(minimum),
        })
    return [feature for index, feature in enumerate(feats) if keep[index]]


def build_scan_features(cloud: PointCloud, tsdf_volume, kps: KeyPoints,
                        cfg: PipelineConfig, up, ground: float,
                        trace=None) -> List[KeyPointFeature]:
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    tree = cloud.tree
    R = _primitive_radius(cfg)
    feats = []
    for i in range(kps.size):
        p = kps.positions[i]
        height = float(p @ up - ground)
        if height < cfg.matching.floor_height:
            continue                            # key point de sol -> ignoré (sec. 6)
        desc = build_scan_descriptor(tsdf_volume, p, cfg.descriptor)
        idx = tree.query_ball_point(p, R)
        pset = detect_primitives(cloud.points[idx], cloud.normals[idx], cfg.primitive, up)
        feats.append(KeyPointFeature(p, kps.normals[i], height,
                                     desc, pset, pset.size_vector(cfg.primitive),
                                     float(kps.responses[i]),
                                     float(
                                         kps.selection_scores[i]
                                         if kps.selection_scores is not None
                                         else kps.responses[i]
                                     ),
                                     float(
                                         kps.wall_scores[i]
                                         if kps.wall_scores is not None else 0.0
                                     ),
                                     float(
                                         kps.object_scores[i]
                                         if kps.object_scores is not None else 0.0
                                     ),
                                     float(
                                         kps.wall_affinities[i]
                                         if kps.wall_affinities is not None else 0.0
                                     ),
                                     float(
                                         kps.wall_proximities[i]
                                         if kps.wall_proximities is not None else 0.0
                                     ),
                                     int(
                                         kps.wall_plane_indices[i]
                                         if kps.wall_plane_indices is not None else -1
                                     )))
    feats = _filter_scan_feature_quality(feats, cfg, trace=trace)
    return _cap_by_response(
        feats, cfg.matching.max_scan_keypoints,
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
        trace=trace,
    )


def _horizontal_angle(direction, up, e1, e2) -> Optional[float]:
    d = np.asarray(direction, float)
    dh = d - (d @ up) * up
    if np.linalg.norm(dh) < 1e-6:
        return None
    return float(np.arctan2(dh @ e2, dh @ e1))


def _candidate_thetas(mf: KeyPointFeature, sf: KeyPointFeature, cfg, up, e1, e2):
    m_angles = [a for a in (_horizontal_angle(d, up, e1, e2) for d in mf.primitives.directions())
                if a is not None]
    s_angles = [a for a in (_horizontal_angle(d, up, e1, e2) for d in sf.primitives.directions())
                if a is not None]
    thetas = []
    if m_angles and s_angles:
        for ma in m_angles:
            for sa in s_angles:
                thetas.append(sa - ma)
                thetas.append(sa - ma + np.pi)      # ambiguïté de signe des plans
    if not thetas or bool(getattr(
        cfg.matching, "primitive_rotation_fallback", False,
    )):
        n = cfg.matching.n_uniform_rotations
        thetas.extend(np.linspace(0, 2 * np.pi, n, endpoint=False))
    # Les directions de primitives peuvent produire plusieurs fois le même
    # angle. Une quantification fine évite de recalculer ces rotations.
    unique = {}
    for theta in thetas:
        wrapped = float(theta) % (2.0 * np.pi)
        unique.setdefault(round(wrapped, 8), wrapped)
    return list(unique.values())


def _size_ratio_ok(a: np.ndarray, b: np.ndarray, limit: float) -> bool:
    r = np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-9)
    return bool(np.all(r <= limit))


def _compatible_primitive_sizes(model_sizes: np.ndarray, scan_size: np.ndarray,
                                limit: float, epsilon: float = 1e-8) -> np.ndarray:
    'Compare only primitives observed on both sides. Missing scan primitives are not evidence of absence: occlusion can hide them. This avoids rejecting partial observations through near-zero size ratios.\n    '
    model_sizes = np.asarray(model_sizes, float)
    scan_size = np.asarray(scan_size, float)
    comparable = (model_sizes > epsilon) & (scan_size[None, :] > epsilon)
    ratios = np.ones_like(model_sizes)
    np.divide(
        np.maximum(model_sizes, scan_size),
        np.minimum(model_sizes, scan_size),
        out=ratios,
        where=comparable,
    )
    return np.all((~comparable) | (ratios <= float(limit)), axis=1)


def match_features(model_feats: List[KeyPointFeature], scan_feats: List[KeyPointFeature],
                   cfg: PipelineConfig, up, diagnostics=None) -> List[Correspondence]:
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    mcfg = cfg.matching
    M = len(model_feats)
    if M == 0:
        return []

    # Tableaux modèle précalculés (filtres vectorisés).
    m_height = np.array([mf.height for mf in model_feats])
    m_size6d = np.array([mf.size6d for mf in model_feats])          # (M,6)
    m_nocc = np.array([max(mf.descriptor.n_occupied, 1) for mf in model_feats])

    corres: List[Correspondence] = []
    stats = {
        "scan_keypoints": len(scan_feats),
        "scan_with_scale_candidates": 0,
        "scan_with_size_candidates": 0,
        "scan_with_confidence_candidates": 0,
        "scan_with_descriptor_candidates": 0,
        "scale_pairs": 0,
        "size_pairs": 0,
        "confidence_pairs": 0,
        "descriptor_pairs": 0,
    }
    per_scan = []
    for sj, sf in enumerate(scan_feats):
        sh = sf.height
        scales = (sh / np.maximum(m_height, 1e-9)) if sh > 1e-3 else np.ones(M)
        scales = np.where(m_height > 1e-3, scales, 1.0)
        scale_ok = (scales >= mcfg.scale_min) & (scales <= mcfg.scale_max)
        # ratio de tailles 6D (rejet si un ratio > limit)
        size_ok = _compatible_primitive_sizes(
            m_size6d, sf.size6d, mcfg.size_ratio_reject,
        )
        # confiance (eq. 5)
        conf = np.minimum(1.0, (sf.descriptor.n_occupied + sf.descriptor.n_unknown) / m_nocc)
        confidence_ok = conf >= mcfg.confidence_threshold
        scale_count = int(np.count_nonzero(scale_ok))
        size_count = int(np.count_nonzero(scale_ok & size_ok))
        confidence_count = int(np.count_nonzero(
            scale_ok & size_ok & confidence_ok
        ))
        stats["scale_pairs"] += scale_count
        stats["size_pairs"] += size_count
        stats["confidence_pairs"] += confidence_count
        stats["scan_with_scale_candidates"] += int(scale_count > 0)
        stats["scan_with_size_candidates"] += int(size_count > 0)
        stats["scan_with_confidence_candidates"] += int(confidence_count > 0)
        ok = np.where(scale_ok & size_ok & confidence_ok)[0]

        cand = []
        best_unthresholded = float("inf")
        for mi in ok:
            mf = model_feats[mi]
            thetas = np.asarray(_candidate_thetas(mf, sf, cfg, up, e1, e2), float)
            # Les hypothèses portent le modèle vers le scan. La lecture UDF
            # fait le trajet inverse : offsets scan vers le repère du modèle.
            dists = descriptor_distance_batch(
                mf.descriptor, sf.descriptor, cfg.descriptor, -thetas, up,
            )
            k = int(np.argmin(dists))
            best_distance = float(dists[k])
            best_unthresholded = min(best_unthresholded, best_distance)
            cand.append((best_distance, int(mi), float(thetas[k]), float(scales[mi])))
        cand = sorted(
            (item for item in cand
             if np.isfinite(item[0]) and item[0] < cfg.ransac.desc_inlier),
            key=lambda x: x[0],
        )
        descriptor_count = len(cand)
        stats["descriptor_pairs"] += descriptor_count
        stats["scan_with_descriptor_candidates"] += int(descriptor_count > 0)
        ratio_threshold = float(
            getattr(mcfg, "descriptor_ratio_threshold", 0.0) or 0.0
        )
        if ratio_threshold > 0.0 and len(cand) > 1:
            best_distance, second_distance = cand[0][0], cand[1][0]
            # Deux distances nulles (ou quasi nulles) ne donnent none preuve
            # sur l'identité du keypoint modèle : on rejette cette ambiguïté.
            if second_distance <= 1e-12:
                continue
            if best_distance / second_distance > ratio_threshold:
                continue
            cand = cand[:1]
        for best_d, mi, best_t, s in cand[: mcfg.max_correspondences_per_kp]:
            T = estimate_from_correspondence(model_feats[mi].position, sf.position, best_t, s, up)
            corres.append(Correspondence(mi, sj, best_t, s, best_d, T))
        if diagnostics is not None:
            per_scan.append({
                "scan_keypoint": int(sj),
                "scale_candidates": scale_count,
                "size_candidates": size_count,
                "confidence_candidates": confidence_count,
                "descriptor_candidates": descriptor_count,
                "best_descriptor_distance": (
                    round(best_unthresholded, 4)
                    if np.isfinite(best_unthresholded) else None
                ),
                "retained_correspondences": min(
                    descriptor_count, int(mcfg.max_correspondences_per_kp),
                ),
            })
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(stats)
        diagnostics["distinct_scan_keypoints"] = len({c.scan_idx for c in corres})
        diagnostics["distinct_model_keypoints"] = len({c.model_idx for c in corres})
        diagnostics["per_scan"] = per_scan
    return corres
