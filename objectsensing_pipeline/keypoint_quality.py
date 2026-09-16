'Scan-only stability filters for RGB-D reconstruction keypoints.'

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from keypoints import KeyPoints, _covariance_normals, _harris_response
from parallelism import kdtree_workers


@dataclass
class KeypointQualityResult:
    keypoints: KeyPoints
    planar_rejected: int = 0
    hole_rejected: int = 0
    repeatability_rejected: int = 0
    score_rejected: int = 0
    nms_rejected: int = 0
    planar_rejected_positions: np.ndarray | None = None
    hole_rejected_positions: np.ndarray | None = None
    repeatability_rejected_positions: np.ndarray | None = None
    planar_rejected_sources: np.ndarray | None = None
    hole_rejected_sources: np.ndarray | None = None
    repeatability_rejected_sources: np.ndarray | None = None
    score_rejected_positions: np.ndarray | None = None
    score_rejected_sources: np.ndarray | None = None
    nms_rejected_positions: np.ndarray | None = None
    nms_rejected_sources: np.ndarray | None = None
    reintroduced_sources: dict | None = None


def _adaptive_target(count, cfg):
    ceiling = max(0, int(getattr(cfg, "quality_filter_min_features", 0)))
    absolute = max(0, int(getattr(
        cfg, "quality_filter_min_absolute", ceiling,
    )))
    ratio = max(0.0, float(getattr(cfg, "quality_filter_min_ratio", 1.0)))
    target = max(absolute, int(np.ceil(count * ratio)))
    if ceiling > 0:
        target = min(target, ceiling)
    return min(count, target)


def _guarded_keep(keypoints, keep, cfg, eligible=None):
    """Apply an adaptive recall floor without restoring certain rejects."""
    keep = np.asarray(keep, bool).copy()
    target = _adaptive_target(len(keep), cfg)
    if int(keep.sum()) >= target:
        return keep, np.zeros(0, int)
    scores = keypoints.selection_scores
    if scores is None:
        scores = keypoints.responses
    if eligible is None:
        eligible = np.ones(len(keep), bool)
    eligible = np.asarray(eligible, bool)
    rejected = np.flatnonzero(~keep & eligible)
    maximum_fraction = max(0.0, float(getattr(
        cfg, "quality_filter_max_reintroduced_fraction", 1.0,
    )))
    maximum = int(np.floor(len(rejected) * maximum_fraction))
    needed = min(target - int(keep.sum()), maximum)
    if needed <= 0:
        return keep, np.zeros(0, int)
    order = rejected[np.argsort(-np.asarray(scores, float)[rejected])]
    restored = order[:needed]
    keep[restored] = True
    return keep, restored


def _planar_interior_mask(cloud, keypoints, cfg):
    count = keypoints.size
    interior = np.zeros(count, bool)
    if not cfg.planar_filter_enabled or not count:
        return interior
    radius = max(float(cfg.planar_filter_radius), 1e-6)
    neighborhoods = cloud.tree.query_ball_point(
        keypoints.positions, radius, workers=kdtree_workers(),
    )
    bins = 12
    for index, neighbors in enumerate(neighborhoods):
        if len(neighbors) < int(cfg.planar_filter_min_neighbors):
            continue
        points = np.asarray(cloud.points[neighbors], float)
        center = points.mean(axis=0)
        covariance = np.cov((points - center).T, bias=True)
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        residual = float(np.sqrt(max(values[0], 0.0)))
        alignment = float(np.median(np.abs(cloud.normals[neighbors] @ normal)))
        if (
            residual > float(cfg.planar_filter_max_residual)
            or alignment < float(cfg.planar_filter_normal_cos)
        ):
            continue
        tangent1 = vectors[:, 2]
        tangent2 = vectors[:, 1]
        relative = points - keypoints.positions[index]
        radial = np.column_stack((relative @ tangent1, relative @ tangent2))
        lengths = np.linalg.norm(radial, axis=1)
        radial = radial[lengths >= radius * 0.25]
        if not len(radial):
            continue
        angles = np.mod(np.arctan2(radial[:, 1], radial[:, 0]), 2 * np.pi)
        occupied = np.unique(np.floor(angles / (2 * np.pi) * bins).astype(int))
        coverage = len(occupied) / bins
        interior[index] = coverage >= float(cfg.planar_filter_angular_coverage)
    return interior


def _hole_boundary_mask(scene, keypoints, cfg):
    count = keypoints.size
    rejected = np.zeros(count, bool)
    sampler = getattr(scene, "sample_visibility_details", None)
    if sampler is None:
        sampler = getattr(getattr(scene, "volume", None),
                          "sample_visibility_details", None)
    if not cfg.hole_boundary_filter_enabled or not count or sampler is None:
        return rejected
    radius = max(float(cfg.hole_boundary_probe_radius), 0.0)
    offsets = np.asarray([
        [0, 0, 0], [radius, 0, 0], [-radius, 0, 0],
        [0, radius, 0], [0, -radius, 0], [0, 0, radius], [0, 0, -radius],
    ], float)
    probes = (keypoints.positions[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    _, boundary = sampler(probes)
    boundary = np.asarray(boundary, bool).reshape(count, len(offsets))
    fraction = boundary.mean(axis=1)
    rejected = boundary[:, 0] | (
        fraction >= float(cfg.hole_boundary_max_fraction)
    )
    return rejected


def _unstable_harris_mask(cloud, keypoints, cfg):
    count = keypoints.size
    unstable = np.zeros(count, bool)
    if not cfg.repeatability_filter_enabled or not count:
        return unstable
    radius = max(
        float(cfg.neighbor_radius) * float(cfg.repeatability_radius_factor),
        float(cfg.neighbor_radius),
    )
    neighborhoods = cloud.tree.query_ball_point(
        keypoints.positions, radius, workers=kdtree_workers(),
    )
    threshold = (
        float(cfg.harris_threshold)
        * float(cfg.repeatability_response_ratio)
    )
    for index, neighbors in enumerate(neighborhoods):
        # Les coins 2D sont validés par leur support planaire étendu, pas par
        # Harris 3D; ils ne doivent pas être détruits par ce test multi-scale.
        if float(keypoints.responses[index]) <= float(cfg.harris_threshold):
            continue
        if len(neighbors) < 6:
            unstable[index] = True
            continue
        covariance = _covariance_normals(cloud.normals[neighbors])
        response = _harris_response(
            covariance, cfg.harris_k, cfg.harris_reference_neighbors,
        )
        unstable[index] = not np.isfinite(response) or response < threshold
    return unstable


def _weak_response_mask(keypoints, cfg):
    'Local 3D Harris threshold; the 2D branch remains independent.'
    responses = np.asarray(keypoints.responses, float)
    three_d = responses > float(cfg.harris_threshold)
    weak = np.zeros(len(responses), bool)
    valid = three_d & np.isfinite(responses)
    if not valid.any():
        return weak
    radius = max(float(cfg.quality_score_radius), 1e-6)
    positions = np.asarray(keypoints.positions, float)
    tree = cKDTree(positions)
    for index in np.flatnonzero(valid):
        neighbors = np.asarray(
            tree.query_ball_point(positions[index], radius), int,
        )
        local = responses[neighbors]
        local = local[
            three_d[neighbors] & np.isfinite(local)
        ]
        # Un coin isolé ne doit pas être pénalisé par une référence distante.
        if len(local) < 4:
            continue
        reference = float(np.percentile(local, 90))
        threshold = max(
            float(cfg.harris_threshold),
            reference * float(cfg.quality_response_ratio),
        )
        weak[index] = responses[index] < threshold
    return weak


def _quality_nms_mask(keypoints, cfg):
    count = keypoints.size
    rejected = np.zeros(count, bool)
    radius = max(0.0, float(cfg.geometric_nms_radius))
    if count < 2 or radius <= 0:
        return rejected
    scores = keypoints.selection_scores
    if scores is None:
        scores = keypoints.responses
    scores = np.asarray(scores, float)
    tree = keypoints.positions
    index = cKDTree(tree)
    suppressed = np.zeros(count, bool)
    for current in np.argsort(-scores):
        if suppressed[current]:
            continue
        for neighbor in index.query_ball_point(tree[current], radius):
            if neighbor != current:
                suppressed[neighbor] = True
    rejected[:] = suppressed
    return rejected


def filter_scene_keypoints(scene, keypoints, cfg) -> KeypointQualityResult:
    'Apply planar, RGB-D hole and Harris repeatability filters in order.'
    if int(getattr(keypoints, "size", 0)) == 0:
        empty_positions = np.zeros((0, 3), float)
        empty_sources = np.zeros(0, int)
        return KeypointQualityResult(
            keypoints,
            planar_rejected_positions=empty_positions,
            hole_rejected_positions=empty_positions,
            repeatability_rejected_positions=empty_positions,
            score_rejected_positions=empty_positions,
            nms_rejected_positions=empty_positions,
            planar_rejected_sources=empty_sources,
            hole_rejected_sources=empty_sources,
            repeatability_rejected_sources=empty_sources,
            score_rejected_sources=empty_sources,
            nms_rejected_sources=empty_sources,
        )
    current = keypoints
    planar = _planar_interior_mask(scene.cloud, current, cfg)
    reintroduced = {}
    keep, restored = _guarded_keep(current, ~planar, cfg)
    reintroduced["planar"] = np.asarray(current.source_index[restored], int)
    planar_positions = np.asarray(current.positions[~keep], float)
    planar_sources = np.asarray(current.source_index[~keep], int)
    planar_count = int((~keep).sum())
    current = current.subset(np.flatnonzero(keep))
    hole = _hole_boundary_mask(scene, current, cfg)
    # A confirmed free/unknown boundary is a hard rejection and must not be
    # restored merely to satisfy a numerical floor.
    keep, restored = _guarded_keep(current, ~hole, cfg, eligible=~hole)
    reintroduced["hole"] = np.asarray(current.source_index[restored], int)
    hole_positions = np.asarray(current.positions[~keep], float)
    hole_sources = np.asarray(current.source_index[~keep], int)
    hole_count = int((~keep).sum())
    current = current.subset(np.flatnonzero(keep))
    unstable = _unstable_harris_mask(scene.cloud, current, cfg)
    keep, restored = _guarded_keep(current, ~unstable, cfg)
    reintroduced["repeatability"] = np.asarray(
        current.source_index[restored], int,
    )
    unstable_positions = np.asarray(current.positions[~keep], float)
    unstable_sources = np.asarray(current.source_index[~keep], int)
    unstable_count = int((~keep).sum())
    current = current.subset(np.flatnonzero(keep))
    weak = _weak_response_mask(current, cfg)
    keep, restored = _guarded_keep(current, ~weak, cfg)
    reintroduced["score"] = np.asarray(current.source_index[restored], int)
    weak_positions = np.asarray(current.positions[~keep], float)
    weak_sources = np.asarray(current.source_index[~keep], int)
    weak_count = int((~keep).sum())
    current = current.subset(np.flatnonzero(keep))
    nms = _quality_nms_mask(current, cfg)
    keep, restored = _guarded_keep(current, ~nms, cfg)
    reintroduced["nms"] = np.asarray(current.source_index[restored], int)
    nms_positions = np.asarray(current.positions[~keep], float)
    nms_sources = np.asarray(current.source_index[~keep], int)
    nms_count = int((~keep).sum())
    current = current.subset(np.flatnonzero(keep))
    return KeypointQualityResult(
        current,
        planar_rejected=planar_count,
        hole_rejected=hole_count,
        repeatability_rejected=unstable_count,
        score_rejected=weak_count,
        nms_rejected=nms_count,
        planar_rejected_positions=planar_positions,
        hole_rejected_positions=hole_positions,
        repeatability_rejected_positions=unstable_positions,
        planar_rejected_sources=planar_sources,
        hole_rejected_sources=hole_sources,
        repeatability_rejected_sources=unstable_sources,
        score_rejected_positions=weak_positions,
        score_rejected_sources=weak_sources,
        nms_rejected_positions=nms_positions,
        nms_rejected_sources=nms_sources,
        reintroduced_sources=reintroduced,
    )
