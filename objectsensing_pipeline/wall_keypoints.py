'Scan-only analysis near large vertical planes. Group nearly horizontal normals by orientation and distance, validate large supports, and measure wall affinity without changing the original Harris response.\n'

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from config import KeypointConfig
from geometry import PointCloud
from keypoints import KeyPoints
from parallelism import kdtree_workers


@dataclass(frozen=True)
class WallPlane:
    'Large vertical plane bounded in tangent/vertical coordinates.'

    normal: np.ndarray
    offset: float
    tangent: np.ndarray
    tangent_min: float
    tangent_max: float
    height_min: float
    height_max: float
    support_count: int
    quality: float

    @property
    def width(self) -> float:
        return float(self.tangent_max - self.tangent_min)

    @property
    def height(self) -> float:
        return float(self.height_max - self.height_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, float) @ self.normal - self.offset

    def contains(self, points: np.ndarray, up: np.ndarray,
                 margin: float = 0.0) -> np.ndarray:
        points = np.asarray(points, float)
        tangent = points @ self.tangent
        height = points @ up
        return (
            (tangent >= self.tangent_min - margin)
            & (tangent <= self.tangent_max + margin)
            & (height >= self.height_min - margin)
            & (height <= self.height_max + margin)
        )

    def corners(self, up: np.ndarray) -> np.ndarray:
        'Return the four corners of the plane support rectangle.'
        origin = self.normal * self.offset
        return np.asarray([
            origin + self.tangent * self.tangent_min + up * self.height_min,
            origin + self.tangent * self.tangent_max + up * self.height_min,
            origin + self.tangent * self.tangent_max + up * self.height_max,
            origin + self.tangent * self.tangent_min + up * self.height_max,
        ])


@dataclass
class WallPlaneAnalysis:
    planes: list[WallPlane]
    surface_plane_index: np.ndarray
    keypoint_plane_index: np.ndarray
    plane_proximity: np.ndarray
    plane_affinity: np.ndarray
    small_support: np.ndarray
    large_support: np.ndarray
    protrusion_score: np.ndarray
    discontinuity_score: np.ndarray
    object_anchor_score: np.ndarray
    object_score: np.ndarray
    wall_score: np.ndarray


@dataclass
class WallFilterResult:
    keypoints: KeyPoints
    analysis: WallPlaneAnalysis | None
    kept_mask: np.ndarray
    rejected_mask: np.ndarray


def _horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ up)) > 0.9:
        axis = np.array([0.0, 0.0, 1.0])
    first = axis - (axis @ up) * up
    first /= np.clip(np.linalg.norm(first), 1e-12, None)
    second = np.cross(up, first)
    second /= np.clip(np.linalg.norm(second), 1e-12, None)
    return first, second


def _angle_distance_mod_pi(a: np.ndarray, b: float) -> np.ndarray:
    delta = np.abs(np.asarray(a) - float(b))
    return np.minimum(delta, math.pi - delta)


def _robust_extent(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 20:
        return float(values.min()), float(values.max())
    low, high = np.percentile(values, [1.0, 99.0])
    return float(low), float(high)


def _canonical_plane(normal: np.ndarray, offset: float,
                     first: np.ndarray, second: np.ndarray):
    angle = math.atan2(float(normal @ second), float(normal @ first))
    if angle < 0.0:
        angle += math.pi
        normal = -normal
        offset = -offset
    elif angle >= math.pi:
        angle -= math.pi
        normal = -normal
        offset = -offset
    return normal, float(offset), angle


def detect_dominant_wall_planes(cloud: PointCloud, cfg: KeypointConfig,
                                up) -> list[WallPlane]:
    'Detect large vertical planes without Open3D/RANSAC. Group normals modulo pi, find offset peaks, refine on the full cloud and validate width, height, area and support count.\n    '
    points = np.asarray(cloud.points, float)
    normals = np.asarray(cloud.normals, float)
    if len(points) == 0 or normals.shape != points.shape:
        return []

    up = np.asarray(up, float)
    up /= np.clip(np.linalg.norm(up), 1e-12, None)
    first, second = _horizontal_basis(up)
    normal_length = np.linalg.norm(normals, axis=1)
    vertical = (
        np.isfinite(normals).all(axis=1)
        & (normal_length > 1e-8)
        & (np.abs(normals @ up) <= cfg.wall_vertical_normal_cos)
    )
    source_indices = np.flatnonzero(vertical)
    if len(source_indices) < cfg.wall_plane_min_points:
        return []

    sample_limit = max(int(cfg.wall_plane_sample_points), 1)
    if len(source_indices) > sample_limit:
        positions = np.linspace(
            0, len(source_indices) - 1, sample_limit, dtype=int,
        )
        sampled_indices = source_indices[positions]
    else:
        sampled_indices = source_indices

    sampled_normals = normals[sampled_indices]
    sampled_horizontal = sampled_normals - (
        sampled_normals @ up
    )[:, None] * up
    sampled_horizontal /= np.clip(
        np.linalg.norm(sampled_horizontal, axis=1, keepdims=True), 1e-12, None,
    )
    sampled_angles = np.mod(np.arctan2(
        sampled_horizontal @ second, sampled_horizontal @ first,
    ), math.pi)

    requested_width = math.radians(max(cfg.wall_plane_angle_degrees, 1.0))
    angle_bins = max(4, int(math.ceil(math.pi / requested_width)))
    angle_width = math.pi / angle_bins
    angle_ids = np.minimum(
        (sampled_angles / angle_width).astype(int), angle_bins - 1,
    )
    distance_bin = max(float(cfg.wall_plane_distance), 1e-4)
    hypotheses = []
    minimum_sample_support = max(
        12,
        int(math.ceil(
            cfg.wall_plane_min_points
            * min(1.0, len(sampled_indices) / max(1, len(source_indices)))
            * 0.45
        )),
    )

    for angle_id in range(angle_bins):
        local = np.flatnonzero(angle_ids == angle_id)
        if len(local) < minimum_sample_support:
            continue
        angles = sampled_angles[local]
        doubled = 2.0 * angles
        angle = 0.5 * math.atan2(
            float(np.sin(doubled).mean()), float(np.cos(doubled).mean()),
        )
        angle %= math.pi
        normal = math.cos(angle) * first + math.sin(angle) * second
        offsets = points[sampled_indices[local]] @ normal
        offset_ids = np.floor(offsets / distance_bin).astype(np.int64)
        unique, counts = np.unique(offset_ids, return_counts=True)
        order = np.argsort(-counts)
        peak_limit = max(4, int(cfg.wall_plane_max_count) * 3)
        for peak in order[:peak_limit]:
            if counts[peak] < minimum_sample_support:
                break
            members = local[offset_ids == unique[peak]]
            offset = float(np.median(points[sampled_indices[members]] @ normal))
            hypotheses.append((int(counts[peak]), normal.copy(), offset, angle))

    if not hypotheses:
        return []

    horizontal = normals - (normals @ up)[:, None] * up
    horizontal /= np.clip(
        np.linalg.norm(horizontal, axis=1, keepdims=True), 1e-12, None,
    )
    candidates = []
    for _, normal, offset, angle in sorted(hypotheses, reverse=True,
                                            key=lambda item: item[0]):
        compatible = vertical & (
            np.abs(horizontal @ normal) >= cfg.wall_plane_normal_cos
        )
        compatible &= np.abs(points @ normal - offset) <= distance_bin
        indices = np.flatnonzero(compatible)
        if len(indices) < cfg.wall_plane_min_points:
            continue

        oriented = horizontal[indices].copy()
        flip = (oriented @ normal) < 0.0
        oriented[flip] *= -1.0
        refined = oriented.mean(axis=0)
        refined -= (refined @ up) * up
        refined /= np.clip(np.linalg.norm(refined), 1e-12, None)
        refined_offset = float(np.median(points[indices] @ refined))
        refined, refined_offset, refined_angle = _canonical_plane(
            refined, refined_offset, first, second,
        )

        compatible = vertical & (
            np.abs(horizontal @ refined) >= cfg.wall_plane_normal_cos
        )
        compatible &= (
            np.abs(points @ refined - refined_offset) <= distance_bin
        )
        indices = np.flatnonzero(compatible)
        if len(indices) < cfg.wall_plane_min_points:
            continue
        tangent = np.cross(up, refined)
        tangent /= np.clip(np.linalg.norm(tangent), 1e-12, None)
        tangent_min, tangent_max = _robust_extent(points[indices] @ tangent)
        height_min, height_max = _robust_extent(points[indices] @ up)
        width = tangent_max - tangent_min
        height = height_max - height_min
        area = width * height
        if (
            width < cfg.wall_plane_min_width
            or height < cfg.wall_plane_min_height
            or area < cfg.wall_plane_min_area
        ):
            continue

        support_quality = min(
            1.0, len(indices) / max(1.0, 4.0 * cfg.wall_plane_min_points),
        )
        extent_quality = min(
            1.0,
            math.sqrt(
                area / max(cfg.wall_plane_min_area, 1e-6)
            ) / 1.5,
        )
        quality = float(math.sqrt(support_quality * extent_quality))
        candidate = WallPlane(
            normal=refined,
            offset=refined_offset,
            tangent=tangent,
            tangent_min=tangent_min,
            tangent_max=tangent_max,
            height_min=height_min,
            height_max=height_max,
            support_count=int(len(indices)),
            quality=quality,
        )

        duplicate = False
        duplicate_distance = max(6.0 * distance_bin, 0.25)
        for existing in candidates:
            aligned_offset = candidate.offset
            if float(candidate.normal @ existing.normal) < 0.0:
                aligned_offset = -aligned_offset
            same_orientation = (
                abs(float(candidate.normal @ existing.normal))
                >= math.cos(2.0 * angle_width)
            )
            if (
                same_orientation
                and abs(aligned_offset - existing.offset) <= duplicate_distance
            ):
                duplicate = True
                break
        if not duplicate:
            candidates.append(candidate)
        if len(candidates) >= int(cfg.wall_plane_max_count):
            break

    robust_min, robust_max = np.percentile(points, [2.0, 98.0], axis=0)
    center = 0.5 * (robust_min + robust_max)
    oriented_planes = []
    for plane in candidates:
        if float(center @ plane.normal - plane.offset) >= 0.0:
            oriented_planes.append(plane)
            continue
        oriented_planes.append(WallPlane(
            normal=-plane.normal,
            offset=-plane.offset,
            tangent=-plane.tangent,
            tangent_min=-plane.tangent_max,
            tangent_max=-plane.tangent_min,
            height_min=plane.height_min,
            height_max=plane.height_max,
            support_count=plane.support_count,
            quality=plane.quality,
        ))
    return oriented_planes


def analyze_wall_planes(cloud: PointCloud, keypoints: KeyPoints,
                        cfg: KeypointConfig, up) -> WallPlaneAnalysis:
    'Measure wall and object evidence per keypoint. Combine plane proximity with two-scale planar support, then reduce wall evidence when geometry protrudes toward the scene interior.\n    '
    up = np.asarray(up, float)
    up /= np.clip(np.linalg.norm(up), 1e-12, None)
    planes = detect_dominant_wall_planes(cloud, cfg, up)
    surface_plane_index = np.full(cloud.size, -1, dtype=int)
    surface_score = np.zeros(cloud.size, dtype=float)
    keypoint_plane_index = np.full(keypoints.size, -1, dtype=int)
    plane_proximity = np.zeros(keypoints.size, dtype=float)
    plane_affinity = np.zeros(keypoints.size, dtype=float)
    small_support = np.zeros(keypoints.size, dtype=float)
    large_support = np.zeros(keypoints.size, dtype=float)
    protrusion_score = np.zeros(keypoints.size, dtype=float)
    discontinuity_score = np.zeros(keypoints.size, dtype=float)
    object_anchor_score = np.zeros(keypoints.size, dtype=float)
    object_score = np.zeros(keypoints.size, dtype=float)
    wall_score = np.zeros(keypoints.size, dtype=float)
    if not planes:
        return WallPlaneAnalysis(
            planes, surface_plane_index, keypoint_plane_index,
            plane_proximity, plane_affinity, small_support, large_support,
            protrusion_score, discontinuity_score, object_anchor_score,
            object_score, wall_score,
        )

    points = np.asarray(cloud.points, float)
    normals = np.asarray(cloud.normals, float)
    kp_points = np.asarray(keypoints.positions, float)
    kp_normals = np.asarray(keypoints.normals, float)
    tolerance = max(float(cfg.wall_plane_distance), 1e-6)
    margin = tolerance * 2.0

    for plane_index, plane in enumerate(planes):
        surface_distance = np.abs(plane.signed_distance(points))
        surface_alignment = np.abs(normals @ plane.normal)
        inside = plane.contains(points, up, margin=margin)
        score = (
            np.exp(-np.square(surface_distance / tolerance))
            * np.clip(
                (surface_alignment - cfg.wall_plane_normal_cos)
                / max(1.0 - cfg.wall_plane_normal_cos, 1e-6),
                0.0, 1.0,
            )
            * plane.quality
            * inside
            * (surface_distance <= 2.0 * tolerance)
        )
        better = score > surface_score
        surface_score[better] = score[better]
        surface_plane_index[better] = plane_index

        if keypoints.size == 0:
            continue
        kp_distance = np.abs(plane.signed_distance(kp_points))
        kp_alignment = np.abs(kp_normals @ plane.normal)
        kp_inside = plane.contains(kp_points, up, margin=margin)
        proximity = (
            np.exp(-np.square(kp_distance / tolerance))
            * plane.quality
            * kp_inside
            * (kp_distance <= 2.0 * tolerance)
        )
        affinity = (
            proximity
            * np.clip(
                (kp_alignment - cfg.wall_vertical_normal_cos)
                / max(1.0 - cfg.wall_vertical_normal_cos, 1e-6),
                0.0, 1.0,
            )
        )
        better = proximity > plane_proximity
        plane_proximity[better] = proximity[better]
        plane_affinity[better] = affinity[better]
        keypoint_plane_index[better] = plane_index

    active = np.flatnonzero(keypoint_plane_index >= 0)
    if len(active):
        small_neighbors = cloud.tree.query_ball_point(
            kp_points[active], max(float(cfg.wall_small_radius), 1e-6),
            workers=kdtree_workers(),
        )
        large_radius = max(
            float(cfg.wall_large_radius), float(cfg.wall_small_radius), 1e-6,
        )
        large_neighbors = cloud.tree.query_ball_point(
            kp_points[active], large_radius, workers=kdtree_workers(),
        )
        protrusion_radius = max(
            float(cfg.wall_protrusion_radius), large_radius, 1e-6,
        )
        protrusion_neighbors = cloud.tree.query_ball_point(
            kp_points[active], protrusion_radius,
            workers=kdtree_workers(),
        )
        normal_cos = float(cfg.wall_plane_normal_cos)
        separation = max(float(cfg.wall_protrusion_distance), tolerance, 1e-6)

        def support_ratio(indices, plane):
            if len(indices) < 6:
                return 0.0
            indices = np.asarray(indices, int)
            close = np.abs(plane.signed_distance(points[indices])) <= tolerance
            aligned = np.abs(normals[indices] @ plane.normal) >= normal_cos
            return float(np.count_nonzero(close & aligned) / len(indices))

        for local, keypoint_index in enumerate(active):
            plane = planes[keypoint_plane_index[keypoint_index]]
            keypoint_signed_distance = float(
                plane.signed_distance(kp_points[keypoint_index:keypoint_index + 1])[0]
            )
            object_anchor_score[keypoint_index] = float(np.clip(
                keypoint_signed_distance / separation, 0.0, 1.0,
            ))
            small_support[keypoint_index] = support_ratio(
                small_neighbors[local], plane,
            )
            large_support[keypoint_index] = support_ratio(
                large_neighbors[local], plane,
            )

            indices = np.asarray(protrusion_neighbors[local], int)
            if len(indices) >= 8:
                signed = plane.signed_distance(points[indices])
                front_extent = max(0.0, float(np.percentile(signed, 90.0)))
                protrusion_score[keypoint_index] = float(np.clip(
                    (front_extent - separation) / separation, 0.0, 1.0,
                ))
                near_fraction = float(
                    np.count_nonzero(np.abs(signed) <= tolerance) / len(signed)
                )
                front_fraction = float(
                    np.count_nonzero(signed >= separation) / len(signed)
                )
                discontinuity_score[keypoint_index] = float(
                    protrusion_score[keypoint_index]
                    * min(1.0, near_fraction / 0.25)
                    * min(1.0, front_fraction / 0.10)
                )

    # Une protrusion voisine ne prouve pas que le keypoint appartient à l'objet.
    # La protection augmente uniquement lorsque le keypoint lui-même se situe
    # devant le plan. Cela évite qu'une chaise proche protège un trou du mur.
    object_score = (
        np.maximum(protrusion_score, discontinuity_score)
        * object_anchor_score
    )
    multi_scale_support = np.sqrt(small_support * large_support)
    normal_factor = 0.5 + 0.5 * np.divide(
        plane_affinity, np.maximum(plane_proximity, 1e-12),
        out=np.zeros_like(plane_affinity), where=plane_proximity > 0.0,
    )
    raw_wall_score = plane_proximity * multi_scale_support * normal_factor
    wall_score = raw_wall_score * (
        1.0 - float(cfg.wall_object_protection) * object_score
    )

    return WallPlaneAnalysis(
        planes=planes,
        surface_plane_index=surface_plane_index,
        keypoint_plane_index=keypoint_plane_index,
        plane_proximity=np.clip(plane_proximity, 0.0, 1.0),
        plane_affinity=np.clip(plane_affinity, 0.0, 1.0),
        small_support=np.clip(small_support, 0.0, 1.0),
        large_support=np.clip(large_support, 0.0, 1.0),
        protrusion_score=np.clip(protrusion_score, 0.0, 1.0),
        discontinuity_score=np.clip(discontinuity_score, 0.0, 1.0),
        object_anchor_score=np.clip(object_anchor_score, 0.0, 1.0),
        object_score=np.clip(object_score, 0.0, 1.0),
        wall_score=np.clip(wall_score, 0.0, 1.0),
    )


def apply_wall_keypoint_filter(cloud: PointCloud, keypoints: KeyPoints,
                               cfg: KeypointConfig, up,
                               force_analysis: bool = False) -> WallFilterResult:
    'Penalize and reject only unambiguous wall keypoints. Preserve original Harris responses; use separate selection_scores for soft penalties and budget ordering.\n    '
    count = keypoints.size
    if count == 0:
        empty = np.zeros(0, dtype=bool)
        return WallFilterResult(keypoints, None, empty, empty)

    if not cfg.wall_filter_enabled and not force_analysis:
        passthrough = KeyPoints(
            positions=keypoints.positions.copy(),
            normals=keypoints.normals.copy(),
            responses=keypoints.responses.copy(),
            source_index=keypoints.source_index.copy(),
            selection_scores=keypoints.responses.copy(),
            wall_scores=np.zeros(count, float),
            object_scores=np.zeros(count, float),
            wall_affinities=np.zeros(count, float),
            wall_proximities=np.zeros(count, float),
            wall_plane_indices=np.full(count, -1, int),
        )
        return WallFilterResult(
            passthrough, None, np.ones(count, bool), np.zeros(count, bool),
        )

    analysis = analyze_wall_planes(cloud, keypoints, cfg, up)
    selection_scores = np.asarray(keypoints.responses, float).copy()
    if cfg.wall_filter_enabled:
        selection_scores -= (
            float(cfg.wall_penalty_weight) * analysis.wall_score
        )
    rejected = np.zeros(count, dtype=bool)
    if cfg.wall_filter_enabled and cfg.wall_hard_reject:
        rejected = (
            (analysis.wall_score >= float(cfg.wall_reject_threshold))
            & (
                analysis.object_score
                <= float(cfg.wall_reject_max_object_score)
            )
        )
    kept = ~rejected
    filtered = KeyPoints(
        positions=keypoints.positions[kept],
        normals=keypoints.normals[kept],
        responses=keypoints.responses[kept],
        source_index=keypoints.source_index[kept],
        selection_scores=selection_scores[kept],
        wall_scores=analysis.wall_score[kept],
        object_scores=analysis.object_score[kept],
        wall_affinities=analysis.plane_affinity[kept],
        wall_proximities=analysis.plane_proximity[kept],
        wall_plane_indices=analysis.keypoint_plane_index[kept],
    )
    return WallFilterResult(filtered, analysis, kept, rejected)
