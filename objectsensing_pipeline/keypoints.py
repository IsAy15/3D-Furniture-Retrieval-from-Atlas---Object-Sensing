'\nKeypoint detection inspired by paper section 4: curvature prefilter, density-normalized Harris response, tangent-plane 2D corner recovery, NMS, iterative geometric adjustment and deduplication. Scan-specific quality filters are applied separately.\n'

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from config import KeypointConfig
from geometry import PointCloud
from parallelism import kdtree_workers


@dataclass
class KeyPoints:
    positions: np.ndarray      # (K, 3)
    normals: np.ndarray        # (K, 3)
    responses: np.ndarray      # (K,) response de Harris (ou 0 pour coins 2D)
    source_index: np.ndarray   # (K,) index du point d'origine dans le nuage
    selection_scores: np.ndarray | None = None  # priorité scan après contexte mural
    wall_scores: np.ndarray | None = None       # score mural scan-only [0, 1]
    object_scores: np.ndarray | None = None     # preuve d'objet scan-only [0, 1]
    wall_affinities: np.ndarray | None = None   # proximité/orientation du plan [0, 1]
    wall_proximities: np.ndarray | None = None  # proximité au plan sans normale [0, 1]
    wall_plane_indices: np.ndarray | None = None  # plan dominant associé, -1 sinon

    def subset(self, indices) -> "KeyPoints":
        'Return a subset while preserving scan-only diagnostics.'
        indices = np.asarray(indices)

        def take(values):
            return None if values is None else np.asarray(values)[indices]

        return KeyPoints(
            positions=take(self.positions), normals=take(self.normals),
            responses=take(self.responses), source_index=take(self.source_index),
            selection_scores=take(self.selection_scores),
            wall_scores=take(self.wall_scores),
            object_scores=take(self.object_scores),
            wall_affinities=take(self.wall_affinities),
            wall_proximities=take(self.wall_proximities),
            wall_plane_indices=take(self.wall_plane_indices),
        )

    @property
    def size(self) -> int:
        return len(self.positions)


def _covariance_normals(normals_neigh: np.ndarray) -> np.ndarray:
    return normals_neigh.T @ normals_neigh        # somme n_j n_j^T


def _harris_response(C: np.ndarray, k: float,
                     reference_neighbors: float = 6.0) -> float:
    'Harris response normalized to a fixed reference neighbor count. Without normalization, determinant and squared-trace terms scale differently with point density, causing noisy dense surfaces to appear as strong corners.\n    '
    trace = float(np.trace(C))
    if trace <= 1e-12:
        return float("-inf")
    C_scaled = C * (float(reference_neighbors) / trace)
    return float(np.linalg.det(C_scaled) - k * np.trace(C_scaled) ** 2)


def _convex_hull_area_2d(points: np.ndarray) -> float:
    'Compute 2D convex hull area entirely in memory.'
    points = np.unique(np.asarray(points, np.float64), axis=0)
    if len(points) < 3:
        return 0.0
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin, a, b):
        oa = a - origin
        ob = b - origin
        return oa[0] * ob[1] - oa[1] * ob[0]

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    return float(0.5 * abs(
        np.dot(hull[:, 0], np.roll(hull[:, 1], -1)) -
        np.dot(hull[:, 1], np.roll(hull[:, 0], -1))
    ))


def _passes_2d_corner(p, n, neigh_pts, radius, hull_ratio):
    '2D corner: the point lies on a local plane and neighbors do not fill the disk. The hull area threshold is hull_ratio * radius^2.'
    rel = neigh_pts - p
    dist_to_plane = rel @ n
    on_plane = np.abs(dist_to_plane) < 0.02 * (radius / 0.05)
    if on_plane.sum() < 6:
        return False
    # planéité locale : la majorité des voisins doivent être proches du plan
    if on_plane.mean() < 0.6:
        return False
    # base tangente
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    proj = rel[on_plane]
    uv = np.column_stack([proj @ t1, proj @ t2])
    if len(uv) < 4:
        return True
    area = _convex_hull_area_2d(uv)
    return area < hull_ratio * radius * radius


def _has_large_planar_support(p, n, neigh_pts, neigh_normals,
                              radius, min_area, normal_cos):
    'Approximate the paper requirement for large planar support. A wider neighborhood must contain a normal-consistent surface covering a minimum area.\n    '
    rel = neigh_pts - p
    plane_tolerance = 0.02 * (radius / 0.15)
    on_plane = (
        (np.abs(rel @ n) < plane_tolerance) &
        (np.abs(neigh_normals @ n) >= normal_cos)
    )
    if on_plane.sum() < 8:
        return False

    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a)
    t1 /= np.clip(np.linalg.norm(t1), 1e-12, None)
    t2 = np.cross(n, t1)
    proj = rel[on_plane]
    uv = np.column_stack([proj @ t1, proj @ t2])
    area = _convex_hull_area_2d(uv)
    return area >= min_area


def _prefilter_2d_candidates(candidate_indices, positions, responses,
                             cell_size, per_cell=2):
    'Reduce convex hull tests by keeping strong weak-Harris candidates per cell before 2D tests. The final exact NMS is unchanged.\n    '
    cell_size = max(float(cell_size), 1e-6)
    cells = {}
    for index in candidate_indices:
        key = tuple(np.floor(positions[index] / cell_size).astype(np.int64))
        cells.setdefault(key, []).append(int(index))
    selected = []
    for indices in cells.values():
        indices.sort(key=lambda i: -responses[i])
        selected.extend(indices[:per_cell])
    return np.asarray(selected, dtype=int)


def _empty_keypoints() -> KeyPoints:
    return KeyPoints(
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, int),
    )


def _finalize_trace(trace, **counts):
    if trace is not None:
        trace["counts"] = {key: int(value) for key, value in counts.items()}


def _pre_harris_visibility_mask(points, cfg, visibility_sampler):
    """Return certain RGB-D hole boundaries before expensive Harris work."""
    count = len(points)
    rejected = np.zeros(count, bool)
    if (
        not bool(getattr(cfg, "pre_harris_visibility_filter_enabled", False))
        or visibility_sampler is None or not count
    ):
        return rejected
    radius = max(
        float(getattr(cfg, "pre_harris_visibility_probe_radius", 0.03)), 0.0,
    )
    offsets = np.asarray([
        [0, 0, 0], [radius, 0, 0], [-radius, 0, 0],
        [0, radius, 0], [0, -radius, 0], [0, 0, radius], [0, 0, -radius],
    ], float)
    probes = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    _, boundary = visibility_sampler(probes)
    boundary = np.asarray(boundary, bool).reshape(count, len(offsets))
    fraction = boundary.mean(axis=1)
    # Unlike the later quality filter, this early gate only rejects strong
    # evidence. Ambiguous candidates still reach Harris and the 2D branch.
    return boundary[:, 0] | (
        fraction >= float(getattr(
            cfg, "pre_harris_visibility_max_fraction", 0.58,
        ))
    )


def detect_keypoints(cloud: PointCloud, cfg: KeypointConfig,
                     trace=None, visibility_sampler=None) -> KeyPoints:
    'Detect keypoints and optionally record decisions. Debug trace indices refer to the source cloud to support unambiguous configuration comparisons.\n    '
    pts = cloud.points
    nrm = cloud.normals
    curv = cloud.curvature
    if curv is None:
        raise ValueError('The cloud must have curvature (estimate_normals_curvature).')
    r = cfg.neighbor_radius
    tree = cloud.tree

    if trace is not None:
        trace.clear()
        trace["schema_version"] = 1
        trace["input_count"] = int(len(pts))
        trace["parameters"] = {
            "neighbor_radius": float(r),
            "curvature_threshold": float(cfg.curvature_threshold),
            "harris_k": float(cfg.harris_k),
            "harris_threshold": float(cfg.harris_threshold),
            "nms_radius": float(cfg.nms_radius),
            "adjust_iterations": int(cfg.adjust_iterations),
            "jitter_reject": float(
                cfg.jitter_reject if cfg.jitter_reject is not None else r
            ),
            "dedup_radius": float(cfg.dedup_radius),
            "pre_harris_visibility_filter_enabled": bool(getattr(
                cfg, "pre_harris_visibility_filter_enabled", False,
            )),
            "pre_harris_visibility_probe_radius": float(getattr(
                cfg, "pre_harris_visibility_probe_radius", 0.03,
            )),
            "pre_harris_visibility_max_fraction": float(getattr(
                cfg, "pre_harris_visibility_max_fraction", 0.58,
            )),
        }

    candidates = np.where(curv > cfg.curvature_threshold)[0]
    if trace is not None:
        trace["curvature_candidates"] = candidates.copy()
    visibility_rejected = _pre_harris_visibility_mask(
        pts[candidates], cfg, visibility_sampler,
    )
    if trace is not None:
        trace["pre_harris_visibility_rejected"] = candidates[
            visibility_rejected
        ]
        trace["pre_harris_visibility_kept"] = candidates[
            ~visibility_rejected
        ]
    candidates = candidates[~visibility_rejected]
    if len(candidates) == 0:
        _finalize_trace(
            trace, input=len(pts), curvature=0, harris_3d=0, corner_2d=0,
            nms=0, adjusted=0, final=0,
        )
        return _empty_keypoints()

    neigh = tree.query_ball_point(pts[candidates], r, workers=kdtree_workers())
    responses = np.full(len(candidates), -np.inf)
    keep = np.zeros(len(candidates), bool)
    insufficient_neighbors = []
    for a, ci in enumerate(candidates):
        idx = neigh[a]
        if len(idx) < 6:
            insufficient_neighbors.append(int(ci))
            continue
        C = _covariance_normals(nrm[idx])
        R = _harris_response(
            C, cfg.harris_k, cfg.harris_reference_neighbors,
        )
        responses[a] = R
        if R > cfg.harris_threshold:
            keep[a] = True

    harris_keep = keep.copy()
    if trace is not None:
        trace["harris_responses"] = responses.copy()
        trace["harris_accepted"] = candidates[np.flatnonzero(harris_keep)]
        trace["harris_rejected_neighbors"] = np.asarray(
            insufficient_neighbors, dtype=int,
        )

    low_response_all = np.where(
        np.isfinite(responses) & (responses <= cfg.harris_threshold)
    )[0]
    low_response = _prefilter_2d_candidates(
        low_response_all, pts[candidates], responses,
        cell_size=max(cfg.nms_radius, r), per_cell=2,
    )
    rejected_2d_hull = []
    rejected_2d_support = []
    accepted_2d = []
    support_radius = r * cfg.corner_plane_radius_factor
    for a in low_response:
        ci = candidates[a]
        idx = neigh[a]
        if not _passes_2d_corner(
            pts[ci], nrm[ci], pts[idx], r, cfg.convex_hull_ratio,
        ):
            rejected_2d_hull.append(int(ci))
            continue
        support_idx = tree.query_ball_point(pts[ci], support_radius)
        has_support = _has_large_planar_support(
            pts[ci], nrm[ci], pts[support_idx], nrm[support_idx],
            support_radius, cfg.corner_plane_min_area,
            cfg.corner_plane_normal_cos,
        )
        keep[a] = has_support
        if has_support:
            accepted_2d.append(int(ci))
        else:
            rejected_2d_support.append(int(ci))

    if trace is not None:
        tested_set = set(int(index) for index in low_response)
        trace["corner_2d_prefilter_rejected"] = candidates[np.asarray([
            index for index in low_response_all if int(index) not in tested_set
        ], dtype=int)]
        trace["corner_2d_tested"] = candidates[low_response]
        trace["corner_2d_accepted"] = np.asarray(accepted_2d, dtype=int)
        trace["corner_2d_rejected_hull"] = np.asarray(
            rejected_2d_hull, dtype=int,
        )
        trace["corner_2d_rejected_support"] = np.asarray(
            rejected_2d_support, dtype=int,
        )

    sel = np.where(keep)[0]
    if trace is not None:
        trace["nms_input"] = candidates[sel]
    if len(sel) == 0:
        _finalize_trace(
            trace, input=len(pts), curvature=len(candidates),
            harris_3d=int(harris_keep.sum()), corner_2d=len(accepted_2d),
            nms=0, adjusted=0, final=0,
        )
        return _empty_keypoints()

    # --- non-maximum suppression sur R (par |R| pour inclure the coins 2D) ---
    sel_pts = pts[candidates[sel]]
    score = np.where(np.isfinite(responses[sel]), responses[sel], 0.0)
    order = np.argsort(-score)
    nms_tree = cKDTree(sel_pts)
    suppressed = np.zeros(len(sel), bool)
    kept_local = []
    for o in order:
        if suppressed[o]:
            continue
        kept_local.append(o)
        for j in nms_tree.query_ball_point(sel_pts[o], cfg.nms_radius):
            if j != o and score[j] <= score[o]:
                suppressed[j] = True
    kept_local = np.array(kept_local, int)
    if trace is not None:
        trace["nms_kept"] = candidates[sel[kept_local]]
        trace["nms_rejected"] = candidates[sel[np.flatnonzero(suppressed)]]

    # --- ajustement itératif vers la position stable du coin ---
    final_pos, final_nrm, final_resp, final_src = [], [], [], []
    adjustment_origin, adjustment_target, adjustment_distance = [], [], []
    adjustment_rejected = {
        "neighbors": [], "singular": [], "non_finite": [], "jitter": [],
    }
    jitter = cfg.jitter_reject if cfg.jitter_reject is not None else r
    for o in kept_local:
        ci = candidates[sel[o]]
        p = pts[ci].copy()
        p0 = p.copy()
        ok = True
        failure = None
        for _ in range(cfg.adjust_iterations):
            idx = tree.query_ball_point(p, r)
            if len(idx) < 6:
                ok = False
                failure = "neighbors"
                break
            C = _covariance_normals(nrm[idx])
            # somme_j (n_j n_j^T) p_j = somme_j n_j (n_j . p_j)
            dots = np.einsum('jk,jk->j', nrm[idx], pts[idx])
            NNt_p = (nrm[idx] * dots[:, None]).sum(axis=0)
            try:
                p_new = np.linalg.solve(C + 1e-6 * np.eye(3), NNt_p)
            except np.linalg.LinAlgError:
                ok = False
                failure = "singular"
                break
            if not np.isfinite(p_new).all():
                ok = False
                failure = "non_finite"
                break
            p = p_new
        distance = float(np.linalg.norm(p - p0)) if np.isfinite(p).all() else float("inf")
        if not ok or distance > jitter:
            adjustment_rejected[failure or "jitter"].append(int(ci))
            continue
        final_pos.append(p)
        final_nrm.append(nrm[ci])
        final_resp.append(score[o])
        final_src.append(int(ci))
        adjustment_origin.append(p0)
        adjustment_target.append(p.copy())
        adjustment_distance.append(distance)

    if trace is not None:
        trace["adjustment_source_index"] = np.asarray(final_src, dtype=int)
        trace["adjustment_origin"] = np.asarray(adjustment_origin, dtype=float).reshape(-1, 3)
        trace["adjustment_target"] = np.asarray(adjustment_target, dtype=float).reshape(-1, 3)
        trace["adjustment_distance"] = np.asarray(adjustment_distance, dtype=float)
        trace["adjustment_rejected"] = {
            reason: np.asarray(indices, dtype=int)
            for reason, indices in adjustment_rejected.items()
        }

    if not final_pos:
        _finalize_trace(
            trace, input=len(pts), curvature=len(candidates),
            harris_3d=int(harris_keep.sum()), corner_2d=len(accepted_2d),
            nms=len(kept_local), adjusted=0, final=0,
        )
        return _empty_keypoints()

    P = np.array(final_pos)
    # --- déduplication (points ayant convergé au même endroit) ---
    keep_idx = _dedup(P, cfg.dedup_radius)
    result = KeyPoints(
        P[keep_idx],
        np.array(final_nrm)[keep_idx],
        np.array(final_resp)[keep_idx],
        np.array(final_src)[keep_idx],
    )
    if trace is not None:
        dedup_mask = np.ones(len(P), dtype=bool)
        dedup_mask[keep_idx] = False
        trace["dedup_kept"] = np.asarray(final_src, dtype=int)[keep_idx]
        trace["dedup_rejected"] = np.asarray(final_src, dtype=int)[dedup_mask]
        trace["final_positions"] = result.positions.copy()
        trace["final_responses"] = result.responses.copy()
        trace["final_source_index"] = result.source_index.copy()
        _finalize_trace(
            trace, input=len(pts), curvature=len(candidates),
            harris_3d=int(harris_keep.sum()), corner_2d=len(accepted_2d),
            nms=len(kept_local), adjusted=len(P), final=result.size,
        )
    return result


def _dedup(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, int)
    tree = cKDTree(points)
    taken = np.zeros(len(points), bool)
    keep = []
    for i in range(len(points)):
        if taken[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(points[i], radius):
            taken[j] = True
    return np.array(keep, int)
