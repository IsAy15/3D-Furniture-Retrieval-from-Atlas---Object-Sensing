"""Soft spatial proposals built from retained scene keypoints.

This module is intentionally independent from retrieval.  It is a POC used to
check whether local sets of keypoints carry more useful object evidence than
isolated keypoints before changing the database descriptors or ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class KeypointGroup:
    member_indices: np.ndarray
    anchor_index: int
    centroid: np.ndarray
    extent: np.ndarray
    score: float
    descriptor: np.ndarray


def build_surface_component_groups(
        surface_points, keypoint_points, *, wall_mask=None, scores=None,
        floor_height=0.06, voxel_size=0.055, connection_radius=0.095,
        assignment_radius=0.14, min_surface_voxels=8, min_members=3,
        max_object_diameter=1.65, max_groups=40):
    """Group keypoints through connected non-structural scene surfaces.

    The surface, rather than sparse keypoint-to-keypoint distance, provides the
    bridge between opposite sides of one object.  Dominant wall points and the
    floor are removed before connected components are computed.
    """
    surface_points = np.asarray(surface_points, float).reshape(-1, 3)
    keypoint_points = np.asarray(keypoint_points, float).reshape(-1, 3)
    if not len(surface_points) or not len(keypoint_points):
        return []
    wall_mask = (
        np.zeros(len(surface_points), bool) if wall_mask is None
        else np.asarray(wall_mask, bool).reshape(-1)
    )
    if len(wall_mask) != len(surface_points):
        raise ValueError('wall_mask must contain one value per surface point')
    scores = (
        np.ones(len(keypoint_points), float) if scores is None
        else np.nan_to_num(np.asarray(scores, float).reshape(-1), nan=0.0)
    )
    if len(scores) != len(keypoint_points):
        raise ValueError('scores must contain one value per keypoint')

    keep = (~wall_mask) & (surface_points[:, 1] >= floor_height)
    object_surface = surface_points[keep]
    if not len(object_surface):
        return []
    voxel_keys = np.floor(object_surface / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
    voxels = object_surface[np.sort(unique_indices)]
    tree = cKDTree(voxels)
    parent = np.arange(len(voxels), dtype=int)
    size = np.ones(len(voxels), dtype=int)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]

    for left, right in tree.query_pairs(connection_radius):
        union(int(left), int(right))
    roots = np.asarray([find(index) for index in range(len(voxels))], int)
    root_sizes = np.bincount(roots, minlength=len(voxels))
    valid_roots = set(np.flatnonzero(root_sizes >= min_surface_voxels).tolist())

    distances, nearest = tree.query(keypoint_points, k=1)
    assignments = np.asarray([
        roots[index] if distance <= assignment_radius and roots[index] in valid_roots
        else -1
        for distance, index in zip(distances, nearest)
    ], int)
    groups = []
    for root in valid_roots:
        members = np.flatnonzero(assignments == root)
        if len(members) < min_members:
            continue
        member_points = keypoint_points[members]
        if (
            len(members) > min_members
            and np.linalg.norm(np.ptp(member_points, axis=0)) > max_object_diameter
        ):
            labels = fclusterdata(
                member_points, t=max_object_diameter,
                criterion="distance", method="complete",
            )
        else:
            labels = np.ones(len(members), int)
        for label in np.unique(labels):
            subgroup = members[labels == label]
            if len(subgroup) < min_members:
                continue
            groups.append(_make_group(subgroup, keypoint_points, scores))
    groups.sort(key=lambda group: (-len(group.member_indices), -group.score))
    return groups[:max_groups]


def _group_descriptor(points, normals, scores):
    centered = points - points.mean(axis=0)
    extent = np.ptp(points, axis=0)
    distances = np.linalg.norm(
        centered[:, None, :] - centered[None, :, :], axis=2,
    )
    upper = distances[np.triu_indices(len(points), 1)]
    scale = max(float(np.linalg.norm(extent)), 1e-9)
    distance_hist, _ = np.histogram(
        upper / scale, bins=np.linspace(0.0, 1.0, 7),
    )
    distance_hist = distance_hist / max(1, distance_hist.sum())
    if normals is None:
        normal_hist = np.zeros(3, float)
    else:
        vertical = np.abs(normals[:, 1])
        normal_hist = np.asarray([
            np.mean(vertical < 0.35),
            np.mean((vertical >= 0.35) & (vertical < 0.8)),
            np.mean(vertical >= 0.8),
        ])
    return np.concatenate([
        np.sort(extent) / scale,
        distance_hist,
        normal_hist,
        [len(points) / 32.0, float(np.mean(scores))],
    ]).astype(np.float32)


def _make_group(member_indices, points, scores):
    member_indices = np.asarray(member_indices, int)
    group_points = points[member_indices]
    centroid = np.average(
        group_points, axis=0,
        weights=np.clip(scores[member_indices], 1e-6, None),
    )
    return KeypointGroup(
        member_indices=member_indices,
        anchor_index=int(member_indices[np.argmax(scores[member_indices])]),
        centroid=centroid,
        extent=np.ptp(group_points, axis=0),
        score=float(np.mean(scores[member_indices]) * np.log1p(len(member_indices))),
        descriptor=_group_descriptor(
            group_points, None, scores[member_indices],
        ),
    )


def refine_surface_groups(
        groups, keypoint_points, *, scores=None, wall_mask=None,
        min_members=3, large_mixed_min=8, mixed_wall_ratio=0.55,
        merge_gap=0.22, part_merge_gap=None,
        horizontal_overlap_tolerance=0.05,
        max_object_diameter=1.65):
    """Remove structural tails, then merge nearby parts of one object."""
    points = np.asarray(keypoint_points, float).reshape(-1, 3)
    scores = (
        np.ones(len(points), float) if scores is None
        else np.asarray(scores, float).reshape(-1)
    )
    wall_mask = (
        np.zeros(len(points), bool) if wall_mask is None
        else np.asarray(wall_mask, bool).reshape(-1)
    )
    refined = []
    for group in groups:
        members = np.asarray(group.member_indices, int)
        wall_ratio = float(np.mean(wall_mask[members]))
        if len(members) >= large_mixed_min and wall_ratio >= mixed_wall_ratio:
            object_members = members[~wall_mask[members]]
            if len(object_members) >= min_members:
                members = object_members
        refined.append(_make_group(members, points, scores))

    def bounds(group):
        values = points[group.member_indices]
        return values.min(axis=0), values.max(axis=0)

    changed = True
    while changed:
        changed = False
        best = None
        for left in range(len(refined)):
            left_min, left_max = bounds(refined[left])
            for right in range(left + 1, len(refined)):
                right_min, right_max = bounds(refined[right])
                separation = np.maximum(
                    0.0,
                    np.maximum(left_min, right_min)
                    - np.minimum(left_max, right_max),
                )
                gap = float(np.linalg.norm(separation))
                horizontal_overlap = (
                    separation[0] <= horizontal_overlap_tolerance
                    or separation[2] <= horizontal_overlap_tolerance
                )
                members = np.unique(np.concatenate([
                    refined[left].member_indices,
                    refined[right].member_indices,
                ]))
                # Desks are elongated; their box diagonal is not an object size.
                diameter = float(np.max(np.ptp(points[members], axis=0)))
                wall_ratio = float(np.mean(wall_mask[members]))
                left_wall_ratio = float(np.mean(
                    wall_mask[refined[left].member_indices],
                ))
                right_wall_ratio = float(np.mean(
                    wall_mask[refined[right].member_indices],
                ))
                allowed_gap = (
                    (merge_gap if part_merge_gap is None else part_merge_gap)
                    if max(left_wall_ratio, right_wall_ratio) <= 0.5
                    else merge_gap
                )
                if (
                    gap <= allowed_gap and horizontal_overlap
                    and diameter <= max_object_diameter
                    and wall_ratio < mixed_wall_ratio
                    and (best is None or gap < best[0])
                ):
                    best = (gap, left, right, members)
        if best is not None:
            _, left, right, members = best
            refined = [
                group for index, group in enumerate(refined)
                if index not in (left, right)
            ]
            refined.append(_make_group(members, points, scores))
            changed = True
    refined.sort(key=lambda group: (-len(group.member_indices), -group.score))
    return refined


def recover_object_evidence_groups(
        groups, keypoint_points, *, scores=None, wall_mask=None,
        object_scores=None, orphan_radius=0.18, rescue_radius=0.50,
        rescue_link_radius=0.42, min_object_score=0.12,
        max_object_diameter=1.65):
    """Recover thin supports and wall-tagged points backed by object evidence.

    Wall-affinity points never seed a group. They may only bridge existing clean
    groups when their object evidence is strong and they remain close to clean
    members. This avoids reconnecting furniture through the dominant wall.
    """
    points = np.asarray(keypoint_points, float).reshape(-1, 3)
    scores = (
        np.ones(len(points), float) if scores is None
        else np.asarray(scores, float).reshape(-1)
    )
    wall_mask = (
        np.zeros(len(points), bool) if wall_mask is None
        else np.asarray(wall_mask, bool).reshape(-1)
    )
    object_scores = (
        np.zeros(len(points), float) if object_scores is None
        else np.asarray(object_scores, float).reshape(-1)
    )
    recovered = list(groups)
    if not recovered:
        return recovered

    used = set(np.concatenate([
        group.member_indices for group in recovered
    ]).tolist())
    # Feet and thin supports can lose their surface component after floor
    # removal. Only clean keypoints at very short range are attached here.
    for orphan in np.flatnonzero(~wall_mask):
        if int(orphan) in used:
            continue
        best = None
        for group_index, group in enumerate(recovered):
            clean_members = group.member_indices[~wall_mask[group.member_indices]]
            if not len(clean_members):
                continue
            distance = float(np.min(np.linalg.norm(
                points[clean_members] - points[orphan], axis=1,
            )))
            combined = np.append(group.member_indices, orphan)
            if (
                distance <= orphan_radius
                and np.max(np.ptp(points[combined], axis=0)) <= max_object_diameter
                and (best is None or distance < best[0])
            ):
                best = (distance, group_index)
        if best is not None:
            _, group_index = best
            members = np.append(recovered[group_index].member_indices, orphan)
            recovered[group_index] = _make_group(members, points, scores)
            used.add(int(orphan))

    clean_members = np.unique(np.concatenate([
        group.member_indices[~wall_mask[group.member_indices]]
        for group in recovered
        if np.any(~wall_mask[group.member_indices])
    ]))
    if not len(clean_members):
        return recovered
    clean_tree = cKDTree(points[clean_members])
    wall_candidates = np.flatnonzero(
        wall_mask & (object_scores >= min_object_score)
    )
    if len(wall_candidates):
        distance, _ = clean_tree.query(points[wall_candidates], k=1)
        rescued = wall_candidates[distance <= rescue_radius]
    else:
        rescued = np.zeros(0, int)
    if not len(rescued):
        recovered.sort(key=lambda group: (-len(group.member_indices), -group.score))
        return recovered

    clean_groups = [
        index for index, group in enumerate(recovered)
        if np.any(~wall_mask[group.member_indices])
    ]
    node_indices = np.unique(np.concatenate([
        rescued,
        *[recovered[index].member_indices[~wall_mask[recovered[index].member_indices]]
          for index in clean_groups],
    ]))
    node_lookup = {int(value): index for index, value in enumerate(node_indices)}
    rescued_set = set(rescued.tolist())
    parent = np.arange(len(node_indices), dtype=int)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    tree = cKDTree(points[node_indices])
    for left, right in tree.query_pairs(rescue_link_radius):
        left_source = int(node_indices[left])
        right_source = int(node_indices[right])
        if left_source in rescued_set or right_source in rescued_set:
            union(int(left), int(right))

    component_groups = {}
    for group_index in clean_groups:
        roots = {
            find(node_lookup[int(member)])
            for member in recovered[group_index].member_indices
            if int(member) in node_lookup and not wall_mask[member]
        }
        for root in roots:
            component_groups.setdefault(root, set()).add(group_index)
    merged_group_indices = set()
    additions = []
    for root, group_indices in component_groups.items():
        component_rescued = [
            source for source in rescued
            if find(node_lookup[int(source)]) == root
        ]
        if len(group_indices) < 2 and not component_rescued:
            continue
        members = np.unique(np.concatenate([
            *[recovered[index].member_indices for index in group_indices],
            np.asarray(component_rescued, int),
        ]))
        if np.max(np.ptp(points[members], axis=0)) <= max_object_diameter:
            additions.append(_make_group(members, points, scores))
            merged_group_indices.update(group_indices)
    recovered = [
        group for index, group in enumerate(recovered)
        if index not in merged_group_indices
    ] + additions
    recovered.sort(key=lambda group: (-len(group.member_indices), -group.score))
    return recovered


def reconcile_keypoint_components(
        groups, keypoint_points, *, scores=None, wall_mask=None,
        link_radius=0.30, min_members=3, max_members=32,
        min_seed_overlap=2, min_seed_fraction=0.5,
        max_wall_ratio=0.80, max_object_diameter=1.65):
    """Expand a surface proposal through a compact keypoint component.

    Surface masking can split thin furniture close to a wall.  This pass only
    trusts a mixed clean/wall keypoint component when an existing surface
    proposal already covers a substantial part of it.  Large wall chains and
    unseeded components therefore cannot create object proposals by themselves.
    """
    points = np.asarray(keypoint_points, float).reshape(-1, 3)
    scores = (
        np.ones(len(points), float) if scores is None
        else np.asarray(scores, float).reshape(-1)
    )
    wall_mask = (
        np.zeros(len(points), bool) if wall_mask is None
        else np.asarray(wall_mask, bool).reshape(-1)
    )
    if not len(points) or not groups:
        return list(groups)

    parent = np.arange(len(points), dtype=int)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    tree = cKDTree(points)
    for left, right in tree.query_pairs(link_radius):
        union(int(left), int(right))
    components = {}
    for index in range(len(points)):
        components.setdefault(find(index), []).append(index)

    replacements = []
    replaced_groups = set()
    for values in components.values():
        members = np.asarray(values, int)
        if not min_members <= len(members) <= max_members:
            continue
        member_walls = wall_mask[members]
        wall_ratio = float(np.mean(member_walls))
        if not np.any(member_walls) or not np.any(~member_walls):
            continue
        if wall_ratio > max_wall_ratio:
            continue
        if float(np.max(np.ptp(points[members], axis=0))) > max_object_diameter:
            continue
        component_set = set(members.tolist())
        seeded = []
        for group_index, group in enumerate(groups):
            group_set = set(np.asarray(group.member_indices, int).tolist())
            overlap = len(component_set & group_set)
            if (
                overlap >= min_seed_overlap
                and overlap / max(1, len(group_set)) >= min_seed_fraction
            ):
                seeded.append(group_index)
        if not seeded:
            continue
        replacements.append(_make_group(members, points, scores))
        replaced_groups.update(seeded)

    reconciled = [
        group for index, group in enumerate(groups)
        if index not in replaced_groups
    ] + replacements
    reconciled.sort(key=lambda group: (-len(group.member_indices), -group.score))
    return reconciled


def partition_overlapping_groups(
        groups, keypoint_points, *, scores=None, min_members=2):
    """Turn soft proposals into disjoint object groups.

    Larger proposals own their shared support surface.  A smaller proposal is
    retained when its exclusive core still contains enough keypoints.  This is
    useful for an object resting on furniture: the desk keeps its tabletop and
    the supported object keeps only its own distinctive geometry.
    """
    points = np.asarray(keypoint_points, float).reshape(-1, 3)
    scores = (
        np.ones(len(points), float) if scores is None
        else np.asarray(scores, float).reshape(-1)
    )
    ordered = sorted(
        groups,
        key=lambda group: (-len(group.member_indices), -group.score),
    )
    assigned = set()
    partitioned = []
    for group in ordered:
        members = np.asarray([
            index for index in group.member_indices
            if int(index) not in assigned
        ], int)
        if len(members) < min_members:
            continue
        partitioned.append(_make_group(members, points, scores))
        assigned.update(members.tolist())
    partitioned.sort(
        key=lambda group: (-len(group.member_indices), -group.score),
    )
    return partitioned


def build_object_keypoint_groups(
        surface_points, keypoint_points, *, surface_wall_mask=None,
        keypoint_wall_mask=None, scores=None, object_scores=None,
        ground=0.0, floor_height=0.06, wall_clearance=0.075,
        clean_restore_radius=0.12, surface_voxel=0.055,
        surface_connection=0.095, assignment_radius=0.14,
        min_surface_voxels=8, min_members=2, max_groups=40,
        max_object_diameter=1.65, component_merge_gap=0.35,
        part_merge_gap=0.45, orphan_attach_radius=0.18,
        object_rescue_radius=0.50, object_rescue_link_radius=0.42,
        object_rescue_score=1.01, keypoint_link_radius=0.30,
        keypoint_component_max_members=32):
    """Build the tuned, disjoint object proposals used by grouped retrieval."""
    surface = np.asarray(surface_points, float).reshape(-1, 3).copy()
    points = np.asarray(keypoint_points, float).reshape(-1, 3).copy()
    if not len(surface) or not len(points):
        return []
    surface[:, 1] -= float(ground)
    points[:, 1] -= float(ground)
    surface_wall = (
        np.zeros(len(surface), bool) if surface_wall_mask is None
        else np.asarray(surface_wall_mask, bool).reshape(-1).copy()
    )
    keypoint_wall = (
        np.zeros(len(points), bool) if keypoint_wall_mask is None
        else np.asarray(keypoint_wall_mask, bool).reshape(-1)
    )
    scores = (
        np.ones(len(points), float) if scores is None
        else np.asarray(scores, float).reshape(-1)
    )
    object_scores = (
        np.zeros(len(points), float) if object_scores is None
        else np.asarray(object_scores, float).reshape(-1)
    )
    if len(surface_wall) != len(surface):
        raise ValueError('surface_wall_mask must match surface points')
    if len(keypoint_wall) != len(points):
        raise ValueError('keypoint_wall_mask must match keypoints')

    wall_points = surface[surface_wall]
    if len(wall_points):
        wall_distance, _ = cKDTree(wall_points).query(surface, k=1)
        surface_wall |= wall_distance <= float(wall_clearance)
    clean_points = points[~keypoint_wall]
    if len(clean_points):
        clean_distance, _ = cKDTree(clean_points).query(surface, k=1)
        surface_wall &= clean_distance > float(clean_restore_radius)

    groups = build_surface_component_groups(
        surface, points, wall_mask=surface_wall, scores=scores,
        floor_height=floor_height, voxel_size=surface_voxel,
        connection_radius=surface_connection,
        assignment_radius=assignment_radius,
        min_surface_voxels=min_surface_voxels,
        max_object_diameter=max_object_diameter,
        min_members=min_members, max_groups=max_groups,
    )
    groups = refine_surface_groups(
        groups, points, scores=scores, wall_mask=keypoint_wall,
        min_members=min_members, merge_gap=component_merge_gap,
        part_merge_gap=part_merge_gap,
        max_object_diameter=max_object_diameter,
    )
    groups = recover_object_evidence_groups(
        groups, points, scores=scores, wall_mask=keypoint_wall,
        object_scores=object_scores, orphan_radius=orphan_attach_radius,
        rescue_radius=object_rescue_radius,
        rescue_link_radius=object_rescue_link_radius,
        min_object_score=object_rescue_score,
        max_object_diameter=max_object_diameter,
    )
    groups = reconcile_keypoint_components(
        groups, points, scores=scores, wall_mask=keypoint_wall,
        link_radius=keypoint_link_radius, min_members=min_members,
        max_members=keypoint_component_max_members,
        max_object_diameter=max_object_diameter,
    )
    groups = [
        group for group in groups
        if not np.all(keypoint_wall[group.member_indices])
    ]
    return partition_overlapping_groups(
        groups, points, scores=scores, min_members=min_members,
    )


def map_groups_to_features(groups, keypoint_points, features, tolerance=1e-6,
                           min_features=2):
    """Map object-proposal members to descriptor features that survived filtering."""
    if not groups or not features:
        return []
    keypoint_points = np.asarray(keypoint_points, float).reshape(-1, 3)
    feature_points = np.asarray([
        feature.position for feature in features
    ], float).reshape(-1, 3)
    distance, nearest = cKDTree(feature_points).query(keypoint_points, k=1)
    keypoint_to_feature = {
        int(index): int(feature_index)
        for index, (value, feature_index) in enumerate(zip(distance, nearest))
        if float(value) <= float(tolerance)
    }
    output = []
    for group in groups:
        indices = sorted({
            keypoint_to_feature[int(index)]
            for index in np.asarray(group.member_indices, int)
            if int(index) in keypoint_to_feature
        })
        if len(indices) >= int(min_features):
            output.append([features[index] for index in indices])
    return output


def build_keypoint_groups(
        points, *, normals=None, scores=None, membership_weights=None,
        radius=0.65, edge_radius=0.24, min_membership_weight=0.5,
        path_budget=0.9, low_membership_penalty=2.5,
        min_members=3, max_members=24, max_groups=40,
        anchor_separation=0.25, duplicate_jaccard=0.8):
    """Create overlapping local object proposals around strong anchors.

    This is deliberately not hard segmentation: an object touching a wall may
    share boundary keypoints with both a furniture proposal and a wall proposal.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    count = len(points)
    if count == 0:
        return []
    normals = None if normals is None else np.asarray(normals, float).reshape(-1, 3)
    scores = (
        np.ones(count, float) if scores is None
        else np.nan_to_num(np.asarray(scores, float).reshape(-1), nan=0.0)
    )
    membership_weights = (
        np.ones(count, float) if membership_weights is None
        else np.asarray(membership_weights, float).reshape(-1)
    )
    if (
        len(scores) != count or len(membership_weights) != count
        or (normals is not None and len(normals) != count)
    ):
        raise ValueError(
            'points, normals, scores and membership_weights must have equal lengths'
        )

    tree = cKDTree(points)
    eligible = membership_weights >= min_membership_weight
    order = np.argsort(-(scores * membership_weights), kind="stable")
    anchors = []
    groups = []
    member_sets = []
    for anchor in order:
        if not eligible[anchor]:
            continue
        if anchors and np.min(np.linalg.norm(
                points[np.asarray(anchors)] - points[anchor], axis=1,
        )) < anchor_separation:
            continue
        local = np.asarray(tree.query_ball_point(points[anchor], radius), int)
        if len(local) < min_members:
            continue
        # A spherical neighborhood can contain two nearby pieces of furniture.
        # Keep only the graph component that is connected to the anchor by
        # short keypoint-to-keypoint hops.
        local_tree = cKDTree(points[local])
        adjacency = local_tree.query_ball_point(points[local], edge_radius)
        anchor_local = int(np.flatnonzero(local == anchor)[0])
        costs = np.full(len(local), np.inf)
        costs[anchor_local] = 0.0
        pending = [(0.0, anchor_local)]
        while pending:
            current_cost, current = heapq.heappop(pending)
            if current_cost != costs[current]:
                continue
            for neighbor in adjacency[current]:
                edge = float(np.linalg.norm(
                    points[local[current]] - points[local[neighbor]],
                ))
                ambiguity = 1.0 - min(
                    membership_weights[local[current]],
                    membership_weights[local[neighbor]],
                )
                next_cost = current_cost + edge * (
                    1.0 + low_membership_penalty * ambiguity
                )
                if next_cost <= path_budget and next_cost < costs[neighbor]:
                    costs[neighbor] = next_cost
                    heapq.heappush(pending, (next_cost, neighbor))
        connected = np.flatnonzero(np.isfinite(costs))
        local_costs = costs[connected]
        local = local[connected]
        if len(local) < min_members:
            continue
        local_dist = np.linalg.norm(points[local] - points[anchor], axis=1)
        weighted_scores = scores[local] * membership_weights[local]
        local_order = np.lexsort((-weighted_scores, local_dist, local_costs))
        local = local[local_order[:max_members]]
        members = frozenset(local.tolist())
        if any(
            len(members & previous) / max(1, len(members | previous))
            >= duplicate_jaccard
            for previous in member_sets
        ):
            continue
        group_points = points[local]
        centroid = np.average(
            group_points, axis=0,
            weights=np.clip(weighted_scores[local_order[:len(local)]], 1e-6, None),
        )
        compactness = 1.0 - min(1.0, float(np.mean(local_dist)) / radius)
        group_score = float(np.mean(
            scores[local] * membership_weights[local]
        ) * (0.5 + 0.5 * compactness))
        groups.append(KeypointGroup(
            member_indices=local,
            anchor_index=int(anchor),
            centroid=centroid,
            extent=np.ptp(group_points, axis=0),
            score=group_score,
            descriptor=_group_descriptor(
                group_points,
                None if normals is None else normals[local],
                scores[local],
            ),
        ))
        member_sets.append(members)
        anchors.append(int(anchor))
        if len(groups) >= max_groups:
            break
    return groups
