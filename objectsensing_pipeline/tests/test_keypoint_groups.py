import numpy as np

from keypoint_groups import (
    KeypointGroup, build_keypoint_groups, build_surface_component_groups,
    partition_overlapping_groups, reconcile_keypoint_components,
    recover_object_evidence_groups, refine_surface_groups,
)


def test_groups_keep_spatially_separated_objects_apart():
    left = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [.1, .1, 0]])
    right = left + np.array([2.0, 0, 0])
    groups = build_keypoint_groups(
        np.vstack([left, right]), radius=.3, min_members=3,
        anchor_separation=.2,
    )

    assert len(groups) == 2
    assert all(
        np.all(group.member_indices < 4) or np.all(group.member_indices >= 4)
        for group in groups
    )


def test_groups_are_soft_and_can_overlap():
    points = np.array([[x, 0, 0] for x in np.linspace(0, 1, 7)])
    groups = build_keypoint_groups(
        points, radius=.45, min_members=3, anchor_separation=.3,
        duplicate_jaccard=.95,
    )

    memberships = [set(group.member_indices.tolist()) for group in groups]
    assert len(groups) >= 2
    assert any(a & b for i, a in enumerate(memberships) for b in memberships[i + 1:])


def test_short_edge_connectivity_does_not_bridge_nearby_objects():
    left = np.array([[0, 0, 0], [.08, 0, 0], [0, .08, 0]])
    right = left + np.array([.45, 0, 0])
    groups = build_keypoint_groups(
        np.vstack([left, right]), radius=.8, edge_radius=.15,
        min_members=3, anchor_separation=.2,
    )

    assert len(groups) == 2
    assert all(len(group.member_indices) == 3 for group in groups)


def test_low_membership_points_do_not_propagate_along_wall_chain():
    points = np.array([
        [0, 0, 0], [.1, 0, 0], [0, .1, 0],
        [.15, .1, 0], [.2, .1, 0], [.25, .1, 0],
    ])
    groups = build_keypoint_groups(
        points, membership_weights=[1, 1, 1, .2, .2, .2],
        radius=.5, edge_radius=.11, path_budget=.5,
        low_membership_penalty=4.0, min_members=3,
    )

    assert groups
    assert all(np.max(group.member_indices) <= 3 for group in groups)


def test_group_descriptor_has_stable_shape():
    points = np.array([[0, 0, 0], [.1, 0, 0], [0, .2, 0], [.1, .2, .1]])
    normals = np.tile([0, 1, 0], (len(points), 1))
    groups = build_keypoint_groups(
        points, normals=normals, radius=.5, min_members=3,
    )

    assert groups[0].descriptor.shape == (14,)
    assert np.isfinite(groups[0].descriptor).all()


def test_surface_components_bridge_sparse_keypoints_on_same_object():
    left_surface = np.array([[x, y, 0] for x in np.linspace(0, 1, 21) for y in (0, .05)])
    right_surface = left_surface + np.array([2.0, 0, 0])
    surface = np.vstack([left_surface, right_surface])
    keypoints = np.array([[0, .05, 0], [1, .05, 0], [.5, .05, 0], [2, .05, 0], [3, .05, 0]])

    groups = build_surface_component_groups(
        surface, keypoints, floor_height=0, voxel_size=.04,
        connection_radius=.08, assignment_radius=.1,
        min_surface_voxels=3, min_members=2,
    )

    assert sorted(len(group.member_indices) for group in groups) == [2, 3]


def test_surface_components_remove_wall_bridge():
    chair = np.array([[x, .2 + y, 0] for x in (0, .05, .1) for y in (0, .05)])
    wall = np.array([[x, .2 + y, .08] for x in np.linspace(0, .8, 17) for y in (0, .05)])
    surface = np.vstack([chair, wall])
    wall_mask = np.arange(len(surface)) >= len(chair)
    keypoints = np.array([[0, .2, 0], [.05, .25, 0], [.1, .2, 0], [.7, .2, .08]])

    groups = build_surface_component_groups(
        surface, keypoints, wall_mask=wall_mask, floor_height=0,
        voxel_size=.04, connection_radius=.08, assignment_radius=.1,
        min_surface_voxels=3, min_members=2,
    )

    assert len(groups) == 1
    assert set(groups[0].member_indices) == {0, 1, 2}


def test_surface_components_reconnect_when_object_bridge_is_restored():
    left = np.array([[x, y, 0.0] for x in np.arange(0.0, 0.31, 0.05)
                     for y in np.arange(0.1, 0.41, 0.05)])
    right = left + np.array([0.50, 0.0, 0.0])
    bridge = np.array([[x, 0.25, 0.0] for x in np.arange(0.30, 0.56, 0.025)])
    surface = np.vstack([left, bridge, right])
    keypoints = np.array([
        [0.1, 0.2, 0.0], [0.3, 0.25, 0.0], [0.75, 0.3, 0.0],
    ])
    wall_mask = np.zeros(len(surface), bool)
    wall_mask[len(left):len(left) + len(bridge)] = True
    kwargs = dict(
        floor_height=0.0, voxel_size=0.04, connection_radius=0.08,
        assignment_radius=0.12, min_surface_voxels=3, min_members=1,
    )

    disconnected = build_surface_component_groups(
        surface, keypoints, wall_mask=wall_mask, **kwargs,
    )
    wall_mask[len(left):len(left) + len(bridge)] = False
    restored = build_surface_component_groups(
        surface, keypoints, wall_mask=wall_mask, **kwargs,
    )

    assert len(disconnected) == 2
    assert len(restored) == 1


def test_oversized_surface_component_is_split_by_object_diameter():
    surface = np.array([[x, .2, 0] for x in np.linspace(0, 3, 61)])
    keypoints = np.array([[x, .2, 0] for x in (0, .2, .4, 2.6, 2.8, 3.0)])

    groups = build_surface_component_groups(
        surface, keypoints, floor_height=0, voxel_size=.04,
        connection_radius=.08, assignment_radius=.1,
        min_surface_voxels=3, min_members=2, max_object_diameter=1.0,
    )

    assert sorted(len(group.member_indices) for group in groups) == [3, 3]


def _fake_group(indices, points):
    indices = np.asarray(list(indices), int)
    values = points[indices]
    return KeypointGroup(
        indices, int(indices[0]), values.mean(axis=0),
        np.ptp(values, axis=0), 1.0, np.zeros(14, np.float32),
    )


def test_surface_refinement_merges_overlapping_object_parts():
    points = np.array([
        [0, 0, 0], [.1, 0, 0], [0, .1, 0],
        [.15, .05, 0], [.25, .05, 0], [.2, .15, 0],
    ])
    groups = [
        _fake_group([0, 1, 2], points),
        _fake_group([3, 4, 5], points),
    ]

    refined = refine_surface_groups(groups, points, merge_gap=.2)

    assert len(refined) == 1
    assert len(refined[0].member_indices) == 6


def test_surface_refinement_strips_large_wall_tail():
    points = np.array([[index * .03, 0, 0] for index in range(18)])
    group = _fake_group(range(18), points)
    wall = np.array([False] * 5 + [True] * 13)

    refined = refine_surface_groups([group], points, wall_mask=wall)

    assert len(refined) == 1
    assert set(refined[0].member_indices) == set(range(5))


def test_object_evidence_bridges_clean_groups_through_supported_wall_points():
    points = np.array([
        [0, 0, 0], [.1, 0, 0], [0, .1, 0],
        [.35, .05, 0],
        [.65, 0, 0], [.75, 0, 0], [.7, .1, 0],
    ])
    groups = [_fake_group(range(3), points), _fake_group(range(4, 7), points)]
    wall = np.array([False, False, False, True, False, False, False])
    object_scores = np.array([0, 0, 0, .4, 0, 0, 0])

    recovered = recover_object_evidence_groups(
        groups, points, wall_mask=wall, object_scores=object_scores,
        rescue_radius=.5, rescue_link_radius=.4,
    )

    assert len(recovered) == 1
    assert set(recovered[0].member_indices) == set(range(7))


def test_object_evidence_does_not_bridge_with_weak_wall_point():
    points = np.array([
        [0, 0, 0], [.1, 0, 0], [0, .1, 0],
        [.35, .05, 0],
        [.65, 0, 0], [.75, 0, 0], [.7, .1, 0],
    ])
    groups = [_fake_group(range(3), points), _fake_group(range(4, 7), points)]
    wall = np.array([False, False, False, True, False, False, False])
    object_scores = np.array([0, 0, 0, .02, 0, 0, 0])

    recovered = recover_object_evidence_groups(
        groups, points, wall_mask=wall, object_scores=object_scores,
        rescue_radius=.5, rescue_link_radius=.4,
    )

    assert len(recovered) == 2


def test_object_evidence_attaches_clean_short_range_orphan():
    points = np.array([
        [0, .3, 0], [.1, .3, 0], [0, .4, 0], [.05, .13, 0],
    ])
    groups = [_fake_group(range(3), points)]

    recovered = recover_object_evidence_groups(
        groups, points, orphan_radius=.18,
    )

    assert set(recovered[0].member_indices) == {0, 1, 2, 3}


def test_surface_refinement_allows_elongated_furniture():
    points = np.array([
        [0, 0, 0], [.1, 0, 0], [0, .1, 0],
        [1.45, 0, 0], [1.55, 0, 0], [1.45, .1, 0],
    ])
    groups = [_fake_group(range(3), points), _fake_group(range(3, 6), points)]

    refined = refine_surface_groups(
        groups, points, merge_gap=1.5, max_object_diameter=1.6,
    )

    assert len(refined) == 1


def test_surface_refinement_keeps_furniture_separate_beyond_part_gap():
    left = np.array([
        [0, 0, 0], [.1, 0, 0], [0, .1, 0],
    ])
    right = left + np.array([0.58, 0, 0])
    points = np.vstack([left, right])
    groups = [_fake_group(range(3), points), _fake_group(range(3, 6), points)]

    refined = refine_surface_groups(
        groups, points, merge_gap=.35, part_merge_gap=.45,
    )

    assert len(refined) == 2


def test_keypoint_component_recovers_mixed_furniture_and_drops_surface_tail():
    furniture = np.array([
        [0.00, 0.0, 0.0], [0.15, 0.0, 0.0], [0.30, 0.0, 0.0],
        [0.45, 0.0, 0.0], [0.60, 0.0, 0.0], [0.75, 0.0, 0.0],
    ])
    tail = np.array([[0.0, 0.8, 0.0], [0.1, 0.8, 0.0]])
    points = np.vstack([furniture, tail])
    seed = _fake_group([0, 1, 2, 6, 7], points)
    wall = np.array([False, False, True, True, True, True, False, False])

    reconciled = reconcile_keypoint_components(
        [seed], points, wall_mask=wall, link_radius=.16,
        min_members=3, max_members=10,
    )

    assert len(reconciled) == 1
    assert set(reconciled[0].member_indices) == set(range(6))


def test_keypoint_component_rejects_large_wall_chain():
    points = np.array([[index * .1, 0, 0] for index in range(20)])
    seed = _fake_group([0, 1, 2], points)
    wall = np.array([False, False] + [True] * 18)

    reconciled = reconcile_keypoint_components(
        [seed], points, wall_mask=wall, link_radius=.11,
        max_members=12,
    )

    assert set(reconciled[0].member_indices) == {0, 1, 2}


def test_overlapping_groups_keep_support_and_exclusive_object_core():
    points = np.array([[index * .1, 0, 0] for index in range(8)])
    support = _fake_group(range(6), points)
    supported_object = _fake_group([4, 5, 6, 7], points)

    partitioned = partition_overlapping_groups(
        [supported_object, support], points, min_members=2,
    )

    assert len(partitioned) == 2
    assert set(partitioned[0].member_indices) == set(range(6))
    assert set(partitioned[1].member_indices) == {6, 7}
    assert not (
        set(partitioned[0].member_indices)
        & set(partitioned[1].member_indices)
    )
