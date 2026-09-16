import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matching import _cap_by_response  # noqa: E402


def _feature(x, response, selection_score=None, wall_affinity=0.0,
             object_score=0.0, wall_plane_index=-1, wall_proximity=0.0):
    return SimpleNamespace(
        position=np.array([x, 0.0, 0.0]),
        response=float(response),
        selection_score=selection_score,
        wall_affinity=float(wall_affinity),
        wall_proximity=float(wall_proximity),
        object_score=float(object_score),
        wall_plane_index=int(wall_plane_index),
    )


def test_spatial_cap_preserves_weak_regions():
    crowded = [_feature(0.01 * i, 100 - i) for i in range(10)]
    weak_region = [_feature(2.0, 1.0), _feature(4.0, 0.5)]
    feats = crowded + weak_region

    global_selection = _cap_by_response(feats, 3)
    spatial_selection = _cap_by_response(
        feats, 3, spatial_balance=True, cell_size=0.5,
    )

    assert all(feat.position[0] < 0.5 for feat in global_selection)
    assert sorted(float(feat.position[0]) for feat in spatial_selection) == [
        0.0, 2.0, 4.0,
    ]


def test_zero_cap_keeps_every_feature():
    feats = [_feature(float(i), float(i)) for i in range(12)]
    assert _cap_by_response(feats, 0, spatial_balance=True) is feats


def test_cap_reserves_budget_for_2d_corners():
    harris = [_feature(float(i), 100.0 - i) for i in range(30)]
    corners_2d = [_feature(100.0 + i, 0.0) for i in range(12)]

    selected = _cap_by_response(
        harris + corners_2d,
        20,
        harris_threshold=0.008,
        min_2d_fraction=0.25,
    )

    assert len(selected) == 20
    assert sum(feat.response <= 0.008 for feat in selected) == 5


def test_cap_uses_contextual_selection_score_without_changing_corner_type():
    wall = _feature(0.0, response=0.9, selection_score=-0.1)
    furniture = _feature(1.0, response=0.4, selection_score=0.4)

    selected = _cap_by_response([wall, furniture], 1)

    assert selected == [furniture]
    assert wall.response == 0.9


def test_wall_budget_limits_unprotected_walls_and_fills_with_objects():
    walls = [
        _feature(
            0.05 * index, 100.0 - index,
            wall_affinity=0.8, wall_plane_index=0,
        )
        for index in range(12)
    ]
    objects = [_feature(2.0 + index, 1.0 - 0.1 * index) for index in range(6)]
    trace = {}

    selected = _cap_by_response(
        walls + objects, 5,
        wall_budget_enabled=True,
        wall_budget_max_fraction=0.20,
        wall_budget_cell_size=0.5,
        wall_budget_max_per_cell=1,
        trace=trace,
    )

    assert len(selected) == 5
    assert sum(item.wall_plane_index >= 0 for item in selected) == 1
    assert len(trace["wall_budget_rejected"]) == 11
    assert trace["wall_budget_quota"] == 1


def test_wall_budget_does_not_limit_object_protected_feature():
    protected = _feature(
        0.0, 10.0, wall_affinity=0.9,
        object_score=0.8, wall_plane_index=0,
    )
    walls = [
        _feature(
            1.0 + index, 9.0 - index,
            wall_affinity=0.9, wall_plane_index=0,
        )
        for index in range(4)
    ]
    objects = [_feature(10.0 + index, 1.0) for index in range(4)]

    selected = _cap_by_response(
        [protected] + walls + objects, 4,
        wall_budget_enabled=True,
        wall_budget_max_object_score=0.35,
        wall_budget_max_fraction=0.0,
    )

    assert protected in selected
    assert all(not any(item is chosen for chosen in selected) for item in walls)


def test_wall_budget_captures_planar_corner_with_unstable_normal():
    wall_corner = _feature(
        0.0, 10.0, wall_affinity=0.0, wall_proximity=0.8,
        wall_plane_index=0,
    )
    objects = [_feature(2.0 + index, 1.0) for index in range(4)]

    selected = _cap_by_response(
        [wall_corner] + objects, 3,
        wall_budget_enabled=True,
        wall_budget_affinity_threshold=0.05,
        wall_budget_proximity_threshold=0.20,
        wall_budget_max_fraction=0.0,
    )

    assert not any(item is wall_corner for item in selected)


def test_wall_budget_applies_when_global_cap_is_not_reached():
    walls = [
        _feature(
            float(index), 10.0 - index,
            wall_affinity=0.8, wall_plane_index=0,
        )
        for index in range(6)
    ]
    objects = [_feature(10.0 + index, 1.0) for index in range(2)]
    trace = {}

    selected = _cap_by_response(
        walls + objects, 10,
        wall_budget_enabled=True,
        wall_budget_max_fraction=0.25,
        wall_budget_cell_size=0.5,
        wall_budget_min_features=0,
        trace=trace,
    )

    assert len(selected) == 4
    assert sum(item.wall_plane_index >= 0 for item in selected) == 2
    assert trace["wall_budget_quota"] == 2
    assert len(trace["wall_budget_rejected"]) == 4


def test_wall_budget_floor_preserves_sparse_scene_recall():
    walls = [
        _feature(
            float(index), 10.0 - index,
            wall_affinity=0.8, wall_plane_index=0,
        )
        for index in range(6)
    ]
    objects = [_feature(10.0 + index, 1.0) for index in range(2)]
    trace = {}

    selected = _cap_by_response(
        walls + objects, 10,
        wall_budget_enabled=True,
        wall_budget_max_fraction=0.25,
        wall_budget_min_features=8,
        trace=trace,
    )

    assert len(selected) == 8
    assert len(trace["wall_budget_reintroduced"]) == 4
    assert trace["wall_budget_rejected"] == []


def test_component_budget_adapts_total_to_spatial_regions():
    features = []
    for center in (0.0, 2.0, 4.0):
        features.extend([
            _feature(center + 0.02 * index, 100.0 - index)
            for index in range(8)
        ])
    trace = {}

    selected = _cap_by_response(
        features, 100,
        component_budget_enabled=True,
        component_budget_radius=0.25,
        component_budget_max_per_component=3,
        component_budget_min_features=0,
        trace=trace,
    )

    assert len(selected) == 9
    assert trace["component_budget_count"] == 3
    assert sorted(trace["component_budget_sizes"]) == [8, 8, 8]
    assert len(trace["component_budget_rejected"]) == 15


def test_component_budget_floor_preserves_small_scene():
    features = [_feature(0.02 * index, 10.0 - index) for index in range(12)]
    trace = {}

    selected = _cap_by_response(
        features, 100,
        component_budget_enabled=True,
        component_budget_radius=0.25,
        component_budget_max_per_component=3,
        component_budget_min_features=12,
        trace=trace,
    )

    assert len(selected) == 12
    assert len(trace["component_budget_reintroduced"]) == 9
    assert trace["component_budget_rejected"] == []


def test_component_budget_preserves_2d_corner_reserve_per_region():
    harris = [_feature(0.01 * index, 10.0 - index) for index in range(20)]
    corners_2d = [
        _feature(0.02 * index, 0.0) for index in range(20, 24)
    ]

    selected = _cap_by_response(
        harris + corners_2d, 100,
        harris_threshold=0.008,
        min_2d_fraction=0.25,
        component_budget_enabled=True,
        component_budget_radius=0.5,
        component_budget_max_per_component=8,
        component_budget_min_features=0,
    )

    assert len(selected) == 8
    assert sum(feature.response <= 0.008 for feature in selected) == 2
