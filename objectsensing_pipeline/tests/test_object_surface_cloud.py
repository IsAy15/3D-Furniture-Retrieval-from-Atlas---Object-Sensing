from types import SimpleNamespace
import numpy as np
from database import DatabaseModel, object_surface_cloud
from geometry import PointCloud
from verify import coverage_score
from transforms import GroundTransform


def model_with_floor():
    rng = np.random.default_rng(4)
    surface = rng.uniform([0, .2, 0], [1, 1, 1], (80, 3))
    floor = rng.uniform([-1, .2, -1], [2, .2, 2], (20, 3))
    cloud = PointCloud(np.vstack([surface, floor]))
    return DatabaseModel('chair_test', '03001627', cloud, None, [], .2), surface, floor


def test_legacy_virtual_floor_is_excluded_without_mutating_descriptors():
    model, surface, _ = model_with_floor()
    assert np.array_equal(object_surface_cloud(model).points, surface)
    assert model.cloud.size == 100


def test_explicit_surface_count_preserves_actual_object_bottom_points():
    model, surface, _ = model_with_floor()
    model.cloud.points[0, 1] = .2
    model.surface_point_count = 80
    actual = object_surface_cloud(model)
    assert actual.size == 80
    assert actual.points[0, 1] == .2


def test_unrecognized_clouds_are_not_trimmed():
    model, _, _ = model_with_floor()
    model.cloud.points[-1, 1] = .3
    assert object_surface_cloud(model) is model.cloud
    other = SimpleNamespace(cloud=model.cloud)
    assert object_surface_cloud(other) is other.cloud


def test_artificial_floor_does_not_count_as_furniture_evidence():
    model, _, floor = model_with_floor()
    scene = PointCloud(floor)
    transform = GroundTransform(0., 1., np.zeros(3), np.array([0., 1., 0.]))
    full = coverage_score(model.cloud, scene, transform, .001)[0]
    actual = coverage_score(object_surface_cloud(model), scene, transform, .001)[0]
    assert full >= .2
    assert actual == 0.
