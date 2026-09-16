from types import SimpleNamespace

import numpy as np
import pytest

from geometry import PointCloud
from transforms import GroundTransform
from sdf_fusion import VIS_FREE, VIS_OCCUPIED, VIS_UNKNOWN
from verify import visibility_coverage_score


def test_visibility_respects_surface_tolerance_without_filling_unknown_space():
    # One occupied point, one free point within fit tolerance, one genuinely
    # free point beyond tolerance, and one unknown point at the surface.
    points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    model = PointCloud(points)
    scan = PointCloud(points + np.array([[0., 0., 0.], [0., .03, 0.],
                                         [0., .2, 0.], [0., 0., 0.]]))
    volume = SimpleNamespace(sample_visibility=lambda pts: np.array(
        [VIS_OCCUPIED, VIS_FREE, VIS_FREE, VIS_UNKNOWN], dtype=np.uint8))
    transform = GroundTransform(0., 1., np.zeros(3), np.array([0., 1., 0.]))
    strict = visibility_coverage_score(model, transform, volume)
    tolerant = visibility_coverage_score(model, transform, volume, scan, .05)
    assert strict == pytest.approx((1 / 3, .75, 2 / 3, 3))
    assert tolerant == pytest.approx((2 / 3, .75, 1 / 3, 3))


def test_visibility_does_not_accept_free_points_outside_fit_tolerance():
    model = PointCloud(np.array([[0., 0., 0.]]))
    scan = PointCloud(np.array([[0., .03, 0.]]))
    volume = SimpleNamespace(sample_visibility=lambda pts: np.full(len(pts), VIS_FREE))
    transform = GroundTransform(0., 1., np.zeros(3), np.array([0., 1., 0.]))
    assert visibility_coverage_score(model, transform, volume, scan, .02) == (0., 1., 1., 1)
