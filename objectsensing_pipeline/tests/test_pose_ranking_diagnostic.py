import numpy as np
import pytest

from tools.diagnose_pose_ranking import centered_seed
from transforms import GroundTransform


@pytest.mark.parametrize('up', [np.array([0., 1., 0.]), np.array([0., 0., 1.])])
def test_yaw_sweep_preserves_bottom_center(up):
    points = np.array([[-1., 0., -2.], [2., 3., 4.], [0., 1., 0.]])
    baseline = GroundTransform(.37, .9, np.array([1., 2., 3.]), up)
    pivot = (points.min(0) + points.max(0)) / 2
    pivot += up * ((points @ up).min() - pivot @ up)
    for angle in (0, 90, 180, 270):
        seed = centered_seed(points, baseline, angle, 1.1)
        np.testing.assert_allclose(seed.apply(pivot[None]), baseline.apply(pivot[None]))
        assert seed.scale == pytest.approx(.99)


def test_same_yaw_and_scale_reproduce_baseline():
    points = np.array([[-1., 0., -2.], [2., 3., 4.]])
    baseline = GroundTransform(.37, .9, np.array([1., 2., 3.]), np.array([0., 1., 0.]))
    seed = centered_seed(points, baseline, np.degrees(baseline.theta))
    np.testing.assert_allclose(seed.matrix(), baseline.matrix(), atol=1e-12)
