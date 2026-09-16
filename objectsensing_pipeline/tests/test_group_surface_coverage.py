from types import SimpleNamespace
import numpy as np
from geometry import PointCloud
from transforms import GroundTransform
from verify import coverage_score, group_surface_coverage
from pipeline import _ranking_reverse_coverage, _tag_group_result
import pytest
from verify import verify_model, verify_model_candidates


def test_small_model_cannot_define_its_own_perfect_region():
    axis = np.linspace(0, 1, 21)
    whole = np.array([[x, .8, z] for x in axis for z in axis])
    scan = PointCloud(whole)
    fragment = PointCloud(whole[whole[:, 0] <= .25])
    transform = GroundTransform(0., 1., np.zeros(3), np.array([0., 1., 0.]))
    group = whole[[0, 20, 420, 440]]
    small = group_surface_coverage(fragment, scan, transform, group, .01)
    complete = group_surface_coverage(scan, scan, transform, group, .01)
    assert coverage_score(fragment, scan, transform, .01)[1] == 1.
    assert .25 < small[0] < .3
    assert complete[0] == 1.
    assert small[1] == complete[1] == len(whole)


def test_group_measurement_never_increases_geometric_rank():
    reg = SimpleNamespace(cov_reverse=.99, _geometric_cov_reverse=.8, _query_group_reverse=.2)
    assert _ranking_reverse_coverage(reg) == .2
    reg._query_group_reverse = .9
    assert _ranking_reverse_coverage(reg) == .8


def test_group_context_survives_registration_and_missing_context_is_ignored():
    reg = SimpleNamespace()
    features = [SimpleNamespace(position=np.array(p)) for p in [[0, 0, 0], [1, 1, 0], [1, 0, 1]]]
    _tag_group_result(reg, 2, features)
    assert reg._query_group_id == 2
    assert reg._query_group_points.shape == (3, 3)
    scan = PointCloud(np.zeros((2, 3)))
    assert group_surface_coverage(scan, scan, None, None) is None
    assert group_surface_coverage(scan, scan, None, reg._query_group_points) is None


@pytest.mark.parametrize('multiple', [False, True])
def test_pose_choice_prefers_the_target_group_over_identical_remote_geometry(multiple):
    points = np.array([[x, .8, z] for x in np.linspace(0, .3, 8) for z in np.linspace(0, .3, 8)])
    target = points + [2., 0., 0.]
    model, scan = PointCloud(points), PointCloud(np.vstack([points, target]))
    poses = [SimpleNamespace(transform=GroundTransform(0., 1., np.array([x,0.,0.]), np.array([0.,1.,0.])), quality=1., inliers=[])
             for x in [0., 2.]]
    kwargs = dict(group_points=target, threshold=.02, min_structure=0., min_thickness=0.)
    if multiple:
        chosen = verify_model_candidates('table_test', model, scan, poses, max_results=2, **kwargs)[0]
    else:
        chosen = verify_model('table_test', model, scan, poses, **kwargs)
    np.testing.assert_allclose(chosen.transform.t, [2., 0., 0.])
