import numpy as np
import pytest
from geometry import PointCloud
from transforms import GroundTransform
from verify import normal_compatible_coverage


def test_normals_reject_coincident_perpendicular_surfaces_but_accept_opposites():
    points = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
    model = PointCloud(points, np.array([[0., 1, 0]]*3))
    scan = PointCloud(points, np.array([[0., -1, 0], [1., 0, 0], [0., 0, 0]]))
    t = GroundTransform(0, 1, np.zeros(3), np.array([0., 1, 0]))
    assert normal_compatible_coverage(model, scan, t, .05, 60) == pytest.approx(2/3)
    assert normal_compatible_coverage(model, scan, t, .05, 0) is None
    assert normal_compatible_coverage(model, PointCloud(points), t, .05, 60) is None


def test_normal_comparison_rotates_with_model_and_keeps_distance_gate():
    model = PointCloud(np.array([[0., 0, 0], [0., 1, 0]]), np.array([[1., 0, 0]]*2))
    scan = PointCloud(np.array([[0., 0, 0], [0., 1.2, 0]]), np.array([[0., 0, -1]]*2))
    t = GroundTransform(np.pi/2, 1, np.zeros(3), np.array([0., 1, 0]))
    assert normal_compatible_coverage(model, scan, t, .05, 60) == .5


def test_final_validation_uses_optional_normal_gate(monkeypatch):
    from types import SimpleNamespace
    import progressive
    from config import PipelineConfig
    from verify import Registration
    points=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1]])
    model=SimpleNamespace(name='test', mesh_path='', cloud=PointCloud(points,np.tile([0.,1,0],(4,1))))
    scene=SimpleNamespace(cloud=PointCloud(points,np.tile([1.,0,0],(4,1))),up=np.array([0.,1,0]))
    monkeypatch.setattr(progressive.db_store,'load_index',lambda _: {'names':['test']})
    monkeypatch.setattr(progressive.db_store,'model_files',lambda _: ['model'])
    monkeypatch.setattr(progressive.db_store,'load_model',lambda _: model)
    cfg=PipelineConfig()
    cfg.matching.min_model_zone_fraction=0
    cfg.matching.min_support_extent_ratio=0
    reg=Registration('test',GroundTransform(0,1,np.zeros(3),scene.up),1,0,1)
    assert progressive._final_revalidate([reg],'db',scene,cfg,.45,0)[0] == [reg]
    cfg.matching.normal_support_angle_deg=60
    kept,diagnostics=progressive._final_revalidate([reg],'db',scene,cfg,.45,0)
    assert not kept
    assert diagnostics[0]['status']=='rejected_normal_support'
