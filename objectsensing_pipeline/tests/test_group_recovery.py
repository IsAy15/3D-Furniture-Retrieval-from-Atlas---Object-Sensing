import numpy as np

from tools.sweep_group_recovery import evaluate


def test_perfect_partition_and_merge_penalty():
    points=np.array([[0.,0,0],[.1,0,0],[1.,0,0],[1.1,0,0]])
    labels=np.array([0,0,1,1])
    separate=[dict(member_indices=[0,1]),dict(member_indices=[2,3])]
    merged=[dict(member_indices=[0,1,2,3])]
    assert evaluate(points,labels,points,separate)["score"] == 1.
    assert evaluate(points,labels,points,merged)["score"] < .5


def test_missing_anchors_and_contamination_are_penalized():
    points=np.array([[0.,0,0],[.3,0,0],[1.,0,0]])
    labels=np.array([0,0,-1])
    subset=points[[0,2]]
    metrics=evaluate(points,labels,subset,[dict(member_indices=[0,1])])
    assert metrics["objects"][0]["precision"] == .5
    assert metrics["objects"][0]["recall"] == .5
    assert metrics["score"] == .5


def test_empty_predictions_score_zero():
    result=evaluate(np.zeros((1,3)),np.array([0]),np.empty((0,3)),[])
    assert result["score"] == 0.


def test_group_controls_roundtrip_and_preview():
    from types import SimpleNamespace
    import debug_visualizer as debug
    parameters=debug.normalized_parameters(dict(group_part_merge_gap=.30,neighbor_radius=.06))
    assert parameters['group_part_merge_gap'] == .30
    assert parameters['neighbor_radius'] == .06
    kp=SimpleNamespace(positions=np.array([[0.,1.,0.],[1.,1.,0.]]),source_index=np.array([10,20]),size=2)
    groups=[dict(member_indices=[0,1])]
    layers,steps=[],[]
    debug._append_group_preview(layers,steps,kp,groups,np.array([0,1,0]),0.)
    debug._append_group_preview(layers,steps,kp,groups,np.array([0,1,0]),0.)
    assert len(layers)==2 and len(steps)==1
    assert layers[0]['source_index']==[10,20]
    assert len(layers[1]['segments'])==2
