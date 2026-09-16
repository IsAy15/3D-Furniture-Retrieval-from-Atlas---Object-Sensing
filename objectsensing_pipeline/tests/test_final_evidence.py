from types import SimpleNamespace

import numpy as np
import pytest

import cli
import pipeline
from config import PipelineConfig
from geometry import PointCloud
from transforms import GroundTransform
from verify import Registration


def test_exported_harmonic_score_matches_final_rank_under_visibility():
    reg = Registration('chair_test', GroundTransform(0, 1, np.zeros(3),
                       np.array([0., 1., 0.])), .8, .01, 4., .8, .99)
    reg._geometric_cov_reverse = .7
    reg._query_group_reverse = .4
    reg._visibility_used = True
    entry = cli._registration_manifest_entry(reg)
    assert entry['balanced_score'] == pytest.approx(.5333)
    assert entry['balanced_score'] == pipeline._trace_registration(reg)['balanced_score']
    assert entry['cov_reverse'] == .99
    assert entry['group_cov_reverse'] == .4


def test_unconfirmed_group_pose_does_not_become_a_final_object():
    cfg = PipelineConfig()
    scene = SimpleNamespace(cloud=PointCloud(np.zeros((2, 3))),
                            up=np.array([0., 1., 0.]))
    reg = SimpleNamespace(model_name='chair_background', coverage=.99,
                          cov_reverse=.99, mean_surface_dist=.001,
                          _pose_seed_fallback=True)
    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics([reg], scene, cfg)
    assert selected == []
    assert diagnostics[0]['status'] == 'rejected_unconfirmed_pose_seed'
    # A confirmed hypothesis remains eligible for ordinary coverage checks.
    reg._pose_seed_fallback = False
    reg.cov_reverse = .01
    _, diagnostics = pipeline._non_max_select_fp_with_diagnostics([reg], scene, cfg)
    assert diagnostics[0]['status'] == 'rejected_null_hypothesis'
