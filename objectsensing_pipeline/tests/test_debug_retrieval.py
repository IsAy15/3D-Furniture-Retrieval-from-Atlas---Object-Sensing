"""Contracts for separate retrieval debug stages and reusable pose artifacts."""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import debug_visualizer as debug
import debug_retrieval as retrieval
import pipeline
import db_store
from geometry import PointCloud
from transforms import GroundTransform
from constellations import Constellation
from verify import Registration
from config import PipelineConfig


@pytest.mark.parametrize("stage", ["matching", "verification", "selection"])
def test_new_stage_parameters(stage):
    parameters = debug.normalized_parameters(stage=stage)
    assert set(parameters) == {spec["id"] for spec in debug.parameter_schema(stage)}
    assert all(spec["detail"] and spec["summary"] for spec in debug.parameter_schema(stage))
    with pytest.raises(ValueError):
        debug.normalized_parameters({"parallel_scenes": 0}, stage)


def test_prepared_verification_never_rematches(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("verification must not run descriptor matching")
    monkeypatch.setattr(pipeline, "prepare_model_matching", forbidden)
    cfg = PipelineConfig()
    model = SimpleNamespace(name="chair_test", synset="03001627", cloud=PointCloud(np.zeros((3, 3))))
    monkeypatch.setattr(pipeline, "verify_model", lambda *args, **kwargs: None)
    assert pipeline._match_model_candidates(model, [], model.cloud, cfg, np.array([0,1,0]),
        .4, prepared_constellations=[object()]) == []
    assert pipeline._match_model_candidates(model, [], model.cloud, cfg, np.array([0,1,0]),
        .4, prepared_constellations=[]) == []


def test_debug_chain_preserves_groups_poses_and_selection(tmp_path, monkeypatch):
    up = np.array([0., 1., 0.])
    points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,1.], [1.,1.,1.]])
    cloud = PointCloud(points)
    scene = SimpleNamespace(cloud=cloud, up=up, ground=0., volume=None)
    features = [SimpleNamespace(position=p) for p in points]
    model = SimpleNamespace(name="chair_test", synset="03001627", features=features,
                            cloud=cloud, mesh_path="test.obj")
    monkeypatch.setattr(db_store, "model_files", lambda path: ["test.pkl"])
    monkeypatch.setattr(db_store, "load_model", lambda path: model)
    pose = GroundTransform(0., 1., np.zeros(3), up)
    calls = []
    def prepare(model, scan, cfg, up, trace, allow_group_pose_seed):
        calls.append(len(scan))
        assert allow_group_pose_seed
        trace.update(status="matched", correspondences=3, constellations=1)
        return [Constellation(pose, [], 1.)]
    monkeypatch.setattr(pipeline, "prepare_model_matching", prepare)
    def verify(*args, **kwargs):
        assert len(args[1]) == 3
        assert len(kwargs["prepared_constellations"]) == 1
        return [Registration("chair_test", pose, 1., 0., 1., 1., 1.)]
    monkeypatch.setattr(pipeline, "_match_model_candidates", verify)
    import progressive
    def revalidate(regs, *args):
        for reg in regs:
            reg._footprint = points
            reg._model_index = 0
        return regs, [{"status": "kept"}]
    monkeypatch.setattr(progressive, "_final_revalidate", revalidate)
    path = tmp_path / "query.pkl"
    path.write_bytes(pickle.dumps(dict(scene=scene, features=features, candidate_indices=[0],
        feature_groups=[dict(id=7, feature_indices=[0,1,2])],
        candidate_group_details={0: dict(best_group=0, selected_for_groups=[0])})))
    results = []
    for stage in ("matching", "verification", "selection"):
        output = tmp_path / f"{stage}.pkl"
        result = debug._scene_result(dict(id="test", path=str(path)), retrieval.defaults(stage),
                                     stage=stage, artifact_path=output)
        assert output.exists()
        assert result["artifacts"][stage]["path"] == str(output.resolve())
        assert result["candidate_details"]
        assert all(step.get("explanation") for step in result["steps"])
        json.dumps(debug.json_compatible(result), allow_nan=False)
        results.append(result)
        path = output
    assert calls == [3]
    assert results[-1]["metrics"]["retained"] == 1
    final = pickle.loads(path.read_bytes())
    assert final["selected_registrations"][0]._query_group_id == 7


@pytest.mark.parametrize("stage,key", [("matching","candidate_indices"),
    ("verification","matching_records"), ("selection","verified_registrations")])
def test_empty_stage_and_incompatible_source(tmp_path, monkeypatch, stage, key):
    monkeypatch.setattr(db_store, "model_files", lambda path: [])
    source = tmp_path / "input.pkl"
    source.write_bytes(pickle.dumps({}))
    with pytest.raises(ValueError, match="(?i)incompatible"):
        retrieval.scene_result(dict(path=str(source)), retrieval.defaults(stage), stage)
    scene = SimpleNamespace(cloud=PointCloud(np.zeros((0,3))), up=np.array([0,1,0]), ground=0.)
    source.write_bytes(pickle.dumps(dict(scene=scene, **{key: []})))
    result = retrieval.scene_result(dict(path=str(source)), retrieval.defaults(stage), stage)
    assert result["metrics"]["retained"] == 0
    assert result["top_k"] == []


def test_frontend_exposes_all_stages():
    from web_assets import web_asset
    text = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    assert 'verification:"selection"' in text
    assert 'candidate_details' in text


def test_final_preview_keeps_dense_geometry_rgb_and_recorded_pose(tmp_path, monkeypatch):
    up = np.array([0., 1., 0.])
    points = np.random.default_rng(7).random((2400, 3))
    colors = np.random.default_rng(8).random((2400, 3))
    scene = SimpleNamespace(cloud=PointCloud(points, colors=colors), up=up, ground=.25)
    features = [SimpleNamespace(position=p) for p in points[:8]]
    model = SimpleNamespace(name="chair_test", synset="03001627", cloud=PointCloud(points), features=features)
    pose = GroundTransform(.3, 1.2, np.array([1., 2., 3.]), up)
    retained = Registration("chair_test", pose, 1., 0., 1., 1., 1.)
    rejected = Registration("chair_test", GroundTransform(0., 1., np.zeros(3), up), .5, 0., .5, .5, .5)
    for reg in (retained, rejected):
        reg._model_index = 0
    monkeypatch.setattr(db_store, "model_files", lambda path: ["test.pkl"])
    monkeypatch.setattr(db_store, "load_model", lambda path: model)
    monkeypatch.setattr(pipeline, "_non_max_select_fp_with_diagnostics", lambda regs, *args, **kwargs:
        ([regs[0]], [{"status": "selected"}, {"status": "rejected_explained_overlap"}]))
    monkeypatch.setattr(pipeline, "_registration_rank", lambda reg: -reg.coverage)
    source = tmp_path / "verified.pkl"
    source.write_bytes(pickle.dumps(dict(scene=scene, features=features, verified_registrations=[retained, rejected])))
    output = tmp_path / "selection.pkl"
    result = retrieval.scene_result(dict(path=str(source)), retrieval.defaults("selection"), "selection", artifact_path=output)
    layers = {layer["id"]: layer for layer in result["layers"]}
    assert layers["surface"]["kind"] == "surface"
    np.testing.assert_allclose(layers["surface"]["rgb_colors"], colors, atol=.00051)
    assert len(layers["pose-0"]["points"]) == len(points) > 360
    np.testing.assert_allclose(layers["pose-0"]["points"], debug._to_display(pose.apply(points), up, scene.ground), atol=.000051)
    np.testing.assert_allclose(layers["pose-0-keypoints"]["points"], debug._to_display(pose.apply(points[:8]), up, scene.ground), atol=.000051)
    assert set(result["steps"][-1]["visible_layers"]) == {"surface", "pose-0", "pose-0-keypoints"}
    saved = pickle.loads(output.read_bytes())["selected_registrations"]
    assert len(saved) == 1
    np.testing.assert_array_equal(saved[0].transform.t, pose.t)
    assert saved[0].transform.theta == pose.theta
    assert saved[0].transform.scale == pose.scale


def test_retained_mesh_preview_uses_database_normalization_without_virtual_ground(monkeypatch):
    import trimesh
    import database
    import viz

    mesh = trimesh.creation.box(extents=[2., 4., 2.])
    original_vertices = np.asarray(mesh.vertices).copy()
    model = SimpleNamespace(mesh_path="source.obj", synset="03001627",
                            cloud=PointCloud(np.array([[8., 0., 8.]])))
    monkeypatch.setattr(viz, "_load_mesh", lambda path: mesh.copy())
    def forbidden(*args, **kwargs):
        pytest.fail("A visual model must not add descriptor-only virtual ground")
    monkeypatch.setattr(database, "_virtual_scan", forbidden)
    first, metadata = retrieval._retained_model_preview(model, 600)
    second, _ = retrieval._retained_model_preview(model, 600)
    assert metadata["geometry_source"] == "normalized-mesh"
    assert len(first) == 600
    assert np.all(np.abs(first[:, [0, 2]]) <= .225 + 1e-10)
    assert np.all((first[:, 1] >= -1e-10) & (first[:, 1] <= .9 + 1e-10))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(model.cloud.points, [[8., 0., 8.]])
