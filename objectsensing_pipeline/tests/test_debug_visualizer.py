import os
import sys
import json
import re
import threading
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import debug_visualizer as debug  # noqa: E402
import geometry as G  # noqa: E402
from descriptor_utility import DescriptorUtility  # noqa: E402
from keypoints import KeyPoints  # noqa: E402
from web_assets import web_asset  # noqa: E402


def _cube_surface(n=180, seed=4):
    rng = np.random.default_rng(seed)
    faces = []
    for axis in range(3):
        for sign in (-0.5, 0.5):
            uv = rng.uniform(-0.5, 0.5, (n, 2))
            points = np.zeros((n, 3))
            other = [value for value in range(3) if value != axis]
            points[:, other] = uv
            points[:, axis] = sign
            faces.append(points)
    return np.vstack(faces)


def test_feature_groups_follow_retained_descriptor_positions():
    keypoints = SimpleNamespace(positions=np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
    ]))
    features = [
        SimpleNamespace(position=np.asarray([2.0, 0.0, 0.0])),
        SimpleNamespace(position=np.asarray([0.0, 0.0, 0.0])),
    ]
    groups = [{"id": 7, "member_indices": [0, 1, 2]}]

    result = debug._feature_groups_from_keypoints(
        groups, keypoints, features,
    )

    assert result == [{
        "id": 7, "member_indices": [0, 1, 2],
        "feature_indices": [0, 1], "descriptor_count": 2,
    }]


def test_keypoint_plugin_emits_generic_scene_schema(tmp_path, monkeypatch):
    points = _cube_surface()
    cloud = G.make_point_cloud(points, radius=0.12)
    source_path = tmp_path / "cube.npz"
    np.savez_compressed(
        source_path, points=cloud.points, normals=cloud.normals,
        curvature=cloud.curvature, up=np.asarray([0.0, 1.0, 0.0]),
        ground=np.asarray(-0.5),
    )
    monkeypatch.setattr(debug, "CACHE_ROOT", tmp_path / "cache")
    parameters = debug.default_parameters()
    parameters.update({
        "neighbor_radius": 0.18, "curvature_threshold": 0.02,
        "nms_radius": 0.20, "dedup_radius": 0.10,
        "surface_preview_points": 500, "layer_preview_points": 300,
        "max_scan_keypoints": 40,
    })

    result = debug._scene_result({
        "id": "cube", "label": "Cube", "path": str(source_path),
        "enabled": True,
    }, parameters)

    assert result["schema_version"] == 1
    assert result["stage"] == "keypoints"
    assert [step["id"] for step in result["steps"]] == [
        item[0] for item in debug.KEYPOINT_STEPS
    ]
    assert {layer["id"] for layer in result["layers"]} >= {
        "surface", "wall_surface", "wall_keypoint_affinity",
        "wall_plane_bounds", "curvature_candidates", "harris_accepted",
        "corner_2d_accepted", "nms_kept", "adjustment_vectors",
        "detected_final", "wall_budget_rejected",
        "component_budget_rejected", "retained",
    }
    assert {"wall_affinity", "wall_score", "object_score", "object_anchor"} <= set(
        result["histograms"]
    )
    assert result["metrics"]["retained"] <= 40
    assert min(point[1] for point in result["layers"][0]["points"]) >= -0.01
    assert all(step.get("description") for step in result["steps"])
    assert all(step.get("assessment") for step in result["steps"])


def test_keypoint_step_definitions_are_explanatory():
    definitions = [debug._step_definition(item) for item in debug.KEYPOINT_STEPS]
    assert all(item["label"] for item in definitions)
    assert all(item["description"] for item in definitions)
    assert all(item["group"] for item in definitions)
    assert all(item["assessment"] for item in definitions)


def test_normalized_payload_uses_enabled_existing_sources(tmp_path):
    source = tmp_path / "scene.npz"
    np.savez(source, points=np.zeros((1, 3)), normals=np.zeros((1, 3)),
             curvature=np.zeros(1))
    payload = debug.normalized_payload({
        "sources": [
            {
                "id": "one", "label": "One", "path": str(source),
                "enabled": True,
                "upstream": {
                    "run_id": "fusion-001", "stage": "fusion",
                    "scene_id": "one", "artifact": "geometry",
                },
            },
            {"id": "off", "label": "Off", "path": "missing", "enabled": False},
        ],
        "parameters": {"parallel_scenes": 1},
    })
    assert [item["id"] for item in payload["sources"]] == ["one"]
    assert payload["parameters"]["parallel_scenes"] == 1
    assert payload["sources"][0]["upstream"] == {
        "run_id": "fusion-001", "stage": "fusion",
        "scene_id": "one", "artifact": "geometry",
    }


def test_normalized_parameters_do_not_require_scene_sources():
    parameters = debug.normalized_parameters({
        "neighbor_radius": 0.075,
        "spatial_keypoint_balance": False,
    })

    assert parameters["neighbor_radius"] == 0.075
    assert parameters["spatial_keypoint_balance"] is False


def test_debug_defaults_include_current_quality_and_visibility_corrections():
    from config import PipelineConfig

    cfg = PipelineConfig()
    fusion = debug.default_parameters("fusion")
    keypoints = debug.default_parameters("keypoints")

    for field in (
        "voxel_size", "truncation", "frame_stride", "min_frames",
        "depth_edge_threshold", "visibility_depth_stride",
    ):
        assert fusion[field] == getattr(cfg.sdf, field)
    for field in (
        "planar_filter_enabled", "hole_boundary_filter_enabled",
        "repeatability_filter_enabled", "quality_response_ratio",
        "quality_score_radius", "geometric_nms_radius",
    ):
        assert keypoints[field] == getattr(cfg.keypoint, field)
    for field in (
        "descriptor_keypoint_filter_enabled", "descriptor_min_occupied",
        "descriptor_max_hole_fraction", "descriptor_keypoint_min_features",
        "quality_nms_radius", "quality_min_score_ratio",
    ):
        assert keypoints[field] == getattr(cfg.matching, field)


def test_worker_reports_dense_volume_limit_without_traceback(
        tmp_path, monkeypatch, capsys):
    from sdf_fusion import DenseVolumeLimitError

    payload = tmp_path / "job.json"
    output = tmp_path / "result.json"
    payload.write_text(json.dumps({
        "source": {"id": "office", "path": "office.zip"},
        "parameters": {},
        "stage": "fusion",
    }), encoding="utf-8")
    error = DenseVolumeLimitError(
        (1114, 787, 1078), 0.005, "cpu", 30_000_000, 0.02,
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(debug, "_scene_result", fail)

    with pytest.raises(SystemExit) as caught:
        debug._write_worker_result(payload, output)

    text = capsys.readouterr().out
    assert caught.value.code == 2
    assert "Configuration rejected" in text
    assert "945." in text
    assert 'Use at least 0.020 m' in text
    assert "Traceback" not in text
    assert not output.exists()


def test_fusion_stage_validates_rgbd_sources_and_parameters(tmp_path):
    source = tmp_path / "office.zip"
    source.write_bytes(b"zip")

    payload = debug.normalized_payload({
        "stage": "fusion",
        "sources": [{
            "id": "office", "label": "Office",
            "path": str(source), "enabled": True,
        }],
        "parameters": {
            "backend": "cpu", "frame_stride": 4, "max_frames": 20,
            "parallel_scenes": 1,
        },
    })

    assert payload["stage"] == "fusion"
    assert payload["parameters"]["backend"] == "cpu"
    assert payload["parameters"]["frame_stride"] == 4
    assert payload["sources"][0]["path"] == str(source.resolve())


def test_fusion_plugin_emits_progressive_frame_layers(
        tmp_path, monkeypatch):
    source = tmp_path / "office.zip"
    source.write_bytes(b"zip")
    cloud = G.make_point_cloud(_cube_surface(n=20), radius=0.15)
    cloud.confidence = np.linspace(1.0, 4.0, cloud.size)

    def fake_scan(_path, _cfg, progress=None, fusion_trace=None):
        visibility = np.asarray([[[0, 1], [2, 0]]], dtype=np.uint8)
        base_volume = SimpleNamespace(
            visibility=visibility,
            weight=np.asarray([[[0.0, 1.0], [1.0, 0.0]]]),
            origin=np.asarray([-0.1, -0.1, -0.1]),
            voxel_size=0.02,
        )
        fusion_trace.update({
            "backend": "cpu",
            "source_total_frames": 16,
            "total_frames": 2,
            "volume": {
                "voxel_count": 4, "unknown_voxels": 2,
                "free_voxels": 1, "occupied_voxels": 1,
            },
            "frames": [
                {
                    "frame_id": "000001",
                    "points": [cloud.points[0].tolist(), [0.1, 0.0, 0.0]],
                    "colors": [[255, 0, 0], [0, 255, 0]],
                    "camera_position": [0.0, 0.5, 0.0],
                },
                {
                    "frame_id": "000002",
                    "points": [[0.2, 0.0, 0.0]],
                    "colors": [[0, 0, 255]],
                    "camera_position": [0.1, 0.5, 0.0],
                },
            ],
        })
        return SimpleNamespace(
            cloud=cloud, up=np.asarray([0.0, 1.0, 0.0]), ground=-0.5,
            alignment_rotation=np.eye(3),
            volume=SimpleNamespace(volume=base_volume),
        )

    monkeypatch.setattr("pipeline.scan_from_zip", fake_scan)
    artifact = tmp_path / "office.geometry.npz"
    result = debug._scene_result({
        "id": "office", "label": "Office", "path": str(source),
        "enabled": True,
    }, debug.default_parameters("fusion"), stage="fusion",
        artifact_path=artifact)

    assert result["stage"] == "fusion"
    assert [step["id"] for step in result["steps"]] == [
        "frame-selection", "frame-000001", "frame-000002", "volume",
        "surface-raw", "gravity", "ground", "normals", "curvature",
    ]
    assert result["steps"][0]["warning"] == "preview_only"
    assert result["steps"][1]["draw_counts"]["fusion_measurements_raw"] == 2
    assert result["steps"][2]["draw_counts"]["fusion_measurements_raw"] == 3
    layers = {layer["id"]: layer for layer in result["layers"]}
    assert layers["fusion_measurements_raw"]["frame_index"] == [1, 1, 2]
    assert len(layers["camera_path_raw"]["segments"]) == 2
    assert layers["volume_unknown"]["total"] == 2
    assert layers["volume_free"]["total"] == 1
    assert layers["volume_occupied"]["total"] == 1
    assert result["metrics"]["rgb_projection_distance"] == pytest.approx(0.09)
    assert result["metrics"]["rgb_surface_available"] is True
    assert len(layers["surface"]["rgb_colors"]) == layers["surface"]["sampled"]
    assert result["metrics"]["source_frames"] == 16
    assert result["metrics"]["integrated_frames"] == 2
    assert result["metrics"]["preview_only"] is True
    assert result["metrics"]["backend"] == "cpu"
    assert result["artifacts"]["geometry"]["path"] == str(artifact.resolve())
    assert result["artifacts"]["geometry"]["points"] == cloud.size
    saved = np.load(str(artifact))
    assert saved["points"].shape == cloud.points.shape
    assert saved["normals"].shape == cloud.normals.shape
    assert saved["curvature"].shape == cloud.curvature.shape
    assert np.allclose(saved["confidence"], cloud.confidence)
    assert saved["colors"].shape == cloud.points.shape
    assert np.allclose(saved["up"], [0.0, 1.0, 0.0])
    assert float(saved["ground"]) == pytest.approx(-0.5)


def test_rgb_samples_are_projected_onto_nearby_surface_points():
    colors, valid = debug._project_rgb_to_surface(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        max_distance=0.2,
    )

    assert valid.tolist() == [True, False]
    assert colors[0, 0] > colors[0, 1]
    assert np.allclose(colors[1], [0.57, 0.60, 0.62])


def test_debug_can_chain_fusion_artifacts_into_keypoint_sources(
        tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_id = "fusion-20260803-120000-deadbeef"
    folder = run_root / run_id
    folder.mkdir(parents=True)
    artifact = folder / "office.geometry.npz"
    np.savez_compressed(
        artifact, points=np.zeros((4, 3), np.float32),
        normals=np.zeros((4, 3), np.float32),
        curvature=np.zeros(4, np.float32),
        up=np.asarray([0.0, 1.0, 0.0], np.float32), ground=np.asarray(0.0),
    )
    result = {
        "stage": "fusion",
        "scene": {"id": "office", "label": "Office"},
        "artifacts": {"geometry": {
            "path": str(artifact), "format": "npz", "points": 4,
        }},
    }
    (folder / "office.json").write_text(
        json.dumps(result), encoding="utf-8",
    )
    (folder / "manifest.json").write_text(json.dumps({
        "id": run_id, "stage": "fusion", "created_at": 42,
        "status": "completed", "sources": [],
        "scenes": {"office": {
            "file": "office.json", "status": "completed",
            "artifacts": result["artifacts"],
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(debug, "RUN_ROOT", run_root)
    manager = object.__new__(debug.DebugVisualizerManager)

    chained = manager.chain_sources(run_id)
    recent = manager.recent_runs()

    assert chained["target_stage"] == "keypoints"
    assert chained["sources"] == [{
        "id": "office", "label": "Office", "path": str(artifact.resolve()),
        "enabled": True,
        "upstream": {
            "run_id": run_id, "stage": "fusion", "scene_id": "office",
            "artifact": "geometry",
        },
    }]
    assert recent[0]["chainable"] is True
    assert recent[0]["chainable_scene_count"] == 1


def test_debug_rejects_chaining_legacy_fusion_without_geometry(
        tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_id = "fusion-legacy"
    folder = run_root / run_id
    folder.mkdir(parents=True)
    (folder / "office.json").write_text(json.dumps({
        "stage": "fusion", "scene": {"id": "office", "label": "Office"},
    }), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({
        "id": run_id, "stage": "fusion", "status": "completed",
        "scenes": {"office": {"file": "office.json"}},
    }), encoding="utf-8")
    monkeypatch.setattr(debug, "RUN_ROOT", run_root)
    manager = object.__new__(debug.DebugVisualizerManager)

    with pytest.raises(ValueError, match='Rerun Fusion'):
        manager.chain_sources(run_id)
    assert manager.recent_runs()[0]["chainable"] is False


def test_debug_can_chain_keypoints_into_descriptor_sources(
        tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_id = "keypoints-20260804-120000-feedface"
    folder = run_root / run_id
    folder.mkdir(parents=True)
    artifact = folder / "office.keypoints.pkl"
    artifact.write_bytes(b"keypoints")
    result = {
        "stage": "keypoints",
        "scene": {"id": "office", "label": "Office"},
        "artifacts": {"keypoints": {
            "path": str(artifact), "format": "pickle", "count": 40,
        }},
    }
    (folder / "office.json").write_text(
        json.dumps(result), encoding="utf-8",
    )
    (folder / "manifest.json").write_text(json.dumps({
        "id": run_id, "stage": "keypoints", "created_at": 42,
        "status": "completed", "sources": [],
        "scenes": {"office": {
            "file": "office.json", "status": "completed",
            "artifacts": result["artifacts"],
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(debug, "RUN_ROOT", run_root)
    manager = object.__new__(debug.DebugVisualizerManager)

    chained = manager.chain_sources(run_id, target_stage="descriptors")
    recent = manager.recent_runs()

    assert chained["target_stage"] == "descriptors"
    assert chained["sources"][0]["path"] == str(artifact.resolve())
    assert chained["sources"][0]["upstream"] == {
        "run_id": run_id, "stage": "keypoints", "scene_id": "office",
        "artifact": "keypoints",
    }
    assert recent[0]["chainable"] is True
    assert recent[0]["next_stage"] == "descriptors"


def test_debug_can_chain_descriptors_into_query_sources(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_id = "descriptors-20260811-120000-cafebabe"
    folder = run_root / run_id
    folder.mkdir(parents=True)
    artifact = folder / "office.descriptors.pkl"
    artifact.write_bytes(b"descriptors")
    result = {
        "stage": "descriptors",
        "scene": {"id": "office", "label": "Office"},
        "artifacts": {"descriptors": {
            "path": str(artifact), "format": "pickle", "count": 40,
        }},
    }
    (folder / "office.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({
        "id": run_id, "stage": "descriptors", "created_at": 42,
        "status": "completed", "sources": [],
        "scenes": {"office": {
            "file": "office.json", "status": "completed",
            "artifacts": result["artifacts"],
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(debug, "RUN_ROOT", run_root)
    manager = object.__new__(debug.DebugVisualizerManager)

    chained = manager.chain_sources(run_id, target_stage="query")
    recent = manager.recent_runs()

    assert chained["target_stage"] == "query"
    assert chained["sources"][0]["path"] == str(artifact.resolve())
    assert chained["sources"][0]["upstream"]["stage"] == "descriptors"
    assert recent[0]["chainable"] is True
    assert recent[0]["next_stage"] == "query"


def test_descriptor_stage_emits_utility_layers(tmp_path, monkeypatch):
    cloud = G.make_point_cloud(_cube_surface(n=4), radius=0.20)
    keypoints = KeyPoints(
        positions=cloud.points[:3], normals=cloud.normals[:3],
        responses=np.asarray([0.9, 0.8, 0.7]),
        source_index=np.asarray([0, 1, 2]),
    )
    source_path = tmp_path / "office.keypoints.pkl"
    with source_path.open("wb") as stream:
        import pickle
        pickle.dump({"scene": SimpleNamespace(
            cloud=cloud, volume=SimpleNamespace(),
            up=np.asarray([0.0, 1.0, 0.0]), ground=-0.5,
        ), "keypoints": keypoints, "keypoint_groups": [{
            "id": 7,
            "member_indices": [0, 1],
            "source_indices": [0, 1],
            "centroid": keypoints.positions[:2].mean(axis=0).tolist(),
            "extent": [0.2, 0.2, 0.2],
            "score": 0.75,
        }]}, stream)

    occ = np.full((4, 4, 4), 2, np.int8)
    occ[1:3, 1:3, 1:3] = 0
    occ[2, 2, 2] = 1
    descriptor = SimpleNamespace(
        res=4, extent=0.20, occ=occ,
        n_occupied=1, n_unknown=56, n_total=64,
    )
    features = [SimpleNamespace(
        position=keypoints.positions[index],
        response=keypoints.responses[index], descriptor=descriptor,
    ) for index in range(3)]
    models = [
        SimpleNamespace(name=f"model-{index}", synset="chair")
        for index in range(3)
    ]
    utilities = [
        DescriptorUtility(
            "informative", 10.0, 30.0, 0.66, 1, 5, 0.1,
            "model-0", "chair",
        ),
        DescriptorUtility(
            "ambiguous", 12.0, 12.5, 0.04, 8, 24, 0.98,
            "model-1", "chair",
        ),
        DescriptorUtility(
            "no_match", 180.0, 190.0, 0.05, 0, 2, 0.0,
            "model-2", "chair",
        ),
    ]
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "candidate_index.npz").write_bytes(b"index")
    monkeypatch.setattr(
        debug, "RUNTIME_PATHS", SimpleNamespace(database_dir=db_dir),
    )
    monkeypatch.setattr("db_store.is_db_dir", lambda path: True)
    monkeypatch.setattr("db_store.model_files", lambda path: ["a", "b", "c"])
    monkeypatch.setattr(
        "db_store.load_model", lambda path: models["abc".index(path)],
    )
    monkeypatch.setattr(
        "candidate_index.load_candidate_index", lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "candidate_index.rank_candidates", lambda *args, **kwargs: [0, 1, 2],
    )
    monkeypatch.setattr(
        "matching.build_scan_features", lambda *args, **kwargs: features,
    )
    monkeypatch.setattr(
        "descriptor_utility.profile_descriptor_utility",
        lambda *args, **kwargs: utilities,
    )

    result = debug._scene_result({
        "id": "office", "label": "Office", "path": str(source_path),
        "enabled": True,
    }, debug.default_parameters("descriptors"), stage="descriptors")

    assert result["stage"] == "descriptors"
    assert [step["id"] for step in result["steps"]] == [
        item[0] for item in debug.DESCRIPTOR_STEPS
    ]
    layers = {layer["id"]: layer for layer in result["layers"]}
    assert layers["descriptor_informative"]["total"] == 1
    assert layers["descriptor_ambiguous"]["total"] == 1
    assert layers["descriptor_no_match"]["total"] == 1
    assert layers["descriptor_raw_unknown"]["total"] == 56
    assert layers["descriptor_raw_free"]["total"] == 7
    assert layers["descriptor_raw_occupied"]["total"] == 1
    assert layers["descriptor_final_occupied"]["total"] == 1
    assert layers["descriptor_grid_bounds"]["total"] == 12
    assert layers["descriptor_groups"]["total"] == 2
    assert layers["descriptor_groups"]["group_id"] == [7, 7]
    assert layers["descriptor_group_links"]["total"] == 2
    assert result["steps"][0]["visible_layers"] == [
        "descriptor_context", "descriptor_grid_bounds", "descriptor_seed",
        "descriptor_raw_unknown", "descriptor_raw_free",
        "descriptor_raw_occupied",
    ]
    assert result["descriptor_focus"]["resolution"] == 4
    assert result["descriptor_focus"]["source_index"] == 0
    assert result["steps"][2]["kept_label"] == 'Grouped'
    assert result["steps"][4]["warning"] == "diagnostic_only"
    assert result["steps"][4]["kept_label"] == 'Correspondences'
    assert result["steps"][5]["kept_label"] == "Informative"
    assert "do not filter" in result["steps"][5]["explanation"]
    assert result["feature_groups"][0]["score"] == 0.75
    assert result["metrics"]["informative"] == 1
    assert result["metrics"]["ambiguous"] == 1
    assert result["metrics"]["no_match"] == 1
    assert "descriptor_distance" in result["histograms"]


def test_query_stage_builds_global_pool_and_local_top_k(tmp_path, monkeypatch):
    import pickle

    cloud = G.make_point_cloud(_cube_surface(n=4), radius=0.20)
    keypoints = KeyPoints(
        positions=cloud.points[:3], normals=cloud.normals[:3],
        responses=np.asarray([0.9, 0.8, 0.7]),
        source_index=np.asarray([0, 1, 2]),
    )
    features = [SimpleNamespace(
        position=keypoints.positions[index],
        response=float(keypoints.responses[index]),
    ) for index in range(3)]
    source_path = tmp_path / "office.descriptors.pkl"
    with source_path.open("wb") as stream:
        pickle.dump({
            "scene": SimpleNamespace(
                cloud=cloud, up=np.asarray([0.0, 1.0, 0.0]), ground=-0.5,
            ),
            "keypoints": keypoints, "features": features,
            "parameters": {},
        }, stream)

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "candidate_index.npz").write_bytes(b"index")
    monkeypatch.setattr(
        debug, "RUNTIME_PATHS", SimpleNamespace(database_dir=db_dir),
    )
    monkeypatch.setattr("db_store.is_db_dir", lambda path: True)
    monkeypatch.setattr(
        "db_store.model_files", lambda path: ["a", "b", "c"],
    )
    monkeypatch.setattr(
        "candidate_index.load_candidate_index",
        lambda path: SimpleNamespace(
            names=np.asarray(["chair-a", "table-b", "desk-c"]),
            synsets=np.asarray(["chair", "table", "table"]),
        ),
    )
    monkeypatch.setattr(
        "candidate_index.rank_candidates", lambda *args, **kwargs: [0, 1, 2],
    )
    monkeypatch.setattr(
        "candidate_index.rerank_candidates_by_local_features",
        lambda *args, **kwargs: [2, 0],
    )
    parameters = debug.default_parameters("query")
    parameters.update({"candidate_pool_k": 3, "candidate_top_k": 2})
    artifact = tmp_path / "office.query.pkl"

    result = debug._scene_result({
        "id": "office", "label": "Office", "path": str(source_path),
        "enabled": True,
    }, parameters, stage="query", artifact_path=artifact)

    assert result["stage"] == "query"
    assert [step["id"] for step in result["steps"]] == [
        item[0] for item in debug.QUERY_STEPS
    ]
    assert [item["name"] for item in result["candidate_pool"]] == [
        "chair-a", "table-b", "desk-c",
    ]
    assert [item["name"] for item in result["top_k"]] == [
        "desk-c", "chair-a",
    ]
    assert result["metrics"]["strategy"] == "two_level"
    assert result["metrics"]["top_k_models"] == 2
    assert artifact.exists()


def test_query_stage_ranks_each_keypoint_group_independently(
        tmp_path, monkeypatch):
    import pickle

    cloud = G.make_point_cloud(_cube_surface(n=4), radius=0.20)
    keypoints = KeyPoints(
        positions=cloud.points[:4], normals=cloud.normals[:4],
        responses=np.asarray([0.9, 0.8, 0.7, 0.6]),
        source_index=np.asarray([0, 1, 2, 3]),
    )
    features = [SimpleNamespace(
        position=keypoints.positions[index],
        response=float(keypoints.responses[index]),
    ) for index in range(4)]
    source_path = tmp_path / "office.grouped.descriptors.pkl"
    with source_path.open("wb") as stream:
        pickle.dump({
            "scene": SimpleNamespace(
                cloud=cloud, up=np.asarray([0.0, 1.0, 0.0]), ground=-0.5,
            ),
            "keypoints": keypoints, "features": features,
            "feature_groups": [
                {"id": 4, "feature_indices": [0, 1],
                 "source_indices": [0, 1], "centroid": [0, 0, 0],
                 "extent": [1, 1, 1], "descriptor_count": 2},
                {"id": 7, "feature_indices": [2, 3],
                 "source_indices": [2, 3], "centroid": [1, 0, 0],
                 "extent": [1, 1, 1], "descriptor_count": 2},
            ],
            "parameters": {},
        }, stream)

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "candidate_index.npz").write_bytes(b"index")
    monkeypatch.setattr(
        debug, "RUNTIME_PATHS", SimpleNamespace(database_dir=db_dir),
    )
    monkeypatch.setattr("db_store.is_db_dir", lambda path: True)
    monkeypatch.setattr("db_store.model_files", lambda path: ["a", "b", "c"])
    monkeypatch.setattr(
        "candidate_index.load_candidate_index",
        lambda path: SimpleNamespace(
            names=np.asarray(["chair-a", "table-b", "desk-c"]),
            synsets=np.asarray(["chair", "table", "table"]),
        ),
    )
    monkeypatch.setattr(
        "candidate_index.rank_candidates_by_feature_groups",
        lambda *args, **kwargs: ([0, 1, 2], [[0, 1, 2], [2, 1, 0]]),
    )
    monkeypatch.setattr(
        "candidate_index.rank_candidates",
        lambda *args, **kwargs: [1, 2, 0],
    )
    reranked_pools = []
    monkeypatch.setattr(
        "candidate_index.rerank_candidates_by_local_feature_groups",
        lambda *args, **kwargs: (reranked_pools.append(list(args[1])) or ([2, 0], {
            0: {"best_group": 0, "best_score": .8,
                "group_scores": [.8, .1]},
            1: {"best_group": 0, "best_score": .4,
                "group_scores": [.4, .3]},
            2: {"best_group": 1, "best_score": .9,
                "group_scores": [.2, .9]},
        })),
    )
    parameters = debug.default_parameters("query")
    parameters.update({
        "candidate_pool_k": 3, "candidate_top_k": 2,
        "candidate_min_group_descriptors": 2,
    })
    artifact = tmp_path / "office.grouped.query.pkl"

    result = debug._scene_result({
        "id": "office", "label": "Office", "path": str(source_path),
        "enabled": True,
    }, parameters, stage="query", artifact_path=artifact)

    assert result["metrics"]["strategy"] == "grouped_two_level"
    assert reranked_pools == [[0, 2, 1]]  # Group reservations precede global fill.
    assert result["metrics"]["global_pool_fraction"] == 0.70
    assert result["metrics"]["object_groups"] == 2
    assert len(result["metrics"]["group_weights"]) == 2
    assert [item["best_group_id"] for item in result["top_k"]] == [7, 4]
    assert [group["id"] for group in result["query_groups"]] == [4, 7]
    assert result["query_groups"][1]["top_k"][0]["name"] == "desk-c"
    assert artifact.exists()
    assert debug.default_parameters("query")["candidate_max_groups"] == 1
    assert debug.default_parameters("query")[
        "candidate_descriptor_ratio_threshold"
    ] == 0.97
    assert debug.default_parameters("query")[
        "candidate_extent_weight"
    ] == 0.90
    assert debug.default_parameters("query")[
        "candidate_extent_quota_fraction"
    ] == 0.70
    assert debug.default_parameters("query")[
        "candidate_min_group_descriptors"
    ] == 6
    assert debug.default_parameters("query")[
        "candidate_pool_min_per_group"
    ] == 20
    assert debug.default_parameters("query")[
        "candidate_top_min_per_group"
    ] == 6


def test_query_group_weights_favor_supported_high_quality_groups():
    records = [{"score": 4.0}, {"score": 0.1}]
    groups = [list(range(20)), list(range(6))]

    weights = debug._query_group_weights(records, groups)

    assert len(weights) == 2
    assert weights[0] > weights[1]
    assert np.isclose(np.mean(weights), 1.0)


def test_query_candidate_detail_places_model_and_exposes_matches(
        tmp_path, monkeypatch):
    import pickle

    class Pose:
        theta = 0.25
        scale = 1.2
        t = np.asarray([0.5, 0.0, -0.25])

        def apply(self, points):
            return np.asarray(points, float) * self.scale + self.t

    pose = Pose()
    model_features = [SimpleNamespace(position=np.asarray([0.0, 0.2, 0.0]))]
    scan_features = [SimpleNamespace(position=np.asarray([0.5, 0.24, -0.25]))]
    model = SimpleNamespace(
        name="chair-a", synset="03001627",
        cloud=SimpleNamespace(points=np.asarray([
            [0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.3, 0.0],
        ])),
        features=model_features, ground=0.0,
    )
    correspondence = SimpleNamespace(
        model_idx=0, scan_idx=0, theta=pose.theta, scale=pose.scale,
        desc_dist=0.2, transform=pose,
    )
    constellation = SimpleNamespace(
        transform=pose, inliers=[correspondence], quality=0.8,
    )
    artifact = tmp_path / "office.query.pkl"
    with artifact.open("wb") as stream:
        pickle.dump({
            "scene": SimpleNamespace(
                up=np.asarray([0.0, 1.0, 0.0]), ground=0.0,
            ),
            "features": scan_features,
            "candidate_pool_indices": [0],
            "feature_groups": [{"id": 7, "feature_indices": [0]}],
            "parameters": {},
        }, stream)
    monkeypatch.setattr("db_store.model_files", lambda path: ["chair.pkl"])
    monkeypatch.setattr("db_store.load_model", lambda path: model)
    monkeypatch.setattr(
        "candidate_index._local_descriptor_score",
        lambda *args, **kwargs: 0.125,
    )
    monkeypatch.setattr(
        "matching.match_features", lambda *args: [correspondence],
    )
    monkeypatch.setattr(
        "constellations.one_point_ransac", lambda *args: [constellation],
    )

    detail = debug._query_candidate_detail(
        artifact, 0, db_dir=tmp_path, group_id=7,
    )

    assert detail["name"] == "chair-a"
    assert detail["placed"] is True
    assert detail["pose"]["source"] == "constellation"
    assert detail["correspondences"] == 1
    assert detail["constellations"] == 1
    assert detail["group_id"] == 7
    assert detail["model_points"] == [[0.5, 0.24, -0.25], [0.74, 0.36, -0.25]]
    assert len(detail["correspondence_segments"]) == 1
    assert detail["matches"][0]["model_keypoint"] == 0


def test_descriptor_grid_visualizes_seed_fill_disconnections():
    cloud = G.make_point_cloud(_cube_surface(n=4), radius=0.20)
    occ = np.full((3, 3, 3), 2, np.int8)
    occ[1, 1, 1] = 1
    descriptor = SimpleNamespace(res=3, extent=0.15, occ=occ)
    feature = SimpleNamespace(
        position=np.zeros(3), descriptor=descriptor,
    )

    class VisibilityVolume:
        @staticmethod
        def sample_visibility(points):
            visibility = np.zeros(len(points), np.uint8)
            visibility[0] = 2
            visibility[13] = 2
            return visibility

        @classmethod
        def sample_visibility_details(cls, points):
            return cls.sample_visibility(points), np.zeros(len(points), bool)

    layers, focus = debug._descriptor_grid_layers(
        feature, VisibilityVolume(), cloud, 42,
        np.asarray([0.0, 1.0, 0.0]), -0.5,
    )
    by_id = {layer["id"]: layer for layer in layers}

    assert focus["raw_occupied"] == 2
    assert focus["final_occupied"] == 1
    assert focus["seed_removed"] == 1
    assert by_id["descriptor_seed_removed"]["total"] == 1
    assert by_id["descriptor_seed_removed"]["source_index"] == [42]


def test_hidden_visualizer_overlays_are_not_forced_visible():
    css = web_asset("debug_visualizer.css").read_text(encoding="utf-8")
    compact = "".join(css.split())
    assert "[hidden]{display:none!important}" in compact


def test_descriptor_diagnostic_labels_support_historical_runs():
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert 'step?.id==="distance"' in script
    assert 'kept:"Correspondences",rejected:"No correspondence"' in script
    assert 'step?.id==="ambiguity"' in script
    assert 'kept:"Informative",rejected:"Others"' in script


def test_every_visualizer_parameter_explains_its_impact():
    expected_fields = {"summary", "detail", "higher", "lower", "cost"}
    for stage, help_text in (
        ("keypoints", debug.PARAMETER_HELP),
        ("fusion", debug.FUSION_PARAMETER_HELP),
        ("descriptors", debug.DESCRIPTOR_PARAMETER_HELP),
        ("query", debug.QUERY_PARAMETER_HELP),
    ):
        schema = debug.parameter_schema(stage)
        assert {item["id"] for item in schema} == set(help_text)
        for item in schema:
            assert expected_fields <= set(item)
            assert all(item[field].strip() for field in expected_fields)


def test_fusion_defaults_use_measured_balanced_profile():
    parameters = debug.default_parameters("fusion")

    assert parameters["voxel_size"] == 0.015
    assert parameters["truncation"] == 0.060
    assert parameters["frame_stride"] == 6
    assert parameters["min_frames"] == 50
    assert parameters["max_frames"] == 0
    assert parameters["depth_trunc"] == 4.5
    assert parameters["iso_sampling"] == 0.01
    assert parameters["max_surface_points"] == 175000
    assert parameters["backend"] == "auto"
    assert parameters["volume_layout"] == "auto"


def test_debug_layout_exposes_persistent_resize_handles():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    for handle_id in (
        "left-panel-resizer", "right-panel-resizer",
        "debug-timeline-resizer",
    ):
        assert f'id="{handle_id}"' in html
    assert html.count('role="separator"') >= 3
    for storage_key in (
        "objectsensing-debug-panel-left",
        "objectsensing-debug-panel-right",
        "objectsensing-debug-timeline",
    ):
        assert storage_key in script


def test_debug_timeline_exposes_named_steps_without_redundant_slider():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    css = web_asset("debug_visualizer.css").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert 'id="step-slider"' not in html
    assert "#step-slider" not in css
    assert '$("#step-slider")' not in script
    assert "major-step" in script
    assert 'id:"frames",label:"RGB-D integration"' in script
    assert 'sourceSelector:"last-fusion-frame"' in script
    assert "function resultStepForDefinition(" in script
    assert "resultStepForDefinition(result,definition)" in script
    assert ".chapter-track.dense .chapter-node span{display:none}" not in css
    assert ".chapter-track.dense .chapter-node{flex:0 0 72px}" in css
    assert "function revealTimelineStep(" in script
    assert "scrollIntoView" not in script
    assert 'setAttribute("aria-current","step")' in script
    assert 'event.key==="Home"' in script
    assert 'event.key==="End"' in script


def test_debug_exposes_live_and_historical_execution_logs(tmp_path, monkeypatch):
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    run_root = tmp_path / "runs"
    run_id = "fusion-log-test"
    folder = run_root / run_id
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({
        "id": run_id, "stage": "fusion", "created_at": 42,
        "status": "completed", "scenes": {},
    }), encoding="utf-8")
    monkeypatch.setattr(debug, "RUN_ROOT", run_root)
    manager = object.__new__(debug.DebugVisualizerManager)
    manager.lock = threading.RLock()
    manager.log_lines = []
    manager.events = []
    manager.next_event_id = 1
    manager.state = {"scenes": {"office": {}}}
    worker_logs = []

    manager._stream_worker_output(
        SimpleNamespace(stdout=[
            "fusion TSDF: trame 1/20\n",
            "fusion TSDF: trame 2/20\n",
        ]),
        run_id, "fusion", "office", worker_logs, folder / "debug.log",
    )
    loaded = manager.load_result(run_id)

    for element_id in (
        "toggle-debug-logs", "debug-log-drawer", "debug-log-count",
        "clear-debug-logs", "debug-logs",
    ):
        assert f'id="{element_id}"' in html
    assert "debug_log" in script
    assert "function appendDebugLog(" in script
    assert "payload.logs||[]" in script
    assert loaded["logs"] == worker_logs
    assert len(worker_logs) == 2
    assert worker_logs[0].endswith("fusion TSDF: trame 1/20")
    assert manager.state["scenes"]["office"]["message"] == (
        "fusion TSDF: trame 2/20"
    )
    assert manager.events[-1]["type"] == "debug_log"
    assert manager.events[-1]["data"]["message"] == worker_logs[-1]


def test_debug_panels_match_run_console_collapse_and_preview_behaviour():
    css = web_asset("debug_visualizer.css").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    shared_script = web_asset("ui_shared.js").read_text(encoding="utf-8")
    scene_script = web_asset("ui_scene3d.js").read_text(encoding="utf-8")

    for class_name in (
        "left-collapsed", "right-collapsed",
        "left-peeking", "right-peeking",
    ):
        assert class_name in css
        assert class_name in script
    assert "bindHoverPanels" in script
    assert "bindTooltips" in script
    assert 'href="/ui_shared.css"' in html
    assert 'class="tooltip ui-tooltip"' in html
    assert 'from "./ui_shared.js"' in script
    assert "bindHoverPreviewPanels" in shared_script
    assert "bindResizableDimension" in shared_script
    assert "installSharedPrimitives();" in script
    assert 'from "./ui_scene3d.js"' in script
    assert "createSceneViewport" in scene_script
    assert "createTooltipController" in shared_script
    assert "bindMultiSelectListInteractions" in shared_script
    assert "bindMultiSelectListInteractions({container:\"#query-candidate-list\"" in script
    assert 'id="query-candidate-search"' in html
    assert 'id="query-candidate-category"' in html
    assert 'id="query-candidate-status"' in html
    assert 'id="query-candidate-group"' in html
    assert "createCandidateFilterController" in script
    assert "function allQueryCandidates()" in script
    assert "function syncQueryGroupFilter()" in script
    assert 'queryGroupByScene:new Map()' in script
    assert "Pool rank" in script
    assert ":is(:hover,:focus-within)" in css
    assert "aria-expanded" in script


def test_debug_camera_sync_waits_for_complete_viewers():
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    scene_script = web_asset("ui_scene3d.js").read_text(encoding="utf-8")

    constructor_end = script.index(
        'this.controls.addEventListener("change",()=>syncCameraFrom(this));'
    )
    viewport_assignment = script.index(
        "this.scene=this.viewport.scene;this.camera=this.viewport.camera;"
    )
    assert viewport_assignment < constructor_end
    assert "onControlsChange:()=>syncCameraFrom(this)" not in script
    assert "!source?.camera||!source?.controls" in script
    assert "!viewer?.camera||!viewer?.controls" in script
    sync_start = script.index("function syncCameraFrom")
    sync_end = script.index("function showPointDetail", sync_start)
    assert "try{" in script[sync_start:sync_end]
    assert "finally{state.syncing=false}" in script
    assert "frameId = requestAnimationFrame(animate);" in scene_script
    assert "running = true;\n    animate();" not in scene_script


def test_shared_3d_camera_keeps_scene_inside_the_view():
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    scene_script = web_asset("ui_scene3d.js").read_text(encoding="utf-8")
    stylesheet = web_asset("debug_visualizer.css").read_text(
        encoding="utf-8")

    assert "const horizontalFov = 2 * Math.atan(" in scene_script
    assert "const tanHorizontal = Math.max(Math.tan(horizontalFov / 2)" in scene_script
    assert "function fitObjects(objects, padding = 1.12" in scene_script
    assert "point.fromBufferAttribute(attribute, index)" in scene_script
    assert "frame.halfRight * safePadding / tanHorizontal" in scene_script
    assert "frame.halfUp * safePadding / tanVertical" in scene_script
    assert "camera.up.set(0, 1, 0)" in scene_script
    assert "padding = 1.12" in scene_script
    assert "updateClippingForBox(box)" in scene_script
    assert 'controls.addEventListener("change", () => updateClipping(root))' in scene_script
    assert "camera.near = nearest > 0" in scene_script
    assert "homeBox = frame.box.clone()" in scene_script
    assert "cameraWasHome && homeFrame" in scene_script
    assert "function isAtHome(tolerance = 1e-8)" in scene_script
    assert "isAtHome," in scene_script
    assert "this.viewport.fitBox(box,1.12,{remember:true})" in script
    assert 'const stageGridVisible=state.stage!=="fusion"' in script
    assert 'this.grid.visible=Boolean(settings.grid)&&stageGridVisible' in script
    assert "const followHome=this.hasFit&&this.viewport.isAtHome()" in script
    assert "this.viewport.fitObjects(visible,1.12,{remember:true})" in script
    assert 'grid.dataset.sceneCount=String(sceneCount);' in script
    assert 'grid.style.setProperty("--scene-count",String(Math.min(3,sceneCount)));' in script
    assert "repeat(var(--scene-count,1),minmax(0,1fr))" in stylesheet
    assert script.index("const sceneCount=Math.max(1,state.viewers.size);") > (
        script.index("function ensureSceneCards(sources)"))


def test_scene_display_controls_bind_their_event_type():
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert "input.addEventListener(eventName,event=>" in script
    assert "input.addEventListener(event=>" not in script


def test_loading_debug_history_restores_its_parameters():
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert "function applyManifestInputs(manifest)" in script
    assert "applyManifestInputs(payload.manifest)" in script
    assert 'parameters[input.dataset.parameter]' in script


def test_debug_normalizes_partial_results_before_reading_sources():
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert "function defaultSources()" in script
    assert "function normalizeDebugResult(payload)" in script
    assert "manifest.sources=Array.isArray(manifest.sources)" in script
    assert "function applyCurrentResults(rawPayload" in script
    assert "const payload=normalizeDebugResult(rawPayload);" in script
    assert "ensureSceneCards(payload.manifest.sources);" in script
    assert "payload.manifest.sources||state.defaults.sources" not in script
    assert "initialRunId=currentMeta?.stage===state.stage" in script
    assert "if(initialRunId){" in script
    assert "await loadRun(initialRunId)" in script
    assert "loading ${initialRunId}…" in script
    assert "if(state.defaults.baseline_run_id)loadBaseline(" in script
    assert script.index('const snapshot=await api("/api/debug/state")') < (
        script.index("await loadRun(initialRunId)"))
    assert ".catch(()=>{})" not in script


def test_debug_json_compatibility_replaces_non_finite_values():
    payload = {
        "finite": 1.5,
        "values": [float("nan"), float("inf"), -float("inf")],
        "numpy": np.float32("nan"),
        "integer": np.int64(7),
    }

    compatible = debug.json_compatible(payload)

    assert compatible == {
        "finite": 1.5,
        "values": [None, None, None],
        "numpy": None,
        "integer": 7,
    }
    assert "NaN" not in json.dumps(compatible, allow_nan=False)


def test_debug_renders_all_wall_filter_parameters():
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert '["Vertical plane detection",[' in script
    assert '["Wall scoring and filtering",[' in script
    for parameter in (
        "wall_vertical_normal_cos", "wall_plane_distance",
        "wall_plane_min_area", "wall_small_radius",
        "wall_protrusion_distance", "wall_filter_enabled",
        "wall_penalty_weight", "wall_hard_reject",
        "wall_reject_threshold", "wall_reject_max_object_score",
        "wall_budget_enabled", "wall_budget_affinity_threshold",
        "wall_budget_proximity_threshold",
        "wall_budget_max_fraction", "wall_budget_cell_size",
        "wall_budget_max_per_cell", "wall_budget_min_features",
        "component_budget_enabled", "component_budget_radius",
        "component_budget_max_per_component",
        "component_budget_min_features",
    ):
        assert f'"{parameter}"' in script


def test_debug_module_url_is_versioned_for_running_consoles():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    version = re.search(r'const debugClientVersion="([^"]+)"', script).group(1)
    assert version
    assert f'src="/debug_visualizer.js?v={version}"' in html


def test_query_candidate_controls_expose_desktop_and_touch_multiselect():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert 'id="query-show-all"' in html
    assert 'id="query-hide-all"' in html
    assert "event.shiftKey" in script
    assert "event.ctrlKey||event.metaKey" in script
    assert "bindMultiSelectListInteractions" in script
    assert "onPaintEnd" in script
    assert "queueQueryCandidateLoads" in script
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    assert 'document.documentElement.dataset.debugClientVersion=' in script
    assert "function replaceNonFiniteJsonNumbers(text)" in script
    assert "const payload=parseApiPayload(await response.text(),path);" in script


def test_debug_uses_config_defaults_without_a_recommended_fusion_profile():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    css = web_asset("debug_visualizer.css").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert "function renderFusionProfile(" not in script
    assert "function fusionProfileSummary(" not in script
    assert "Profil recommandé" not in html
    assert '"depth_edge_background_weight"' in script
    assert '"max_surface_points"' in script
    assert "voxel 25 mm" not in script
    assert "objectsensing-debug-stage" in script
    assert "localStorage.setItem(stageStorage,stage)" in script
    assert "#fusion-profile-state" not in css
    assert "without an additional Debug preset" in html


def test_debug_runs_keep_label_and_favorite_outside_manifest(
        tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_id = "debug-keypoints-001"
    folder = run_root / run_id
    folder.mkdir(parents=True)
    manifest = {
        "id": run_id, "stage": "keypoints", "created_at": 42,
        "status": "completed", "scenes": {},
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    monkeypatch.setattr(debug, "RUN_ROOT", run_root)
    manager = object.__new__(debug.DebugVisualizerManager)
    manager._emit = lambda *args, **kwargs: None

    manager.set_favorite(run_id, True)
    renamed = manager.rename_run(run_id, "Office - murs réduits")
    recent = manager.recent_runs()

    assert renamed["manifest"]["label"] == "Office - murs réduits"
    assert renamed["manifest"]["favorite"] is True
    assert recent[0]["label"] == "Office - murs réduits"
    assert recent[0]["favorite"] is True
    assert json.loads((folder / "manifest.json").read_text(
        encoding="utf-8",
    )) == manifest


def test_debug_history_exposes_rename_and_favorite_controls():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    css = web_asset("debug_visualizer.css").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert 'id="debug-rename-dialog"' in html
    assert 'id="debug-rename-form"' in html
    assert 'api("/api/debug/run/rename"' in script
    assert 'api("/api/debug/run/favorite"' in script
    assert "function toggleDebugFavorite(" in script
    assert ".history-item.favorite" in css
    assert ".favorite-action.active" in css


def test_debug_history_tooltip_describes_the_run():
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert 'load.dataset.tooltip="Charger cette exécution"' not in script
    assert 'load.dataset.tooltip=run.label||run.id' in script
    assert "Stage: ${runStage}" in script
    assert "Status: ${run.status||\"unknown\"}" in script
    assert "ID: ${run.id}" in script
    assert "Can chain to ${stageLabel(run.next_stage)}" in script


def test_point_legend_controls_every_rendered_layer_role():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    for role in (
        "surface", "fusion", "unknown", "free", "occupied", "normal",
        "candidate", "wall", "accepted", "retained", "rejected", "baseline",
        "vector",
    ):
        assert f'data-layer-role="{role}"' in html
    assert "objectsensing-debug-layer-visibility" in script
    assert "layerVisible(role)" in script
    assert 'setLayerVisibility("rejected"' in script
    assert 'setLayerVisibility("baseline"' in script


def test_debug_visualizer_uses_local_browser_dependencies():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")

    assert "https://unpkg.com" not in html
    assert "https://cdn.jsdelivr.net" not in html
    assert 'src="/vendor/lucide/lucide.min.js"' in html
    assert '"/vendor/three/three.module.js"' in html


def test_large_pipeline_chapters_only_exist_in_debug_timeline():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")

    assert 'id="pipeline-tabs"' not in html
    assert 'id="stage-list"' not in html
    assert 'id="debug-stage-track"' in html
    assert 'id="chapter-track"' in html
    assert 'id="step-title"' in html


def test_debug_stage_navigation_and_switches_use_shared_contract():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    css = web_asset("debug_visualizer.css").read_text(encoding="utf-8")
    shared_css = web_asset("ui_shared.css").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")

    assert 'class="legend-toggle fusion-only"' in html
    assert "function selectStage(" in script
    assert "stage_steps_by_stage" in script
    assert "draw_counts" in script
    assert "transform:translateX(12px)" not in "".join(css.split())
    assert ".switch-row input:checked + .switch::after" in shared_css


def test_debug_ui_exposes_explicit_fusion_to_keypoint_chaining():
    html = web_asset("debug_visualizer.html").read_text(encoding="utf-8")
    css = web_asset("debug_visualizer.css").read_text(encoding="utf-8")
    script = web_asset("debug_visualizer.js").read_text(encoding="utf-8")
    server = (debug.ROOT / "run_console.py").read_text(encoding="utf-8")

    for element_id in (
        "upstream-fusion-run", "chain-fusion-run", "chain-status",
        "chain-current-fusion", "chain-current-status", "chain-target-stage",
    ):
        assert f'id="{element_id}"' in html
    assert 'api("/api/debug/chain"' in script
    assert "function chainFusionRun(" in script
    assert "sourceLineage:new Map()" in script
    assert "source.upstream=upstream" in script
    assert "legacy run without artifact" in script
    assert "function chainStageRun(" in script
    assert 'nextStageByStage={fusion:"keypoints",keypoints:"descriptors",descriptors:"query",query:"matching",matching:"verification",verification:"selection"}' in script
    assert "function continueRunUntil(" in script
    assert "function runToSelectedStage(" in script
    assert ".chain-section" in css
    assert 'self.path == "/api/debug/chain"' in server
    assert "DEBUG_MANAGER.chain_sources(" in server
