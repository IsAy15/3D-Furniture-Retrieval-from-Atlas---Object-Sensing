import json
import io
import os
import json
import pickle
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_events import LiveEventWriter, emit_progress_event  # noqa: E402
from pipeline import (  # noqa: E402
    _trace_detected_keypoints,
    _trace_retained_keypoints,
)
import run_console  # noqa: E402
from run_console import (  # noqa: E402
    ExclusiveThreadingHTTPServer,
    RunManager,
    _estimated_trace_ground,
    add_scene_catalog_entry,
    apply_debug_parameters_to_execution,
    browse_local_path,
    build_replay,
    execution_default_config,
    execution_paths,
    generated_run_name,
    normalized_config,
    pick_local_path,
    rename_run,
    resolve_step_ids,
    retrieval_command,
    runtime_health,
    run_details,
    scene_catalog,
    set_run_favorite,
)
from runtime_paths import RuntimePaths  # noqa: E402
from web_assets import web_asset  # noqa: E402


def _config(tmp_path):
    scene = tmp_path / "office.zip"
    scene.write_bytes(b"zip")
    db = tmp_path / "db"
    db.mkdir()
    (db / "index.json").write_text("{}", encoding="utf-8")
    return normalized_config({
        "scene_zip": str(scene),
        "db_dir": str(db),
        "candidate_index": str(db / "candidate_index.npz"),
        "run_root": str(tmp_path / "runs"),
        "jobs": 2,
    })


def test_run_console_defaults_use_measured_balanced_fusion_profile():
    config = run_console.default_config()

    assert config["voxel"] == 0.015
    assert config["truncation"] == 0.060
    assert config["max_surface_points"] == 175000
    assert config["frame_stride"] == 6
    assert config["min_frames"] == 50
    assert config["max_frames"] == 0
    assert config["fusion_backend"] == "auto"
    assert config["volume_layout"] == "auto"


def test_multi_scene_coordinator_runs_each_scene_sequentially(tmp_path):
    class FakeManager:
        def __init__(self):
            self.state = {"status": "idle"}
            self.worker = None
            self.started = []
            self.events = []

        def snapshot(self, include_live=False):
            result = dict(self.state)
            result["running"] = bool(self.worker and self.worker.is_alive())
            return result

        def start(self, config, steps):
            self.started.append((config["scene_zip"], list(steps)))
            self.state = {"status": "running", "run_id": Path(config["scene_zip"]).stem}

            def finish():
                time.sleep(0.01)
                self.state["status"] = "completed"

            self.worker = threading.Thread(target=finish)
            self.worker.start()

        def stop(self):
            return None

        def _publish(self, kind, data):
            self.events.append((kind, data))

    scene_a = tmp_path / "office.zip"
    scene_b = tmp_path / "single-chair.zip"
    scene_a.write_bytes(b"zip")
    scene_b.write_bytes(b"zip")
    manager = FakeManager()
    coordinator = run_console.MultiSceneRunCoordinator(manager)

    coordinator.start(
        _config(tmp_path), [str(scene_a), str(scene_b)], ["fusion"],
    )
    coordinator.worker.join(timeout=2)

    assert [Path(path).name for path, _ in manager.started] == [
        "office.zip", "single-chair.zip",
    ]
    assert coordinator.batch["completed"] == 2
    assert coordinator.batch["failed"] == 0
    assert coordinator.batch["status"] == "completed"
    assert manager.events[-1][0] == "batch_completed"


def test_multi_scene_controls_are_exposed_in_run_console():
    root = Path(__file__).parents[1]
    html = (root / "web" / "run-console" / "run_console.html").read_text(
        encoding="utf-8",
    )
    javascript = (root / "web" / "run-console" / "run_console.js").read_text(
        encoding="utf-8",
    )

    assert 'id="multi-scene-enabled"' in html
    assert 'id="multi-scene-list"' in html
    assert '"/api/run/batch"' in javascript


def test_run_console_server_refuses_port_reuse():
    assert ExclusiveThreadingHTTPServer.allow_reuse_address is False
    assert ExclusiveThreadingHTTPServer.allow_reuse_port is False


def test_generated_name_is_stable_and_parameter_sensitive(tmp_path):
    config = _config(tmp_path)
    first = generated_run_name(config)
    assert first == generated_run_name(dict(config))
    changed = dict(config, candidate_top_k=config["candidate_top_k"] + 1)
    assert first != generated_run_name(changed)
    assert "local_prog_k100-pool500" in first


def test_scene_catalog_discovers_and_persists_rgbd_zips(tmp_path, monkeypatch):
    scene_root = tmp_path / "scenes"
    scene_root.mkdir()
    office = scene_root / "office.zip"
    ikea = scene_root / "ikea-table.zip"
    custom = tmp_path / "meeting-room.zip"
    for path in (office, ikea, custom):
        path.write_bytes(b"zip")
    state_root = tmp_path / "work" / "run_console"
    monkeypatch.setattr("run_console.ROOT", tmp_path)
    monkeypatch.setattr("run_console.SCENE_ROOT", scene_root)
    monkeypatch.setattr("run_console.STATE_ROOT", state_root)
    monkeypatch.setattr(
        "run_console.SCENE_CATALOG_PATH", state_root / "scene_catalog.json",
    )

    before = scene_catalog()
    added = add_scene_catalog_entry("Meeting room", custom)
    after = scene_catalog()

    assert {item["label"] for item in before} >= {"Office", "IKEA Table"}
    assert added["scene"]["label"] == "Meeting room"
    assert added["scene"]["path"] == str(custom.resolve())
    assert any(item["path"] == str(custom.resolve()) for item in after)
    stored = json.loads(
        (state_root / "scene_catalog.json").read_text(encoding="utf-8"),
    )
    assert stored["scenes"] == [{
        "label": "Meeting room", "path": str(custom.resolve()),
    }]


def test_scene_catalog_rejects_missing_or_non_zip_files(tmp_path, monkeypatch):
    state_root = tmp_path / "work" / "run_console"
    monkeypatch.setattr("run_console.STATE_ROOT", state_root)
    monkeypatch.setattr(
        "run_console.SCENE_CATALOG_PATH", state_root / "scene_catalog.json",
    )
    invalid = tmp_path / "scene.ply"
    invalid.write_text("ply", encoding="ascii")

    with np.testing.assert_raises_regex(ValueError, r"\.zip"):
        add_scene_catalog_entry("Invalid", invalid)
    with np.testing.assert_raises_regex(ValueError, "not found"):
        add_scene_catalog_entry("Missing", tmp_path / "missing.zip")


def test_local_path_picker_validates_kind_extension_and_cancellation(
        tmp_path, monkeypatch):
    index = tmp_path / "candidate_index.npz"
    index.write_bytes(b"npz")
    (tmp_path / "nested").mkdir()
    monkeypatch.setattr("run_console.BROWSE_ROOTS", (tmp_path.resolve(),))

    listing = browse_local_path("file", "candidate_index", tmp_path)
    selected = pick_local_path("file", "candidate_index", index)

    assert selected == {"cancelled": False, "path": str(index.resolve())}
    assert {entry["name"] for entry in listing["entries"]} == {
        "candidate_index.npz", "nested",
    }
    assert listing["select_current"] is False
    assert pick_local_path("folder", "run_root", None) == {
        "cancelled": True, "path": None,
    }
    assert pick_local_path("folder", "run_root", tmp_path) == {
        "cancelled": False, "path": str(tmp_path.resolve()),
    }
    with np.testing.assert_raises_regex(ValueError, "picker type"):
        pick_local_path("unknown", "run_root", tmp_path)
    with np.testing.assert_raises_regex(ValueError, "incompatible"):
        browse_local_path("folder", "candidate_index", tmp_path)


def test_execution_folder_and_command_are_self_contained(tmp_path):
    config = _config(tmp_path)
    paths = execution_paths(config)
    paths["config"].write_text(json.dumps(config), encoding="utf-8")
    reused = execution_paths(config)
    assert paths["folder"] == reused["folder"]
    command = retrieval_command(config, paths)
    assert command[1:3] == [str(Path(__file__).parents[1] / "cli.py"), "run-progressive"]
    assert command[command.index("--out") + 1] == str(paths["aln"])
    assert command[command.index("--resume-dir") + 1] == str(paths["resume"])
    assert command[command.index("--min-frames") + 1] == "50"
    assert command[command.index("--volume-layout") + 1] == "auto"
    assert command[command.index("--max-surface-points") + 1] == "175000"
    assert command[command.index("--surface-extraction") + 1] == "hybrid"
    assert command[command.index("--surface-rescue-min-weight") + 1] == "5.0"
    assert command[command.index("--surface-field") + 1] == "raw"
    assert command[
        command.index("--visibility-depth-stride") + 1
    ] == "2"
    assert command[command.index("--live-events") + 1] == str(paths["events"])
    assert command[command.index("--stop-after") + 1] == "selection"
    assert command[command.index("--query-mode") + 1] == "single"
    assert command[command.index("--stage-cache-dir") + 1] == str(
        paths["stage_cache"],
    )
    assert command[command.index("--neighbor-radius") + 1] == "0.06"
    assert command[command.index("--harris-threshold") + 1] == "0.008"
    assert command[command.index("--min-2d-corner-fraction") + 1] == "0.25"
    assert command[command.index("--jitter-reject") + 1] == "auto"
    assert command[command.index("--wall-penalty-weight") + 1] == "0.12"
    assert command[command.index("--wall-budget-max-fraction") + 1] == "0.05"
    assert command[command.index("--wall-budget-proximity-threshold") + 1] == "0.05"
    assert command[command.index("--wall-budget-min-features") + 1] == "40"
    assert "--wall-budget" in command
    assert command[command.index("--component-budget-radius") + 1] == "0.2"
    assert command[
        command.index("--component-budget-max-per-component") + 1
    ] == "64"
    assert command[command.index("--component-budget-min-features") + 1] == "40"
    assert "--component-budget" in command
    assert command[command.index("--max-scan-keypoints") + 1] == "350"
    assert float(command[command.index("--curvature-threshold") + 1]) == 0.105
    assert float(command[command.index("--floor-height") + 1]) == 0.0
    assert command[command.index("--min-scan-inliers") + 1] == "3"
    assert command[command.index("--min-model-inliers") + 1] == "3"
    assert command[command.index("--min-scan-cells") + 1] == "3"
    assert command[command.index("--wall-reject-threshold") + 1] == "0.2"
    assert command[
        command.index("--wall-reject-max-object-score") + 1
    ] == "0.35"
    assert "--wall-filter" in command
    assert "--wall-hard-reject" in command
    assert "--spatial-keypoints" in command


def test_execution_command_can_disable_wall_filter(tmp_path):
    config = _config(tmp_path)
    config["wall_filter_enabled"] = False
    config["wall_hard_reject"] = False

    command = retrieval_command(config, execution_paths(config))

    assert "--no-wall-filter" in command
    assert "--no-wall-hard-reject" in command


def test_debug_parameters_are_persisted_for_future_executions(
        tmp_path, monkeypatch):
    state_root = tmp_path / "run_console"
    preset_path = state_root / "execution_parameters.json"
    monkeypatch.setattr("run_console.STATE_ROOT", state_root)
    monkeypatch.setattr(
        "run_console.EXECUTION_PARAMETERS_PATH", preset_path,
    )

    result = apply_debug_parameters_to_execution({
        "neighbor_radius": 0.075,
        "curvature_threshold": 0.082,
        "harris_threshold": 0.011,
        "spatial_keypoint_balance": False,
        "parallel_scenes": 3,
    })
    config = execution_default_config()

    assert result["applied"]["neighbor_radius"] == 0.075
    assert result["applied"]["spatial_keypoints"] is False
    assert {item["parameter"] for item in result["ignored"]} == {
        "surface_preview_points", "layer_preview_points", "parallel_scenes",
    }
    assert config["neighbor_radius"] == 0.075
    assert config["curvature_threshold"] == 0.082
    assert config["harris_threshold"] == 0.011
    assert config["spatial_keypoints"] is False
    assert "parallel_scenes" not in config
    stored = json.loads(preset_path.read_text(encoding="utf-8"))
    assert stored["source"] == "debug_visualizer"
    assert stored["parameters"]["neighbor_radius"] == 0.075


def test_fusion_debug_parameters_are_persisted_for_future_executions(
        tmp_path, monkeypatch):
    state_root = tmp_path / "run_console"
    preset_path = state_root / "execution_parameters.json"
    monkeypatch.setattr("run_console.STATE_ROOT", state_root)
    monkeypatch.setattr(
        "run_console.EXECUTION_PARAMETERS_PATH", preset_path,
    )

    result = apply_debug_parameters_to_execution({
        "voxel_size": 0.015,
        "truncation": 0.08,
        "frame_stride": 4,
        "min_frames": 60,
        "max_frames": 120,
        "backend": "cpu",
        "depth_trunc": 4.5,
        "iso_sampling": 0.01,
        "parallel_scenes": 2,
    }, stage="fusion")
    config = execution_default_config()

    assert {
        key: result["applied"][key]
        for key in (
            "voxel", "truncation", "frame_stride", "min_frames",
            "max_frames", "fusion_backend", "depth_trunc",
            "iso_sampling", "volume_layout",
        )
    } == {
        "voxel": 0.015,
        "truncation": 0.08,
        "frame_stride": 4,
        "min_frames": 60,
        "max_frames": 120,
        "fusion_backend": "cpu",
        "depth_trunc": 4.5,
        "iso_sampling": 0.01,
        "volume_layout": "auto",
    }
    assert {item["parameter"] for item in result["ignored"]} == {
        "parallel_scenes", "surface_preview_points",
        "volume_preview_points", "normal_preview_points",
    }
    assert config["voxel"] == 0.015
    assert config["fusion_backend"] == "cpu"


def test_descriptor_debug_parameters_are_persisted_for_future_executions(
        tmp_path, monkeypatch):
    state_root = tmp_path / "run_console"
    preset_path = state_root / "execution_parameters.json"
    monkeypatch.setattr("run_console.STATE_ROOT", state_root)
    monkeypatch.setattr(
        "run_console.EXECUTION_PARAMETERS_PATH", preset_path,
    )

    result = apply_debug_parameters_to_execution({
        "utility_distance_threshold": 7.0,
        "descriptor_ratio_threshold": 0.965,
    }, stage="descriptors")
    config = execution_default_config()

    assert result["applied"] == {
        "descriptor_distance_threshold": 7.0,
        "descriptor_ratio_threshold": 0.965,
    }
    assert config["descriptor_distance_threshold"] == 7.0
    assert config["descriptor_ratio_threshold"] == 0.965


def test_scientific_target_expands_ordered_dependencies():
    assert resolve_step_ids(["matching"]) == [
        "fusion", "keypoints", "query", "matching",
    ]
    assert resolve_step_ids(["validate", "query", "visualize"]) == [
        "validate", "fusion", "keypoints", "query",
    ]


def test_live_event_writer_flushes_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = LiveEventWriter(path)

    emit_progress_event(SimpleNamespace(event=writer.emit), "keypoints", {"retained": 42})
    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["type"] == "keypoints"
    assert event["data"]["retained"] == 42


def test_keypoint_trace_preserves_detected_and_retained_sets():
    detected = SimpleNamespace(
        positions=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        responses=np.asarray([0.9, 0.4]),
    )
    retained = [SimpleNamespace(
        position=np.asarray([1.0, 2.0, 3.0]), response=0.9, height=1.5,
    )]

    detected_trace = _trace_detected_keypoints(
        detected, np.asarray([0.0, 1.0, 0.0]), 0.5,
    )
    retained_trace = _trace_retained_keypoints(retained)

    assert [item["position"] for item in detected_trace] == [
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
    ]
    assert detected_trace[0]["height"] == 1.5
    assert [item["position"] for item in retained_trace] == [[1.0, 2.0, 3.0]]


def test_legacy_ground_uses_dominant_low_horizontal_layer():
    frames = [{
        "points": (
            [[index * 0.01, -1.2 + (index % 3) * 0.003, 0.0]
             for index in range(120)]
            + [[0.0, 0.4 + index * 0.01, 0.0] for index in range(30)]
        ),
    }]

    assert _estimated_trace_ground(frames) == -1.2


def test_manager_runs_validation_as_manual_step(tmp_path):
    config = _config(tmp_path)
    manager = RunManager()
    manager.start(config, ["validate"])
    deadline = time.time() + 5
    while manager.snapshot()["running"] and time.time() < deadline:
        time.sleep(0.02)
    state = manager.snapshot()
    assert state["status"] == "completed"
    assert state["steps"][0]["status"] == "completed"


def test_runtime_health_distinguishes_liveness_and_readiness(
        tmp_path, monkeypatch):
    runtime = RuntimePaths(
        project_root=tmp_path,
        scene_dir=tmp_path / "scenes",
        database_dir=tmp_path / "db",
        run_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
        debug_dir=tmp_path / "debug",
        headless=True,
    )
    runtime.scene_dir.mkdir()
    runtime.database_dir.mkdir()
    runtime.ensure_writable_directories()
    monkeypatch.setattr("run_console.RUNTIME_PATHS", runtime)

    assert runtime_health(require_ready=False)["ok"] is True
    assert runtime_health(require_ready=True)["ok"] is False

    (runtime.database_dir / "index.json").write_text("{}", encoding="utf-8")
    ready = runtime_health(require_ready=True)
    assert ready["ok"] is True
    assert ready["status"] == "ready"


def test_process_tree_stop_escalates_after_grace_period(monkeypatch):
    signals = []

    class Process:
        pid = 42

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout):
            raise subprocess.TimeoutExpired("demo", timeout)

        @staticmethod
        def terminate():
            raise AssertionError("The process group should be terminated")

    monkeypatch.setattr("run_console.IS_WINDOWS", False)
    monkeypatch.setattr(
        run_console.os, "getpgid", lambda pid: pid, raising=False,
    )
    monkeypatch.setattr(
        run_console.os, "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
        raising=False,
    )
    monkeypatch.setattr(
        run_console.signal, "SIGKILL", 9, raising=False,
    )

    RunManager._terminate_process_tree(Process(), grace_seconds=0)

    assert signals == [(42, signal.SIGTERM), (42, 9)]


def test_manager_runs_scientific_dependencies_as_one_target(tmp_path, monkeypatch):
    config = _config(tmp_path)
    manager = RunManager()
    commands = []

    def fake_run_process(command, paths, step_id, live=False):
        commands.append((command, step_id, live))
        return "Frontière atteinte"

    monkeypatch.setattr(manager, "_run_process", fake_run_process)
    manager.start(config, ["matching"])
    deadline = time.time() + 5
    while manager.snapshot()["running"] and time.time() < deadline:
        time.sleep(0.02)

    state = manager.snapshot()
    assert state["status"] == "completed"
    assert [step["id"] for step in state["steps"]] == [
        "fusion", "keypoints", "query", "matching",
    ]
    assert len(commands) == 1
    command, step_id, live = commands[0]
    assert step_id == "matching" and live is True
    assert command[command.index("--stop-after") + 1] == "matching"


def test_selection_generates_final_visualization_automatically(tmp_path, monkeypatch):
    config = _config(tmp_path)
    manager = RunManager()
    commands = []

    def fake_run_process(command, paths, step_id, live=False):
        commands.append((command, step_id, live))
        if "--stop-after" in command:
            paths["json"].write_text("{}", encoding="utf-8")
            paths["scan"].write_text("ply\n", encoding="ascii")
        else:
            paths["viz"].write_text("ply\n", encoding="ascii")
            paths["html"].write_text("<html></html>", encoding="utf-8")
        return 'Completed'

    monkeypatch.setattr(manager, "_run_process", fake_run_process)
    manager.start(config, ["selection"])
    deadline = time.time() + 5
    while manager.snapshot()["running"] and time.time() < deadline:
        time.sleep(0.02)

    state = manager.snapshot()
    assert state["status"] == "completed"
    assert len(commands) == 2
    assert commands[0][1:] == ("selection", True)
    assert commands[1][0][2] == "visualize"
    assert commands[1][1:] == ("selection", False)


def test_existing_partial_run_can_resume_in_same_folder(tmp_path, monkeypatch):
    config = _config(tmp_path)
    paths = execution_paths(config)
    paths["config"].write_text(json.dumps(config), encoding="utf-8")
    paths["progress"].write_text(
        json.dumps({"completed_stage": "keypoints", "steps": [{}]}),
        encoding="utf-8",
    )
    manager = RunManager()
    commands = []

    def fake_run_process(command, used_paths, step_id, live=False):
        commands.append((command, used_paths, step_id, live))
        return "Frontière atteinte"

    monkeypatch.setattr("run_console.DEFAULT_RUN_ROOT", Path(config["run_root"]))
    monkeypatch.setattr(manager, "_run_process", fake_run_process)
    manager.start(None, ["query"], existing_run_id=paths["id"])
    deadline = time.time() + 5
    while manager.snapshot()["running"] and time.time() < deadline:
        time.sleep(0.02)

    state = manager.snapshot()
    command, used_paths, step_id, live = commands[0]
    assert state["status"] == "completed"
    assert state["run_id"] == paths["id"]
    assert used_paths["folder"] == paths["folder"]
    assert command[command.index("--stage-cache-dir") + 1] == str(
        paths["stage_cache"],
    )
    assert command[command.index("--stop-after") + 1] == "query"
    assert step_id == "query" and live is True


def test_run_details_and_rename_keep_technical_id(tmp_path, monkeypatch):
    config = _config(tmp_path)
    paths = execution_paths(config)
    paths["config"].write_text(json.dumps(config), encoding="utf-8")
    paths["progress"].write_text(
        json.dumps({"completed_stage": "keypoints"}), encoding="utf-8",
    )
    monkeypatch.setattr("run_console.DEFAULT_RUN_ROOT", Path(config["run_root"]))

    before = run_details(paths["id"])
    renamed = rename_run(paths["id"], "Office - keypoints validés")
    after = run_details(paths["id"])

    assert before["run"]["completed_stage"] == "keypoints"
    assert before["run"]["next_stage"] == "query"
    assert before["run"]["resumable"] is True
    assert before["run"]["stage_cache"] == {
        "fusion_available": False,
        "fusion_count": 0,
        "fusion_frame_counts": [],
    }
    assert renamed["run"]["id"] == paths["id"]
    assert after["run"]["label"] == "Office - keypoints validés"
    assert paths["folder"].exists()
    assert paths["config"].exists()


def test_run_favorite_and_label_share_persistent_metadata(tmp_path, monkeypatch):
    config = _config(tmp_path)
    paths = execution_paths(config)
    paths["config"].write_text(json.dumps(config), encoding="utf-8")
    paths["progress"].write_text(
        json.dumps({"completed_stage": "keypoints"}), encoding="utf-8",
    )
    monkeypatch.setattr("run_console.DEFAULT_RUN_ROOT", Path(config["run_root"]))

    favorite = set_run_favorite(paths["id"], True)
    renamed = rename_run(paths["id"], "Office retenu")
    unfavorite = set_run_favorite(paths["id"], False)

    assert favorite["run"]["favorite"] is True
    assert renamed["run"]["favorite"] is True
    assert renamed["run"]["label"] == "Office retenu"
    assert unfavorite["run"]["favorite"] is False
    assert unfavorite["run"]["label"] == "Office retenu"
    assert paths["folder"].name == paths["id"]


def test_run_details_distinguishes_history_from_reusable_fusion_cache(
        tmp_path, monkeypatch):
    config = _config(tmp_path)
    paths = execution_paths(config)
    paths["config"].write_text(json.dumps(config), encoding="utf-8")
    paths["progress"].write_text(
        json.dumps({"completed_stage": "keypoints"}), encoding="utf-8",
    )
    paths["stage_cache"].mkdir()
    (paths["stage_cache"] / "scene_000176.pkl").write_bytes(b"cache")
    (paths["stage_cache"] / "scene_000020.pkl").write_bytes(b"cache")
    monkeypatch.setattr("run_console.DEFAULT_RUN_ROOT", Path(config["run_root"]))

    details = run_details(paths["id"])

    assert details["run"]["completed_stage"] == "keypoints"
    assert details["run"]["stage_cache"] == {
        "fusion_available": True,
        "fusion_count": 2,
        "fusion_frame_counts": [20, 176],
    }


def test_build_replay_compacts_visual_trace(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    folder = runs / "sample_run"
    folder.mkdir(parents=True)
    progress = {
        "fusion_trace": {
            "total_frames": 1,
            "frames": [{
                "frame_id": "000001", "frame_index": 1,
                "points": [[0, 0, 0]], "colors": [[255, 0, 0]],
                "camera_position": [0, 0, 0],
            }],
        },
        "execution_trace": [{
            "frame_count": 1,
            "full": {
                "scan": {
                    "detected_keypoints": 2, "retained_keypoints": 1,
                    "detected_keypoint_points": [
                        {"position": [0, 0, 0], "response": 1.0},
                        {"position": [1, 0, 0], "response": 0.5},
                    ],
                    "retained_keypoint_points": [
                        {"position": [0, 0, 0], "response": 1.0},
                    ],
                    "keypoints": [{"position": [0, 0, 0], "response": 1.0}],
                },
                "preselection": {
                    "strategy": "global", "database_count": 10,
                    "pool": ["chair_a", "table_b"], "top_k": ["chair_a"],
                },
                "matching": [{
                    "model": "chair_a", "status": "candidate",
                    "model_preview": [[index, 0, 0] for index in range(500)],
                    "correspondence_preview": [[0, 1]] * 500,
                    "verification": {"poses": [{}] * 50, "best_score": 0.8},
                    "registration": {"translation": [0, 0, 0], "scale": 1.0},
                }],
                "checkpoint_selection": [], "checkpoint_selected": [],
            },
        }],
        "selected": [], "final_selection": [],
    }
    (folder / "sample_run_progress.json").write_text(
        json.dumps(progress), encoding="utf-8",
    )
    (folder / "run_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("run_console.DEFAULT_RUN_ROOT", runs)

    replay = build_replay("sample_run")

    assert len(replay["events"]) == 7
    assert replay["ground"] == 0.0
    assert [chapter["id"] for chapter in replay["chapters"]] == [
        "fusion", "keypoints", "query", "matching", "verification", "selection",
    ]
    keypoints = next(event for event in replay["events"] if event["type"] == "keypoints")
    assert len(keypoints["data"]["detected_points"]) == 2
    assert len(keypoints["data"]["retained_points"]) == 1
    matching = next(event for event in replay["events"] if event["type"] == "matching_result")
    candidate = matching["data"]["candidate"]
    assert len(candidate["model_preview"]) <= 240
    assert "correspondence_preview" not in candidate
    assert "poses" not in candidate["verification"]


def test_legacy_progress_uses_final_visualization_instead_of_blank_replay(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "legacy_progress.json").write_text(
        json.dumps({"selected": [{"model": "chair_a"}]}), encoding="utf-8",
    )
    (runs / "legacy_viz.ply").write_text("ply\n", encoding="ascii")
    monkeypatch.setattr("run_console.DEFAULT_RUN_ROOT", runs)

    replay = build_replay("legacy")

    assert replay["detailed"] is False
    assert replay["events"] == []
    assert replay["run"]["viz_url"].endswith("/legacy/viz")


def test_console_ui_keeps_contextual_guidance_contract():
    root = Path(__file__).parents[1]
    html = web_asset("run_console.html").read_text(encoding="utf-8")
    css = web_asset("run_console.css").read_text(encoding="utf-8")
    javascript = web_asset("run_console.js").read_text(encoding="utf-8")
    shared_css = web_asset("ui_shared.css").read_text(encoding="utf-8")
    shared_javascript = web_asset("ui_shared.js").read_text(encoding="utf-8")
    scene_javascript = web_asset("ui_scene3d.js").read_text(encoding="utf-8")
    timeline_javascript = web_asset("ui_timeline.js").read_text(encoding="utf-8")

    assert 'id="app-tooltip"' in html
    assert 'href="/ui_shared.css"' in html
    assert 'class="app-tooltip ui-tooltip"' in html
    assert 'from "./ui_shared.js"' in javascript
    assert "createTooltipController" in shared_javascript
    assert "createNotifier" in shared_javascript
    assert "bindHoverPreviewPanels" in shared_javascript
    assert "bindResizableDimension" in shared_javascript
    assert "installSharedPrimitives" in shared_javascript
    assert "installSharedPrimitives();" in javascript
    assert 'from "./ui_scene3d.js"' in javascript
    assert "createSceneViewport" in scene_javascript
    assert "SCENE3D_THEME" in scene_javascript
    assert "viewport.fitBox(box,1.12,{remember:true})" in javascript
    assert "fitCameraToBox(scanBounds,1.12" in (
        root / "timeline_viewer.py"
    ).read_text(encoding="utf-8")
    assert 'from "./ui_timeline.js"' in javascript
    assert "candidateTimelineGroups" in timeline_javascript
    assert "renderCandidateTimelineNavigation" in timeline_javascript
    assert "applyRelatedTimelineState" in timeline_javascript
    assert "function listValues(value)" in timeline_javascript
    assert "...(event?.data?.candidates || [])" not in timeline_javascript
    assert 'id="run-candidate-timeline"' in javascript
    assert "candidateRelatedEvents" in javascript
    assert ".ui-tooltip" in shared_css
    assert ".ui-candidate-timeline" in shared_css
    assert ".ui-toast" in shared_css
    assert ".ui-tabs button" in shared_css
    assert ".ui-control" in shared_css
    assert ".ui-status-badge" in shared_css
    assert 'id="stage-insight"' in html
    assert 'id="phase-label"' in html
    assert 'id="stage-detail-drawer"' in html
    assert 'id="event-ruler-cells"' in html
    assert 'id="chapter-segments" aria-label="Run chapters"' in html
    assert 'id="event-position-detail"' in html
    assert 'id="keypoint-compare"' in html
    assert 'data-layer="keypoints-detected"' in html
    assert 'data-layer="keypoints-retained"' in html
    assert "workflow-panel:is(:hover,:focus-within)" in css
    assert ".chapter-segments{z-index:6" in css
    assert ".chapter-segment{min-width:12px" in css
    assert "inspector-panel:is(:hover,:focus-within)" in css
    assert "function bindTooltips()" in javascript
    assert "function updateStageInsight(" in javascript
    assert 'button.className="event-cell"' in javascript
    assert "button.textContent=String(position)" in javascript
    assert "rulerZoom=1" in javascript
    assert "data.detected_points" in javascript
    assert "function fitReplayCamera()" in javascript
    assert "showFusionFrames(fusionEvents.map(event=>event.data),false)" in javascript
    assert "fitReplayCamera();renderReplay(0)" in javascript
    assert "function setSceneGround(value)" in javascript
    assert "Number(position[1])-sceneGround" in javascript
    assert 'class="dependency-policy"' in html
    assert 'id="resume-context"' in html
    assert 'id="rename-dialog"' in html
    assert 'groupLabels={setup:"Preparation",pipeline:"Scientific pipeline"}' in javascript
    assert 'workflowStepHelp={validate:' in javascript
    assert 'matching:"Test correspondences and constellations.' in javascript
    assert "function loadExistingRun(" in javascript
    assert 'data-viewer-run="${escapeHtml(run.id)}"' in javascript
    assert 'target="_blank" aria-label="Open final viewer"' not in javascript
    assert 'sessionStorage.setItem("objectsensing-selected-run"' in javascript
    assert "function activateReplay(" in javascript
    assert "Promise.all([api(`/api/run?id=" in javascript
    assert "applyState({...payload.state,live:undefined})" in javascript
    assert "logs:undefined" not in javascript
    assert 'id="workflow-resizer"' in html
    assert 'id="inspector-resizer"' in html
    assert 'id="timeline-resizer"' in html
    assert "function bindPanelResizers()" in javascript
    assert "function bindTimelineResizer()" in javascript
    assert 'draggable="true"' not in javascript
    assert "dragstart" not in javascript
    assert "--workflow-width" in css
    assert "--inspector-width" in css
    assert "--timeline-height" in css
    assert ".workflow-panel:is(:hover,:focus-within) .playlist-divider{display:flex}" in css
    assert ".workflow-panel:is(:hover,:focus-within) .playlist-item.scientific::after{display:block}" in css
    assert ".inspector-panel:is(:hover,:focus-within) .candidate-panel.active{display:grid}" in css
    assert "function openRenameDialog(" in javascript
    assert 'data-continue-run=' in javascript
    assert 'data-rename-run=' in javascript
    assert 'data-favorite-run=' in javascript
    assert 'api("/api/run/favorite"' in javascript
    assert ".history-row.favorite" in css
    assert 'id="scene-select" data-config="scene_zip"' in html
    assert 'id="add-scene-dialog"' in html
    assert 'api("/api/scenes"' in javascript
    assert "function renderSceneOptions(" in javascript
    assert ".config-control-row" in css
    assert html.count("data-pick-path") == 4
    assert 'data-purpose="db_dir"' in html
    assert 'data-purpose="candidate_index"' in html
    assert 'data-purpose="run_root"' in html
    assert 'data-purpose="scene_zip"' in html
    assert 'id="path-browser-dialog"' in html
    assert 'api("/api/browse-path"' in javascript
    assert 'api("/api/pick-path"' in javascript
    assert "function bindPathPickers()" in javascript
    assert "https://unpkg.com" not in html
    assert "https://cdn.jsdelivr.net" not in html
    assert 'src="/vendor/lucide/lucide.min.js"' in html
    assert '"/vendor/three/three.module.js"' in html
    assert (root / "vendor" / "lucide" / "lucide.min.js").is_file()
    assert (root / "vendor" / "three" / "three.module.js").is_file()


def test_console_serves_every_local_module_dependency():
    root = Path(run_console.__file__).resolve().parent
    javascript = web_asset("run_console.js").read_text(encoding="utf-8")
    local_imports = {
        f"/{Path(match).name}"
        for match in re.findall(r'from\s+"(\./[^"]+)"', javascript)
    }

    assert local_imports
    assert local_imports <= run_console.STATIC_ASSETS
    assert "/ui_timeline.js" in run_console.STATIC_ASSETS


def test_console_disables_browser_cache_for_frontend_assets():
    server = Path(run_console.__file__).read_text(encoding="utf-8")

    assert 'if path.suffix.lower() in {".html", ".js", ".css"}' in server
    assert 'self.send_header("Cache-Control", cache_control)' in server
    assert "/ui_scene3d.js" in run_console.STATIC_ASSETS


def test_debug_sources_include_every_annotation_scene():
    source_ids = {
        source["id"] for source in run_console.DEBUG_MANAGER.defaults()["sources"]
    }

    assert set(run_console.ANNOTATION_TARGETS) <= source_ids


def test_shapenet_annotation_storage_normalizes_duplicates(tmp_path, monkeypatch):
    path = tmp_path / "target_annotations.json"
    quality_path = tmp_path / "target_annotation_quality.json"
    monkeypatch.setattr(run_console, "TARGET_ANNOTATIONS_PATH", path)
    monkeypatch.setattr(
        run_console, "TARGET_ANNOTATION_QUALITY_PATH", quality_path,
    )

    saved = run_console.save_target_annotations({
        "office": {"desk": ["table_a", "table_a", "table_b"]},
    })

    assert saved["annotations"]["office"]["desk"] == ["table_a", "table_b"]
    assert run_console.load_target_annotations()["office"]["desk"] == [
        "table_a", "table_b",
    ]
    assert saved["path"] == str(path.resolve())
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 2
    assert stored["scenes"]["office"]["desk"] == {
        "table_a": 1, "table_b": 1,
    }

    quality = run_console.save_target_annotation_quality({
        "office": {"desk": {"table_a": 2, "table_b": "1"}},
    })
    assert quality["quality"]["office"]["desk"] == {
        "table_a": 2, "table_b": 1,
    }
    assert quality["quality"]["office"]["chair"] == {}
    assert quality["quality"]["ikea-table"] == {"table": {}, "chair": {}}
    assert quality["quality"]["chairs"] == {
        "left-wood-chair": {},
        "center-folding-chair": {},
        "right-wood-seat": {},
    }
    assert run_console.load_target_annotation_quality() == quality["quality"]
    assert not quality_path.exists()

    with pytest.raises(ValueError, match="0, 1 or 2"):
        run_console.save_target_annotation_quality({
            "office": {"desk": {"table_a": 3}},
        })


def test_shapenet_annotation_storage_migrates_legacy_files(tmp_path, monkeypatch):
    path = tmp_path / "target_annotations.json"
    quality_path = tmp_path / "target_annotation_quality.json"
    path.write_text(json.dumps({
        "office": {"desk": ["table_a", "table_b"]},
    }), encoding="utf-8")
    quality_path.write_text(json.dumps({
        "office": {"desk": {"table_a": 2, "table_c": 0}},
    }), encoding="utf-8")
    monkeypatch.setattr(run_console, "TARGET_ANNOTATIONS_PATH", path)
    monkeypatch.setattr(
        run_console, "TARGET_ANNOTATION_QUALITY_PATH", quality_path,
    )

    assert run_console.load_target_annotation_quality() == {
        "office": {"desk": {"table_a": 2, "table_c": 0, "table_b": 1}},
    }
    annotations = run_console.load_target_annotations()
    assert annotations["office"]["desk"] == ["table_a", "table_b"]
    assert annotations["office"]["chair"] == []
    assert annotations["ikea-table"] == {"table": [], "chair": []}
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert not quality_path.exists()
    assert quality_path.with_suffix(".json.legacy.bak").exists()


def test_terminal_console_debug_toggle_and_restart(monkeypatch):
    console = run_console.TerminalConsole(
        "127.0.0.1", 8770, debug=False, interactive=False,
    )
    console.request("GET", "/api/health", ('request', "200", "-"))
    assert console.requests == 1
    assert console.last_request == "GET /api/health -> 200"
    console._restart.set()
    assert console.consume_restart() is True
    assert console.consume_restart() is False


def test_terminal_console_pins_status_and_help_around_scroll_region(monkeypatch):
    console = run_console.TerminalConsole(
        "127.0.0.1", 8770, debug=True, interactive=False,
    )
    output = io.StringIO()
    monkeypatch.setattr(run_console.sys, "stdout", output)
    monkeypatch.setattr(console, "_dimensions", lambda: (80, 20))
    console._tui = True

    console.banner()
    rendered = output.getvalue()

    assert "\033[4;18r" in rendered
    assert "ObjectSensing Run Console" in rendered
    assert "logs DEBUG" in rendered
    assert "[R] Restart" in rendered
    assert 'HTTP server ready' in rendered
    console.close()


def test_shapenet_browser_lists_and_loads_models(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "index.json").write_text(json.dumps({
        "n": 2,
        "names": ["chair_alpha", "table_beta"],
        "synsets": ["chair", "table"],
        "mesh_paths": ["a.obj", "b.obj"],
    }), encoding="utf-8")
    for index, (name, synset) in enumerate((
            ("chair_alpha", "chair"), ("table_beta", "table"))):
        model = SimpleNamespace(
            name=name, synset=synset, mesh_path=f"{name}.obj",
            cloud=SimpleNamespace(
                points=np.asarray([[0, 0, 0], [1, 1, 1]], float), size=2,
            ),
            features=[1, 2],
        )
        with (db_dir / f"model_{index:05d}.pkl").open("wb") as stream:
            pickle.dump(model, stream)
    monkeypatch.setattr(
        type(run_console.RUNTIME_PATHS), "browse_roots",
        lambda self: (tmp_path,),
    )

    page = run_console.shapenet_model_page(
        db_dir, query="beta", synset="table",
    )
    detail = run_console.shapenet_model_detail(db_dir, 1)

    assert page["total_filtered"] == 1
    assert page["models"][0]["name"] == "table_beta"
    assert detail["name"] == "table_beta"
    assert detail["point_count"] == 2
    assert detail["keypoint_count"] == 2

    included = run_console.shapenet_model_page(
        db_dir, include_names={"table_beta"},
    )
    empty_group = run_console.shapenet_model_page(
        db_dir, include_names=[],
    )
    excluded = run_console.shapenet_model_page(
        db_dir, exclude_names={"table_beta"},
    )

    assert [model["name"] for model in included["models"]] == [
        "table_beta",
    ]
    assert empty_group["total_filtered"] == 0
    assert all(
        model["name"] != "table_beta" for model in excluded["models"]
    )


