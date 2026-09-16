from pathlib import Path

from runtime_paths import RuntimePaths, path_is_within


def test_runtime_paths_use_environment_overrides(tmp_path):
    root = tmp_path / "project"
    env = {
        "OBJECTSENSING_SCENE_DIR": str(tmp_path / "mounted-scenes"),
        "OBJECTSENSING_DB_DIR": str(tmp_path / "mounted-db"),
        "OBJECTSENSING_RUN_DIR": str(tmp_path / "persistent-runs"),
        "OBJECTSENSING_STATE_DIR": str(tmp_path / "persistent-state"),
        "OBJECTSENSING_HEADLESS": "true",
    }

    runtime = RuntimePaths.from_env(env, project_root=root)

    assert runtime.scene_dir == (tmp_path / "mounted-scenes").resolve()
    assert runtime.database_dir == (tmp_path / "mounted-db").resolve()
    assert runtime.run_dir == (tmp_path / "persistent-runs").resolve()
    assert runtime.state_dir == (tmp_path / "persistent-state").resolve()
    assert runtime.debug_dir == (
        tmp_path / "persistent-state" / ".." / "debug_visualizer"
    ).resolve()
    assert runtime.headless is True
    assert root.resolve() not in runtime.browse_roots(env)


def test_runtime_creates_only_writable_directories(tmp_path):
    runtime = RuntimePaths(
        project_root=tmp_path / "project",
        scene_dir=tmp_path / "read-only-scenes",
        database_dir=tmp_path / "read-only-db",
        run_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
        debug_dir=tmp_path / "debug",
        headless=True,
    )

    runtime.ensure_writable_directories()

    assert runtime.run_dir.is_dir()
    assert runtime.state_dir.is_dir()
    assert runtime.debug_dir.is_dir()
    assert not runtime.scene_dir.exists()
    assert not runtime.database_dir.exists()


def test_path_is_within_resolved_root(tmp_path):
    root = tmp_path / "root"
    child = root / "nested" / "file.zip"
    outside = tmp_path / "outside"

    assert path_is_within(child, [root])
    assert not path_is_within(outside, [root])
