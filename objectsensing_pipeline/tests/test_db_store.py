import os
import pickle
import sys
import tempfile
import json
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import db_store  # noqa: E402
from config import PipelineConfig  # noqa: E402


def test_resume_index_can_be_rebuilt_from_model_files():
    tmp = tempfile.mkdtemp()
    models = [
        SimpleNamespace(name="chair_a", synset="03001627", mesh_path="a.obj"),
        SimpleNamespace(name="table_b", synset="04379243", mesh_path="b.obj"),
    ]
    for i, model in enumerate(models):
        with open(os.path.join(tmp, f"model_{i:05d}.pkl"), "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = db_store.load_resume_index(tmp)

    assert meta["n"] == 2
    assert meta["names"] == ["chair_a", "table_b"]
    assert meta["synsets"] == ["03001627", "04379243"]
    assert meta["mesh_paths"] == ["a.obj", "b.obj"]
    assert meta["format_version"] == 1
    assert meta["model_descriptor_schema"] == db_store.MODEL_DESCRIPTOR_SCHEMA


def test_legacy_index_is_loaded_with_current_schema_defaults():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"n": 0, "names": [], "synsets": [], "mesh_paths": []}, f)

    meta = db_store.load_index(tmp)

    assert meta["format_version"] == 1
    assert meta["model_descriptor_schema"] == db_store.MODEL_DESCRIPTOR_SCHEMA


def test_future_database_format_is_rejected():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"format_version": db_store.DB_FORMAT_VERSION + 1}, f)

    try:
        db_store.load_index(tmp)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("A future format must be rejected")


def test_load_cfg_migrates_legacy_metric_descriptor_distance():
    tmp = tempfile.mkdtemp()
    cfg = PipelineConfig()
    cfg.descriptor.distance_unit = 1.0
    cfg.ransac.desc_inlier = 128.0
    with open(os.path.join(tmp, "cfg.pkl"), "wb") as handle:
        pickle.dump(cfg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = db_store.load_cfg(tmp)

    assert loaded.descriptor.distance_unit == 0.0
    assert loaded.ransac.desc_inlier == 128.0


def test_load_cfg_preserves_explicit_nonlegacy_descriptor_distance():
    tmp = tempfile.mkdtemp()
    cfg = PipelineConfig()
    cfg.descriptor.distance_unit = 0.02
    cfg.ransac.desc_inlier = 42.0
    with open(os.path.join(tmp, "cfg.pkl"), "wb") as handle:
        pickle.dump(cfg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = db_store.load_cfg(tmp)

    assert loaded.descriptor.distance_unit == 0.02
    assert loaded.ransac.desc_inlier == 42.0
