'\nOne-file-per-model storage for memory-efficient parallel matching. Layout: index.json stores model metadata; cfg.pkl stores PipelineConfig; model_00000.pkl and subsequent files store DatabaseModel instances.\n'

from __future__ import annotations

import glob
import json
import os
import pickle
from typing import List


DB_FORMAT_VERSION = 2
MODEL_DESCRIPTOR_SCHEMA = "udf-local-v1"


def _with_schema(meta: dict, default_version=DB_FORMAT_VERSION) -> dict:
    meta = dict(meta)
    meta.setdefault("format_version", default_version)
    meta.setdefault("model_descriptor_schema", MODEL_DESCRIPTOR_SCHEMA)
    return meta


def save_db_dir(db, out_dir: str):
    'Write an in-memory ModelDatabase in one-file-per-model format.'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cfg.pkl"), "wb") as f:
        pickle.dump(db.cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "format_version": DB_FORMAT_VERSION,
        "model_descriptor_schema": MODEL_DESCRIPTOR_SCHEMA,
        "n": len(db.models),
        "names": [m.name for m in db.models],
        "synsets": [m.synset for m in db.models],
        "mesh_paths": [m.mesh_path for m in db.models],
    }
    for i, m in enumerate(db.models):
        with open(os.path.join(out_dir, f"model_{i:05d}.pkl"), "wb") as f:
            pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def is_db_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "index.json"))


def load_index(db_dir: str) -> dict:
    with open(os.path.join(db_dir, "index.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    version = int(meta.get("format_version", 1))
    if version > DB_FORMAT_VERSION:
        raise ValueError(
            f"Base format v{version} unsupported (maximum v{DB_FORMAT_VERSION})"
        )
    return _with_schema(meta, default_version=1)


def load_cfg(db_dir: str):
    with open(os.path.join(db_dir, "cfg.pkl"), "rb") as f:
        cfg = pickle.load(f)
    descriptor = getattr(cfg, "descriptor", None)
    if descriptor is not None:
        # Les bases v1/v2 ont sérialisé 1 m comme unité tout en conservant leurs
        # UDF en mètres. La conversion en cellules est donc une migration de
        # configuration et ne demande pas de reconstruire the models.
        unit = float(getattr(descriptor, "distance_unit", 1.0) or 0.0)
        if unit == 1.0:
            descriptor.distance_unit = 0.0
    matching = getattr(cfg, "matching", None)
    if matching is not None and "descriptor_ratio_threshold" not in vars(matching):
        matching.descriptor_ratio_threshold = 0.0
    return cfg


def model_files(db_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(db_dir, "model_*.pkl")))


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_cfg(db_dir: str, cfg):
    os.makedirs(db_dir, exist_ok=True)
    with open(os.path.join(db_dir, "cfg.pkl"), "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_index(db_dir: str, meta: dict, partial: bool = False):
    os.makedirs(db_dir, exist_ok=True)
    name = "index.partial.json" if partial else "index.json"
    with open(os.path.join(db_dir, name), "w", encoding="utf-8") as f:
        json.dump(_with_schema(meta), f, indent=2)


def load_resume_index(db_dir: str) -> dict:
    'Load the complete or partial index, or recover it from existing model files to resume a long build.'
    final = os.path.join(db_dir, "index.json")
    partial = os.path.join(db_dir, "index.partial.json")
    src = final if os.path.exists(final) else partial
    if os.path.exists(src):
        if src == final:
            return load_index(db_dir)
        with open(src, "r", encoding="utf-8") as f:
            return _with_schema(json.load(f), default_version=1)

    names, synsets, mesh_paths = [], [], []
    for path in model_files(db_dir):
        model = load_model(path)
        names.append(model.name)
        synsets.append(model.synset)
        mesh_paths.append(model.mesh_path)
    return _with_schema({
        "n": len(names), "names": names,
        "synsets": synsets, "mesh_paths": mesh_paths,
    }, default_version=1)


def convert_pkl_to_dir(pkl_path: str, out_dir: str, progress=None):
    'Convert a monolithic db.pkl into one-file-per-model storage without reprocessing geometry.'
    with open(pkl_path, "rb") as f:
        db = pickle.load(f)
    save_db_dir(db, out_dir)
    if progress:
        progress(f"{len(db.models)} models written to {out_dir}")
    return len(db.models)
