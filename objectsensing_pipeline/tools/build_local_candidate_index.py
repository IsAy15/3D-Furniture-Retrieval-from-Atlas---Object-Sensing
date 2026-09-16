"""Build a resumable local UDF proposal index alongside a model database."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import db_store
from candidate_index import _top_feature_indices
from local_preselection import source_fingerprint


def build(db_dir, output=None):
    files = db_store.model_files(str(db_dir))
    names = db_store.load_index(str(db_dir))["names"]
    root = Path(output or Path(db_dir) / "local_candidate_index")
    manifest = {
        "names": names, "resolution": 6, "max_features": 32,
        "models": len(files), "source_fingerprint": source_fingerprint(db_dir),
        "schema_version": 1,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Source index changed; build into a new output directory")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    started = time.monotonic()
    for begin in range(0, len(files), 1000):
        target = root / f"{begin:05d}.npz"
        if target.exists():
            continue
        grids, ids, heights, occupancy, extents = [], [], [], [], []
        for model_index in range(begin, min(begin + 1000, len(files))):
            model = db_store.load_model(files[model_index])
            if model.name != names[model_index]:
                raise ValueError("Model file ordering differs from the database index")
            for feature_index in _top_feature_indices(model.features, 32):
                feature = model.features[feature_index]
                descriptor = feature.descriptor
                if descriptor.udf is None:
                    continue
                if descriptor.res != 16 or not np.isclose(descriptor.extent, .12):
                    raise ValueError("This compact index supports res=16, extent=.12 UDFs")
                axis = np.arange(0, 16, 3)
                grids.append(descriptor.udf[np.ix_(axis, axis, axis)].reshape(-1))
                ids.append(model_index)
                heights.append(feature.height)
                occupancy.append(descriptor.n_occupied)
                extents.append(descriptor.extent)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary, udf=np.asarray(grids, dtype=np.float16).reshape(-1, 216),
            model=np.asarray(ids, dtype=np.int32),
            height=np.asarray(heights, dtype=np.float32),
            occupancy=np.asarray(occupancy, dtype=np.float32),
            extent=np.asarray(extents, dtype=np.float32),
        )
        temporary.replace(target)
        print(f"{min(begin + 1000, len(files))}/{len(files)} — "
              f"{time.monotonic() - started:.0f}s", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    print(build(args.db, args.out))
