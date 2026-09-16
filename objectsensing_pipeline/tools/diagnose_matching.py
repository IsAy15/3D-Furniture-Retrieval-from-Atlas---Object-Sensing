"""Compare matching fixes on recorded features, without fusion or keypoint changes.

The optional legacy variants reproduce the former angle and boundary conventions.
They are diagnostics only; the production thresholds and seed budgets stay fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db_store
import descriptors
import matching
from constellations import one_point_ransac


def evaluate(model, scan_features, cfg, up, variant):
    original_distance = matching.descriptor_distance_batch
    original_lookup = descriptors._udf_lookup

    def legacy_lookup(descriptor, offsets):
        values = original_lookup(descriptor, offsets)
        coords = (offsets + descriptor.extent) / (2 * descriptor.extent) * (descriptor.res - 1)
        lower = np.floor(coords).astype(int)
        valid = np.all((lower >= 0) & (lower + 1 < descriptor.res), axis=1)
        values[~valid] = 2 * descriptor.extent
        return values

    if variant == "baseline":
        matching.descriptor_distance_batch = (
            lambda model, scan, config, angles, axis:
            original_distance(model, scan, config, -np.asarray(angles), axis)
        )
    if variant in ("baseline", "rotation"):
        descriptors._udf_lookup = legacy_lookup
    started = time.perf_counter()
    match_trace, ransac_trace = {}, {}
    try:
        pairs = matching.match_features(model.features, scan_features, cfg, up, diagnostics=match_trace)
        constellations = one_point_ransac(pairs, model.features, scan_features, cfg, up, diagnostics=ransac_trace)
    finally:
        matching.descriptor_distance_batch = original_distance
        descriptors._udf_lookup = original_lookup
    return {
        "variant": variant,
        "seconds": round(time.perf_counter() - started, 3),
        "correspondences": len(pairs),
        "constellations": len(constellations),
        "distinct_scan": len({pair.scan_idx for pair in pairs}),
        "distinct_model": len({pair.model_idx for pair in pairs}),
        "max_constellation_inliers": max((len(c.inliers) for c in constellations), default=0),
        "best_pose": ({"theta": float(constellations[0].transform.theta),
                       "scale": float(constellations[0].transform.scale),
                       "translation": constellations[0].transform.t.tolist()}
                      if constellations else None),
        "matching": match_trace,
        "ransac": ransac_trace,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-run", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["office", "single-chair"])
    parser.add_argument("--limit", type=int, default=8, help="Model/group records per scene; 0 means all.")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "rotation", "corrected"],
                        default=["baseline", "rotation", "corrected"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"source": str(args.matching_run.resolve()), "limit": args.limit, "records": [], "summary": {}}
    for scene_name in args.scenes:
        with (args.matching_run / f"{scene_name}.matching.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        cfg = payload["retrieval_config"]
        files = db_store.model_files(payload["database_dir"])
        records = payload["matching_records"]
        if args.limit > 0:
            records = records[:args.limit]
        totals = {variant: Counter() for variant in args.variants}
        for index, record in enumerate(records):
            model = db_store.load_model(files[int(record["model_index"])])
            scan_features = [payload["features"][int(i)] for i in record["feature_indices"]]
            result = {"scene": scene_name, "model": model.name, "group_id": record["group_id"],
                      "scan_features": len(scan_features), "model_features": len(model.features),
                      "historical_ransac": record["trace"].get("ransac"), "variants": []}
            for variant in args.variants:
                entry = evaluate(model, scan_features, cfg, payload["scene"].up, variant)
                result["variants"].append(entry)
                totals[variant].update({"records": 1, "with_constellation": int(entry["constellations"] > 0),
                                        "constellations": entry["constellations"],
                                        "correspondences": entry["correspondences"],
                                        "distinct_scan_sum": entry["distinct_scan"], "seconds": entry["seconds"]})
            report["records"].append(result)
            print(json.dumps({"scene": scene_name, "record": index + 1, "model": model.name,
                              "constellations": {v["variant"]: v["constellations"] for v in result["variants"]}}), flush=True)
        report["summary"][scene_name] = {variant: dict(counts) for variant, counts in totals.items()}
        print(json.dumps({"scene": scene_name, "summary": report["summary"][scene_name]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
