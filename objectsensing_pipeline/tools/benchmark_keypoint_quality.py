"""Quickly measure keypoint selection on debug scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from debug_visualizer import (  # noqa: E402
    _keypoint_scene_result,
    default_parameters,
    default_sources,
    normalized_parameters,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--budgets", type=int, nargs="+",
        help="max_scan_keypoints budgets to compare (e.g. 600 400 300 200)",
    )
    args = parser.parse_args()
    base_parameters = normalized_parameters(
        default_parameters("keypoints"), stage="keypoints",
    )
    rows = []
    budgets = args.budgets or [int(base_parameters["max_scan_keypoints"])]
    for budget in budgets:
        parameters = dict(base_parameters)
        parameters["max_scan_keypoints"] = max(0, int(budget))
        for source in default_sources():
            if not Path(source["path"]).exists():
                continue
            print(
                f"[{source['label']}] budget {budget}: detecting...",
                flush=True,
            )
            result = _keypoint_scene_result(source, parameters)
            metrics = result["metrics"]
            row = {
                "scene": source["label"],
                "budget": budget,
                "detected": metrics["detected"],
                "wall_rejected": metrics["wall_rejected"],
                "planar_rejected": metrics["planar_rejected"],
                "hole_boundary_rejected": metrics["hole_boundary_rejected"],
                "repeatability_rejected": metrics["repeatability_rejected"],
                "score_rejected": metrics["score_rejected"],
                "nms_rejected": metrics["nms_rejected"],
                "after_quality_filter": metrics["after_quality_filter"],
                "retained": metrics["retained"],
                "seconds": result["duration_seconds"],
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {
        "base_parameters": base_parameters,
        "budgets": budgets,
        "scenes": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
