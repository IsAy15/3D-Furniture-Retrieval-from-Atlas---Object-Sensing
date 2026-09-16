"""Turn an existing keypoint debug run into a visual grouping POC."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keypoint_groups import (
    build_keypoint_groups, build_object_keypoint_groups,
)
from runtime_paths import RUNTIME_PATHS


PALETTE = np.asarray([
    [0.35, 0.78, 0.86], [0.96, 0.63, 0.30], [0.53, 0.82, 0.51],
    [0.72, 0.52, 0.94], [0.94, 0.45, 0.52], [0.95, 0.80, 0.34],
])


def _latest_keypoint_run(run_root):
    candidates = []
    for path in run_root.glob("*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("stage") == "keypoints" and manifest.get("status") == "completed":
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No completed Keypoints run")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def _source_set(result, layer_id):
    layer = next(
        (item for item in result.get("layers", []) if item.get("id") == layer_id),
        None,
    )
    return set(layer.get("source_index", [])) if layer else set()


def _full_geometry_path(source_meta):
    raw_path = Path(source_meta.get("path", ""))
    if raw_path.name.endswith(".scene.pkl"):
        candidate = raw_path.with_name(
            raw_path.name.removesuffix(".scene.pkl") + ".geometry.npz"
        )
        if candidate.exists():
            return candidate
    return None


def _enrich_scene(result, args, geometry_path=None):
    retained = next(
        (layer for layer in result.get("layers", []) if layer.get("id") == "retained"),
        None,
    )
    if not retained or not retained.get("points"):
        return result, []
    points = np.asarray(retained["points"], float)
    scores = np.asarray(retained.get("selection_score", retained.get("response", [])), float)
    if len(scores) != len(points):
        scores = np.ones(len(points), float)
    finite = scores[np.isfinite(scores)]
    if len(finite):
        low, high = np.min(finite), np.max(finite)
        scores = (np.nan_to_num(scores, nan=low) - low) / max(high - low, 1e-9)
    source = retained.get("source_index", list(range(len(points))))
    wall_affinity = _source_set(result, "wall_keypoint_affinity")
    object_protected = _source_set(result, "wall_object_protected")
    membership_weights = np.asarray([
        0.9 if value in object_protected else 0.35 if value in wall_affinity else 1.0
        for value in source
    ], float)
    if args.mode == "surface":
        surface = next(
            layer for layer in result.get("layers", [])
            if layer.get("id") == "surface"
        )
        if geometry_path is not None:
            with np.load(geometry_path, allow_pickle=False) as geometry:
                surface_points = np.asarray(geometry["points"], float)
                ground = float(np.asarray(geometry["ground"]).reshape(()))
            surface_points[:, 1] -= ground
            surface_sources = np.arange(len(surface_points), dtype=int)
        else:
            surface_points = np.asarray(surface.get("points", []), float)
            surface_sources = surface.get(
                "source_index", list(range(len(surface_points)))
            )
        wall_sources = _source_set(result, "wall_surface")
        wall_layer = next(
            (layer for layer in result.get("layers", []) if layer.get("id") == "wall_surface"),
            None,
        )
        wall_mask = np.asarray(
            [value in wall_sources for value in surface_sources], bool,
        )
        if wall_layer and wall_layer.get("points") and len(surface_points):
            wall_tree = cKDTree(np.asarray(wall_layer["points"], float))
            wall_distance, _ = wall_tree.query(surface_points, k=1)
            wall_mask |= wall_distance <= args.wall_clearance
        retained_wall = np.asarray([
            value in wall_affinity for value in source
        ], bool)
        affinity_layer = next(
            (layer for layer in result.get("layers", [])
             if layer.get("id") == "wall_keypoint_affinity"),
            None,
        )
        object_score_by_source = {
            int(source_index): float(object_score)
            for source_index, object_score in zip(
                affinity_layer.get("source_index", []),
                affinity_layer.get("object_score", []),
            )
        } if affinity_layer else {}
        retained_object_scores = np.asarray([
            object_score_by_source.get(int(value), 0.0) for value in source
        ], float)
        groups = build_object_keypoint_groups(
            surface_points, points, surface_wall_mask=wall_mask,
            keypoint_wall_mask=retained_wall, scores=scores,
            object_scores=retained_object_scores,
            floor_height=args.floor_height, wall_clearance=0.0,
            clean_restore_radius=args.clean_restore_radius,
            surface_voxel=args.surface_voxel,
            surface_connection=args.surface_connection,
            assignment_radius=args.assignment_radius,
            min_surface_voxels=args.min_surface_voxels,
            min_members=args.min_members, max_groups=args.max_groups,
            max_object_diameter=args.max_object_diameter,
            component_merge_gap=args.component_merge_gap,
            part_merge_gap=args.part_merge_gap,
            orphan_attach_radius=args.orphan_attach_radius,
            object_rescue_radius=args.object_rescue_radius,
            object_rescue_link_radius=args.object_rescue_link_radius,
            object_rescue_score=args.object_rescue_score,
            keypoint_link_radius=args.keypoint_link_radius,
            keypoint_component_max_members=(
                args.keypoint_component_max_members
            ),
        )
    else:
        groups = build_keypoint_groups(
            points, scores=scores, membership_weights=membership_weights,
            radius=args.radius, edge_radius=args.edge_radius,
            path_budget=args.path_budget,
            low_membership_penalty=args.wall_penalty,
            min_members=args.min_members, max_members=args.max_members,
            max_groups=args.max_groups,
        )
    grouped_points, colors, source_indices, segments = [], [], [], []
    summaries = []
    for group_id, group in enumerate(groups):
        color = PALETTE[group_id % len(PALETTE)]
        for index in group.member_indices:
            grouped_points.append(points[index].tolist())
            colors.append(color.tolist())
            source_indices.append(int(source[index]))
            segments.append([group.centroid.tolist(), points[index].tolist()])
        summaries.append({
            "id": group_id,
            "members": group.member_indices.astype(int).tolist(),
            "source_indices": [int(source[index]) for index in group.member_indices],
            "anchor": int(group.anchor_index),
            "centroid": np.round(group.centroid, 4).tolist(),
            "extent": np.round(group.extent, 4).tolist(),
            "score": round(group.score, 6),
            "wall_affinity_members": int(sum(
                source[index] in wall_affinity for index in group.member_indices
            )),
            "descriptor": np.round(group.descriptor, 6).tolist(),
        })
    result["layers"].extend([
        {
            "id": "object_group_members", "label": "Groupes POC",
            "kind": "points", "role": "candidate", "total": len(grouped_points),
            "sampled": len(grouped_points), "points": grouped_points,
            "colors": colors, "source_index": source_indices,
        },
        {
            "id": "object_group_links", "label": 'Group links',
            "kind": "lines", "role": "vector", "total": len(segments),
            "segments": segments,
        },
    ])
    result["steps"].append({
        "id": "object_groups_poc", "label": "Object groups (POC)",
        "description": "Propositions locales chevauchantes construites autour des keypoints retenus.",
        "group": "POC", "assessment": "Diagnostic uniquement : none effet sur le top-k.",
        "input": len(points), "kept": len(groups), "rejected": 0,
        "visible_layers": ["surface", "retained", "object_group_members", "object_group_links"],
    })
    result.setdefault("metrics", {})["object_group_count"] = len(groups)
    result["object_groups_poc"] = summaries
    return result, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Identifiant ou dossier d'un run Keypoints")
    parser.add_argument("--radius", type=float, default=0.65)
    parser.add_argument("--edge-radius", type=float, default=0.24)
    parser.add_argument("--path-budget", type=float, default=0.9)
    parser.add_argument("--wall-penalty", type=float, default=2.5)
    parser.add_argument("--min-members", type=int, default=2)
    parser.add_argument("--max-members", type=int, default=24)
    parser.add_argument("--max-groups", type=int, default=40)
    parser.add_argument("--mode", choices=("surface", "geodesic"), default="surface")
    parser.add_argument("--floor-height", type=float, default=0.06)
    parser.add_argument("--wall-clearance", type=float, default=0.075)
    parser.add_argument("--surface-voxel", type=float, default=0.055)
    parser.add_argument("--surface-connection", type=float, default=0.095)
    parser.add_argument("--assignment-radius", type=float, default=0.14)
    parser.add_argument("--min-surface-voxels", type=int, default=8)
    parser.add_argument("--max-object-diameter", type=float, default=1.65)
    parser.add_argument("--clean-restore-radius", type=float, default=0.12)
    parser.add_argument("--component-merge-gap", type=float, default=0.35)
    parser.add_argument("--part-merge-gap", type=float, default=0.45)
    parser.add_argument("--orphan-attach-radius", type=float, default=0.18)
    parser.add_argument("--object-rescue-radius", type=float, default=0.50)
    parser.add_argument("--object-rescue-link-radius", type=float, default=0.42)
    parser.add_argument("--object-rescue-score", type=float, default=1.01)
    parser.add_argument("--keypoint-link-radius", type=float, default=0.30)
    parser.add_argument(
        "--keypoint-component-max-members", type=int, default=32,
    )
    parser.add_argument(
        "--scenes", nargs="+",
        help="Scene IDs to include (default: all)",
    )
    args = parser.parse_args()

    run_root = RUNTIME_PATHS.debug_dir / "runs"
    source = Path(args.run).resolve() if args.run and Path(args.run).exists() else (
        run_root / args.run if args.run else _latest_keypoint_run(run_root)
    )
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "keypoints":
        raise ValueError("Source run must be a Keypoints run")
    run_id = f"keypoint-groups-poc-{time.strftime('%Y%m%d-%H%M%S')}"
    target = run_root / run_id
    target.mkdir(parents=True)
    reports = {}
    selected_scenes = {
        scene_id: meta
        for scene_id, meta in manifest.get("scenes", {}).items()
        if not args.scenes or scene_id in set(args.scenes)
    }
    if not selected_scenes:
        raise ValueError("None of the requested scenes exists in the source run")
    source_by_scene = {
        item.get("id"): item for item in manifest.get("sources", [])
    }
    for scene_id, scene_meta in selected_scenes.items():
        source_json = source / scene_meta.get("file", f"{scene_id}.json")
        result = json.loads(source_json.read_text(encoding="utf-8"))
        geometry_path = _full_geometry_path(source_by_scene.get(scene_id, {}))
        result, reports[scene_id] = _enrich_scene(
            result, args, geometry_path=geometry_path,
        )
        (target / f"{scene_id}.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8",
        )
    output_manifest = {
        **manifest, "id": run_id, "created_at": time.time(),
        "finished_at": time.time(), "status": "completed",
        "poc": {
            "source_run": source.name, "radius": args.radius,
            "edge_radius": args.edge_radius,
            "path_budget": args.path_budget,
            "wall_penalty": args.wall_penalty,
            "mode": args.mode,
            "surface_voxel": args.surface_voxel,
            "surface_connection": args.surface_connection,
            "assignment_radius": args.assignment_radius,
            "wall_clearance": args.wall_clearance,
            "clean_restore_radius": args.clean_restore_radius,
            "floor_height": args.floor_height,
            "max_object_diameter": args.max_object_diameter,
            "component_merge_gap": args.component_merge_gap,
            "part_merge_gap": args.part_merge_gap,
            "orphan_attach_radius": args.orphan_attach_radius,
            "object_rescue_radius": args.object_rescue_radius,
            "object_rescue_link_radius": args.object_rescue_link_radius,
            "object_rescue_score": args.object_rescue_score,
            "keypoint_link_radius": args.keypoint_link_radius,
            "keypoint_component_max_members": args.keypoint_component_max_members,
            "surface_geometry": "full_npz",
        },
        "scenes": {
            scene_id: {**meta, "file": f"{scene_id}.json"}
            for scene_id, meta in selected_scenes.items()
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (target / "run_meta.json").write_text(json.dumps({
        "label": f"POC groupes · {run_id.removeprefix('keypoint-groups-poc-')}",
        "favorite": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "groups.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if (source / "debug.log").exists():
        shutil.copy2(source / "debug.log", target / "debug.log")
    print(target)


if __name__ == "__main__":
    main()
