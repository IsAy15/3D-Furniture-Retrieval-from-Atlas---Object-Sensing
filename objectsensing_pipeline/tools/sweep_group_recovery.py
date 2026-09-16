"""Compare keypoint/group settings against spatially transferred reference groups.

Reference labels are used only for evaluation, never passed to the detector.
Source indices belong to one reconstruction: comparisons use display coordinates.
"""
import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import debug_visualizer as debug
from keypoint_groups import build_object_keypoint_groups


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(debug.json_compatible(data), ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(reference_points, reference_labels, points, groups, tolerance=.10):
    """One-to-one group assignment; distant new points count as uncertain support.

    Precision is a conservative proximity proxy, not manually labeled new data.
    Recall includes reference anchors lost by detection. Hungarian assignment
    prevents one merged prediction from earning full credit for several objects.
    """
    labels = sorted(set(reference_labels) - {-1})
    if not len(points):
        return dict(score=0., objects=[], groups=len(groups), keypoints=0)
    distances, nearest = cKDTree(reference_points).query(points)
    assigned = np.where(distances <= tolerance, reference_labels[nearest], -1)
    anchor_distance, anchor_nearest = cKDTree(points).query(reference_points)
    scores = np.zeros((len(labels), len(groups)))
    measurements = {}
    for i, label in enumerate(labels):
        anchors = reference_labels == label
        for j, group in enumerate(groups):
            members = np.asarray(group["member_indices"], int)
            precision = float(np.mean(assigned[members] == label)) if len(members) else 0.
            recall = float(np.mean((anchor_distance[anchors] <= tolerance) & np.isin(anchor_nearest[anchors], members)))
            f1 = 2*precision*recall/max(precision+recall, 1e-12)
            scores[i,j] = f1
            measurements[i,j] = dict(label=int(label), group=j, precision=precision, recall=recall,
                f1=f1, members=len(members), uncertain=int(np.sum(distances[members] > tolerance)))
    rows, cols = linear_sum_assignment(-scores) if len(groups) else ([], [])
    matches = {int(i): measurements[i,int(j)] for i,j in zip(rows,cols)}
    objects = [matches.get(i, dict(label=int(label), group=None, precision=0., recall=0., f1=0.)) for i,label in enumerate(labels)]
    return dict(score=float(np.mean([item["f1"] for item in objects])), objects=objects,
        groups=len(groups), keypoints=len(points), anchor_coverage=float(np.mean(anchor_distance[reference_labels>=0] <= tolerance)))


def add_visual_groups(result, payload):
    debug._append_group_preview(result["layers"], result["steps"], payload["keypoints"],
        payload["keypoint_groups"], payload["scene"].up, payload["scene"].ground)
    result["object_groups"] = payload["keypoint_groups"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--inputs", required=True, help="JSON scene -> latest keypoint run id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--plan", help="Optional JSON with variants and group_variants lists")
    args=parser.parse_args()
    runs=debug.RUN_ROOT; reference=runs/args.reference
    inputs=read_json(args.inputs); old_groups=read_json(reference/"groups.json")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    refs={}
    for scene in inputs:
        result=read_json(reference/f"{scene}.json")
        layer=next(layer for layer in result["layers"] if layer["id"]=="retained")
        sources=layer["source_index"]
        # Explicit chair and PC memberships came from the user's annotations.
        # Other Office memberships are the accepted POC, a weaker reference.
        selected = ([g for g in old_groups[scene] if g["id"] in (0,1,3,12)]
                    if scene=="office" else [old_groups[scene][0]])
        labels=np.full(len(sources),-1,int)
        for g in selected:
            labels[np.isin(sources,g["source_indices"])]=g["id"]
        refs[scene]=(np.asarray(layer["points"],float),labels)
    variants=[
        ("baseline",{}),
        ("nms04",dict(nms_radius=.04)),
        ("nms07",dict(nms_radius=.07)),
        ("neighbors06",dict(neighbor_radius=.06)),
        ("neighbors07",dict(neighbor_radius=.07)),
        ("quality06",dict(quality_nms_radius=.06)),
        ("quality10",dict(quality_nms_radius=.10)),
        ("wall_strict",dict(wall_plane_distance=.045,wall_plane_normal_cos=.97)),
        ("wall_wide",dict(wall_plane_distance=.08)),
        ("curvature03",dict(curvature_threshold=.03)),
    ]
    group_variants=[("original",{}), ("link22",dict(keypoint_link_radius=.22)),
        ("link26",dict(keypoint_link_radius=.26)),("link35",dict(keypoint_link_radius=.35)),
        ("wall05",dict(wall_clearance=.05)),("parts30",dict(part_merge_gap=.30))]
    if args.plan:
        plan=read_json(args.plan)
        variants=plan.get("variants",variants)
        group_variants=plan.get("group_variants",group_variants)
    stamp=time.strftime("%Y%m%d-%H%M%S"); report=[]; best=None
    for name,changes in variants:
        captured={}; detections={}
        for scene,run in inputs.items():
            manifest=read_json(runs/run/"manifest.json")
            source=next(s for s in manifest["sources"] if s["id"]==scene)
            parameters={**debug.default_parameters("keypoints"),**manifest["parameters"],**changes,"parallel_scenes":1}
            def capture(*a,**kw):
                captured[scene]=(a,kw)
                return build_object_keypoint_groups(*a,**kw)
            debug.build_object_keypoint_groups=capture
            scratch=out/f"{scene}.keypoints.pkl"
            print(f"{name} / {scene}: keypoints",flush=True)
            result=debug._scene_result(source,parameters,stage="keypoints",artifact_path=scratch)
            with scratch.open("rb") as stream: payload=pickle.load(stream)
            detections[scene]=(result,payload,source,parameters)
        debug.build_object_keypoint_groups=build_object_keypoint_groups
        for group_name,options in group_variants:
            metrics={}; records={}
            for scene,(result,payload,source,parameters) in detections.items():
                a,kw=captured[scene]
                groups=build_object_keypoint_groups(*a,**{**kw,**options})
                records[scene]=debug._serialize_object_groups(groups,payload["keypoints"])
                points=debug._to_display(payload["keypoints"].positions,payload["scene"].up,payload["scene"].ground)
                metrics[scene]=evaluate(*refs[scene],points,records[scene])
            # Balance scenes; the worst scene matters as much as the mean.
            values=[m["score"] for m in metrics.values()]
            score=.5*min(values)+.5*float(np.mean(values))
            row=dict(variant=f"{name}-{group_name}",score=score,keypoint_changes=changes,group_changes=options,scenes=metrics)
            report.append(row); print(json.dumps(row),flush=True)
            if best is None or score>best["score"]+1e-9:
                best=row
                run_id=f"keypoints-recovery-{stamp}-{name}-{group_name}"
                target=runs/run_id; target.mkdir(exist_ok=True)
                scene_meta={}; source_meta=[]
                for scene,(result,payload,source,parameters) in detections.items():
                    parameters={**parameters,**{"group_"+key:value for key,value in options.items()}}
                    result=copy.deepcopy(result); payload={**payload,"keypoint_groups":records[scene],"group_parameters":options,"parameters":parameters}
                    artifact=target/f"{scene}.keypoints.pkl"; debug._atomic_pickle(artifact,payload)
                    result["artifacts"]["keypoints"]["path"]=str(artifact.resolve())
                    result["metrics"]["object_groups"]=len(records[scene]); result["evaluation"]=metrics[scene]
                    add_visual_groups(result,payload); write_json(target/f"{scene}.json",result)
                    scene_meta[scene]=dict(file=f"{scene}.json",status="completed",artifacts=result["artifacts"])
                    source_meta.append(source)
                write_json(target/"manifest.json",dict(id=run_id,stage="keypoints",status="completed",created_at=time.time(),finished_at=time.time(),parameters=parameters,sources=source_meta,scenes=scene_meta,group_parameters=options,evaluation=row))
                write_json(target/"run_meta.json",dict(label=f"Group recovery - {name} / {group_name}",favorite=True))
                best={**row,"run_id":run_id}
            write_json(out/"report.json",dict(reference=args.reference,inputs=inputs,metric="Spatial proximity proxy at 10cm; not independent validation",trials=report,best=best))
    print("BEST "+json.dumps(best),flush=True)


if __name__=="__main__":
    main()
