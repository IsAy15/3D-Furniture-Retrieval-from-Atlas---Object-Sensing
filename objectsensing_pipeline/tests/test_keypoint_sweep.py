import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from debug_visualizer import default_parameters  # noqa: E402
from keypoint_sweep import (  # noqa: E402
    aggregate_trial,
    parameter_combinations,
    pareto_front_trials,
    parse_assignment,
    parse_grid,
    run_keypoint_sweep,
    select_pareto_profiles,
    settings_from_json,
    summarize_scene,
)


def _layer(layer_id, indices, points=None):
    if points is None:
        points = [[float(value), 0.0, 0.0] for value in indices]
    return {
        "id": layer_id, "source_index": list(indices), "points": points,
    }


def test_grid_parsing_preserves_parameter_types():
    defaults = default_parameters("keypoints")
    assert parse_assignment("component_budget_enabled=false", defaults) == (
        "component_budget_enabled", False,
    )
    assert parse_grid("component_budget_radius=0.25,0.35", defaults) == (
        "component_budget_radius", [0.25, 0.35],
    )
    combinations = parameter_combinations({
        "component_budget_radius": [0.25, 0.35],
        "component_budget_max_per_component": [16, 24],
    })
    assert len(combinations) == 4
    assert combinations[-1]["component_budget_max_per_component"] == 24


def test_settings_json_accepts_run_console_aliases(tmp_path):
    profile = tmp_path / "execution.json"
    profile.write_text(json.dumps({"parameters": {
        "neighbor_radius": 0.06,
        "spatial_keypoints": False,
        "candidate_top_k": 100,
    }}), encoding="utf-8")

    settings = settings_from_json(profile, default_parameters("keypoints"))

    assert settings == {
        "neighbor_radius": 0.06,
        "spatial_keypoint_balance": False,
    }


def test_scene_summary_measures_retained_feature_types_and_cells():
    result = {
        "duration_seconds": 1.25,
        "metrics": {
            "detected": 8, "after_wall_filter": 6, "retained": 4,
            "pre_harris_visibility_rejected": 3,
            "quality_reintroduced": 1,
            "wall_budget_reintroduced": 1,
            "component_budget_reintroduced": 0,
        },
        "layers": [
            _layer("retained", [1, 2, 3, 4], [
                [0.01, 0, 0], [0.12, 0, 0], [0.8, 0, 0], [0.8, 0.4, 0],
            ]),
            _layer("harris_accepted", [1, 3, 7]),
            _layer("corner_2d_accepted", [2, 4]),
            _layer("wall_keypoint_affinity", [1, 7]),
            _layer("wall_score_high", [1]),
        ],
    }
    summary = summarize_scene(result, coverage_cell=0.35)
    assert summary["retained_harris_3d"] == 2
    assert summary["retained_corner_2d"] == 2
    assert summary["retained_high_wall"] == 1
    assert summary["spatial_cells"] == 3
    assert summary["pre_harris_visibility_rejected"] == 3
    assert summary["total_reintroduced"] == 2


def test_aggregate_requires_recall_on_every_scene():
    baseline = [
        {"scene": "a", "retained": 100, "spatial_cells": 10,
         "retained_harris_3d": 20, "retained_corner_2d": 10},
        {"scene": "b", "retained": 50, "spatial_cells": 8,
         "retained_harris_3d": 8, "retained_corner_2d": 4},
    ]
    candidate = [
        {**baseline[0], "retained": 60, "spatial_cells": 9,
         "retained_harris_3d": 18, "retained_corner_2d": 8,
         "duration_seconds": 1.0},
        {**baseline[1], "retained": 30, "spatial_cells": 7,
         "retained_harris_3d": 6, "retained_corner_2d": 3,
         "duration_seconds": 1.0},
    ]
    aggregate = aggregate_trial(candidate, baseline)
    assert aggregate["feasible"] is True
    assert aggregate["reintroduced_ratio"] == 0.0
    candidate[1]["spatial_cells"] = 6
    assert aggregate_trial(candidate, baseline)["feasible"] is False


def test_aggregate_requires_nonwall_descriptor_utility_when_available():
    baseline = [{
        "scene": "a", "retained": 100, "spatial_cells": 10,
        "retained_harris_3d": 20, "retained_corner_2d": 10,
        "descriptor_evaluated": 100, "descriptor_informative": 40,
        "descriptor_informative_nonwall": 32,
        "descriptor_informative_wall": 8,
    }]
    candidate = [{
        **baseline[0], "retained": 80, "spatial_cells": 9,
        "retained_harris_3d": 18, "retained_corner_2d": 8,
        "descriptor_informative": 32,
        "descriptor_informative_nonwall": 25,
        "descriptor_informative_wall": 7,
        "duration_seconds": 1.0,
    }]

    aggregate = aggregate_trial(candidate, baseline)
    assert aggregate["descriptor_utility_available"] is True
    assert aggregate["feasible"] is True
    candidate[0]["descriptor_informative_nonwall"] = 20
    assert aggregate_trial(candidate, baseline)["feasible"] is False


def test_aggregate_requires_full_annotated_candidate_recall():
    baseline = [{
        "scene": "office", "retained": 100, "spatial_cells": 10,
        "retained_harris_3d": 20, "retained_corner_2d": 10,
    }]
    candidate = [{
        **baseline[0], "retained": 80, "spatial_cells": 9,
        "retained_harris_3d": 18, "retained_corner_2d": 8,
        "candidate_recall_evaluated": True,
        "candidate_target_recall": 0.5,
        "duration_seconds": 1.0,
    }]

    aggregate = aggregate_trial(candidate, baseline)
    assert aggregate["candidate_recall_available"] is True
    assert aggregate["min_candidate_target_recall"] == 0.5
    assert aggregate["feasible"] is False
    candidate[0]["candidate_target_recall"] = 1.0
    assert aggregate_trial(candidate, baseline)["feasible"] is True


def test_sweep_checkpoints_and_resumes(tmp_path):
    source_path = tmp_path / "scene.pkl"
    source_path.write_bytes(b"cache")
    source = {
        "id": "scene", "label": "Scene", "path": str(source_path),
        "enabled": True,
    }
    calls = []

    def fake_runner(_source, parameters, progress=None):
        calls.append(dict(parameters))
        cap = int(parameters["component_budget_max_per_component"])
        retained = 8 if not parameters["component_budget_enabled"] else cap // 2
        indices = list(range(retained))
        return {
            "duration_seconds": 0.1,
            "metrics": {"detected": 10, "after_wall_filter": 8,
                        "retained": retained},
            "layers": [
                _layer("retained", indices),
                _layer("harris_accepted", list(range(4))),
                _layer("corner_2d_accepted", list(range(4, 8))),
                _layer("wall_keypoint_affinity", []),
                _layer("wall_score_high", []),
            ],
        }

    out = tmp_path / "sweep.json"
    payload = run_keypoint_sweep(
        [source], {}, {"component_budget_max_per_component": [12, 16]},
        out, scene_runner=fake_runner,
    )
    assert out.exists()
    assert len(payload["trials"]) == 2
    assert all(item["status"] == "completed" for item in payload["trials"])
    assert len(calls) == 3

    resumed = run_keypoint_sweep(
        [source], {}, {"component_budget_max_per_component": [12, 16]},
        out, scene_runner=fake_runner,
    )
    assert len(resumed["trials"]) == 2
    assert len(calls) == 3
    assert json.loads(out.read_text(encoding="utf-8"))["recommended_trial"]


def test_budget_sweep_uses_largest_budget_for_baseline(tmp_path):
    source_path = tmp_path / "scene.pkl"
    source_path.write_bytes(b"cache")
    source = {
        "id": "scene", "label": "Scene", "path": str(source_path),
        "enabled": True,
    }
    calls = []

    def fake_runner(_source, parameters, progress=None):
        calls.append(dict(parameters))
        count = int(parameters["max_scan_keypoints"])
        indices = list(range(min(count, 10)))
        return {
            "duration_seconds": 0.1,
            "metrics": {"detected": 10, "after_wall_filter": 10,
                        "retained": len(indices)},
            "layers": [
                _layer("retained", indices),
                _layer("harris_accepted", indices),
                _layer("corner_2d_accepted", []),
                _layer("wall_keypoint_affinity", []),
                _layer("wall_score_high", []),
            ],
        }

    run_keypoint_sweep(
        [source], {"max_scan_keypoints": 350},
        {"max_scan_keypoints": [200, 600, 400]},
        tmp_path / "budget.json", scene_runner=fake_runner,
    )

    assert calls[0]["max_scan_keypoints"] == 600
    assert calls[0]["component_budget_enabled"] is False


def test_sweep_does_not_recommend_an_infeasible_trial(tmp_path):
    source_path = tmp_path / "scene.pkl"
    source_path.write_bytes(b"cache")
    source = {
        "id": "scene", "label": "Scene", "path": str(source_path),
        "enabled": True,
    }

    def fake_runner(_source, parameters, progress=None):
        baseline = not parameters["component_budget_enabled"]
        indices = list(range(10 if baseline else 2))
        return {
            "duration_seconds": 0.1,
            "metrics": {"detected": 10, "after_wall_filter": 10,
                        "retained": len(indices)},
            "layers": [
                _layer("retained", indices),
                _layer("harris_accepted", list(range(10))),
                _layer("corner_2d_accepted", []),
                _layer("wall_keypoint_affinity", []),
                _layer("wall_score_high", []),
            ],
        }

    payload = run_keypoint_sweep(
        [source], {}, {"component_budget_max_per_component": [2]},
        tmp_path / "infeasible.json", scene_runner=fake_runner,
    )

    assert payload["recommended_trial"] is None
    assert payload["best_effort_trial"] == payload["trials"][0]["id"]


def test_sweep_merges_optional_descriptor_metrics(tmp_path):
    source_path = tmp_path / "scene.pkl"
    source_path.write_bytes(b"scene")
    source = {
        "id": "scene", "label": "Scene", "path": str(source_path),
        "enabled": True,
    }

    def fake_runner(_source, parameters, progress=None, artifact_path=None):
        artifact_path.write_bytes(b"artifact")
        count = 8 if not parameters["component_budget_enabled"] else 6
        return {
            "duration_seconds": 0.1,
            "metrics": {"detected": 10, "after_wall_filter": 8,
                        "retained": count},
            "layers": [
                _layer("retained", list(range(count))),
                _layer("harris_accepted", list(range(6))),
                _layer("corner_2d_accepted", list(range(2))),
                _layer("wall_keypoint_affinity", []),
                _layer("wall_score_high", []),
            ],
        }

    class FakeEvaluator:
        def signature(self):
            return {"kind": "fake"}

        def evaluate(self, artifact_path, progress=None, source=None):
            assert artifact_path.read_bytes() == b"artifact"
            assert source is not None
            return {
                "descriptor_evaluated": 6,
                "descriptor_informative": 5,
                "descriptor_informative_nonwall": 4,
                "descriptor_informative_wall": 1,
            }

    payload = run_keypoint_sweep(
        [source], {}, {"component_budget_max_per_component": [12]},
        tmp_path / "utility.json", scene_runner=fake_runner,
        descriptor_evaluator=FakeEvaluator(),
    )

    assert payload["descriptor_utility"] == {"kind": "fake"}
    assert payload["trials"][0]["aggregate"][
        "descriptor_utility_available"
    ] is True
    assert payload["trials"][0]["scenes"][0][
        "descriptor_informative_nonwall"
    ] == 4


def test_pareto_front_keeps_quality_balance_and_efficiency():
    def trial(trial_id, retained, spatial, descriptor, wall):
        return {
            "id": trial_id, "status": "completed",
            "aggregate": {
                "feasible": True,
                "total_retained": retained,
                "min_spatial_recall": spatial,
                "min_harris_recall": 0.9,
                "descriptor_utility_available": True,
                "min_descriptor_nonwall_recall": descriptor,
                "descriptor_informative_wall_ratio": wall,
                "retained_high_wall_ratio": wall,
            },
        }

    trials = [
        trial("efficiency", 200, 0.90, 0.80, 0.10),
        trial("balanced", 300, 0.95, 0.90, 0.08),
        trial("quality", 400, 1.00, 0.95, 0.05),
        trial("dominated", 450, 0.90, 0.80, 0.20),
    ]

    front = pareto_front_trials(trials, feasible_only=True)
    assert {item["id"] for item in front} == {
        "efficiency", "balanced", "quality",
    }
    profiles = select_pareto_profiles(front)
    assert {item["role"] for item in profiles} == {
        "quality", "balanced", "efficiency",
    }
    assert {item["trial"] for item in profiles} == {
        "efficiency", "balanced", "quality",
    }


def test_duplicate_pareto_knee_uses_complementary_extreme():
    def trial(trial_id, retained, spatial, descriptor):
        return {
            "id": trial_id, "status": "completed",
            "aggregate": {
                "feasible": False, "total_retained": retained,
                "min_spatial_recall": spatial,
                "min_harris_recall": spatial,
                "descriptor_utility_available": True,
                "min_descriptor_nonwall_recall": descriptor,
                "descriptor_informative_wall_ratio": 0.0,
                "retained_high_wall_ratio": 0.0,
            },
        }

    front = pareto_front_trials([
        trial("k600", 1139, 0.92, 0.67),
        trial("k400", 843, 0.80, 1.00),
        trial("k300", 643, 0.73, 0.67),
        trial("k200", 443, 0.61, 1.00),
    ])
    profiles = select_pareto_profiles(front)

    assert profiles == [
        {"role": "quality", "trial": "k400"},
        {"role": "balanced", "trial": "k600"},
        {"role": "efficiency", "trial": "k200"},
    ]
