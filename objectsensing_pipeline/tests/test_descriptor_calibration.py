import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from descriptor_calibration import (  # noqa: E402
    CalibrationRecord,
    evaluate_setting,
    select_setting,
)


def _record(scene, best, second, positive=True, wall=0.0):
    return CalibrationRecord(
        scene=scene, scan_index=0, model="model", synset="synset",
        best_distance=best, second_distance=second, wall_score=wall,
        expected_category=positive, scan_position=(0.0, 0.0, 0.0),
    )


def test_ratio_is_only_applied_when_second_match_survives_threshold():
    records = [
        _record("office", 20.0, 200.0),
        _record("office", 20.0, 21.0),
    ]

    result = evaluate_setting(records, threshold=128.0, ratio=0.95)

    assert result["tp"] == 1
    assert result["fn"] == 1


def test_calibration_prefers_setting_that_separates_categories():
    records = [
        _record("office", 20.0, 40.0, positive=True),
        _record("office", 24.0, 48.0, positive=True),
        _record("single-chair", 18.0, 36.0, positive=True),
        _record("single-chair", 22.0, 44.0, positive=True),
        _record("office", 90.0, 95.0, positive=False),
        _record("single-chair", 100.0, 105.0, positive=False),
    ]

    recommended, _ = select_setting(
        records, thresholds=(32.0, 128.0), ratios=(0.0, 0.95),
        min_scene_recall=0.5, min_scene_positives=1,
        min_model_correspondences=1, min_viable_models_per_category=1,
    )

    assert recommended["threshold"] == 32.0
    assert recommended["tp"] == 4
    assert recommended["fp"] == 0


def test_wall_supported_expected_category_is_negative_proxy():
    records = [_record("office", 10.0, 20.0, positive=True, wall=0.8)]

    result = evaluate_setting(records, threshold=32.0, ratio=0.0)

    assert result["tp"] == 0
    assert result["fp"] == 1


def test_operational_metrics_count_distinct_scan_correspondences():
    records = [
        CalibrationRecord(
            scene="office", scan_index=index, model="chair_a",
            synset="03001627", best_distance=10.0,
            second_distance=20.0, wall_score=0.0,
            expected_category=True,
            scan_position=(0.1 * index, 0.0, 0.0),
            model_feature_index=index,
        )
        for index in range(4)
    ]

    result = evaluate_setting(
        records, threshold=32.0, ratio=0.0,
        min_model_correspondences=4,
    )

    assert result["viable_expected_models"] == 1
    assert result["per_scene"][0]["viable_expected_by_synset"] == {
        "03001627": 1,
    }


def test_operational_metrics_reject_repeated_model_keypoint():
    records = [
        CalibrationRecord(
            scene="office", scan_index=index, model="chair_a",
            synset="03001627", best_distance=10.0,
            second_distance=20.0, wall_score=0.0,
            expected_category=True,
            scan_position=(0.1 * index, 0.0, 0.0),
            model_feature_index=0,
        )
        for index in range(4)
    ]

    result = evaluate_setting(
        records, threshold=32.0, ratio=0.0,
        min_model_correspondences=4,
    )

    assert result["viable_expected_models"] == 0
