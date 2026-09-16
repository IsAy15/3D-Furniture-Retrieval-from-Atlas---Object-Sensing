from types import SimpleNamespace

import numpy as np

import descriptor_utility as utility
from config import PipelineConfig
from descriptors import LocalDescriptor
from matching import KeyPointFeature
from primitives import PrimitiveSet


def _descriptor(kind, occupied, udf_value=0.0):
    resolution = 4
    offsets = np.asarray(occupied, float).reshape(-1, 3)
    if kind == "scan":
        return LocalDescriptor(
            kind, resolution, 0.12, np.zeros(3),
            occ=np.zeros((resolution,) * 3, np.int8),
            occupied_offsets=offsets,
            n_occupied=len(offsets), n_unknown=resolution ** 3,
            n_total=resolution ** 3,
        )
    return LocalDescriptor(
        kind, resolution, 0.12, np.zeros(3),
        udf=np.full((resolution,) * 3, udf_value, np.float32),
        n_occupied=max(1, len(offsets)), n_total=resolution ** 3,
    )


def _feature(descriptor):
    return KeyPointFeature(
        position=np.asarray([0.0, 0.5, 0.0]),
        normal=np.asarray([0.0, 1.0, 0.0]),
        height=0.5,
        descriptor=descriptor,
        primitives=PrimitiveSet(None, []),
        size6d=np.ones(6),
        response=1.0,
    )


def test_descriptor_utility_separates_distinctive_ambiguous_and_no_match():
    cfg = PipelineConfig()
    cfg.descriptor.distance_unit = 1.0
    cfg.ransac.desc_inlier = 1.0
    scan = _feature(_descriptor("scan", [[0.0, 0.0, 0.0]]))
    distinctive_models = [
        SimpleNamespace(
            name="best", synset="chair",
            features=[_feature(_descriptor("model", [], 0.05))],
        ),
        SimpleNamespace(
            name="second", synset="table",
            features=[_feature(_descriptor("model", [], 0.8))],
        ),
    ]
    distinctive = utility.profile_descriptor_utility(
        [scan], distinctive_models, cfg, [0.0, 1.0, 0.0], rotations=1,
    )[0]

    ambiguous_models = [
        SimpleNamespace(
            name=f"model-{index}", synset="chair",
            features=[_feature(_descriptor("model", [], 0.4))],
        )
        for index in range(8)
    ]
    ambiguous = utility.profile_descriptor_utility(
        [scan], ambiguous_models, cfg, [0.0, 1.0, 0.0], rotations=1,
        max_valid_models=2, margin_threshold=0.2,
        entropy_threshold=0.5,
    )[0]

    no_match = utility.profile_descriptor_utility(
        [scan], [SimpleNamespace(
            name="far", synset="table",
            features=[_feature(_descriptor("model", [], 2.0))],
        )], cfg, [0.0, 1.0, 0.0], rotations=1,
    )[0]

    assert distinctive.label == utility.INFORMATIVE
    assert distinctive.best_model == "best"
    assert distinctive.margin > 0.9
    assert ambiguous.label == utility.AMBIGUOUS
    assert ambiguous.valid_model_count == 8
    assert ambiguous.entropy > 0.99
    assert no_match.label == utility.NO_MATCH
    assert no_match.valid_model_count == 0


def test_descriptor_summary_separates_nonwall_utility_from_wall_matches():
    rows = [
        utility.DescriptorUtility(
            utility.INFORMATIVE, 1.0, 3.0, 0.66, 1, 4, 0.1, "a", "chair",
        ),
        utility.DescriptorUtility(
            utility.INFORMATIVE, 1.0, 2.0, 0.50, 2, 5, 0.2, "b", "table",
        ),
        utility.DescriptorUtility(
            utility.AMBIGUOUS, 1.0, 1.1, 0.09, 8, 9, 0.9, "c", "table",
        ),
        utility.DescriptorUtility(
            utility.NO_MATCH, 9.0, 10.0, 0.0, 0, 0, 0.0, None, None,
        ),
    ]

    summary = utility.summarize_descriptor_utilities(
        rows, wall_scores=[0.1, 0.8, 0.2, 0.9],
        hole_boundary_removed=[0, 3, 0, 2],
    )

    assert summary["descriptor_informative"] == 2
    assert summary["descriptor_informative_nonwall"] == 1
    assert summary["descriptor_informative_wall"] == 1
    assert summary["descriptor_informative_wall_ratio"] == 0.5
    assert summary["descriptor_ambiguous"] == 1
    assert summary["descriptor_no_match"] == 1
    assert summary["descriptor_hole_boundary_cells_removed"] == 5


def test_candidate_recall_reports_hits_ranks_and_missing_targets():
    summary = utility.summarize_candidate_recall(
        ["chair_a", "table_b", "chair_c"],
        {
            "desk": ["table_a", "table_b"],
            "cabinet": ["table_missing"],
        },
    )

    assert summary["candidate_recall_evaluated"] is True
    assert summary["candidate_target_recall"] == 0.5
    assert summary["candidate_target_first_rank"] == 2
    assert summary["candidate_target_hit_names"] == {
        "desk": {"name": "table_b", "rank": 2},
    }
    assert summary["candidate_target_missing_names"] == ["cabinet"]


def test_candidate_recall_is_explicitly_unevaluated_without_annotations():
    summary = utility.summarize_candidate_recall(["chair_a"], None)

    assert summary == {
        "candidate_recall_evaluated": False,
        "candidate_pool_size": 1,
    }
