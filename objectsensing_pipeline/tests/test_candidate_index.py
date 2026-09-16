import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import db_store  # noqa: E402
import candidate_index as candidate_module  # noqa: E402
from candidate_index import (  # noqa: E402
    build_candidate_index, load_candidate_index, rank_candidates,
    rank_candidates_by_feature_groups,
    rerank_candidates_by_local_feature_groups, save_candidate_index,
)
from config import PipelineConfig  # noqa: E402
from database import ModelDatabase, preprocess_mesh  # noqa: E402


def _position_features(points):
    return [
        SimpleNamespace(position=np.asarray(point, float), response=1.0)
        for point in points
    ]


def test_group_shape_signature_is_yaw_invariant():
    points = np.asarray([
        [-.5, 0., -.3], [.5, 0., -.3], [-.5, 0., .3], [.5, 0., .3],
        [-.5, .8, .3], [.5, .8, .3],
    ])
    rotated = points[:, [2, 1, 0]] * np.asarray([1.0, 1.0, -1.0])
    left = candidate_module._shape_signature_from_features(
        _position_features(points),
    )
    right = candidate_module._shape_signature_from_features(
        _position_features(rotated),
    )
    assert np.allclose(left, right, atol=1e-6)


def test_group_shape_compatibility_prefers_matching_layout():
    chair = _position_features([
        [-.4, 0., -.3], [.4, 0., -.3], [-.4, 0., .3], [.4, 0., .3],
        [-.4, .8, .3], [.4, .8, .3], [0., .4, .3],
    ])
    yaw_chair = _position_features([
        [.3, 0., -.4], [.3, 0., .4], [-.3, 0., -.4], [-.3, 0., .4],
        [-.3, .8, -.4], [-.3, .8, .4], [-.3, .4, 0.],
    ])
    flat = _position_features([
        [-.8, 0., -.1], [-.4, 0., -.1], [0., 0., -.1], [.4, 0., -.1],
        [.8, 0., -.1], [-.8, .1, .1], [.8, .1, .1],
    ])
    matching = candidate_module._group_shape_compatibility(
        chair, yaw_chair, np.asarray([0., 1., 0.]),
    )
    mismatch = candidate_module._group_shape_compatibility(
        flat, yaw_chair, np.asarray([0., 1., 0.]),
    )
    assert matching > mismatch


def test_local_rerank_uses_registration_threshold_when_candidate_is_auto(
        monkeypatch):
    cfg = PipelineConfig()
    cfg.ransac.desc_inlier = 2048.0
    cfg.matching.candidate_descriptor_distance_threshold = 0.0
    descriptor = SimpleNamespace(n_occupied=10, n_unknown=0)
    feature = SimpleNamespace(
        position=np.zeros(3), response=1.0, height=1.0,
        size6d=np.ones(6), descriptor=descriptor,
    )
    monkeypatch.setattr(
        candidate_module, "descriptor_distance_batch",
        lambda *args, **kwargs: np.asarray([1300.0]),
    )
    evidence = candidate_module._local_descriptor_score(
        [feature], [feature], cfg, np.asarray([0., 1., 0.]),
        return_details=True,
    )
    assert evidence["matches"] == 1
    assert evidence["mean_similarity"] > 0.45


def test_candidate_index_roundtrip_and_rank():
    cfg = PipelineConfig()
    tmp = tempfile.mkdtemp()
    chair_mesh = trimesh.creation.box(extents=[0.6, 0.9, 0.6])
    table_mesh = trimesh.creation.box(extents=[1.2, 0.75, 0.8])
    chair = preprocess_mesh(chair_mesh, "chair_box", "03001627", cfg, n_points=5000, add_ground=False)
    table = preprocess_mesh(table_mesh, "table_box", "04379243", cfg, n_points=5000, add_ground=False)
    db_dir = os.path.join(tmp, "db")
    db_store.save_db_dir(ModelDatabase(models=[chair, table], cfg=cfg), db_dir)

    index = build_candidate_index(db_dir)
    out = os.path.join(tmp, "candidate_index.npz")
    save_candidate_index(index, out)
    loaded = load_candidate_index(out)
    ranked = rank_candidates(loaded, chair.features, cfg, top_k=1)
    diverse = rank_candidates(loaded, chair.features, cfg, top_k=2, diversify=True)
    forced = rank_candidates(loaded, chair.features, cfg, top_k=1, include_names=["table_box"])

    assert loaded.names == ["chair_box", "table_box"]
    assert loaded.height_hist.shape == (2, 32)
    assert len(ranked) == 1
    assert 0 <= ranked[0] < 2
    assert len(diverse) == 2
    assert ranked[0] in diverse
    assert forced == [1]
    assert np.isfinite(loaded.height_hist).all()


def test_group_rank_fusion_and_local_rerank_keep_best_object_group(
        monkeypatch):
    calls = []

    def fake_rank(index, features, cfg, top_k, **kwargs):
        calls.append(features)
        return [0, 1, 2] if features[0] == "desk" else [2, 1, 0]

    monkeypatch.setattr(candidate_module, "rank_candidates", fake_rank)
    pool, rankings = rank_candidates_by_feature_groups(
        SimpleNamespace(), [["desk", "desk-2"], ["chair", "chair-2"]],
        PipelineConfig(), top_k=3,
    )

    assert len(calls) == 2
    assert rankings == [[0, 1, 2], [2, 1, 0]]
    assert pool == [0, 2, 1]

    monkeypatch.setattr(db_store, "model_files", lambda path: ["a", "b", "c"])
    monkeypatch.setattr(
        db_store, "load_model",
        lambda path: SimpleNamespace(features=[path]),
    )

    def fake_score(model_features, scan_features, *args, **kwargs):
        return {
            ("a", "desk"): .9, ("a", "chair"): .1,
            ("b", "desk"): .2, ("b", "chair"): .8,
            ("c", "desk"): .3, ("c", "chair"): .4,
        }[(model_features[0], scan_features[0])]

    monkeypatch.setattr(candidate_module, "_local_descriptor_score", fake_score)
    selected, details = rerank_candidates_by_local_feature_groups(
        "db", pool, [["desk", "desk-2"], ["chair", "chair-2"]],
        PipelineConfig(), top_k=2,
        group_rankings=[[0, 2, 1], [1, 2, 0]],
        max_groups_per_candidate=1,
    )

    assert selected == [0, 1]
    assert details[0]["best_group"] == 0
    assert details[1]["best_group"] == 1
    assert details[0]["evaluated_groups"] == [0]
    assert details[1]["evaluated_groups"] == [1]


def test_group_rank_fusion_reserves_candidates_for_each_group(monkeypatch):
    rankings = iter([
        [0, 1, 2, 3, 4],
        [0, 1, 5, 6, 7],
    ])
    monkeypatch.setattr(
        candidate_module, "rank_candidates",
        lambda *args, **kwargs: next(rankings),
    )

    pool, _ = rank_candidates_by_feature_groups(
        SimpleNamespace(), [["desk", "a"], ["chair", "b"]],
        PipelineConfig(), top_k=4, group_weights=[2.0, 0.5],
        min_candidates_per_group=1,
    )

    assert len(pool) == 4
    assert any(candidate in pool for candidate in (5, 6, 7))


def test_local_descriptor_score_uses_distinct_matches(monkeypatch):
    cfg = PipelineConfig()
    cfg.matching.descriptor_ratio_threshold = 1.0

    def feature(identifier, x):
        descriptor = SimpleNamespace(identifier=identifier, n_occupied=8,
                                     n_unknown=0)
        return SimpleNamespace(
            height=0.5, size6d=np.ones(6), descriptor=descriptor,
            position=np.asarray([x, 0.5, 0.0]), response=1.0,
        )

    model = [feature("m0", 0.0), feature("m1", 1.0), feature("m2", 2.0)]
    scan = [feature("s0", 0.0), feature("s1", 1.0), feature("s2", 2.0)]
    distances = {
        ("m0", "s0"): 0.1, ("m1", "s0"): 0.5, ("m2", "s0"): 0.6,
        ("m0", "s1"): 0.1, ("m1", "s1"): 0.2, ("m2", "s1"): 0.6,
        ("m0", "s2"): 0.1, ("m1", "s2"): 0.5, ("m2", "s2"): 0.2,
    }
    monkeypatch.setattr(
        candidate_module, "descriptor_distance_batch",
        lambda model_desc, scan_desc, *args, **kwargs: np.asarray([
            distances[(model_desc.identifier, scan_desc.identifier)]
        ]),
    )

    evidence = candidate_module._local_descriptor_score(
        model, scan, cfg, np.asarray([0.0, 1.0, 0.0]),
        return_details=True,
    )

    assert evidence["matches"] == 3
    assert evidence["scan_coverage"] == 1.0
    assert evidence["model_coverage"] == 1.0
    assert evidence["spatial_spread"] == 1.0


def test_local_reranking_score_is_independent_from_ransac_threshold(monkeypatch):
    cfg = PipelineConfig()
    cfg.matching.descriptor_ratio_threshold = 1.0
    descriptor = SimpleNamespace(n_occupied=8, n_unknown=0)
    feature = SimpleNamespace(
        height=0.5, size6d=np.ones(6), descriptor=descriptor,
        position=np.asarray([0.0, 0.5, 0.0]), response=1.0,
    )
    monkeypatch.setattr(
        candidate_module, "descriptor_distance_batch",
        lambda *args, **kwargs: np.asarray([64.0]),
    )

    baseline = candidate_module._local_descriptor_score(
        [feature], [feature], cfg, np.asarray([0.0, 1.0, 0.0]),
    )
    cfg.ransac.desc_inlier = 2048.0
    recalibrated = candidate_module._local_descriptor_score(
        [feature], [feature], cfg, np.asarray([0.0, 1.0, 0.0]),
    )

    assert recalibrated == baseline


def test_local_group_rerank_reserves_top_candidate_per_group(monkeypatch):
    monkeypatch.setattr(db_store, "model_files", lambda path: ["a", "b", "c"])
    monkeypatch.setattr(
        db_store, "load_model",
        lambda path: SimpleNamespace(features=[path]),
    )
    scores = {
        ("a", "desk"): .95, ("a", "chair"): .10,
        ("b", "desk"): .90, ("b", "chair"): .20,
        ("c", "desk"): .15, ("c", "chair"): .70,
    }
    monkeypatch.setattr(
        candidate_module, "_local_descriptor_score",
        lambda model_features, scan_features, *args, **kwargs: scores[
            (model_features[0], scan_features[0])
        ],
    )

    selected, _ = rerank_candidates_by_local_feature_groups(
        "db", [0, 1, 2], [["desk", "a"], ["chair", "b"]],
        PipelineConfig(), top_k=2, group_weights=[1.0, 1.0],
        min_candidates_per_group=1, max_groups_per_candidate=2,
    )

    assert selected == [0, 2]


def test_limited_group_rerank_never_fills_with_arbitrary_groups(monkeypatch):
    monkeypatch.setattr(db_store, "model_files", lambda path: ["a"])
    monkeypatch.setattr(
        db_store, "load_model",
        lambda path: SimpleNamespace(features=[path]),
    )
    evaluated = []

    def fake_score(model_features, scan_features, *args, **kwargs):
        evaluated.append(scan_features[0])
        return .5

    monkeypatch.setattr(candidate_module, "_local_descriptor_score", fake_score)
    _, details = rerank_candidates_by_local_feature_groups(
        "db", [0], [["g0", "a"], ["g1", "b"], ["g2", "c"]],
        PipelineConfig(), top_k=1,
        group_rankings=[[], [], [0]], max_groups_per_candidate=1,
    )

    assert evaluated == ["g2"]
    assert details[0]["origin_groups"] == [2]
    assert details[0]["evaluated_groups"] == [2]


def test_limited_group_rerank_evaluates_global_only_candidate(monkeypatch):
    monkeypatch.setattr(db_store, "model_files", lambda path: ["a"])
    monkeypatch.setattr(
        db_store, "load_model",
        lambda path: SimpleNamespace(features=[path]),
    )
    evaluated = []

    def fake_score(model_features, scan_features, *args, **kwargs):
        evaluated.append(scan_features[0])
        return .5

    monkeypatch.setattr(candidate_module, "_local_descriptor_score", fake_score)
    _, details = rerank_candidates_by_local_feature_groups(
        "db", [0], [["g0", "a"], ["g1", "b"]],
        PipelineConfig(), top_k=1,
        group_rankings=[[], []], global_ranking=[0],
        max_groups_per_candidate=1,
    )

    assert evaluated == ["g0", "g1"]
    assert details[0]["origin_groups"] == []
    assert details[0]["evaluated_groups"] == [0, 1]
    assert details[0]["global_preselection_rank"] == 1
    assert details[0]["preselection_scores"] == [1.0, 1.0]


def test_unlimited_group_rerank_keeps_all_origin_groups_only(monkeypatch):
    monkeypatch.setattr(db_store, "model_files", lambda path: ["a"])
    monkeypatch.setattr(
        db_store, "load_model",
        lambda path: SimpleNamespace(features=[path]),
    )
    evaluated = []

    def fake_score(model_features, scan_features, *args, **kwargs):
        evaluated.append(scan_features[0])
        return .5

    monkeypatch.setattr(candidate_module, "_local_descriptor_score", fake_score)
    _, details = rerank_candidates_by_local_feature_groups(
        "db", [0], [["g0", "a"], ["g1", "b"], ["g2", "c"]],
        PipelineConfig(), top_k=1,
        group_rankings=[[0], [], [0]], max_groups_per_candidate=0,
    )

    assert evaluated == ["g0", "g2"]
    assert details[0]["origin_groups"] == [0, 2]
    assert details[0]["evaluated_groups"] == [0, 2]


def test_group_rerank_records_origin_local_and_protected_groups(monkeypatch):
    monkeypatch.setattr(db_store, "model_files", lambda path: ["a", "b"])
    monkeypatch.setattr(
        db_store, "load_model",
        lambda path: SimpleNamespace(features=[path]),
    )
    scores = {
        ("a", "g0"): .20, ("a", "g1"): .90,
        ("b", "g0"): .80, ("b", "g1"): .10,
    }
    monkeypatch.setattr(
        candidate_module, "_local_descriptor_score",
        lambda model_features, scan_features, *args, **kwargs: scores[
            (model_features[0], scan_features[0])
        ],
    )

    selected, details = rerank_candidates_by_local_feature_groups(
        "db", [0, 1], [["g0", "a"], ["g1", "b"]],
        PipelineConfig(), top_k=2,
        group_rankings=[[0, 1], [1, 0]], max_groups_per_candidate=0,
        min_candidates_per_group=1, local_weight=.65,
    )

    assert set(selected) == {0, 1}
    assert details[0]["origin_groups"] == [0, 1]
    assert details[0]["best_group"] == 1
    assert len(details[0]["combined_group_scores"]) == 2
    assert any(
        details[index]["selected_for_groups"] for index in selected
    )


def test_group_rerank_keeps_independent_top_k_per_group(monkeypatch):
    monkeypatch.setattr(db_store, "model_files", lambda path: ["a", "b", "c"])
    monkeypatch.setattr(
        db_store, "load_model",
        lambda path: SimpleNamespace(features=[path]),
    )
    scores = {
        ("a", "g0"): .95, ("a", "g1"): .10,
        ("b", "g0"): .90, ("b", "g1"): .20,
        ("c", "g0"): .15, ("c", "g1"): .99,
    }
    monkeypatch.setattr(
        candidate_module, "_local_descriptor_score",
        lambda model_features, scan_features, *args, **kwargs: scores[
            (model_features[0], scan_features[0])
        ],
    )

    selected, details = rerank_candidates_by_local_feature_groups(
        "db", [0, 1, 2], [["g0", "x"], ["g1", "y"]],
        PipelineConfig(), top_k=1, candidates_per_group=1,
    )

    assert selected == [2, 0]
    assert details[0]["selected_for_groups"] == [0]
    assert details[2]["selected_for_groups"] == [1]


def test_group_rerank_can_protect_extent_channel(monkeypatch):
    files = ["a", "b", "c"]
    sizes = {"a": 1.0, "b": 2.0, "c": 3.0}
    monkeypatch.setattr(db_store, "model_files", lambda path: files)
    monkeypatch.setattr(
        db_store, "load_model",
        lambda path: SimpleNamespace(
            features=[path],
            cloud=SimpleNamespace(points=np.asarray([
                [0.0, 0.0, 0.0], [sizes[path], 1.0, 1.0],
            ])),
        ),
    )
    monkeypatch.setattr(
        candidate_module, "_local_descriptor_score",
        lambda *args, **kwargs: {
            "score": .5, "matches": 2, "coverage": .1,
        },
    )
    monkeypatch.setattr(
        candidate_module, "_extent_compatibility",
        lambda model_extent, *args: float(np.max(model_extent)),
    )

    selected, details = rerank_candidates_by_local_feature_groups(
        "db", [0, 1, 2], [["group", "feature"]],
        PipelineConfig(), top_k=1,
        group_rankings=[[0, 1, 2]], local_weight=0.0,
        extent_quota_fraction=1.0,
    )

    assert selected == [2]
    assert details[2]["selected_for_extent_groups"] == [0]


def test_extent_compatibility_penalizes_oversized_furniture():
    cfg = PipelineConfig()
    scan_extent = np.asarray([0.58, 0.62, 0.77])
    chair_extent = np.asarray([0.70, 0.80, 0.90])
    couch_extent = np.asarray([1.10, 2.30, 0.80])

    chair_score = candidate_module._extent_compatibility(
        chair_extent, scan_extent, cfg,
    )
    couch_score = candidate_module._extent_compatibility(
        couch_extent, scan_extent, cfg,
    )

    assert 0.0 < couch_score < chair_score <= 1.0
    assert chair_score - couch_score > 0.15


def test_virtual_ground_cannot_change_reranking_extent(monkeypatch):
    from geometry import PointCloud

    points = np.array([
        [x, y, z] for x in [-.5, .5] for y in [0., 1.] for z in [-.5, .5]
    ])
    features = _position_features(points)
    models = {
        "clean": SimpleNamespace(features=features, cloud=PointCloud(points)),
        "padded": SimpleNamespace(
            features=features,
            cloud=PointCloud(np.vstack([points, [[-10, 0, -10], [10, 0, 10]]])),
            surface_point_count=len(points),
        ),
    }
    monkeypatch.setattr(db_store, "model_files", lambda _: ["clean", "padded"])
    monkeypatch.setattr(db_store, "load_model", lambda path: models[path])
    monkeypatch.setattr(
        candidate_module, "_local_descriptor_score", lambda *a, **k: {"score": .5},
    )
    _, details = rerank_candidates_by_local_feature_groups(
        "db", [0, 1], [features], PipelineConfig(), 2, local_weight=1,
    )
    assert (
        details[0]["group_evidence"][0]["extent_similarity"]
        == details[1]["group_evidence"][0]["extent_similarity"]
    )
    assert details[0]["best_score"] == details[1]["best_score"]
