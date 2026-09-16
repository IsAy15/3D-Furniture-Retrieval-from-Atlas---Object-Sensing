"Test d'intégration : on place un modèle (forme en L, asymétrique) dans une\nscene avec une transformation sol connue (rotation+translation+scale), puis on\nrecouvre cette transformation via matching -> 1-Point RANSAC -> vérification."
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry as G  # noqa: E402
import pipeline  # noqa: E402
from config import PipelineConfig  # noqa: E402
from keypoints import detect_keypoints  # noqa: E402
from matching import build_model_features, build_scan_features, match_features  # noqa: E402
from constellations import one_point_ransac  # noqa: E402
from verify import verify_model  # noqa: E402
from transforms import GroundTransform  # noqa: E402
from verify import Registration  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def test_matching_checkpoint_rejects_changed_scientific_code(tmp_path, monkeypatch):
    import pytest
    for name in ("pipeline.py", "matching.py", "descriptors.py", "constellations.py",
                 "transforms.py", "geometry.py", "verify.py", "database.py",
                 "candidate_index.py", "local_preselection.py", "progressive.py"):
        (tmp_path / name).write_text("# original implementation\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "pipeline.py"))
    scene = SimpleNamespace(cloud=G.PointCloud(np.zeros((2, 3))))
    cfg = PipelineConfig()
    tasks = [(0, "model.pkl")]
    original = pipeline._resume_signature(tasks, scene, cfg, .45)
    checkpoint = tmp_path / "resume.pkl"
    state = {"version": 1, "signature": original, "results": {0: None}}
    pipeline._save_resume_state(checkpoint, state)
    assert pipeline._load_resume_state(checkpoint, original, tasks)["results"] == {0: None}

    (tmp_path / "descriptors.py").write_text("# corrected distance\n", encoding="utf-8")
    updated = pipeline._resume_signature(tasks, scene, cfg, .45)
    assert updated != original
    with pytest.raises(ValueError, match="Checkpoint incompatible"):
        pipeline._load_resume_state(checkpoint, updated, tasks)


def test_final_rank_does_not_reward_visibility_instead_of_explained_geometry():
    good = SimpleNamespace(coverage=.80, cov_reverse=.43,
                           _geometric_cov_reverse=.74, mean_surface_dist=.02)
    poor = SimpleNamespace(coverage=.55, cov_reverse=.95,
                           _geometric_cov_reverse=.45, mean_surface_dist=.02)
    assert pipeline._registration_rank(good) < pipeline._registration_rank(poor)
    legacy = SimpleNamespace(coverage=.8, cov_reverse=.6, mean_surface_dist=.02)
    assert pipeline._ranking_reverse_coverage(legacy) == .6


def test_null_hypothesis_uses_same_geometric_metric_as_ranking():
    cfg = PipelineConfig()
    cfg.matching.null_hypothesis_score = .6
    reg = SimpleNamespace(model_name='chair_false', coverage=.7, cov_reverse=.99,
                          _geometric_cov_reverse=.25, mean_surface_dist=.02)
    scene = SimpleNamespace(cloud=G.PointCloud(np.zeros((2, 3))), up=np.array([0.,1.,0.]))
    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics([reg], scene, cfg)
    assert selected == []
    assert diagnostics[0]['status'] == 'rejected_null_hypothesis'


def test_worker_matches_candidate_groups_separately(monkeypatch):
    model = SimpleNamespace(
        mesh_path="model.obj", cloud=G.PointCloud(np.zeros((1, 3))),
    )
    monkeypatch.setattr(pipeline.db_store, "load_model", lambda path: model)
    monkeypatch.setattr(
        pipeline, "_footprint", lambda registration, loaded: np.zeros((1, 3)),
    )
    calls = []
    group_seed_flags = []

    def fake_match(loaded, scan_features, *args, trace=None, **kwargs):
        calls.append(list(scan_features))
        group_seed_flags.append(kwargs.get("allow_group_pose_seed"))
        if trace is not None:
            trace["model"] = "chair_test"
        return SimpleNamespace()

    monkeypatch.setattr(pipeline, "_match_model", fake_match)
    pipeline._WORKER.clear()
    pipeline._WORKER.update({
        "sfeats": ["all"], "scene": G.PointCloud(np.zeros((1, 3))),
        "cfg": PipelineConfig(), "up": np.array([0.0, 1.0, 0.0]),
        "thr": 0.4, "capture_trace": True,
        "candidate_feature_groups": {
            7: [(2, ["group-2-a", "group-2-b"]), (5, ["group-5"])],
        },
    })

    result = pipeline._work_file((7, "model.pkl"))

    assert calls == [["group-2-a", "group-2-b"], ["group-5"]]
    assert group_seed_flags == [True, True]
    assert [trace["query_group_id"] for trace in result["traces"]] == [2, 5]
    assert [reg._query_group_id for reg in result["registrations"]] == [2, 5]


def test_worker_falls_back_to_full_scene_without_group(monkeypatch):
    pipeline._WORKER.clear()
    pipeline._WORKER.update({
        "sfeats": ["all-a", "all-b"], "candidate_feature_groups": {},
    })

    assert pipeline._worker_feature_groups(4) == [
        (None, ["all-a", "all-b"]),
    ]


def test_group_pose_seed_recovers_a_coarse_yaw_alignment():
    cfg = PipelineConfig()
    cfg.matching.group_pose_seed_count = 6
    model_points = np.asarray([
        [-.4, 0., -.3], [.4, 0., -.3], [-.4, 0., .3], [.4, 0., .3],
        [-.4, .8, .3], [.4, .8, .3], [0., .4, .3],
    ])
    transform = GroundTransform(
        np.pi / 2, 1.1, np.asarray([1.2, .1, -.7]),
        np.asarray([0., 1., 0.]),
    )
    scan_points = transform.apply(model_points)
    features = lambda points: [
        SimpleNamespace(position=point) for point in points
    ]

    seeds = pipeline._group_pose_constellations(
        features(model_points), features(scan_points), cfg,
        np.asarray([0., 1., 0.]),
    )

    assert len(seeds) == 6
    recovered = seeds[0].transform.apply(model_points)
    assert np.mean(cKDTree(scan_points).query(recovered, k=1)[0]) < 0.08
    assert len(seeds[0].inliers) >= 3
    assert len({pair.model_idx for pair in seeds[0].inliers}) == len(
        seeds[0].inliers
    )
    assert len({pair.scan_idx for pair in seeds[0].inliers}) == len(
        seeds[0].inliers
    )


def test_scene_visibility_sampler_uses_the_aligned_volume_contract():
    expected = np.array([2, 0], np.uint8)
    volume = SimpleNamespace(
        sample_visibility_details=lambda positions: (
            expected, np.zeros(len(positions), bool),
        ),
    )
    scene = SimpleNamespace(volume=volume)

    sampler = pipeline._scene_visibility_sampler(scene)
    labels, uncertain = sampler(np.zeros((2, 3)))

    assert np.array_equal(labels, expected)
    assert not uncertain.any()


def test_matching_trace_keeps_all_scene_constellation_supports():
    up = np.array([0.0, 1.0, 0.0])
    transform = GroundTransform(0.0, 1.0, np.zeros(3), up)
    model_features = [SimpleNamespace(position=np.array([0.0, 0.2, 0.0]))]
    scan_features = [SimpleNamespace(position=np.array([0.1, 0.2, 0.1]))]
    correspondence = SimpleNamespace(
        model_idx=0, scan_idx=0, desc_dist=2.0, theta=0.0,
        scale=1.0, transform=transform,
    )
    constellations = [SimpleNamespace(
        quality=float(100 - index), transform=transform,
        inliers=[correspondence],
    ) for index in range(12)]

    correspondence_preview, constellation_preview, scene_support = (
        pipeline._trace_matching_geometry(
            [correspondence], constellations, model_features, scan_features,
        )
    )

    assert len(correspondence_preview) == 1
    assert len(constellation_preview) == 8
    assert len(scene_support) == 12
    assert scene_support[0] == {"quality": 100.0, "scan_keypoints": [0]}


def test_standard_matching_verifies_configured_shortlist(monkeypatch):
    cfg = PipelineConfig()
    cfg.matching.verification_top_constellations = 37
    captured = {}

    monkeypatch.setattr(pipeline, "match_features", lambda *args: [object()])
    monkeypatch.setattr(
        pipeline, "one_point_ransac", lambda *args, **kwargs: [object()],
    )

    def fake_verify(*args, **kwargs):
        captured["top"] = kwargs["top"]
        return None

    monkeypatch.setattr(pipeline, "verify_model", fake_verify)
    model = type("Model", (), {
        "name": "model",
        "features": [],
        "cloud": G.PointCloud(np.zeros((1, 3))),
    })()

    result = pipeline._match_model(
        model, [], G.PointCloud(np.zeros((1, 3))), cfg,
        np.array([0.0, 1.0, 0.0]), coverage_threshold=0.4,
    )

    assert result is None
    assert captured["top"] == 37


def test_non_max_select_respects_category_limits():
    cfg = PipelineConfig()
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()

    regs = []
    for i, name in enumerate(["chair_a", "chair_b", "chair_c", "table_a"]):
        reg = Registration(
            name,
            GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
            coverage=1.0 - 0.1 * i,
            mean_surface_dist=0.0,
            quality=1.0,
            cov_forward=1.0 - 0.1 * i,
            cov_reverse=1.0,
        )
        reg._footprint = np.array([[float(i), 0.0, 0.0]])
        regs.append(reg)

    selected = pipeline._non_max_select_fp(
        regs, scene, cfg, max_per_category={"chair": 2},
    )
    selected_with_diag, diagnostics = pipeline._non_max_select_fp_with_diagnostics(
        regs, scene, cfg, max_per_category={"chair": 2},
    )

    assert [reg.model_name for reg in selected] == ["chair_a", "chair_b", "table_a"]
    assert [reg.model_name for reg in selected_with_diag] == [
        "chair_a", "chair_b", "table_a",
    ]
    rejected = [entry for entry in diagnostics if entry["status"] != "selected"]
    assert rejected[0]["status"] == "rejected_category_limit"
    assert rejected[0]["model"] == "chair_c"


def test_non_max_select_prioritizes_symmetric_coverage():
    cfg = PipelineConfig()
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()

    regs = []
    for name, coverage, reverse, support_x in [
        ("chair_partial", 0.90, 0.20, 0.0),
        ("chair_balanced", 0.60, 0.50, 2.0),
    ]:
        reg = Registration(
            name,
            GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
            coverage=coverage,
            mean_surface_dist=0.0,
            quality=1.0,
            cov_forward=coverage,
            cov_reverse=reverse,
        )
        reg._footprint = np.array([[support_x, 0.0, 0.0]])
        regs.append(reg)

    selected = pipeline._non_max_select_fp(
        regs, scene, cfg, max_per_category={"chair": 1},
    )

    assert pipeline._symmetric_score(0.60, 0.50) > pipeline._symmetric_score(0.90, 0.20)
    assert [reg.model_name for reg in selected] == ["chair_balanced"]


def test_non_max_select_horizontal_iou_is_category_aware():
    cfg = PipelineConfig()
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()

    regs = []
    for name, coverage, support_x in [
        ("chair_a", 1.0, 0.0),
        ("table_a", 0.9, 1.0),
    ]:
        reg = Registration(
            name,
            GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
            coverage=coverage,
            mean_surface_dist=0.0,
            quality=1.0,
            cov_forward=coverage,
            cov_reverse=1.0,
        )
        reg._footprint = np.array([
            [support_x - 0.4, 0.0, -0.4],
            [support_x - 0.4, 0.0, 0.4],
            [support_x + 0.4, 0.0, -0.4],
            [support_x + 0.4, 0.0, 0.4],
            [support_x, 0.0, 0.0],
        ])
        regs.append(reg)

    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics(
        regs, scene, cfg,
    )

    assert [reg.model_name for reg in selected] == ["chair_a", "table_a"]
    assert [entry["status"] for entry in diagnostics] == ["selected", "selected"]


def test_non_max_select_rejects_near_cross_category_collision():
    cfg = PipelineConfig()
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([
            [-0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()

    regs = []
    for name, coverage, points in [
        ("chair_a", 1.0, [[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        ("table_a", 0.9, [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    ]:
        reg = Registration(
            name,
            GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
            coverage=coverage,
            mean_surface_dist=0.0,
            quality=1.0,
            cov_forward=coverage,
            cov_reverse=1.0,
        )
        reg._footprint = np.asarray(points, float)
        regs.append(reg)

    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics(
        regs, scene, cfg,
        global_overlap_thresh=0.9,
        cross_category_center_distance=0.55,
        cross_category_overlap_thresh=0.25,
    )

    assert [reg.model_name for reg in selected] == ["chair_a"]
    assert [entry["status"] for entry in diagnostics] == [
        "selected",
        "rejected_cross_category_collision",
    ]
    assert diagnostics[1]["conflict_model"] == "chair_a"


def test_non_max_select_rejects_cross_category_global_duplicate():
    cfg = PipelineConfig()
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([
            [0.0, 0.0, 0.0],
        ])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()

    regs = []
    for name, coverage in [
        ("chair_a", 1.0),
        ("table_a", 0.9),
    ]:
        reg = Registration(
            name,
            GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
            coverage=coverage,
            mean_surface_dist=0.0,
            quality=1.0,
            cov_forward=coverage,
            cov_reverse=1.0,
        )
        reg._footprint = np.array([[0.0, 0.0, 0.0]])
        regs.append(reg)

    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics(
        regs, scene, cfg,
    )

    assert [reg.model_name for reg in selected] == ["chair_a"]
    assert [entry["status"] for entry in diagnostics] == [
        "selected",
        "rejected_global_explained_overlap",
    ]


def test_non_max_select_rejects_category_min_coverage():
    cfg = PipelineConfig()
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([
            [0.0, 0.0, 0.0],
        ])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()
    reg = Registration(
        "table_a",
        GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
        coverage=0.59,
        mean_surface_dist=0.0,
        quality=1.0,
        cov_forward=0.59,
        cov_reverse=1.0,
    )
    reg._footprint = np.array([[0.0, 0.0, 0.0]])

    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics(
        [reg], scene, cfg, min_coverage_by_category={"table": 0.6},
    )

    assert selected == []
    assert diagnostics[0]["status"] == "rejected_category_min_coverage"
    assert diagnostics[0]["coverage"] == 0.59
    assert diagnostics[0]["category_min_coverage"] == 0.6


def test_non_max_select_rejects_min_symmetric_score():
    cfg = PipelineConfig()
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([[0.0, 0.0, 0.0]])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()
    reg = Registration(
        "couch_a",
        GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
        coverage=0.9,
        mean_surface_dist=0.0,
        quality=1.0,
        cov_forward=0.9,
        cov_reverse=0.25,
    )
    reg._footprint = np.array([[0.0, 0.0, 0.0]])

    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics(
        [reg], scene, cfg, min_symmetric_score=0.5,
    )

    assert selected == []
    assert diagnostics[0]["status"] == "rejected_min_symmetric_score"


def test_non_max_select_prefers_explicit_null_hypothesis_by_default():
    cfg = PipelineConfig()
    cfg.matching.null_hypothesis_score = 0.55
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([[0.0, 0.0, 0.0]])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()
    reg = Registration(
        "chair_weak",
        GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
        coverage=0.5, mean_surface_dist=0.0, quality=1.0,
        cov_forward=0.5, cov_reverse=0.5,
    )
    reg._footprint = np.array([[0.0, 0.0, 0.0]])

    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics(
        [reg], scene, cfg,
    )

    assert selected == []
    assert diagnostics[0]["status"] == "rejected_null_hypothesis"


def test_default_null_hypothesis_keeps_borderline_distributed_candidate():
    cfg = PipelineConfig()
    scene = type("Scene", (), {
        "cloud": G.PointCloud(np.array([[0.0, 0.0, 0.0]])),
        "up": np.array([0.0, 1.0, 0.0]),
    })()
    reg = Registration(
        "chair_borderline",
        GroundTransform(0.0, 1.0, np.zeros(3), scene.up),
        coverage=0.70, mean_surface_dist=0.02, quality=1.0,
        cov_forward=0.70, cov_reverse=0.36,
    )
    reg._footprint = np.array([[0.0, 0.0, 0.0]])

    selected, diagnostics = pipeline._non_max_select_fp_with_diagnostics(
        [reg], scene, cfg,
    )

    assert selected == [reg]
    assert diagnostics[0]["status"] == "selected"


class SimpleTSDF:
    """Mini-TSDF auto-suffisant pour le test (interface .sample compatible avec
    descriptors.build_scan_descriptor)."""

    def __init__(self, points, normals, voxel, trunc):
        self.voxel = voxel; self.trunc = trunc
        self.origin = points.min(0) - 3 * voxel
        bmax = points.max(0) + 3 * voxel
        self.dims = np.ceil((bmax - self.origin) / voxel).astype(int) + 1
        ax = [self.origin[a] + np.arange(self.dims[a]) * voxel for a in range(3)]
        gx, gy, gz = np.meshgrid(*ax, indexing="ij")
        centers = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        tree = cKDTree(points)
        dist, idx = tree.query(centers, workers=-1)
        sgn = np.sign(np.einsum('ij,ij->i', centers - points[idx], normals[idx]))
        sgn[sgn == 0] = 1.0
        self.tsdf = np.clip(sgn * dist, -trunc, trunc).reshape(self.dims).astype(np.float32)
        self.weight = (dist < 3 * trunc).reshape(self.dims).astype(np.float32)

    def sample(self, positions):
        rel = (np.asarray(positions, float) - self.origin) / self.voxel
        ijk = np.round(rel).astype(int)
        inb = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        t = np.full(len(positions), self.trunc, np.float32)
        w = np.zeros(len(positions), np.float32)
        g = ijk[inb]
        t[inb] = self.tsdf[g[:, 0], g[:, 1], g[:, 2]]
        w[inb] = self.weight[g[:, 0], g[:, 1], g[:, 2]]
        return t, w

    def sample_visibility(self, positions):
        t, w = self.sample(positions)
        labels = np.zeros(len(t), np.uint8)
        labels[(w > 0) & (t > 0.75 * self.voxel)] = 1
        labels[(w > 0) & (np.abs(t) <= 0.75 * self.voxel)] = 2
        return labels


def box_faces(lo, hi, n, rng):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    pts = []
    for axis in range(3):
        for val in (lo[axis], hi[axis]):
            uv = rng.uniform(0, 1, (n, 2))
            p = np.zeros((n, 3))
            others = [a for a in range(3) if a != axis]
            for k, a in enumerate(others):
                p[:, a] = lo[a] + uv[:, k] * (hi[a] - lo[a])
            p[:, axis] = val
            pts.append(p)
    return np.vstack(pts)


def inside(pts, lo, hi, eps=0.01):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return np.all((pts > lo + eps) & (pts < hi - eps), axis=1)


def sample_L(seed=0, n=1500):
    """Solide en L (union de deux boîtes), surface extérieure uniquement."""
    rng = np.random.default_rng(seed)
    A = ([0, 0, 0], [0.6, 0.5, 0.2])
    B = ([0, 0, 0], [0.2, 0.5, 0.6])
    pa = box_faces(*A, n, rng); pa = pa[~inside(pa, *B)]
    pb = box_faces(*B, n, rng); pb = pb[~inside(pb, *A)]
    return np.vstack([pa, pb])


def angle_diff(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def test_recover_transform():
    cfg = PipelineConfig()
    # SimpleTSDF approxime une distance au nuage sans lancer the rayons RGB-D.
    # Ce test cible la récupération de pose, pas la calibration UDF/visibilité.
    cfg.descriptor.distance_unit = 1.0
    cfg.ransac.desc_inlier = 128.0
    cfg.matching.descriptor_ratio_threshold = 0.0
    cfg.keypoint.neighbor_radius = 0.07
    cfg.keypoint.nms_radius = 0.12
    cfg.keypoint.dedup_radius = 0.08
    cfg.keypoint.curvature_threshold = 0.03
    up = np.array([0.0, 1.0, 0.0])

    # --- modèle ---
    mpts = sample_L(seed=0)
    model = G.make_point_cloud(mpts, radius=0.06)
    mkps = detect_keypoints(model, cfg.keypoint)
    ground_m = mpts[:, 1].min()
    mfeats = build_model_features(model, mkps, cfg, up, ground_m)
    print(f"modele: {model.size} pts, {mkps.size} key points")

    # --- scene = modèle transformé (transformation sol connue) ---
    T0 = GroundTransform(theta=0.7, scale=1.15, t=np.array([1.2, 0.0, -0.6]), up=up)
    R = G.rotation_about_axis(up, T0.theta)
    spts = T0.apply(mpts)
    snrm = model.normals @ R.T
    scene = G.make_point_cloud(spts, radius=0.06)
    ground_s = spts[:, 1].min()
    stsdf = SimpleTSDF(spts, scene.normals, voxel=0.01, trunc=0.04)
    skps = detect_keypoints(scene, cfg.keypoint)
    sfeats = build_scan_features(scene, stsdf, skps, cfg, up, ground_s)
    print(f"scene : {scene.size} pts, {skps.size} key points")

    # --- matching + RANSAC + vérification ---
    corres = match_features(mfeats, sfeats, cfg, up)
    print(f"correspondences putatives: {len(corres)}")
    cons = one_point_ransac(corres, mfeats, sfeats, cfg, up)
    print(f"constellations: {len(cons)}")
    reg = verify_model("L", model, scene, cons, threshold=0.05)
    assert reg is not None, "none registration trouvee"
    T = reg.transform
    print(f"VRAI : theta={T0.theta:.3f} scale={T0.scale:.3f} t={np.round(T0.t,3)}")
    print(f"ESTIM: theta={T.theta:.3f} scale={T.scale:.3f} t={np.round(T.t,3)}")
    print(f"couverture={reg.coverage:.2f} dist_surface={reg.mean_surface_dist*1000:.1f}mm")

    assert reg.coverage > 0.8, f"couverture trop faible: {reg.coverage}"
    assert angle_diff(T.theta, T0.theta) < np.radians(8), "theta mal recouvre"
    assert abs(T.scale - T0.scale) < 0.12, "echelle mal recouvree"
    assert np.linalg.norm(T.t - T0.t) < 0.1, "translation mal recouvree"


if __name__ == "__main__":
    test_recover_transform()
    print("OK pipeline registration")
