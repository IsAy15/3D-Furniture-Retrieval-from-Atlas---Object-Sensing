import os
import sys
from types import SimpleNamespace
import json
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import progressive  # noqa: E402
from config import PipelineConfig  # noqa: E402
from geometry import PointCloud  # noqa: E402
from sdf_fusion import TSDFVolume, VIS_OCCUPIED  # noqa: E402
from transforms import GroundTransform  # noqa: E402
from verify import Registration  # noqa: E402


def test_progressive_checkpoints_include_final_frame():
    assert progressive.progressive_checkpoints(80, 20, 30) == [20, 50, 80]
    assert progressive.progressive_checkpoints(65, 20, 30) == [20, 50, 65]
    assert progressive.progressive_checkpoints(10, 20, 30) == [10]


def test_progressive_can_stop_after_fusion(monkeypatch):
    cfg = PipelineConfig()
    cfg.sdf.max_frames = 0
    scene = SimpleNamespace(
        cloud=SimpleNamespace(size=42),
        fusion_trace={"total_frames": 10},
    )
    monkeypatch.setattr(progressive, "selected_frame_count", lambda *args: 10)
    monkeypatch.setattr(
        progressive.pipeline, "scan_from_zip", lambda *args, **kwargs: scene,
    )

    result = progressive.progressive_retrieve_dir(
        "office.zip", "db", cfg, stop_after="fusion",
    )

    assert result.completed_stage == "fusion"
    assert result.scene is scene
    assert result.steps == []
    assert result.registrations == []


def test_single_query_fusion_materializes_only_final_scene(monkeypatch):
    cfg = PipelineConfig()
    cfg.sdf.max_frames = 0
    calls = []

    def fake_scan(_zip, step_cfg, progress=None, alignment_rotation=None,
                  fusion_trace=None):
        calls.append(int(step_cfg.sdf.max_frames))
        return SimpleNamespace(
            cloud=SimpleNamespace(size=42),
            alignment_rotation=np.eye(3),
            fusion_trace={"total_frames": int(step_cfg.sdf.max_frames)},
        )

    monkeypatch.setattr(progressive, "selected_frame_count", lambda *args: 50)
    monkeypatch.setattr(progressive.pipeline, "scan_from_zip", fake_scan)

    result = progressive.progressive_retrieve_dir(
        "office.zip", "db", cfg, query_mode="single",
        start_frames=20, interval_frames=10, stop_after="fusion",
    )

    assert result.completed_stage == "fusion"
    assert calls == [50]


def test_fusion_boundary_prepares_every_progressive_checkpoint(monkeypatch):
    cfg = PipelineConfig()
    cfg.sdf.max_frames = 0
    calls = []
    final_rotation = np.eye(3)

    def fake_scan(_zip, step_cfg, progress=None, alignment_rotation=None,
                  fusion_trace=None):
        frame_count = int(step_cfg.sdf.max_frames)
        calls.append((frame_count, alignment_rotation))
        return SimpleNamespace(
            cloud=SimpleNamespace(size=frame_count),
            alignment_rotation=final_rotation,
            fusion_trace={"total_frames": frame_count},
        )

    monkeypatch.setattr(progressive, "selected_frame_count", lambda *args: 50)
    monkeypatch.setattr(progressive.pipeline, "scan_from_zip", fake_scan)

    result = progressive.progressive_retrieve_dir(
        "office.zip", "db", cfg, start_frames=20, interval_frames=20,
        stop_after="fusion", query_mode="progressive",
    )

    assert result.completed_stage == "fusion"
    assert [frame_count for frame_count, _ in calls] == [50, 20, 40]
    assert calls[1][1] is final_rotation
    assert calls[2][1] is final_rotation


def test_progressive_returns_query_trace_at_partial_boundary(monkeypatch):
    cfg = PipelineConfig()
    cfg.sdf.max_frames = 0
    scene = SimpleNamespace(
        cloud=PointCloud(np.zeros((4, 3))),
        volume=None,
        ground=0.0,
        up=np.array([0.0, 1.0, 0.0]),
        alignment_rotation=np.eye(3),
        fusion_trace={"total_frames": 20},
    )
    monkeypatch.setattr(progressive, "selected_frame_count", lambda *args: 20)
    monkeypatch.setattr(
        progressive.pipeline, "scan_from_zip", lambda *args, **kwargs: scene,
    )

    def stop_at_query(*args, trace=None, **kwargs):
        trace.update({"preselection": {"top_k": ["chair_a"]}})
        raise progressive.pipeline.RetrievalStageComplete("query", trace=trace)

    monkeypatch.setattr(progressive.pipeline, "retrieve_dir", stop_at_query)

    result = progressive.progressive_retrieve_dir(
        "office.zip", "db", cfg, stop_after="query",
    )

    assert result.completed_stage == "query"
    assert len(result.steps) == 1
    assert result.steps[0].trace["full"]["preselection"]["top_k"] == [
        "chair_a",
    ]


def test_scene_cache_reuses_volume_and_invalidates_changed_sdf(tmp_path, monkeypatch):
    source = tmp_path / "office.zip"
    source.write_bytes(b"rgbd")
    cache_dir = tmp_path / "stage_cache"
    cfg = PipelineConfig()
    cfg.sdf.max_frames = 10
    calls = []

    def fake_scan(*args, **kwargs):
        calls.append(cfg.sdf.voxel_size)
        volume = TSDFVolume(
            [-0.02, -0.02, -0.02], [0.02, 0.02, 0.02],
            cfg.sdf.voxel_size, cfg.sdf.truncation,
        )
        volume.tsdf.fill(0.25)
        volume.weight.fill(1.0)
        aligned_volume = progressive.pipeline._AlignedVolume(volume, np.eye(3))
        return progressive.pipeline.SceneScan(
            PointCloud(np.asarray([[0.0, 0.0, 0.0]])),
            aligned_volume, 0.0, np.asarray([0.0, 1.0, 0.0]), np.eye(3),
            {"total_frames": 10},
        )

    monkeypatch.setattr(progressive.pipeline, "scan_from_zip", fake_scan)

    first = progressive._scan_from_cache_or_zip(
        source, cfg, 10, stage_cache_dir=cache_dir,
    )
    second = progressive._scan_from_cache_or_zip(
        source, cfg, 10, stage_cache_dir=cache_dir,
    )

    assert len(calls) == 1
    assert np.array_equal(first.volume.volume.tsdf, second.volume.volume.tsdf)
    assert second.volume.volume.backend == "cpu"
    assert second.volume.volume.xp is np

    cfg.sdf.voxel_size = 0.01
    progressive._scan_from_cache_or_zip(
        source, cfg, 10, stage_cache_dir=cache_dir,
    )
    assert len(calls) == 2


def test_progressive_queries_share_alignment_and_grow_cache(monkeypatch):
    cfg = PipelineConfig()
    cfg.sdf.max_frames = 0
    rotation = np.eye(3)
    scans = []
    retrieval_calls = []
    resume_calls = []
    exhaustive_flags = []

    monkeypatch.setattr(progressive, "selected_frame_count", lambda *args: 50)

    def fake_scan(zip_path, step_cfg, progress=None, alignment_rotation=None,
                  fusion_trace=None):
        scans.append((step_cfg.sdf.max_frames, alignment_rotation))
        used_rotation = rotation if alignment_rotation is None else alignment_rotation
        return SimpleNamespace(
            cloud=PointCloud(np.zeros((4, 3))),
            volume=None,
            ground=0.0,
            up=np.array([0.0, 1.0, 0.0]),
            alignment_rotation=used_rotation,
        )

    def fake_retrieve(scene, db_dir, step_cfg, **kwargs):
        resume_calls.append((
            kwargs.get("resume_dir"), kwargs.get("resume_key"),
            kwargs.get("resume_batch_size"),
        ))
        only = list(kwargs.get("candidate_only_names") or [])
        if only:
            retrieval_calls.append((step_cfg.sdf.max_frames, ["cache", *only]))
            return []
        exhaustive_flags.append(kwargs.get("exhaustive"))
        retrieval_calls.append((
            step_cfg.sdf.max_frames,
            list(kwargs["candidate_include_names"]),
        ))
        if step_cfg.sdf.max_frames == 20:
            return [SimpleNamespace(model_name="chair_a")]
        return [SimpleNamespace(model_name="table_b")]

    monkeypatch.setattr(progressive.pipeline, "scan_from_zip", fake_scan)
    monkeypatch.setattr(progressive.pipeline, "retrieve_dir", fake_retrieve)
    monkeypatch.setattr(
        progressive.pipeline, "_non_max_select_fp",
        lambda regs, scene, cfg, **kwargs: regs,
    )
    monkeypatch.setattr(
        progressive.pipeline, "_non_max_select_fp_with_diagnostics",
        lambda regs, scene, cfg, **kwargs: (
            regs,
            [
                {
                    "registration": reg,
                    "status": "selected",
                    "reason": "selected",
                }
                for reg in regs
            ],
        ),
    )

    result = progressive.progressive_retrieve_dir(
        "office.zip", "db", cfg,
        start_frames=20, interval_frames=30,
        query_mode="progressive",
        initial_cache_names=["seed_model"],
        exhaustive=True,
        resume_dir="resume",
        resume_batch_size=7,
    )

    assert [step.frame_count for step in result.steps] == [20, 50]
    assert scans[0] == (50, None)
    assert scans[1][0] == 20
    assert np.array_equal(scans[1][1], rotation)
    assert retrieval_calls == [
        (20, ["cache", "seed_model"]),
        (20, ["seed_model"]),
        (50, ["cache", "seed_model", "chair_a"]),
        (50, ["seed_model", "chair_a"]),
    ]
    assert result.steps[-1].cache_names == [
        "seed_model", "chair_a", "table_b",
    ]
    assert [reg.model_name for reg in result.final_candidates] == [
        "chair_a", "table_b",
    ]
    assert [entry["status"] for entry in result.final_diagnostics] == [
        "kept", "kept",
    ]
    assert [entry["status"] for entry in result.final_selection_diagnostics] == [
        "selected", "selected",
    ]
    assert exhaustive_flags == [True, True]
    assert resume_calls == [
        ("resume", "frames_000020_cache", 7),
        ("resume", "frames_000020_full", 7),
        ("resume", "frames_000050_cache", 7),
        ("resume", "frames_000050_full", 7),
    ]


def test_single_query_revalidates_candidates_hidden_by_checkpoint_nms(monkeypatch):
    cfg = PipelineConfig()
    cfg.sdf.max_frames = 0
    scene = SimpleNamespace(
        cloud=PointCloud(np.zeros((4, 3))), volume=None, ground=0.0,
        up=np.array([0.0, 1.0, 0.0]), alignment_rotation=np.eye(3),
    )
    table = SimpleNamespace(model_name="table_false")
    chair = SimpleNamespace(model_name="chair_good")
    seen = []

    monkeypatch.setattr(progressive, "selected_frame_count", lambda *args: 42)
    monkeypatch.setattr(
        progressive.pipeline, "scan_from_zip", lambda *args, **kwargs: scene,
    )
    retrieve_calls = []

    def retrieve(*args, **kwargs):
        retrieve_calls.append(kwargs)
        return [table, chair]

    monkeypatch.setattr(progressive.pipeline, "retrieve_dir", retrieve)
    monkeypatch.setattr(
        progressive.pipeline, "_non_max_select_fp",
        lambda regs, *args, **kwargs: [table],
    )

    def revalidate(registrations, *args, **kwargs):
        seen.extend(registrations)
        return [chair], []

    monkeypatch.setattr(progressive, "_final_revalidate", revalidate)
    monkeypatch.setattr(
        progressive.pipeline, "_non_max_select_fp_with_diagnostics",
        lambda regs, *args, **kwargs: (list(regs), []),
    )

    result = progressive.progressive_retrieve_dir(
        "single-chair.zip", "db", cfg, query_mode="single",
    )

    assert seen == [table, chair]
    assert result.registrations == [chair]
    assert retrieve_calls[0]["return_all_candidates"] is True


def test_final_revalidate_rejects_low_reverse_coverage(tmp_path):
    cfg = PipelineConfig()
    cfg.keypoint.neighbor_radius = 0.05
    up = np.array([0.0, 1.0, 0.0])
    model_pts = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0],
    ])
    clutter = np.array([
        [0.08, 0.08, 0.08],
        [0.08, 0.08, -0.08],
        [-0.08, 0.08, 0.08],
        [-0.08, 0.08, -0.08],
        [0.05, 0.05, 0.08],
        [0.05, 0.05, -0.08],
    ])
    scene = SimpleNamespace(
        cloud=PointCloud(np.vstack([model_pts, clutter])),
        up=up,
    )
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    meta = {
        "n": 1,
        "names": ["chair_a"],
        "synsets": ["03001627"],
        "mesh_paths": ["chair_a.obj"],
    }
    (db_dir / "index.json").write_text(json.dumps(meta), encoding="utf-8")
    model = SimpleNamespace(
        name="chair_a",
        cloud=PointCloud(model_pts),
        mesh_path="chair_a.obj",
    )
    with open(db_dir / "model_00000.pkl", "wb") as f:
        pickle.dump(model, f)

    reg = Registration(
        "chair_a",
        GroundTransform(0.0, 1.0, np.zeros(3), up),
        coverage=1.0,
        mean_surface_dist=0.0,
        quality=1.0,
        cov_forward=1.0,
        cov_reverse=1.0,
    )

    kept, diagnostics = progressive._final_revalidate(
        [reg], str(db_dir), scene, cfg, coverage_threshold=0.4,
        reverse_gate=0.8,
    )
    assert kept == []
    assert diagnostics[0]["status"] == "rejected_reverse"
    assert diagnostics[0]["reason"] == "cov_reverse < 0.8"


def test_final_revalidate_uses_visibility_instead_of_unknown_scene_surface(
        tmp_path):
    cfg = PipelineConfig()
    cfg.keypoint.neighbor_radius = 0.05
    cfg.matching.visibility_min_known_points = 1
    cfg.matching.visibility_min_known_fraction = 0.0
    up = np.array([0.0, 1.0, 0.0])
    model_pts = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0],
    ])
    clutter = np.array([
        [0.08, 0.08, 0.08],
        [0.08, 0.08, -0.08],
        [-0.08, 0.08, 0.08],
        [-0.08, 0.08, -0.08],
        [0.05, 0.05, 0.08],
        [0.05, 0.05, -0.08],
    ])

    class OccupiedVolume:
        def sample_visibility(self, points):
            return np.full(len(points), VIS_OCCUPIED, np.uint8)

    scene = SimpleNamespace(
        cloud=PointCloud(np.vstack([model_pts, clutter])),
        volume=OccupiedVolume(),
        up=up,
    )
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    meta = {
        "n": 1,
        "names": ["table_desk"],
        "synsets": ["04379243"],
        "mesh_paths": ["table_desk.obj"],
    }
    (db_dir / "index.json").write_text(json.dumps(meta), encoding="utf-8")
    model = SimpleNamespace(
        name="table_desk",
        cloud=PointCloud(model_pts),
        mesh_path="table_desk.obj",
    )
    with open(db_dir / "model_00000.pkl", "wb") as handle:
        pickle.dump(model, handle)
    reg = Registration(
        "table_desk",
        GroundTransform(0.0, 1.0, np.zeros(3), up),
        coverage=1.0,
        mean_surface_dist=0.0,
        quality=1.0,
        cov_forward=1.0,
        cov_reverse=1.0,
    )

    kept, diagnostics = progressive._final_revalidate(
        [reg], str(db_dir), scene, cfg, coverage_threshold=0.4,
        reverse_gate=0.8,
    )

    assert kept == [reg]
    assert reg.cov_reverse == 1.0
    assert reg._geometric_cov_reverse < 0.8
    assert reg._visibility_used is True
    assert diagnostics[0]["status"] == "kept"
    assert diagnostics[0]["visibility_support"] == 1.0
    assert diagnostics[0]["visibility_used"] is True


def test_final_revalidation_records_local_support_rejections(monkeypatch):
    """Incomplete support must produce an inspectable rejection, not TypeError."""
    cfg = PipelineConfig()
    cfg.matching.visibility_reverse_enabled = False
    up = np.array([0., 1., 0.])
    axis = np.linspace(0., 1., 8)
    points = np.array([[x, y, z] for x in axis for y in axis for z in axis])
    model = SimpleNamespace(cloud=PointCloud(points), mesh_path="table.obj")
    scene = SimpleNamespace(
        cloud=PointCloud(points[np.all(points <= .4, axis=1)]), up=up,
    )
    monkeypatch.setattr(progressive.db_store, "load_index", lambda _: {"names": ["table_test"]})
    monkeypatch.setattr(progressive.db_store, "model_files", lambda _: ["model.pkl"])
    monkeypatch.setattr(progressive.db_store, "load_model", lambda _: model)
    for zones, extent, expected in [
        (.35, .2, "rejected_model_zones"),
        (0., .8, "rejected_support_locality"),
    ]:
        cfg.matching.min_model_zone_fraction = zones
        cfg.matching.min_support_extent_ratio = extent
        reg = Registration("table_test", GroundTransform(0., 1., np.zeros(3), up), 1., 0., 1.)
        kept, diagnostics = progressive._final_revalidate(
            [reg], "unused", scene, cfg, coverage_threshold=0., reverse_gate=0.,
        )
        assert kept == []
        assert diagnostics[0]["status"] == expected
        assert diagnostics[0]["supported_model_zones"] == 1
        assert diagnostics[0]["model_zone_count"] == 8
        assert diagnostics[0]["model_zone_fraction"] == .125
        assert 0. < diagnostics[0]["support_extent_ratio"] < .4
