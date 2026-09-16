"""The orchestration must not truncate the adaptive fusion frame schedule."""
from types import SimpleNamespace

import pytest

import progressive
from config import PipelineConfig
from sdf_fusion import _select_frame_ids


@pytest.mark.parametrize("available,maximum,expected,last", [
    (200, 0, 50, 196), (250, 0, 50, 245), (1405, 0, 235, 1404),
    (200, 20, 20, 76), (12, 0, 12, 11),
])
def test_final_fusion_preserves_adaptive_schedule(
        monkeypatch, available, maximum, expected, last):
    cfg = PipelineConfig()
    cfg.sdf.frame_stride = 6
    cfg.sdf.min_frames = 50
    cfg.sdf.max_frames = maximum
    closed = []
    monkeypatch.setattr(progressive, "RGBDSequence", lambda *a, **kw:
                        SimpleNamespace(frame_ids=list(range(available)),
                                        close=lambda: closed.append(True)))
    total = progressive.selected_frame_count("scene.zip", cfg)
    assert total == expected
    # This is the orchestration's final fusion cap, not another subsampling.
    cfg.sdf.max_frames = total
    selected, _ = _select_frame_ids(range(available), cfg.sdf)
    assert len(selected) == expected
    assert selected[-1] == last
    assert closed == [True]


def test_scene_cache_invalidates_sampling_and_surface_changes(tmp_path):
    source = tmp_path / "scene.zip"
    source.write_bytes(b"fixture")
    cfg = PipelineConfig()
    signature = progressive._scene_cache_signature(source, cfg, 20, None)
    cfg.sdf.min_frames = 0
    assert progressive._scene_cache_signature(source, cfg, 20, None) != signature
    cfg.sdf.min_frames = 50
    cfg.sdf.surface_extraction = "zero_crossing"
    assert progressive._scene_cache_signature(source, cfg, 20, None) != signature
