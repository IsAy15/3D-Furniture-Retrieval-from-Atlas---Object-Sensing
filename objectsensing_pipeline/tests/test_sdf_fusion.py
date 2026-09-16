"""Test synthétique de la fusion TSDF : on rend la profondeur d'une sphère depuis
plusieurs poses connues, on fusionne, et on vérifie que l'iso-surface recouvre la
sphère (rayon correct) avec des normales radiales."""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PipelineConfig  # noqa: E402
from sdf_fusion import (  # noqa: E402
    DenseVolumeLimitError,
    RGBDFrame,
    SparseTSDFVolume,
    TSDFVolume,
    VIS_FREE,
    VIS_OCCUPIED,
    VIS_UNKNOWN,
    _curvature_from_point_neighborhoods,
    _depth_integration_weights,
    _smooth_observed_tsdf,
    _select_frame_ids,
    _trace_frame_sample,
    _visibility_hole_edge_image,
    _zero_crossing_points,
)


def test_adaptive_frame_selection_preserves_short_scene_view_budget():
    cfg = PipelineConfig().sdf
    cfg.frame_stride = 6
    cfg.min_frames = 50
    cfg.max_frames = 0

    office, office_stride = _select_frame_ids(list(range(1405)), cfg)
    ikea, ikea_stride = _select_frame_ids(list(range(200)), cfg)
    chair, chair_stride = _select_frame_ids(list(range(250)), cfg)

    assert (office_stride, len(office)) == (6, 235)
    assert (ikea_stride, len(ikea)) == (4, 50)
    assert (chair_stride, len(chair)) == (5, 50)


def test_frame_selection_can_disable_adaptive_minimum_and_apply_cap():
    cfg = PipelineConfig().sdf
    cfg.frame_stride = 6
    cfg.min_frames = 0
    cfg.max_frames = 20

    selected, effective_stride = _select_frame_ids(list(range(200)), cfg)

    assert effective_stride == 6
    assert selected == list(range(0, 120, 6))


def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)          # +Z caméra regarde la cible
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    pose = np.eye(4)
    pose[:3, 0] = r; pose[:3, 1] = -u; pose[:3, 2] = f   # X droite, Y bas, Z avant
    pose[:3, 3] = eye
    return pose


def render_sphere_depth(pose, K, H, W, center, radius):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, float)], axis=-1)
    R = pose[:3, :3]; o = pose[:3, 3]
    d_world = dirs @ R.T                                # directions (non normalisées) ; z=1 base
    oc = o - np.asarray(center, float)
    a = np.einsum('ijk,ijk->ij', d_world, d_world)
    b = 2 * np.einsum('ijk,k->ij', d_world, oc)
    c = oc @ oc - radius * radius
    disc = b * b - 4 * a * c
    depth = np.zeros((H, W), np.float32)
    hit = disc >= 0
    t = np.zeros((H, W))
    t[hit] = (-b[hit] - np.sqrt(disc[hit])) / (2 * a[hit])   # racine proche
    valid = hit & (t > 0)
    depth[valid] = t[valid]                              # t == profondeur z caméra (dir z=1)
    return depth


def test_fuse_sphere():
    center = np.array([0.0, 0.0, 0.0]); radius = 0.5
    H, W = 120, 160
    K = np.array([[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1]])
    vol = TSDFVolume([-0.7, -0.7, -0.7], [0.7, 0.7, 0.7], voxel_size=0.02, truncation=0.06)
    n_views = 12
    for i in range(n_views):
        ang = 2 * np.pi * i / n_views
        eye = center + 2.0 * np.array([np.cos(ang), 0.3, np.sin(ang)])
        pose = look_at(eye, center)
        depth = render_sphere_depth(pose, K, H, W, center, radius)
        color = np.zeros((H, W, 3), np.uint8)
        vol.integrate(RGBDFrame(str(i), depth, color, pose), K, depth_trunc=4.0)
    cloud = vol.extract_surface(sampling=0.02)
    assert cloud.size > 500, f"trop peu de points: {cloud.size}"
    assert cloud.confidence.shape == (cloud.size,)
    assert np.all(cloud.confidence > 0)
    r = np.linalg.norm(cloud.points - center, axis=1)
    err = np.abs(r - radius)
    # normales radiales : n . (p-c)/|p-c| ~ +-1
    rad = (cloud.points - center) / np.clip(r[:, None], 1e-9, None)
    align = np.abs(np.einsum('ij,ij->i', cloud.normals, rad))
    print(f"points={cloud.size} rayon moyen={r.mean():.3f} (attendu {radius}) "
          f"err_med={np.median(err):.3f}m  alignement_normales_median={np.median(align):.2f}")
    assert np.median(err) < 0.03, "iso-surface trop loin de la sphere"
    assert np.median(align) > 0.9, "normales pas assez radiales"


def test_tsdf_sampling_is_trilinear():
    vol = TSDFVolume([0, 0, 0], [0.08, 0.08, 0.08], voxel_size=0.02, truncation=1.0)
    gx, gy, gz = np.meshgrid(vol._ax[0], vol._ax[1], vol._ax[2], indexing="ij")
    vol.tsdf = (gx + 2.0 * gy + 3.0 * gz).astype(np.float32)
    vol.weight[:] = 1.0

    p = np.array([[0.031, 0.037, 0.043]])
    t, w = vol.sample(p)
    grad = vol.sample_gradient(p)

    assert np.allclose(t[0], p[0, 0] + 2.0 * p[0, 1] + 3.0 * p[0, 2], atol=1e-6)
    assert np.allclose(w[0], 1.0)
    assert np.allclose(grad[0], [1.0, 2.0, 3.0], atol=1e-5)


def test_hybrid_surface_recovers_supported_near_zero_voxels():
    kwargs = dict(
        bounds_min=[0, 0, 0], bounds_max=[0.12, 0.12, 0.12],
        voxel_size=0.02, truncation=0.06,
        surface_rescue_min_weight=5.0,
    )
    strict = TSDFVolume(**kwargs, surface_extraction="zero_crossing")
    hybrid = TSDFVolume(**kwargs, surface_extraction="hybrid")
    for volume in (strict, hybrid):
        x = np.arange(volume.dims[0], dtype=np.float32)[:, None, None]
        volume.tsdf[:] = 0.004 + x * 0.001
        volume.weight[:] = 6.0

    strict_cloud = strict.extract_surface(sampling=0.02)
    hybrid_cloud = hybrid.extract_surface(sampling=0.02)

    assert strict_cloud.size == 0
    assert hybrid_cloud.size > 0
    assert hybrid.surface_extraction_stats["rescued_points"] > 0


@pytest.mark.smoke
def test_visibility_distinguishes_free_surface_and_occluded_space():
    H, W = 24, 32
    K = np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]])
    depth = np.full((H, W), 1.0, np.float32)
    frame = RGBDFrame("0", depth, np.zeros((H, W, 3), np.uint8), np.eye(4))
    vol = TSDFVolume(
        [-0.1, -0.1, 0.2], [0.1, 0.1, 1.3],
        voxel_size=0.02, truncation=0.06, backend="cpu",
    )
    vol.integrate(frame, K, depth_trunc=2.0)

    labels = vol.sample_visibility(np.array([
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.2],
    ]))

    assert labels.tolist() == [VIS_FREE, VIS_OCCUPIED, VIS_UNKNOWN]


def test_hole_boundary_requires_no_reliable_surface_observation():
    height, width = 24, 32
    K = np.array([
        [30.0, 0, width / 2],
        [0, 30.0, height / 2],
        [0, 0, 1],
    ])
    depth_with_hole = np.full((height, width), 1.0, np.float32)
    depth_with_hole[height // 2, width // 2 + 1] = 0.0
    frame = RGBDFrame(
        "edge", depth_with_hole,
        np.zeros((height, width, 3), np.uint8), np.eye(4),
    )
    volume = TSDFVolume(
        [-0.1, -0.1, 0.8], [0.1, 0.1, 1.2],
        voxel_size=0.02, truncation=0.06, backend="cpu",
    )
    volume.integrate(frame, K, depth_trunc=2.0)

    labels, boundary = volume.sample_visibility_details(
        np.array([[0.0, 0.0, 1.0]]),
    )
    assert labels.tolist() == [VIS_OCCUPIED]
    assert boundary.tolist() == [True]

    clean = RGBDFrame(
        "clean", np.full((height, width), 1.0, np.float32),
        np.zeros((height, width, 3), np.uint8), np.eye(4),
    )
    volume.integrate(clean, K, depth_trunc=2.0)
    _, confirmed_boundary = volume.sample_visibility_details(
        np.array([[0.0, 0.0, 1.0]]),
    )
    assert confirmed_boundary.tolist() == [False]


def test_hole_edge_mask_marks_valid_pixels_next_to_missing_depth():
    depth = np.full((5, 5), 1000, np.uint16)
    depth[2, 2] = 0
    edge = _visibility_hole_edge_image(depth, radius=1)

    assert edge[2, 2] == 0
    assert edge[2, 1] == 1
    assert edge[0, 0] == 1


def test_sparse_volume_preserves_surface_visibility_and_serialization():
    pytest.importorskip("open3d")
    height, width = 24, 32
    K = np.array([
        [30.0, 0, width / 2],
        [0, 30.0, height / 2],
        [0, 0, 1],
    ])
    depth = np.full((height, width), 1.0, np.float32)
    frame = RGBDFrame(
        "0", depth, np.zeros((height, width, 3), np.uint8),
        np.eye(4),
    )
    volume = SparseTSDFVolume(
        [-0.5, -0.5, 0.2], [0.5, 0.5, 1.3],
        voxel_size=0.025, truncation=0.075,
        K=K, depth_trunc=2.0, backend="cpu",
        block_count=1000, visibility_depth_stride=2,
        depth_edge_threshold=0.0,
    )
    volume.integrate(frame)
    labels = volume.sample_visibility(np.array([
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.2],
    ]))
    restored = pickle.loads(pickle.dumps(volume))
    restored_labels = restored.sample_visibility(np.array([
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.2],
    ]))

    center = np.zeros(3)
    radius = 0.5
    sphere_height, sphere_width = 90, 120
    sphere_K = np.array([
        [120.0, 0, sphere_width / 2],
        [0, 120.0, sphere_height / 2],
        [0, 0, 1],
    ])
    sphere_volume = SparseTSDFVolume(
        [-0.7, -0.7, -0.7], [0.7, 0.7, 0.7],
        voxel_size=0.025, truncation=0.075,
        K=sphere_K, depth_trunc=4.0, backend="cpu",
        block_count=2000, depth_edge_threshold=0.0,
    )
    for index in range(8):
        angle = 2 * np.pi * index / 8
        eye = 2.0 * np.array([
            np.cos(angle), 0.25, np.sin(angle),
        ])
        pose = look_at(eye, center)
        sphere_depth = render_sphere_depth(
            pose, sphere_K, sphere_height, sphere_width,
            center, radius,
        )
        sphere_volume.integrate(RGBDFrame(
            str(index), sphere_depth,
            np.zeros((sphere_height, sphere_width, 3), np.uint8),
            pose,
        ))
    cloud = sphere_volume.extract_surface(sampling=0.025)

    assert cloud.size > 100
    assert cloud.confidence.shape == (cloud.size,)
    assert np.all(cloud.confidence > 0)
    assert labels.tolist() == [VIS_FREE, VIS_OCCUPIED, VIS_UNKNOWN]
    assert restored_labels.tolist() == labels.tolist()
    assert restored.visibility_summary()["layout"] == "sparse"


def test_depth_edge_weights_preserve_foreground_and_reduce_background():
    depth = np.full((7, 9), 2.0, np.float32)
    depth[:, 4] = 1.0

    weights, background = _depth_integration_weights(
        depth, depth_trunc=4.5,
        edge_threshold=0.05, background_weight=0.10, radius=1,
    )

    assert np.all(weights[:, 4] == 1.0)
    assert np.all(weights[:, 3] == 0.10)
    assert np.all(weights[:, 5] == 0.10)
    assert np.all(weights[:, :3] == 1.0)
    assert np.all(background[:, 3])
    assert not np.any(background[:, 4])


def test_saturated_tsdf_average_stays_inside_truncation_band():
    height, width = 12, 16
    K = np.array([
        [20.0, 0, width / 2],
        [0, 20.0, height / 2],
        [0, 0, 1],
    ])
    frame = RGBDFrame(
        "0",
        np.full((height, width), 1.0, np.float32),
        np.zeros((height, width, 3), np.uint8),
        np.eye(4),
    )
    vol = TSDFVolume(
        [-0.1, -0.1, 0.5], [0.1, 0.1, 1.1],
        voxel_size=0.04, truncation=0.08,
        depth_edge_threshold=0.0,
    )

    for _ in range(80):
        vol.integrate(frame, K, depth_trunc=2.0, max_weight=8.0)

    observed = vol.weight > 0
    assert np.max(np.abs(vol.tsdf[observed])) <= vol.trunc + 1e-6
    assert np.max(vol.weight) <= 8.0


def test_auto_backend_falls_back_to_cpu_without_cuda():
    vol = TSDFVolume([0, 0, 0], [0.1, 0.1, 0.1], 0.02, 0.06, backend="auto")
    assert vol.backend in {"cpu", "cuda"}


def test_dense_volume_limit_fails_before_large_allocation(monkeypatch):
    monkeypatch.setenv("OBJECTSENSING_MAX_DENSE_VOXELS", "1000")

    with pytest.raises(DenseVolumeLimitError) as caught:
        TSDFVolume(
            [0, 0, 0], [1, 1, 1],
            voxel_size=0.02, truncation=0.06, backend="cpu",
        )

    error = caught.value
    assert error.voxel_count > error.limit
    assert error.recommended_voxel_size > error.voxel_size
    assert "Dense TSDF volume too large" in str(error)
    assert 'Use at least' in str(error)


def test_voxel_centers_use_compact_float32_storage():
    vol = TSDFVolume(
        [0, 0, 0], [0.08, 0.08, 0.08],
        voxel_size=0.02, truncation=0.06, backend="cpu",
    )

    centers = vol.voxel_centers()

    assert centers.dtype == np.float32
    assert centers.shape == (int(np.prod(vol.dims)), 3)
    assert np.allclose(centers[0], [axis[0] for axis in vol._ax])
    assert np.allclose(centers[-1], [axis[-1] for axis in vol._ax])


def test_trace_frame_sample_keeps_world_points_and_colors():
    height, width = 20, 30
    K = np.array([[30.0, 0, width / 2], [0, 30.0, height / 2], [0, 0, 1]])
    depth = np.full((height, width), 1.0, np.float32)
    color = np.zeros((height, width, 3), np.uint8)
    color[..., 0] = 120
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]

    trace = _trace_frame_sample(
        RGBDFrame("demo", depth, color, pose), K, 2.0, max_points=25,
    )

    assert trace["frame_id"] == "demo"
    assert 0 < len(trace["points"]) <= 25
    assert len(trace["points"]) == len(trace["colors"])
    assert trace["camera_position"] == [1.0, 2.0, 3.0]
    assert all(rgb == [120, 0, 0] for rgb in trace["colors"])


def test_tsdf_smoothing_does_not_create_gradient_at_unknown_boundary():
    shape = (9, 9, 9)
    tsdf = np.ones(shape, np.float32)
    weight = np.zeros(shape, np.float32)
    z = np.linspace(-0.2, 0.2, shape[2], dtype=np.float32)
    tsdf[:5, :, :] = z[None, None, :]
    weight[:5, :, :] = 1.0

    smoothed, support = _smooth_observed_tsdf(tsdf, weight)
    gx, _, gz = np.gradient(smoothed, 0.05)

    assert support[4, 4, 4] >= 0.35
    assert abs(float(gx[4, 4, 4])) < 0.05
    assert float(gz[4, 4, 4]) > 0.5


def test_raw_surface_preserves_feature_erased_by_smoothed_field():
    vol = TSDFVolume(
        [0, 0, 0], [0.12, 0.12, 0.12],
        voxel_size=0.02, truncation=0.06,
        surface_field="raw", surface_min_support=0.0,
        gradient_smoothing_sigma=0.8,
    )
    vol.tsdf[:] = 1.0
    vol.weight[:] = 1.0
    center = int(vol.dims[0] // 2)
    vol.tsdf[center, :, :] = -0.2

    raw_points, _ = vol.zero_crossing_surface("raw")
    smoothed_points, _ = vol.zero_crossing_surface("smoothed")

    assert len(raw_points) > 0
    assert len(smoothed_points) == 0


def test_surface_settings_are_validated_and_serialized():
    with pytest.raises(ValueError, match="surface_field"):
        TSDFVolume(
            [0, 0, 0], [0.1, 0.1, 0.1],
            0.02, 0.06, surface_field="invalid",
        )

    cfg = PipelineConfig().sdf
    assert cfg.surface_field == "raw"
    assert cfg.surface_min_support == pytest.approx(0.20)
    assert cfg.gradient_smoothing_sigma == pytest.approx(0.8)


def test_position_pca_keeps_a_noisy_normal_plane_flat():
    grid = np.linspace(-0.2, 0.2, 21)
    xx, yy = np.meshgrid(grid, grid)
    plane = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    curvature = _curvature_from_point_neighborhoods(plane, radius=0.08)

    assert np.quantile(curvature, 0.95) < 1e-6


def test_zero_crossing_extraction_returns_one_thin_plane():
    shape = (9, 10, 11)
    voxel_size = 0.02
    origin = np.array([-0.08, -0.09, -0.10])
    axes = [origin[i] + np.arange(shape[i]) * voxel_size for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    plane_x = 0.013
    field = (gx - plane_x).astype(np.float32)
    observed = np.ones(shape, bool)
    support = np.ones(shape, np.float32)
    gradients = (
        np.ones(shape, np.float32),
        np.zeros(shape, np.float32),
        np.zeros(shape, np.float32),
    )

    points, normals = _zero_crossing_points(
        field, observed, support, gradients, origin, voxel_size,
    )

    assert len(points) == shape[1] * shape[2]
    assert np.allclose(points[:, 0], plane_x, atol=1e-7)
    assert np.allclose(normals, [1.0, 0.0, 0.0])


def test_batched_position_curvature_matches_scalar_pca():
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(12)
    points = rng.normal(size=(180, 3)) * [0.12, 0.08, 0.04]
    radius = 0.11
    actual = _curvature_from_point_neighborhoods(points, radius)

    expected = np.zeros(len(points))
    tree = cKDTree(points)
    for i, idx in enumerate(tree.query_ball_point(points, radius)):
        if len(idx) < 6:
            continue
        centered = points[idx] - points[idx].mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
        expected[i] = eigenvalues[0] / eigenvalues.sum()

    assert np.allclose(actual, expected, atol=1e-10)


if __name__ == "__main__":
    test_fuse_sphere()
    print("OK sdf_fusion")
