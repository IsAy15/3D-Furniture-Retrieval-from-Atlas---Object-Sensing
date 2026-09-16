'\nPlace retrieved models from the JSON summary into the scene and export a colored PLY: grey scan, distinct model colors. Supports MeshLab and the bundled final HTML viewer.\n'

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import trimesh

# Hauteurs physiques par catégorie (identiques à database.py / preprocessing).
CATEGORY_HEIGHT = {"chair": 0.90, "table": 0.75, "couch": 0.80}

PALETTE = np.array([
    [230, 160, 60], [90, 200, 140], [230, 120, 120], [150, 140, 250],
    [70, 200, 200], [220, 200, 90], [240, 130, 190], [140, 210, 110],
], dtype=np.uint8)


def _load_mesh(mesh_ref: str):
    'Load a mesh from an OBJ path or archive.zip!member reference.'
    if "!" in mesh_ref:
        zp, member = mesh_ref.split("!", 1)
        with zipfile.ZipFile(zp) as zf:
            data = zf.read(member)
        m = trimesh.load(io.BytesIO(data), file_type="obj", force="mesh", process=False)
    else:
        m = trimesh.load(mesh_ref, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def _normalize_vertices(V: np.ndarray, category: str) -> np.ndarray:
    'Use preprocessing normalization: category scale, minimum Y at ground and horizontal centering.'
    V = np.asarray(V, float).copy()
    h = V[:, 1].max() - V[:, 1].min()
    s = (CATEGORY_HEIGHT.get(category, 0.8) / h) if h > 1e-6 else 1.0
    V *= s
    V[:, 1] -= V[:, 1].min()
    V[:, 0] -= (V[:, 0].max() + V[:, 0].min()) / 2
    V[:, 2] -= (V[:, 2].max() + V[:, 2].min()) / 2
    return V


def _transform_matrix(entry, up=(0, 1, 0)) -> np.ndarray:
    'Reconstruct the 4x4 model-to-scene transform from a JSON summary.'
    from geometry import rotation_about_axis
    theta = np.radians(entry["theta_deg"])
    scale = entry["scale"]
    t = np.asarray(entry["translation"], float)
    M = np.eye(4)
    M[:3, :3] = rotation_about_axis(up, theta) * scale
    M[:3, 3] = t
    return M


def write_points_ply(path, points, colors=None):
    points = np.asarray(points, np.float32)
    n = len(points)
    if colors is None:
        colors = np.full((n, 3), 180, np.uint8)
    colors = np.asarray(colors, np.uint8)
    with open(path, "wb") as f:
        f.write((
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii"))
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        v = np.empty(n, dt)
        v["x"], v["y"], v["z"] = points[:, 0], points[:, 1], points[:, 2]
        v["r"], v["g"], v["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
        v.tofile(f)


def read_points_ply(path):
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
        n = int([l for l in header.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
        data = f.read()
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a = np.frombuffer(data[: n * dt.itemsize], dtype=dt)
    pts = np.column_stack([a["x"], a["y"], a["z"]]).astype(np.float32)
    col = np.column_stack([a["r"], a["g"], a["b"]]).astype(np.uint8)
    return pts, col


def build_visualization(json_path, out_ply, scan_ply=None, n_per_model=30000, progress=None):
    summary = json.loads(Path(json_path).read_text(encoding="utf-8"))
    all_pts, all_col = [], []
    if scan_ply and Path(scan_ply).exists():
        sp, sc = read_points_ply(scan_ply)
        all_pts.append(sp)
        all_col.append(np.full((len(sp), 3), 150, np.uint8))     # scan en gris
        if progress:
            progress(f"scan: {len(sp)} points")
    rng_state = np.random.get_state()
    np.random.seed(0)
    try:
        for i, entry in enumerate(summary):
            try:
                mesh = _load_mesh(entry["mesh"])
            except Exception as exc:
                if progress:
                    progress(f"[skip] {entry['model']}: {exc}")
                continue
            category = entry["model"].split("_")[0]
            V = _normalize_vertices(mesh.vertices, category)
            m2 = trimesh.Trimesh(V, mesh.faces, process=False)
            pts, _ = trimesh.sample.sample_surface(m2, n_per_model)
            M = _transform_matrix(entry)
            pts = (M[:3, :3] @ np.asarray(pts).T).T + M[:3, 3]
            all_pts.append(pts.astype(np.float32))
            all_col.append(np.tile(PALETTE[i % len(PALETTE)], (len(pts), 1)))
            if progress:
                progress(f"{entry['model']}: placed ({len(pts)} pts)")
    finally:
        np.random.set_state(rng_state)
    if not all_pts:
        raise ValueError('Nothing to visualize.')
    write_points_ply(out_ply, np.vstack(all_pts), np.vstack(all_col))
    return out_ply
