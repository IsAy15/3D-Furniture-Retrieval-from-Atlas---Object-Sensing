'\nCAD model preprocessing: load ShapeNetCore, normalize scale and ground alignment, sample geometry, detect keypoints and compute descriptors. Optional UDF k-means clustering accelerates matching. Expected layout: <root>/<synset>/<model_id>/models/model_normalized.obj.\n'

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import PipelineConfig
from geometry import PointCloud, make_point_cloud
from keypoints import KeyPoints, detect_keypoints
from matching import KeyPointFeature, build_model_features
from shapenet_loader import ModelSource, find_model_sources

# Hauteurs physiques plausibles par catégorie (approxime [SCB*14] pour
# l'initialisation d'scale). Synset WordNet -> hauteur cible en mètres.
CATEGORY_HEIGHT = {
    "03001627": 0.90,   # chair
    "04379243": 0.75,   # table
    "04256520": 0.80,   # sofa/couch
}
CATEGORY_NAME = {"03001627": "chair", "04379243": "table", "04256520": "couch"}


@dataclass
class DatabaseModel:
    name: str
    synset: str
    cloud: PointCloud
    keypoints: KeyPoints
    features: List[KeyPointFeature]
    ground: float
    mesh_path: str = ""
    surface_point_count: Optional[int] = None


def object_surface_cloud(model):
    """Geometry for registration, excluding the descriptor-only virtual floor.

    Old database pickles preserve _virtual_scan's ordered surface + 25% floor.
    Recognize that layout only when the entire suffix is on the minimum plane
    and extends beyond the object's horizontal bounds. Other clouds stay intact.
    """
    cloud = model.cloud
    points = cloud.points
    count = getattr(model, "surface_point_count", None)
    if count is None and isinstance(model, DatabaseModel) and len(points) >= 80:
        candidate = len(points) * 4 // 5
        if candidate + candidate // 4 == len(points):
            surface, floor = points[:candidate], points[candidate:]
            on_ground = np.all(floor[:, 1] == points[:, 1].min())
            outside = np.any((floor[:, [0, 2]] < surface[:, [0, 2]].min(0)) |
                             (floor[:, [0, 2]] > surface[:, [0, 2]].max(0)))
            if on_ground and outside:
                count = candidate
    if count is None or not 0 < int(count) < len(points):
        return cloud
    return cloud.subset(np.arange(int(count)))


@dataclass
class ModelDatabase:
    models: List[DatabaseModel] = field(default_factory=list)
    cfg: Optional[PipelineConfig] = None
    cluster_reps: Optional[np.ndarray] = None      # (C, D) représentants de clusters


def _list_obj_files(root: str, synset: str) -> List[str]:
    base = os.path.join(root, synset)
    cand = glob.glob(os.path.join(base, "*", "models", "model_normalized.obj"))
    if not cand:
        cand = glob.glob(os.path.join(base, "**", "*.obj"), recursive=True)
    return sorted(cand)


def _virtual_scan(mesh, n_points: int, add_ground: bool, rng) -> np.ndarray:
    'Sample mesh surfaces and add a virtual ground patch below the object.'
    pts, _ = _sample_surface(mesh, n_points, rng)
    if add_ground:
        lo = pts.min(0); hi = pts.max(0)
        ext = (hi - lo)
        m = max(ext[0], ext[2]) * 0.6 + 0.1
        cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
        g = rng.uniform([-m, 0, -m], [m, 0, m], size=(n_points // 4, 3))
        g[:, 0] += cx; g[:, 2] += cz; g[:, 1] = lo[1]
        pts = np.vstack([pts, g])
    return pts


def _sample_surface(mesh, n, rng):
    'Uniform sampling weighted by triangle area.'
    import contextlib
    import io
    import trimesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    # sample_surface_even évite the paquets aléatoires qui font disparaître des
    # coins de l'objet dans le virtual scan, tout en restant déterministe.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pts, fid = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    if len(pts) < n:
        extra_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        extra, extra_fid = trimesh.sample.sample_surface(mesh, n - len(pts), seed=extra_seed)
        pts = np.vstack([pts, extra])
        fid = np.concatenate([fid, extra_fid])
    return np.asarray(pts), fid


def _normalize_mesh(mesh, synset: str):
    'Keep Y-up orientation, place the base at Y=0, center horizontally and scale to a plausible physical height.'
    if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        return None
    V = np.asarray(mesh.vertices, float)
    height = V[:, 1].max() - V[:, 1].min()
    target = CATEGORY_HEIGHT.get(synset, 0.8)
    s = (target / height) if height > 1e-6 else 1.0
    V = V * s
    V[:, 1] -= V[:, 1].min()
    V[:, 0] -= (V[:, 0].max() + V[:, 0].min()) / 2
    V[:, 2] -= (V[:, 2].max() + V[:, 2].min()) / 2
    mesh.vertices = V
    return mesh


def preprocess_mesh(mesh, name: str, synset: str, cfg: PipelineConfig,
                    n_points: int = 6000, add_ground: bool = True, seed: int = 0,
                    mesh_path: str = "", n_surface: int = 80000) -> Optional[DatabaseModel]:
    'Preprocess a loaded mesh: normalization, surface sampling, keypoints and descriptors.'
    mesh = _normalize_mesh(mesh, synset)
    if mesh is None:
        return None
    rng = np.random.default_rng(seed)
    pts = _virtual_scan(mesh, n_points, add_ground, rng)
    up = np.asarray(cfg.up_axis, float); up = up / np.linalg.norm(up)
    cloud = make_point_cloud(pts, radius=cfg.keypoint.neighbor_radius)
    ground = float((pts @ up).min())
    kps = detect_keypoints(cloud, cfg.keypoint)
    # NB : UDF locale calculée sur le nuage du modèle (cloud.tree), de DENSITÉ
    # comparable au scan. Un échantillonnage dense rendrait l'UDF plus lisse mais
    # gonflerait |O_occupied| du modèle (cellules udf<cell), ce qui ferait chuter
    # la confiance eq.5 (dénominateur) sous t_c=0.8 et rejetterait tous the matchs.
    feats = build_model_features(cloud, kps, cfg, up, ground)
    return DatabaseModel(name, synset, cloud, kps, feats, ground, mesh_path,
                         surface_point_count=n_points)


def preprocess_model(path: str, synset: str, cfg: PipelineConfig,
                     n_points: int = 6000, add_ground: bool = True,
                     seed: int = 0) -> Optional[DatabaseModel]:
    'Preprocess an OBJ path. Database construction uses shapenet_loader sources.'
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    mid = os.path.basename(os.path.dirname(os.path.dirname(path)))
    name = f"{CATEGORY_NAME.get(synset, synset)}_{mid}"
    return preprocess_mesh(mesh, name, synset, cfg, n_points, add_ground, seed, mesh_path=path)


def build_database(root: str, cfg: PipelineConfig, synsets=None,
                   max_per_synset: int = 50, n_points: int = 6000,
                   progress=None) -> ModelDatabase:
    'Build a database from extracted ShapeNetCore directories or per-synset ZIP archives. Each model stores a preprocessed surface sample.'
    synsets = synsets or cfg.shapenet_synsets
    db = ModelDatabase(cfg=cfg)
    for syn in synsets:
        sources = find_model_sources(root, syn, limit=max_per_synset)
        if progress:
            progress(f"[{syn}] {len(sources)} modeles a traiter")
        for i, src in enumerate(sources):
            kp_count = 0
            try:
                name = f"{CATEGORY_NAME.get(syn, syn)}_{src.model_id}"
                ref = src.path if src.kind == "file" else f"{src.path}!{src.member}"
                m = preprocess_mesh(src.load(), name, syn, cfg, n_points=n_points,
                                    mesh_path=ref)
                if m is not None and m.keypoints.size > 0:
                    db.models.append(m)
                    kp_count = m.keypoints.size
            except Exception as exc:        # modèle illisible -> on saute
                if progress:
                    progress(f"[skip] {src.model_id}: {exc}")
                continue
            if progress and (i % 20 == 0 or i == len(sources) - 1):
                progress(f"[{syn}] {i+1}/{len(sources)}  ({len(db.models)} modeles, "
                         f"dernier: {kp_count} kp)")
    return db


def cluster_descriptors(db: ModelDatabase, threshold: Optional[float] = None):
    'Optional k-means clustering of flattened UDF descriptors. Store cluster representatives and retain references from model features.'
    from sklearn.cluster import KMeans
    cfg = db.cfg
    thr = threshold if threshold is not None else cfg.cluster.cluster_threshold
    vecs, refs = [], []
    for mi, m in enumerate(db.models):
        for ki, f in enumerate(m.features):
            if f.descriptor.udf is not None:
                vecs.append(f.descriptor.udf.reshape(-1))
                refs.append((mi, ki))
    if not vecs:
        return
    X = np.array(vecs)
    n_clusters = max(1, min(len(X), int(len(X) / 9) + 1))   # ~9x de réduction (papier)
    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=0).fit(X)
    db.cluster_reps = km.cluster_centers_
    for (mi, ki), lab in zip(refs, km.labels_):
        db.models[mi].features[ki].descriptor.cluster_id = int(lab)
