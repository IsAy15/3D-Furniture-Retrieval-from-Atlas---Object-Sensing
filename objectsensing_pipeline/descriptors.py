'\nLocal keypoint descriptors and distance/confidence (paper sections 5.1 and 5.2). Model: unsigned distance field on a local cubic grid. Scan: free/occupied/unknown cells obtained from RGB-D visibility and seed-fill. Compare occupied scan offsets with the model UDF under yaw rotation; average normalized distance raised to alpha. Serialized UDFs remain in meters; the automatic distance unit follows the local grid cell size.\n'

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from config import DescriptorConfig
from parallelism import kdtree_workers
from sdf_fusion import VIS_FREE, VIS_OCCUPIED

EMPTY, OCCUPIED, UNKNOWN = 0, 1, 2


@dataclass
class LocalDescriptor:
    kind: str                       # "model" | "scan"
    res: int
    extent: float
    keypoint: np.ndarray            # (3,) position monde
    udf: Optional[np.ndarray] = None        # (res,res,res) distances (modèle)
    occ: Optional[np.ndarray] = None        # (res,res,res) int8 (occupation)
    occupied_offsets: Optional[np.ndarray] = None   # (M,3) offsets des cellules occupées
    n_occupied: int = 0
    n_unknown: int = 0
    n_total: int = 0
    n_hole_boundary_removed: int = 0

    @property
    def cell_size(self) -> float:
        return 2.0 * self.extent / self.res


def _grid_offsets(res: int, extent: float) -> np.ndarray:
    'Cell-center offsets (res^3, 3) in [-extent, extent]^3.'
    ax = np.linspace(-extent, extent, res)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def build_model_descriptor(surface_tree, keypoint, cfg: DescriptorConfig) -> LocalDescriptor:
    'Local model UDF: distance from each cell to the sampled model surface using a KD-tree.'
    res, extent = cfg.udf_grid_res, cfg.udf_extent
    offsets = _grid_offsets(res, extent)
    grid_pts = keypoint + offsets
    dist, _ = surface_tree.query(grid_pts, workers=kdtree_workers())
    udf = dist.reshape(res, res, res).astype(np.float32)
    cell = 2.0 * extent / res
    occ = np.where(udf < cell, OCCUPIED, EMPTY).astype(np.int8)
    occ_mask = occ.reshape(-1) == OCCUPIED
    return LocalDescriptor(
        "model", res, extent, np.asarray(keypoint, float),
        udf=udf, occ=occ, occupied_offsets=offsets[occ_mask],
        n_occupied=int(occ_mask.sum()), n_unknown=0, n_total=res ** 3,
    )


def build_scan_descriptor(tsdf_volume, keypoint, cfg: DescriptorConfig) -> LocalDescriptor:
    'Free/occupied/unknown scan grid with seed-fill from the keypoint.'
    res, extent = cfg.occ_grid_res, cfg.occ_extent
    offsets = _grid_offsets(res, extent)
    grid_pts = keypoint + offsets
    occ = np.full(res ** 3, UNKNOWN, np.int8)
    filter_holes = bool(getattr(cfg, "filter_hole_boundaries", True))
    if filter_holes and hasattr(tsdf_volume, "sample_visibility_details"):
        visibility, hole_boundary = tsdf_volume.sample_visibility_details(
            grid_pts,
        )
    else:
        visibility = tsdf_volume.sample_visibility(grid_pts)
        hole_boundary = np.zeros(len(grid_pts), bool)
    occ[visibility == VIS_FREE] = EMPTY
    occ[visibility == VIS_OCCUPIED] = OCCUPIED
    occ_grid = occ.reshape(res, res, res)

    # Un bord de trou RGB-D peut produire une mince surface TSDF qui ressemble
    # artificiellement à un coin. On ne l'efface que lorsqu'none autre vue ne
    # confirme la cellule loin d'un pixel de profondeur invalide.
    boundary_grid = np.asarray(hole_boundary, bool).reshape(res, res, res)
    boundary_occupied = boundary_grid & (occ_grid == OCCUPIED)
    occ_grid[boundary_occupied] = UNKNOWN

    # Le papier effectue un seed-fill depuis le key point pour ne conserver que
    # la géométrie qui le supporte. On choisit la composante occupée 26-connexe
    # la plus proche du centre; the autres surfaces redeviennent inconnues.
    occupied = occ_grid == OCCUPIED
    if np.any(occupied):
        labels, _ = ndimage.label(occupied, structure=np.ones((3, 3, 3), np.uint8))
        occupied_idx = np.argwhere(occupied)
        center = np.full(3, (res - 1) / 2.0)
        seed_idx = occupied_idx[np.argmin(np.sum((occupied_idx - center) ** 2, axis=1))]
        seed_label = labels[tuple(seed_idx)]
        disconnected = occupied & (labels != seed_label)
        occ_grid[disconnected] = UNKNOWN

    occ = occ_grid.reshape(-1)
    occ_mask = occ == OCCUPIED
    occ_off = offsets[occ_mask]
    full_n = int(occ_mask.sum())
    # Plafond de cellules occupées (échantillon déterministe) pour la perf : la
    # distance eq.3 reste une moyenne, donc un sous-ensemble l'estime bien.
    cap = getattr(cfg, "max_occupied", 0)
    if cap and len(occ_off) > cap:
        occ_off = occ_off[np.linspace(0, len(occ_off) - 1, cap).astype(int)]
    return LocalDescriptor(
        "scan", res, extent, np.asarray(keypoint, float),
        udf=None, occ=occ_grid, occupied_offsets=occ_off,
        n_occupied=full_n,
        n_unknown=int((occ == UNKNOWN).sum()), n_total=res ** 3,
        n_hole_boundary_removed=int(boundary_occupied.sum()),
    )


# ---------------------------------------------------------------------------
# Distance descripteur et confiance (sec. 5.2)
# ---------------------------------------------------------------------------
def _rotate_about_up(offsets: np.ndarray, theta: float, up) -> np.ndarray:
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    c, s = np.cos(theta), np.sin(theta)
    ux, uy, uz = up
    R = np.array([
        [c + ux * ux * (1 - c),      ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
        [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c),      uy * uz * (1 - c) - ux * s],
        [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
    ])
    return offsets @ R.T


def _udf_lookup(model_desc: LocalDescriptor, offsets: np.ndarray) -> np.ndarray:
    'Read model UDF at supplied offsets using trilinear interpolation.'
    res, extent = model_desc.res, model_desc.extent
    coord = (offsets + extent) / (2 * extent) * (res - 1)
    # _grid_offsets includes both endpoints. At +extent the final grid sample
    # is valid: interpolate from the last interval with fraction 1, rather than
    # assigning the out-of-grid penalty. Tolerate only floating-point rotation
    # noise at the boundary, never a geometrically out-of-domain observation.
    tolerance = 32 * np.finfo(float).eps * max(res - 1, 1)
    inb = np.all((coord >= -tolerance) & (coord <= res - 1 + tolerance), axis=1)
    coord = np.clip(coord, 0, res - 1)
    lo = np.minimum(np.floor(coord).astype(int), res - 2)
    hi = lo + 1
    out = np.full(len(offsets), 2 * extent, np.float32)     # hors grille = loin
    if not np.any(inb):
        return out
    g0 = lo[inb]
    g1 = hi[inb]
    frac = coord[inb] - g0
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    udf = model_desc.udf
    c000 = udf[g0[:, 0], g0[:, 1], g0[:, 2]]
    c100 = udf[g1[:, 0], g0[:, 1], g0[:, 2]]
    c010 = udf[g0[:, 0], g1[:, 1], g0[:, 2]]
    c110 = udf[g1[:, 0], g1[:, 1], g0[:, 2]]
    c001 = udf[g0[:, 0], g0[:, 1], g1[:, 2]]
    c101 = udf[g1[:, 0], g0[:, 1], g1[:, 2]]
    c011 = udf[g0[:, 0], g1[:, 1], g1[:, 2]]
    c111 = udf[g1[:, 0], g1[:, 1], g1[:, 2]]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    out[inb] = c0 * (1 - fz) + c1 * fz
    return out


def effective_distance_unit(cfg: DescriptorConfig) -> float:
    'Metric normalization unit for dimensionless UDF comparison. Serialized UDF values stay in meters; automatic mode uses the local model grid cell size.\n    '
    configured = float(getattr(cfg, "distance_unit", 0.0) or 0.0)
    if configured > 0.0:
        return configured
    resolution = max(int(getattr(cfg, "udf_grid_res", 1)), 1)
    extent = max(float(getattr(cfg, "udf_extent", 0.0)), 1e-12)
    return 2.0 * extent / resolution


def descriptor_distance(model_desc: LocalDescriptor, scan_desc: LocalDescriptor,
                        cfg: DescriptorConfig, theta: float, up) -> float:
    'd(Ki, Kj, theta), equation 3. Ki is the model UDF; Kj contains occupied scan cells.'
    occ = scan_desc.occupied_offsets
    if occ is None or len(occ) == 0:
        return float("inf")
    rot = _rotate_about_up(occ, theta, up)
    distance_unit = effective_distance_unit(cfg)
    d = _udf_lookup(model_desc, rot) / distance_unit
    return float((d ** cfg.distance_exponent).sum() / len(occ))


def matching_confidence(model_desc: LocalDescriptor, scan_desc: LocalDescriptor) -> float:
    """c(OLi, OLj) — eq. 5."""
    denom = model_desc.n_occupied
    if denom <= 0:
        return 0.0
    return min(1.0, (scan_desc.n_occupied + scan_desc.n_unknown) / denom)


# ---------------------------------------------------------------------------
# Version vectorisée : distance pour TOUTES the rotations d'un coup (perf)
# ---------------------------------------------------------------------------
def rot_matrices_about(up, thetas) -> np.ndarray:
    'Rotation matrices (T,3,3) around up for an array of angles.'
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    ux, uy, uz = up
    thetas = np.asarray(thetas, float)
    c = np.cos(thetas); s = np.sin(thetas); C = 1 - c
    R = np.empty((len(thetas), 3, 3))
    R[:, 0, 0] = c + ux * ux * C; R[:, 0, 1] = ux * uy * C - uz * s; R[:, 0, 2] = ux * uz * C + uy * s
    R[:, 1, 0] = uy * ux * C + uz * s; R[:, 1, 1] = c + uy * uy * C; R[:, 1, 2] = uy * uz * C - ux * s
    R[:, 2, 0] = uz * ux * C - uy * s; R[:, 2, 1] = uz * uy * C + ux * s; R[:, 2, 2] = c + uz * uz * C
    return R


def descriptor_distance_batch(model_desc: LocalDescriptor, scan_desc: LocalDescriptor,
                              cfg: DescriptorConfig, thetas, up) -> np.ndarray:
    'Vectorized d(Ki, Kj, theta) for all angles. Return one distance per entry in thetas.'
    occ = scan_desc.occupied_offsets
    thetas = np.asarray(thetas, float)
    if occ is None or len(occ) == 0:
        return np.full(len(thetas), np.inf)
    R = rot_matrices_about(up, thetas)                  # (T,3,3)
    rot = np.einsum('tij,mj->tmi', R, occ)              # (T,M,3)
    distance_unit = effective_distance_unit(cfg)
    d = np.vstack([_udf_lookup(model_desc, r) for r in rot]) / distance_unit
    return (d ** cfg.distance_exponent).sum(axis=1) / len(occ)
