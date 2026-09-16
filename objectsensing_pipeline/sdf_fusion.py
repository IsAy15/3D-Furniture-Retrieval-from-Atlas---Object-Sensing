'\nVolumetric TSDF fusion of ObjectSensing RGB-D sequences. ZIP input: frame-XXXXXX.color.png (RGB), frame-XXXXXX.depth.png (16-bit millimeters, zero invalid), frame-XXXXXX.pose.txt (4x4 camera-to-world), depthIntrinsics.txt and colorIntrinsics.txt. Outputs surface points, SDF-gradient normals and curvature. Dense NumPy/CuPy and sparse Open3D layouts retain visibility evidence.\n'

from __future__ import annotations

import io
import math
import os
import re
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, minimum_filter

from config import SDFConfig
from geometry import PointCloud
from live_events import emit_progress_event
from parallelism import kdtree_workers


VIS_UNKNOWN = np.uint8(0)
VIS_FREE = np.uint8(1)
VIS_OCCUPIED = np.uint8(2)
VIS_HOLE_BOUNDARY = np.uint8(4)
VIS_RELIABLE_SURFACE = np.uint8(8)

DEFAULT_DENSE_VOXEL_LIMIT = {
    "cpu": 30_000_000,
    "cuda": 50_000_000,
}


def _validate_surface_extraction(value):
    value = str(value)
    if value not in {"zero_crossing", "hybrid", "near_zero"}:
        raise ValueError(
            'surface_extraction must be zero_crossing, hybrid or near_zero'
        )
    return value


class DenseVolumeLimitError(MemoryError):
    'Reject impractical dense volumes before large allocations.'

    def __init__(self, dims, voxel_size, backend, limit, recommended):
        self.dims = tuple(int(value) for value in dims)
        self.voxel_count = int(np.prod(self.dims, dtype=np.int64))
        self.voxel_size = float(voxel_size)
        self.backend = str(backend)
        self.limit = int(limit)
        self.recommended_voxel_size = float(recommended)
        peak_bytes = self.voxel_count * (96 if backend == "cpu" else 64)
        self.estimated_peak_bytes = int(peak_bytes)
        gib = peak_bytes / (1024 ** 3)
        count_m = self.voxel_count / 1_000_000
        limit_m = self.limit / 1_000_000
        dimensions = " x ".join(str(value) for value in self.dims)
        super().__init__(
            f'Dense TSDF volume too large for backend {backend}: {dimensions} = {count_m:.1f} M voxels (limit {limit_m:.1f} M, estimated working memory ~ {gib:.1f} GiB). Requested voxel size: {self.voxel_size:.3f} m. Use at least {self.recommended_voxel_size:.3f} m for these bounds, or use a sparse voxel-hashed volume.'
        )


def _dense_volume_layout(bounds_min, bounds_max, voxel_size):
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('voxel_size must be strictly positive')
    origin = np.asarray(bounds_min, np.float64) - 3 * voxel_size
    bmax = np.asarray(bounds_max, np.float64) + 3 * voxel_size
    dims = np.ceil((bmax - origin) / voxel_size).astype(np.int64) + 1
    if np.any(dims <= 0):
        raise ValueError('Invalid TSDF bounds or dimensions')
    return origin, dims


def _dense_voxel_limit(backend):
    override = os.environ.get("OBJECTSENSING_MAX_DENSE_VOXELS")
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_DENSE_VOXEL_LIMIT[backend]


def _recommended_dense_voxel_size(bounds_min, bounds_max, voxel_size, limit):
    _, dims = _dense_volume_layout(bounds_min, bounds_max, voxel_size)
    count = int(np.prod(dims, dtype=np.int64))
    estimate = float(voxel_size) * max(
        1.0, (count / limit) ** (1.0 / 3.0),
    )
    step = 0.005
    candidate = math.ceil((estimate - 1e-12) / step) * step
    while True:
        _, candidate_dims = _dense_volume_layout(
            bounds_min, bounds_max, candidate,
        )
        if int(np.prod(candidate_dims, dtype=np.int64)) <= limit:
            return candidate
        candidate += step


def _smooth_observed_tsdf(tsdf: np.ndarray, weight: np.ndarray,
                          sigma: float = 0.8):
    'Smooth TSDF without mixing observed and unknown voxels. A ratio of convolutions extrapolates the field using observed support to avoid false gradients at visibility boundaries.\n    '
    tsdf = np.asarray(tsdf, np.float32)
    confidence = np.clip(np.asarray(weight, np.float32), 0.0, 8.0)
    observed = confidence > 0
    if sigma <= 0:
        return tsdf.copy(), observed.astype(np.float32)
    numerator = gaussian_filter(
        np.where(observed, tsdf * confidence, 0.0),
        sigma=sigma, mode="constant", cval=0.0,
    )
    denominator = gaussian_filter(
        confidence, sigma=sigma, mode="constant", cval=0.0,
    )
    smoothed = np.zeros_like(tsdf)
    supported = denominator > 1e-6
    smoothed[supported] = numerator[supported] / denominator[supported]
    local_support = gaussian_filter(
        observed.astype(np.float32), sigma=sigma,
        mode="constant", cval=0.0,
    )
    return smoothed, local_support


def _depth_integration_weights(depth: np.ndarray, depth_trunc: float,
                               edge_threshold: float = 0.05,
                               background_weight: float = 0.10,
                               radius: int = 1):
    'Downweight background depth near foreground silhouettes. Local minimum depth identifies the front layer; significantly deeper pixels receive reduced weight while foreground pixels retain full weight.\n    '
    depth = np.asarray(depth, np.float32)
    threshold = float(edge_threshold)
    background_weight = float(background_weight)
    radius = int(radius)
    if threshold < 0:
        raise ValueError('depth_edge_threshold must be >= 0')
    if not 0.0 <= background_weight <= 1.0:
        raise ValueError(
            'depth_edge_background_weight must be between 0 and 1'
        )
    if radius < 0:
        raise ValueError('depth_edge_radius must be >= 0')

    valid = (depth > 0.1) & (depth < float(depth_trunc))
    weights = np.zeros(depth.shape, np.float32)
    weights[valid] = 1.0
    if (
        threshold == 0.0 or radius == 0
        or background_weight >= 1.0 or not np.any(valid)
    ):
        return weights, np.zeros(depth.shape, bool)

    local_depth = minimum_filter(
        np.where(valid, depth, np.inf),
        size=2 * radius + 1,
        mode="constant",
        cval=np.inf,
    )
    background_edge = (
        valid & np.isfinite(local_depth)
        & ((depth - local_depth) > threshold)
    )
    weights[background_edge] = background_weight
    return weights, background_edge


def _array_module(backend: str):
    'Return NumPy or CuPy without requiring CUDA.'
    requested = str(backend or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown fusion backend: {backend}")
    if requested != "cpu":
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp, "cuda"
        except Exception:
            if requested == "cuda":
                raise RuntimeError(
                    'CUDA backend requested but CuPy/CUDA is unavailable'
                )
    return np, "cpu"


def _open3d_sparse_device(backend: str):
    """Return an Open3D device suitable for sparse TSDF integration."""
    try:
        import open3d as o3d
    except ImportError:
        return None, None
    requested = str(backend or "auto").lower()
    if requested == "cuda" and o3d.core.cuda.is_available():
        return o3d, o3d.core.Device("CUDA:0")
    return o3d, o3d.core.Device("CPU:0")


def _resolve_volume_layout(cfg, bounds_min, bounds_max):
    requested = str(getattr(cfg, "volume_layout", "auto")).lower()
    if requested not in {"auto", "dense", "sparse"}:
        raise ValueError(
            'volume_layout must be auto, dense or sparse'
        )
    if requested == "dense":
        return "dense"
    o3d, device = _open3d_sparse_device(getattr(cfg, "backend", "auto"))
    if requested == "sparse":
        if o3d is None:
            raise RuntimeError(
                'Sparse volume requested but Open3D is not installed'
            )
        return "sparse"

    # Sur CPU, l'intégration voxel-hashée Open3D évite de reprojeter tous the
    # voxels vides à chaque trame. Sur CUDA, le dense CuPy reste préférable
    # tant que sa boîte englobante respecte la limit mémoire.
    _, dense_backend = _array_module(getattr(cfg, "backend", "auto"))
    _, dims = _dense_volume_layout(
        bounds_min, bounds_max, cfg.voxel_size,
    )
    dense_count = int(np.prod(dims, dtype=np.int64))
    dense_fits = dense_count <= _dense_voxel_limit(dense_backend)
    if o3d is not None and (
        dense_backend == "cpu" or not dense_fits
    ):
        return "sparse"
    return "dense"


# ---------------------------------------------------------------------------
# Play de l'RGB-D archive
# ---------------------------------------------------------------------------
def _frame_id(name: str) -> Optional[str]:
    m = re.search(r"frame-(\d+)\.", name)
    return m.group(1) if m else None


def _read_intrinsics(text: str) -> Tuple[float, float, float, float]:
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]
    if len(nums) >= 9:
        return nums[0], nums[4], nums[2], nums[5]      # fx, fy, cx, cy (matrice 3x3)
    if len(nums) >= 4:
        return nums[0], nums[1], nums[2], nums[3]
    return 525.0, 525.0, 319.5, 239.5


def _read_pose(text: str) -> Optional[np.ndarray]:
    if re.search(r"(?i)\b(?:[-+]?inf|nan|1\.#)", text):
        return None                                     # frame perdue (tracking échoué)
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]
    if len(nums) < 16:
        return None
    M = np.asarray(nums[:16], dtype=np.float64).reshape(4, 4)
    return M if np.isfinite(M).all() else None


@dataclass
class RGBDFrame:
    fid: str
    depth: np.ndarray      # (H, W) float32 en mètres (0 = invalide)
    color: np.ndarray      # (H, W, 3) uint8
    pose: np.ndarray       # (4, 4) caméra -> monde


class RGBDSequence:
    'Lazy access to frames in an ObjectSensing ZIP archive.'

    def __init__(self, zip_path, depth_scale=1000.0):
        self.zip_path = Path(zip_path)
        self.depth_scale = depth_scale
        self._zf = zipfile.ZipFile(self.zip_path)
        names = self._zf.namelist()
        self.fx, self.fy, self.cx, self.cy = 525.0, 525.0, 319.5, 239.5
        di = next((n for n in names if n.lower().endswith("depthintrinsics.txt")), None)
        if di:
            fx, fy, cx, cy = _read_intrinsics(self._zf.read(di).decode("utf-8", "ignore"))
            # Garde-fou : certains depthIntrinsics.txt (ex: office) se lisent en
            # fx/fy = 0 -> division par zéro. On retombe alors sur the valeurs par
            # défaut PrimeSense/Kinect 640x480 (cf. papier : intrinsèques par défaut).
            if np.isfinite([fx, fy, cx, cy]).all() and fx > 1.0 and fy > 1.0:
                self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self._depth, self._color, self._pose = {}, {}, {}
        for n in names:
            fid = _frame_id(n)
            if not fid:
                continue
            low = n.lower()
            if low.endswith(".depth.png"):
                self._depth[fid] = n
            elif low.endswith(".color.png"):
                self._color[fid] = n
            elif low.endswith(".pose.txt"):
                self._pose[fid] = n
        self.frame_ids = sorted(set(self._depth) & set(self._color) & set(self._pose))

    def intrinsics(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def load(self, fid: str) -> Optional[RGBDFrame]:
        pose = _read_pose(self._zf.read(self._pose[fid]).decode("utf-8", "ignore"))
        if pose is None:
            return None
        d = np.asarray(Image.open(io.BytesIO(self._zf.read(self._depth[fid]))))
        if d.ndim == 3:
            d = d[..., 0]
        depth = d.astype(np.float32) / self.depth_scale
        color = np.asarray(Image.open(io.BytesIO(self._zf.read(self._color[fid]))).convert("RGB"))
        return RGBDFrame(fid, depth, color, pose)

    def close(self):
        self._zf.close()


# ---------------------------------------------------------------------------
# TSDF volume
# ---------------------------------------------------------------------------
class TSDFVolume:
    'Dense TSDF volume. Truncated signed distances are stored in meters.'

    def __init__(self, bounds_min, bounds_max, voxel_size, truncation,
                 backend="cpu", surface_field="raw",
                 surface_min_support=0.20,
                 surface_extraction="hybrid",
                 surface_band_factor=1.5,
                 surface_rescue_distance_factor=1.0,
                 surface_rescue_min_weight=5.0,
                 gradient_smoothing_sigma=0.8,
                 depth_edge_threshold=0.05,
                 depth_edge_background_weight=0.10,
                 depth_edge_radius=1,
                 visibility_surface_band_factor=0.75):
        self.voxel_size = float(voxel_size)
        self.trunc = float(truncation)
        if surface_field not in {"raw", "smoothed"}:
            raise ValueError(
                'surface_field must be raw or smoothed'
            )
        self.surface_field = str(surface_field)
        self.surface_min_support = float(surface_min_support)
        self.surface_extraction = _validate_surface_extraction(
            surface_extraction,
        )
        self.surface_band_factor = max(0.0, float(surface_band_factor))
        self.surface_rescue_distance_factor = max(
            0.0, float(surface_rescue_distance_factor),
        )
        self.surface_rescue_min_weight = max(
            0.0, float(surface_rescue_min_weight),
        )
        self.gradient_smoothing_sigma = float(gradient_smoothing_sigma)
        self.depth_edge_threshold = float(depth_edge_threshold)
        self.depth_edge_background_weight = float(
            depth_edge_background_weight
        )
        self.depth_edge_radius = int(depth_edge_radius)
        self.visibility_surface_band_factor = max(
            0.05, float(visibility_surface_band_factor),
        )
        self.integration_stats = {
            "valid_depth_pixels": 0,
            "downweighted_background_pixels": 0,
        }
        self.origin, self.dims = _dense_volume_layout(
            bounds_min, bounds_max, self.voxel_size,
        )
        nx, ny, nz = self.dims
        self.xp, self.backend = _array_module(backend)
        voxel_count = int(np.prod(self.dims, dtype=np.int64))
        voxel_limit = _dense_voxel_limit(self.backend)
        if voxel_count > voxel_limit:
            recommended = _recommended_dense_voxel_size(
                bounds_min, bounds_max, self.voxel_size, voxel_limit,
            )
            raise DenseVolumeLimitError(
                self.dims, self.voxel_size, self.backend,
                voxel_limit, recommended,
            )
        self.tsdf = self.xp.ones((nx, ny, nz), self.xp.float32)
        self.weight = self.xp.zeros((nx, ny, nz), self.xp.float32)
        self.visibility = self.xp.zeros((nx, ny, nz), self.xp.uint8)
        # Coordonnées monde du centre de chaque voxel (précalcul des axes).
        self._ax = [
            (
                self.origin[axis]
                + np.arange(self.dims[axis], dtype=np.float32)
                * self.voxel_size
            ).astype(np.float32, copy=False)
            for axis in range(3)
        ]

    def __getstate__(self):
        """Serialize the durable CPU volume without derived working buffers."""
        self.to_cpu()
        state = dict(self.__dict__)
        state.pop("xp", None)
        for name in (
            "_centers", "_grad", "_surface_field", "_smoothed_field",
            "_gradient_support",
        ):
            state.pop(name, None)
        state["backend"] = "cpu"
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.xp = np
        self.backend = "cpu"

    def voxel_centers(self) -> np.ndarray:
        if getattr(self, "_centers", None) is None:
            nx, ny, nz = (int(value) for value in self.dims)
            centers = np.empty((nx * ny * nz, 3), np.float32)
            centers[:, 0] = np.repeat(self._ax[0], ny * nz)
            centers[:, 1] = np.tile(np.repeat(self._ax[1], nz), nx)
            centers[:, 2] = np.tile(self._ax[2], nx * ny)
            self._centers = self.xp.asarray(centers)
        return self._centers

    def to_cpu(self):
        'Materialize CUDA fusion results for downstream CPU stages.'
        if self.backend == "cuda":
            self.tsdf = self.xp.asnumpy(self.tsdf)
            self.weight = self.xp.asnumpy(self.weight)
            self.visibility = self.xp.asnumpy(self.visibility)
            if getattr(self, "_centers", None) is not None:
                self._centers = self.xp.asnumpy(self._centers)
            self.xp, self.backend = np, "cpu"
        return self

    def gradient_fields(self):
        'TSDF gradient in metric coordinates. The smoothed field supplies normals while raw geometry remains available for surface extraction.\n        '
        if getattr(self, "_grad", None) is None:
            self.to_cpu()
            field, self._gradient_support = _smooth_observed_tsdf(
                self.tsdf, self.weight,
                sigma=float(getattr(
                    self, "gradient_smoothing_sigma", 0.8,
                )),
            )
            self._smoothed_field = field
            self._grad = np.gradient(field, self.voxel_size)
        return self._grad

    def sample_gradient(self, positions: np.ndarray) -> np.ndarray:
        'Sample the TSDF gradient with trilinear interpolation.'
        positions = np.asarray(positions, np.float64)
        gx, gy, gz = self.gradient_fields()
        return np.column_stack([
            self._sample_grid(gx, positions, default=0.0),
            self._sample_grid(gy, positions, default=0.0),
            self._sample_grid(gz, positions, default=0.0),
        ])

    def integrate(self, frame: RGBDFrame, K, depth_trunc, max_weight=64.0):
        'Integrate a frame: project voxels into camera coordinates, compute depth minus voxel depth, and update the weighted mean.'
        nx, ny, nz = self.dims
        xp = self.xp
        centers = self.voxel_centers()                    # (V, 3) monde
        # monde -> caméra : world2cam = inv(pose)
        w2c = xp.asarray(np.linalg.inv(frame.pose))
        cam = (w2c[:3, :3] @ centers.T + w2c[:3, 3:4]).T   # (V, 3)
        z = cam[:, 2]
        valid = z > 1e-4
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = xp.full(len(cam), -1.0); v = xp.full(len(cam), -1.0)
        u[valid] = fx * cam[valid, 0] / z[valid] + cx
        v[valid] = fy * cam[valid, 1] / z[valid] + cy
        H, W = frame.depth.shape
        ui = xp.rint(u).astype(xp.int32); vi = xp.rint(v).astype(xp.int32)
        inb = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        d = xp.zeros(len(cam), xp.float32)
        observation_weight = xp.zeros(len(cam), xp.float32)
        depth = xp.asarray(frame.depth)
        depth_weights_cpu, background_edge = _depth_integration_weights(
            frame.depth, depth_trunc,
            edge_threshold=float(getattr(
                self, "depth_edge_threshold", 0.05,
            )),
            background_weight=float(getattr(
                self, "depth_edge_background_weight", 0.10,
            )),
            radius=int(getattr(self, "depth_edge_radius", 1)),
        )
        depth_weights = xp.asarray(depth_weights_cpu)
        d[inb] = depth[vi[inb], ui[inb]]
        observation_weight[inb] = depth_weights[vi[inb], ui[inb]]
        meas = (
            inb & (d > 0.1) & (d < depth_trunc)
            & (observation_weight > 0.0)
        )
        sdf = d - z                                       # >0 devant la surface
        upd = meas & (sdf >= -self.trunc)                 # on ignore loin derrière
        tval = xp.clip(sdf, -self.trunc, self.trunc).astype(xp.float32)
        idx = xp.where(upd)[0]
        if len(idx) == 0:
            return
        flat_t = self.tsdf.reshape(-1)
        flat_w = self.weight.reshape(-1)
        w_old = flat_w[idx]
        w_observation = observation_weight[idx]
        # Une moyenne glissante bornée continue de s'adapter sans dépasser le
        # poids maximal. L'ancienne formule ajoutait encore le numérateur après
        # saturation, ce qui pouvait faire sortir le TSDF de sa bande.
        w_history = xp.minimum(
            w_old, xp.maximum(max_weight - w_observation, 0.0),
        )
        w_new = w_history + w_observation
        flat_t[idx] = (
            flat_t[idx] * w_history + tval[idx] * w_observation
        ) / xp.maximum(w_new, 1e-6)
        flat_w[idx] = xp.minimum(w_old + w_observation, max_weight)
        self._grad = None
        self._smoothed_field = None
        stats = getattr(self, "integration_stats", None)
        if stats is None:
            stats = {
                "valid_depth_pixels": 0,
                "downweighted_background_pixels": 0,
            }
            self.integration_stats = stats
        stats["valid_depth_pixels"] += int(np.count_nonzero(
            depth_weights_cpu > 0.0,
        ))
        stats["downweighted_background_pixels"] += int(
            np.count_nonzero(background_edge)
        )

        # Visibility issue du rayon de profondeur : devant la surface = libre,
        # bande de surface = occupée, derrière = inconnu (occlusion).
        surface_band = max(
            self.voxel_size * self.visibility_surface_band_factor, 1e-6,
        )
        flat_v = self.visibility.reshape(-1)
        free_idx = xp.where(meas & (sdf > surface_band))[0]
        occ_idx = xp.where(meas & (xp.abs(sdf) <= surface_band))[0]
        flat_v[free_idx] |= VIS_FREE
        flat_v[occ_idx] |= VIS_OCCUPIED
        if len(occ_idx):
            hole_edge = xp.asarray(_visibility_hole_edge_image(
                _visibility_depth_image(frame.depth, 1, depth_trunc),
                radius=int(getattr(self, "depth_edge_radius", 1)),
            ))
            uncertain = hole_edge[vi[occ_idx], ui[occ_idx]] > 0
            flat_v[occ_idx[uncertain]] |= VIS_HOLE_BOUNDARY
            flat_v[occ_idx[~uncertain]] |= VIS_RELIABLE_SURFACE

    @classmethod
    def from_cloud(cls, points, normals, voxel_size, truncation):
        'Build an approximate TSDF from an oriented cloud for models or synthetic tests. Normal direction determines the sign.'
        from scipy.spatial import cKDTree
        points = np.asarray(points, np.float64)
        normals = np.asarray(normals, np.float64)
        vol = cls(points.min(0), points.max(0), voxel_size, truncation)
        centers = vol.voxel_centers()
        tree = cKDTree(points)
        dist, idx = tree.query(centers, workers=kdtree_workers())
        sgn = np.sign(np.einsum('ij,ij->i', centers - points[idx], normals[idx]))
        sgn[sgn == 0] = 1.0
        sd = np.clip(sgn * dist, -truncation, truncation).astype(np.float32)
        band = dist < 3 * truncation
        flat_t = vol.tsdf.reshape(-1); flat_w = vol.weight.reshape(-1)
        flat_t[band] = sd[band]
        flat_w[band] = 1.0
        flat_v = vol.visibility.reshape(-1)
        surface_band = voxel_size * vol.visibility_surface_band_factor
        flat_v[band & (dist <= surface_band)] |= VIS_OCCUPIED
        flat_v[band & (dist <= surface_band)] |= VIS_RELIABLE_SURFACE
        flat_v[band & (dist > surface_band)] |= VIS_FREE
        return vol

    def sample_visibility_details(self, positions: np.ndarray):
        'Sample free/occupied/unknown state from the fused TSDF. Reclassifying the final field allows surface-band adjustments without fusion and avoids keeping cells behind the final surface occupied.\n        '
        self.to_cpu()
        positions = np.asarray(positions, np.float64)
        coord = np.rint((positions - self.origin) / self.voxel_size).astype(int)
        inb = np.all((coord >= 0) & (coord < self.dims), axis=1)
        out = np.full(len(positions), VIS_UNKNOWN, np.uint8)
        hole_boundary = np.zeros(len(positions), bool)
        if np.any(inb):
            c = coord[inb]
            values = self.tsdf[c[:, 0], c[:, 1], c[:, 2]]
            weights = self.weight[c[:, 0], c[:, 1], c[:, 2]]
            observed = weights > 0
            band = max(
                self.voxel_size * float(getattr(
                    self, "visibility_surface_band_factor", 0.75,
                )),
                1e-6,
            )
            labels = np.full(len(values), VIS_UNKNOWN, np.uint8)
            labels[observed & (values > band)] = VIS_FREE
            labels[observed & (np.abs(values) <= band)] = VIS_OCCUPIED
            out[inb] = labels
            history = self.visibility[c[:, 0], c[:, 1], c[:, 2]]
            boundary = (history & VIS_HOLE_BOUNDARY) != 0
            reliable = (history & VIS_RELIABLE_SURFACE) != 0
            hole_boundary[inb] = boundary & ~reliable
        return out, hole_boundary

    def sample_visibility(self, positions: np.ndarray) -> np.ndarray:
        return self.sample_visibility_details(positions)[0]

    def visibility_summary(self):
        self.to_cpu()
        values = np.asarray(self.tsdf, np.float32)
        observed = np.asarray(self.weight, np.float32) > 0
        band = max(
            self.voxel_size * float(getattr(
                self, "visibility_surface_band_factor", 0.75,
            )),
            1e-6,
        )
        occupied = observed & (np.abs(values) <= band)
        free = observed & (values > band)
        return {
            "dims": [int(value) for value in self.dims],
            "voxel_count": int(values.size),
            "unknown_voxels": int(np.count_nonzero(~occupied & ~free)),
            "free_voxels": int(np.count_nonzero(free)),
            "occupied_voxels": int(np.count_nonzero(occupied)),
            "active_blocks": None,
            "layout": "dense",
            "integration": dict(getattr(
                self, "integration_stats", {},
            )),
        }

    def sample(self, positions: np.ndarray):
        'Trilinearly sample (tsdf, weight) at world positions to classify occupancy grids.'
        self.to_cpu()
        return (
            self._sample_grid(self.tsdf, positions, default=self.trunc).astype(np.float32),
            self._sample_grid(self.weight, positions, default=0.0).astype(np.float32),
        )

    def _sample_grid(self, grid: np.ndarray, positions: np.ndarray, default: float) -> np.ndarray:
        'Trilinear voxel-grid interpolation at world positions.'
        positions = np.asarray(positions, np.float64)
        coord = (positions - self.origin) / self.voxel_size
        inb = np.all((coord >= 0.0) & (coord <= (self.dims - 1)), axis=1)
        out = np.full(len(positions), default, np.float64)
        if not np.any(inb):
            return out

        c = coord[inb]
        lo = np.floor(c).astype(int)
        lo = np.minimum(lo, self.dims - 2)
        hi = lo + 1
        frac = c - lo
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

        c000 = grid[lo[:, 0], lo[:, 1], lo[:, 2]]
        c100 = grid[hi[:, 0], lo[:, 1], lo[:, 2]]
        c010 = grid[lo[:, 0], hi[:, 1], lo[:, 2]]
        c110 = grid[hi[:, 0], hi[:, 1], lo[:, 2]]
        c001 = grid[lo[:, 0], lo[:, 1], hi[:, 2]]
        c101 = grid[hi[:, 0], lo[:, 1], hi[:, 2]]
        c011 = grid[lo[:, 0], hi[:, 1], hi[:, 2]]
        c111 = grid[hi[:, 0], hi[:, 1], hi[:, 2]]
        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        out[inb] = c0 * (1 - fz) + c1 * fz
        return out

    def zero_crossing_surface(self, field_mode=None, min_support=None):
        'Return zero crossings without curvature. Raw TSDF carries geometry; the smoothed field supplies normals and legacy diagnostics.\n        '
        gradients = self.gradient_fields()
        mode = str(
            field_mode or getattr(self, "surface_field", "raw")
        )
        if mode == "raw":
            field = np.asarray(self.tsdf, np.float32)
        elif mode == "smoothed":
            field = getattr(self, "_smoothed_field")
        else:
            raise ValueError(
                'field_mode must be raw or smoothed'
            )
        observed = np.asarray(self.weight) > 0
        support = getattr(
            self, "_gradient_support", observed.astype(np.float32),
        )
        threshold = (
            float(getattr(self, "surface_min_support", 0.20))
            if min_support is None else float(min_support)
        )
        return _zero_crossing_points(
            field, observed, support, gradients,
            self.origin, self.voxel_size, min_support=threshold,
        )

    def near_zero_surface(self, field_mode=None, min_support=None,
                          strict_points=None):
        'Recover reliable near-zero voxels without thickening the scan.'
        from scipy.spatial import cKDTree

        gradients = self.gradient_fields()
        mode = str(field_mode or getattr(self, "surface_field", "raw"))
        if mode == "raw":
            field = np.asarray(self.tsdf, np.float32)
        elif mode == "smoothed":
            field = np.asarray(self._smoothed_field, np.float32)
        else:
            raise ValueError('field_mode must be raw or smoothed')
        observed = np.asarray(self.weight, np.float32)
        support = getattr(
            self, "_gradient_support", (observed > 0).astype(np.float32),
        )
        threshold = (
            float(getattr(self, "surface_min_support", 0.20))
            if min_support is None else float(min_support)
        )
        mask = (
            (observed >= self.surface_rescue_min_weight)
            & (support >= threshold)
            & (np.abs(field) <= self.voxel_size * self.surface_band_factor)
        )
        indices = np.column_stack(np.where(mask))
        if not len(indices):
            return np.zeros((0, 3)), np.zeros((0, 3))
        normals = np.column_stack([
            component[mask] for component in gradients
        ])
        magnitude = np.linalg.norm(normals, axis=1)
        valid = magnitude > 1e-6
        indices = indices[valid]
        values = field[mask][valid]
        normals = normals[valid] / magnitude[valid, None]
        points = (
            self.origin + indices.astype(np.float64) * self.voxel_size
            - values[:, None] * normals
        )
        if (
            self.surface_extraction == "hybrid"
            and strict_points is not None and len(strict_points)
        ):
            distance = cKDTree(strict_points).query(
                points, workers=kdtree_workers(),
            )[0]
            rescue = (
                distance
                >= self.voxel_size * self.surface_rescue_distance_factor
            )
            points, normals = points[rescue], normals[rescue]
        return points, normals

    def extract_surface(self, sampling=0.01, max_points=200000, seed=0,
                        field_mode=None, min_support=None):
        del seed
        'Extract isosurface points with gradient normals. Interpolate sign-changing edges between observed voxels, avoiding volumetric wall layers produced by a simple absolute-TSDF band.\n        '
        strict_points, strict_normals = self.zero_crossing_surface(
            field_mode=field_mode, min_support=min_support,
        )
        mode = getattr(self, "surface_extraction", "zero_crossing")
        if mode == "zero_crossing":
            pts, normals = strict_points, strict_normals
            rescued_count = 0
        else:
            rescued_points, rescued_normals = self.near_zero_surface(
                field_mode=field_mode, min_support=min_support,
                strict_points=strict_points,
            )
            if mode == "near_zero":
                pts, normals = rescued_points, rescued_normals
            else:
                pts = np.vstack([strict_points, rescued_points])
                normals = np.vstack([strict_normals, rescued_normals])
            rescued_count = int(len(rescued_points))
        self.surface_extraction_stats = {
            "mode": mode,
            "strict_points": int(len(strict_points)),
            "rescued_points": rescued_count,
            "combined_points": int(len(pts)),
        }
        if len(pts) == 0:
            return PointCloud(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        # sous-échantillonnage par grille (voxel `sampling`) pour homogénéiser
        if sampling and sampling > 0:
            key = np.floor(pts / sampling).astype(np.int64)
            _, uniq = np.unique(key, axis=0, return_index=True)
            pts, normals = pts[uniq], normals[uniq]
        if len(pts) > max_points:
            sel = _sample_evenly(len(pts), max_points)
            pts, normals = pts[sel], normals[sel]
        curvature = _curvature_from_point_neighborhoods(
            pts, radius=max(self.voxel_size * 2.5, sampling * 2.0),
        )
        _, confidence = self.sample(pts)
        self.surface_extraction_stats["output_points"] = int(len(pts))
        return PointCloud(pts, normals, curvature, confidence)


def _visibility_depth_image(depth, stride, depth_trunc):
    """Conservative min-pooling for later free/unknown ray queries."""
    depth = np.asarray(depth, np.float32)
    stride = max(1, int(stride))
    valid = (depth > 0.1) & (depth < float(depth_trunc))
    if stride == 1:
        pooled = np.where(valid, depth, 0.0)
    else:
        height = (depth.shape[0] // stride) * stride
        width = (depth.shape[1] // stride) * stride
        values = np.where(
            valid[:height, :width],
            depth[:height, :width],
            np.inf,
        )
        values = values.reshape(
            height // stride, stride, width // stride, stride,
        )
        pooled = values.min(axis=(1, 3))
        pooled[~np.isfinite(pooled)] = 0.0
    return np.rint(pooled * 1000.0).astype(np.uint16)


def _visibility_hole_edge_image(depth_mm, radius=1):
    """Valid pixels adjacent to missing depth, at visibility-map resolution."""
    valid = np.asarray(depth_mm, np.uint16) > 0
    radius = max(0, int(radius))
    if radius == 0 or not np.any(valid):
        return np.zeros(valid.shape, np.uint8)
    fully_valid = minimum_filter(
        valid.astype(np.uint8), size=2 * radius + 1,
        mode="constant", cval=0,
    ).astype(bool)
    return (valid & ~fully_valid).astype(np.uint8)


class SparseTSDFVolume:
    """Voxel-hashed Open3D TSDF with observation-backed visibility queries.

    Open3D stores only blocks touched by depth rays. Free/unknown classification
    is therefore evaluated lazily against the retained RGB-D observations,
    rather than approximated from a dense bounding box.
    """

    def __init__(self, bounds_min, bounds_max, voxel_size, truncation,
                 K, depth_trunc, backend="auto",
                 block_resolution=16, block_count=50000,
                 visibility_depth_stride=2,
                 surface_field="raw", surface_min_support=0.20,
                 surface_extraction="hybrid",
                 surface_band_factor=1.5,
                 surface_rescue_distance_factor=1.0,
                 surface_rescue_min_weight=5.0,
                 gradient_smoothing_sigma=0.8,
                 depth_edge_threshold=0.05,
                 depth_edge_background_weight=0.10,
                 depth_edge_radius=1,
                 visibility_surface_band_factor=0.75):
        o3d, device = _open3d_sparse_device(backend)
        if o3d is None:
            raise RuntimeError(
                "Sparse volumes require Open3D >= 0.19"
            )
        if surface_field not in {"raw", "smoothed"}:
            raise ValueError(
                'surface_field must be raw or smoothed'
            )
        self.voxel_size = float(voxel_size)
        self.trunc = float(truncation)
        self.depth_trunc = float(depth_trunc)
        self.surface_field = str(surface_field)
        self.surface_min_support = float(surface_min_support)
        self.surface_extraction = _validate_surface_extraction(
            surface_extraction,
        )
        self.surface_band_factor = max(0.0, float(surface_band_factor))
        self.surface_rescue_distance_factor = max(
            0.0, float(surface_rescue_distance_factor),
        )
        self.surface_rescue_min_weight = max(
            0.0, float(surface_rescue_min_weight),
        )
        self.gradient_smoothing_sigma = float(
            gradient_smoothing_sigma
        )
        self.depth_edge_threshold = float(depth_edge_threshold)
        self.depth_edge_background_weight = float(
            depth_edge_background_weight
        )
        self.depth_edge_radius = int(depth_edge_radius)
        self.block_resolution = int(block_resolution)
        self.block_count = int(block_count)
        self.visibility_depth_stride = max(
            1, int(visibility_depth_stride),
        )
        self.visibility_surface_band_factor = max(
            0.05, float(visibility_surface_band_factor),
        )
        if self.block_resolution <= 0 or self.block_count <= 0:
            raise ValueError(
                'Sparse volume dimensions must be positive'
            )
        self.origin, self.dims = _dense_volume_layout(
            bounds_min, bounds_max, self.voxel_size,
        )
        self.K = np.asarray(K, np.float64)
        self.backend = (
            "cuda" if str(device).upper().startswith("CUDA") else "cpu"
        )
        self.layout = "sparse"
        self.integration_stats = {
            "valid_depth_pixels": 0,
            "downweighted_background_pixels": 0,
            "integrated_depth_pixels": 0,
        }
        self._o3d = o3d
        self._device = device
        self._grid = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight"),
            attr_dtypes=(o3d.core.float32, o3d.core.float32),
            attr_channels=((1,), (1,)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=self.block_count,
            device=device,
        )
        self._intrinsic = o3d.core.Tensor(
            self.K, o3d.core.float64, device,
        )
        self._observation_depths = []
        self._observation_hole_edges = []
        self._observation_w2c = []
        self._integrated_frames = 0
        self._surface = None
        self._active_coordinates = None
        self._active_tsdf = None
        self._active_weight = None
        self._active_block_count = None

    def __getstate__(self):
        self._materialize_voxels()
        self._stack_observations()
        state = dict(self.__dict__)
        depths = np.asarray(state.pop("_observation_depths"), np.uint16)
        state["_observation_depths_compressed"] = {
            "shape": depths.shape,
            "data": zlib.compress(depths.tobytes(), level=1),
        }
        hole_edges = np.asarray(
            state.pop("_observation_hole_edges"), np.uint8,
        )
        state["_observation_hole_edges_compressed"] = {
            "shape": hole_edges.shape,
            "data": zlib.compress(hole_edges.tobytes(), level=1),
        }
        for name in (
            "_o3d", "_device", "_grid", "_intrinsic",
            "_active_tree", "_surface_tree",
        ):
            state.pop(name, None)
        state["backend"] = "cpu"
        return state

    def __setstate__(self, state):
        compressed = state.pop("_observation_depths_compressed", None)
        compressed_edges = state.pop(
            "_observation_hole_edges_compressed", None,
        )
        self.__dict__.update(state)
        if compressed is not None:
            self._observation_depths = np.frombuffer(
                zlib.decompress(compressed["data"]), dtype=np.uint16,
            ).reshape(compressed["shape"]).copy()
        if compressed_edges is not None:
            self._observation_hole_edges = np.frombuffer(
                zlib.decompress(compressed_edges["data"]), dtype=np.uint8,
            ).reshape(compressed_edges["shape"]).copy()
        elif not hasattr(self, "_observation_hole_edges"):
            self._observation_hole_edges = None
        self._o3d = None
        self._device = None
        self._grid = None
        self._intrinsic = None

    def _edge_preserving_depth(self, depth):
        weights, background_edge = _depth_integration_weights(
            depth, self.depth_trunc,
            edge_threshold=self.depth_edge_threshold,
            background_weight=self.depth_edge_background_weight,
            radius=self.depth_edge_radius,
        )
        filtered = np.asarray(depth, np.float32).copy()
        if np.any(background_edge):
            yy, xx = np.nonzero(background_edge)
            frame = np.uint64(self._integrated_frames + 1)
            hashed = (
                np.asarray(xx, np.uint64) * np.uint64(73856093)
                ^ np.asarray(yy, np.uint64) * np.uint64(19349663)
                ^ frame * np.uint64(83492791)
            )
            fraction = (
                hashed & np.uint64(0xFFFFFFFF)
            ).astype(np.float64) / float(2 ** 32)
            keep = fraction < self.depth_edge_background_weight
            reject_y = yy[~keep]
            reject_x = xx[~keep]
            filtered[reject_y, reject_x] = 0.0
        self.integration_stats["valid_depth_pixels"] += int(
            np.count_nonzero(weights > 0.0)
        )
        self.integration_stats[
            "downweighted_background_pixels"
        ] += int(np.count_nonzero(background_edge))
        self.integration_stats["integrated_depth_pixels"] += int(
            np.count_nonzero(
                (filtered > 0.1) & (filtered < self.depth_trunc)
            )
        )
        return filtered

    def integrate(self, frame: RGBDFrame, K=None, depth_trunc=None,
                  max_weight=64.0):
        del max_weight
        if self._grid is None:
            raise RuntimeError(
                'Serialized sparse volumes cannot integrate additional frames'
            )
        visibility_depth = _visibility_depth_image(
            frame.depth, self.visibility_depth_stride, self.depth_trunc,
        )
        self._observation_depths.append(visibility_depth)
        self._observation_hole_edges.append(_visibility_hole_edge_image(
            visibility_depth,
            radius=int(getattr(self, "depth_edge_radius", 1)),
        ))
        w2c = np.linalg.inv(frame.pose)
        self._observation_w2c.append(w2c.astype(np.float32))
        filtered_depth = self._edge_preserving_depth(frame.depth)
        depth_image = self._o3d.t.geometry.Image(
            self._o3d.core.Tensor(
                filtered_depth, self._o3d.core.float32, self._device,
            )
        )
        extrinsic = self._o3d.core.Tensor(
            w2c, self._o3d.core.float64, self._device,
        )
        trunc_multiplier = self.trunc / self.voxel_size
        blocks = self._grid.compute_unique_block_coordinates(
            depth_image, self._intrinsic, extrinsic,
            depth_scale=1.0,
            depth_max=self.depth_trunc,
            trunc_voxel_multiplier=trunc_multiplier,
        )
        self._grid.integrate(
            blocks, depth_image, self._intrinsic, extrinsic,
            depth_scale=1.0,
            depth_max=self.depth_trunc,
            trunc_voxel_multiplier=trunc_multiplier,
        )
        self._integrated_frames += 1

    def _stack_observations(self):
        if isinstance(self._observation_depths, list):
            if self._observation_depths:
                self._observation_depths = np.stack(
                    self._observation_depths,
                )
                self._observation_w2c = np.stack(
                    self._observation_w2c,
                )
                self._observation_hole_edges = np.stack(
                    self._observation_hole_edges,
                )
            else:
                self._observation_depths = np.zeros(
                    (0, 0, 0), np.uint16,
                )
                self._observation_w2c = np.zeros(
                    (0, 4, 4), np.float32,
                )
                self._observation_hole_edges = np.zeros(
                    (0, 0, 0), np.uint8,
                )
        if getattr(self, "_observation_hole_edges", None) is None:
            self._observation_hole_edges = np.asarray([
                _visibility_hole_edge_image(
                    depth,
                    radius=int(getattr(self, "depth_edge_radius", 1)),
                )
                for depth in self._observation_depths
            ], np.uint8)

    def _materialize_voxels(self):
        if self._active_coordinates is not None:
            return
        if self._grid is None:
            self._active_coordinates = np.zeros((0, 3), np.float32)
            self._active_tsdf = np.zeros(0, np.float32)
            self._active_weight = np.zeros(0, np.float32)
            return
        active_buffers = self._grid.hashmap().active_buf_indices()
        weight_attribute = self._grid.attribute("weight").reshape((-1, 1))
        tsdf_attribute = self._grid.attribute("tsdf").reshape((-1, 1))
        coordinate_chunks = []
        tsdf_chunks = []
        weight_chunks = []
        block_chunk = 256
        active_buffer_count = int(active_buffers.shape[0])
        for start in range(0, active_buffer_count, block_chunk):
            buffers = active_buffers[start:start + block_chunk]
            coordinates, indices = (
                self._grid.voxel_coordinates_and_flattened_indices(
                    buffers,
                )
            )
            weights = (
                weight_attribute[indices].cpu().numpy().reshape(-1)
            )
            active = weights > 0
            if not np.any(active):
                continue
            coordinate_chunks.append(
                coordinates.cpu().numpy()[active].astype(
                    np.float32, copy=False,
                )
            )
            tsdf_chunks.append(
                tsdf_attribute[indices].cpu().numpy().reshape(-1)[
                    active
                ].astype(np.float32, copy=False)
            )
            weight_chunks.append(
                weights[active].astype(np.float32, copy=False)
            )
        self._active_coordinates = (
            np.concatenate(coordinate_chunks)
            if coordinate_chunks else np.zeros((0, 3), np.float32)
        )
        self._active_tsdf = (
            np.concatenate(tsdf_chunks)
            if tsdf_chunks else np.zeros(0, np.float32)
        )
        self._active_weight = (
            np.concatenate(weight_chunks)
            if weight_chunks else np.zeros(0, np.float32)
        )

    def _extract_open3d_surface(self):
        if self._surface is not None:
            return self._surface
        if self._grid is None:
            self._surface = PointCloud(
                np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0),
            )
            return self._surface
        try:
            # Open3D keeps voxels whose weight is strictly greater than the
            # threshold. Zero therefore includes surfaces seen once, which is
            # required by progressive prefixes and one-frame unit tests.
            point_cloud = self._grid.extract_point_cloud(
                weight_threshold=0.0,
            )
        except RuntimeError as exc:
            # Open3D 0.19 tries to attach a shape-(0,) color tensor when an
            # otherwise valid depth-only volume contains no zero crossing.
            if "Tensor has shape {0}" not in str(exc):
                raise
            self._surface = PointCloud(
                np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0),
            )
            return self._surface
        points = point_cloud.point.positions.cpu().numpy()
        try:
            normals = point_cloud.point.normals.cpu().numpy()
        except Exception:
            normals = np.zeros_like(points)
        curvature = np.zeros(len(points), np.float64)
        self._surface = PointCloud(points, normals, curvature)
        return self._surface

    def _near_zero_surface(self, strict_surface):
        'Project observed near-zero voxels onto the local surface. Recover thin or grazing-angle observations missed by strict sign crossings, constrained by weight and distance to avoid thick walls.\n        '
        from scipy.spatial import cKDTree

        self._materialize_voxels()
        if not len(self._active_coordinates):
            return np.zeros((0, 3)), np.zeros((0, 3))
        signed_distance = self._active_tsdf * self.trunc
        mask = (
            (self._active_weight >= self.surface_rescue_min_weight)
            & (np.abs(signed_distance)
               <= self.voxel_size * self.surface_band_factor)
        )
        coordinates = self._active_coordinates[mask]
        signed_distance = signed_distance[mask]
        if not len(coordinates) or not len(strict_surface.points):
            return np.zeros((0, 3)), np.zeros((0, 3))

        strict_normals = np.asarray(strict_surface.normals, np.float64)
        normal_valid = np.linalg.norm(strict_normals, axis=1) > 1e-6
        if not np.any(normal_valid):
            return np.zeros((0, 3)), np.zeros((0, 3))
        strict_points = strict_surface.points[normal_valid]
        strict_normals = strict_normals[normal_valid]
        tree = cKDTree(strict_points)
        _, nearest = tree.query(
            coordinates, workers=kdtree_workers(),
        )
        normals = strict_normals[nearest]
        projected = coordinates - signed_distance[:, None] * normals

        if self.surface_extraction == "hybrid":
            distance, _ = tree.query(
                projected, workers=kdtree_workers(),
            )
            rescue = (
                distance
                >= self.voxel_size * self.surface_rescue_distance_factor
            )
            projected = projected[rescue]
            normals = normals[rescue]
        return projected, normals

    def _release_open3d_grid(self):
        if self._grid is None:
            return
        self._active_block_count = int(self._grid.hashmap().size())
        self._materialize_voxels()
        self._grid = None
        self._intrinsic = None
        self._device = None
        self._o3d = None

    def to_cpu(self):
        return self

    def extract_surface(self, sampling=0.01, max_points=200000, seed=0,
                        field_mode=None, min_support=None):
        del field_mode, min_support, seed
        strict_surface = self._extract_open3d_surface()
        mode = getattr(self, "surface_extraction", "zero_crossing")
        if mode == "zero_crossing":
            points = strict_surface.points
            normals = strict_surface.normals
            rescued_count = 0
        else:
            rescued_points, rescued_normals = self._near_zero_surface(
                strict_surface,
            )
            if mode == "near_zero":
                points, normals = rescued_points, rescued_normals
            else:
                points = np.vstack([
                    strict_surface.points, rescued_points,
                ])
                normals = np.vstack([
                    strict_surface.normals, rescued_normals,
                ])
            rescued_count = int(len(rescued_points))
        self.surface_extraction_stats = {
            "mode": mode,
            "strict_points": int(len(strict_surface.points)),
            "rescued_points": rescued_count,
            "combined_points": int(len(points)),
        }
        if sampling and sampling > 0 and len(points):
            key = np.floor(points / sampling).astype(np.int64)
            _, selection = np.unique(
                key, axis=0, return_index=True,
            )
            points = points[selection]
            normals = normals[selection]
        if len(points) > max_points:
            selection = _sample_evenly(len(points), max_points)
            points = points[selection]
            normals = normals[selection]
        curvature = _curvature_from_point_neighborhoods(
            points,
            radius=max(
                self.voxel_size * 2.5,
                float(sampling or 0.0) * 2.0,
            ),
        )
        _, confidence = self.sample(points)
        result = PointCloud(points, normals, curvature, confidence)
        self.surface_extraction_stats["output_points"] = int(len(points))
        self._surface = result
        self._release_open3d_grid()
        return result

    def zero_crossing_surface(self, field_mode=None, min_support=None):
        del field_mode, min_support
        cloud = self._extract_open3d_surface()
        return cloud.points, cloud.normals

    def _active_kdtree(self):
        if getattr(self, "_active_tree", None) is None:
            from scipy.spatial import cKDTree
            self._materialize_voxels()
            self._active_tree = (
                cKDTree(self._active_coordinates)
                if len(self._active_coordinates) else None
            )
        return self._active_tree

    def sample(self, positions):
        positions = np.asarray(positions, np.float64).reshape(-1, 3)
        tsdf = np.full(len(positions), self.trunc, np.float32)
        weight = np.zeros(len(positions), np.float32)
        tree = self._active_kdtree()
        if tree is None or not len(positions):
            return tsdf, weight
        distance, index = tree.query(
            positions,
            distance_upper_bound=self.voxel_size * math.sqrt(3.0),
            workers=kdtree_workers(),
        )
        valid = np.isfinite(distance)
        tsdf[valid] = self._active_tsdf[index[valid]] * self.trunc
        weight[valid] = self._active_weight[index[valid]]
        return tsdf, weight

    def sample_gradient(self, positions):
        positions = np.asarray(positions, np.float64).reshape(-1, 3)
        output = np.zeros((len(positions), 3), np.float64)
        cloud = self._extract_open3d_surface()
        if not len(points := cloud.points) or not len(positions):
            return output
        if getattr(self, "_surface_tree", None) is None:
            from scipy.spatial import cKDTree
            self._surface_tree = cKDTree(points)
        _, index = self._surface_tree.query(
            positions, workers=kdtree_workers(),
        )
        output[:] = cloud.normals[index]
        return output

    def sample_visibility_details(self, positions):
        positions = np.asarray(positions, np.float64).reshape(-1, 3)
        result = np.full(len(positions), VIS_UNKNOWN, np.uint8)
        hole_boundary = np.zeros(len(positions), bool)
        if not len(positions):
            return result, hole_boundary
        self._stack_observations()
        depths = self._observation_depths
        hole_edges = self._observation_hole_edges
        transforms = self._observation_w2c
        if not len(depths):
            return result, hole_boundary
        stride = self.visibility_depth_stride
        height, width = depths.shape[1:]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        band = max(
            self.voxel_size * float(getattr(
                self, "visibility_surface_band_factor", 0.75,
            )),
            1e-6,
        )
        frame_index = np.arange(len(depths))[:, None]
        chunk_size = 4096
        for start in range(0, len(positions), chunk_size):
            stop = min(len(positions), start + chunk_size)
            points = positions[start:stop]
            camera = np.einsum(
                "fij,nj->fni", transforms[:, :3, :3], points,
            )
            camera += transforms[:, None, :3, 3]
            z = camera[:, :, 2]
            valid_z = z > 1e-4
            safe_z = np.where(valid_z, z, 1.0)
            u = fx * camera[:, :, 0] / safe_z + cx
            v = fy * camera[:, :, 1] / safe_z + cy
            ui = np.floor(u / stride).astype(np.int32)
            vi = np.floor(v / stride).astype(np.int32)
            in_bounds = (
                valid_z & (ui >= 0) & (ui < width)
                & (vi >= 0) & (vi < height)
            )
            ui = np.clip(ui, 0, width - 1)
            vi = np.clip(vi, 0, height - 1)
            measured = (
                depths[frame_index, vi, ui].astype(np.float32)
                * 0.001
            )
            observed = in_bounds & (measured > 0.1)
            delta = measured - z
            occupied = observed & (np.abs(delta) <= band)
            free = observed & (delta > band)
            local = np.full(stop - start, VIS_UNKNOWN, np.uint8)
            local[np.any(free, axis=0)] = VIS_FREE
            local[np.any(occupied, axis=0)] = VIS_OCCUPIED
            result[start:stop] = local
            edge_observation = occupied & (
                hole_edges[frame_index, vi, ui] > 0
            )
            reliable_observation = occupied & ~(
                hole_edges[frame_index, vi, ui] > 0
            )
            hole_boundary[start:stop] = (
                np.any(edge_observation, axis=0)
                & ~np.any(reliable_observation, axis=0)
            )
        return result, hole_boundary

    def sample_visibility(self, positions):
        return self.sample_visibility_details(positions)[0]

    def visibility_summary(self):
        self._materialize_voxels()
        normalized_band = min(1.0, max(
            self.voxel_size * float(getattr(
                self, "visibility_surface_band_factor", 0.75,
            )),
            1e-6,
        ) / self.trunc)
        occupied = np.abs(self._active_tsdf) <= normalized_band
        free = self._active_tsdf > normalized_band
        voxel_count = int(np.prod(self.dims, dtype=np.int64))
        observed = int(np.count_nonzero(occupied | free))
        active_blocks = (
            int(self._grid.hashmap().size())
            if self._grid is not None else self._active_block_count
        )
        return {
            "dims": [int(value) for value in self.dims],
            "voxel_count": voxel_count,
            "unknown_voxels": max(0, voxel_count - observed),
            "free_voxels": int(np.count_nonzero(free)),
            "occupied_voxels": int(np.count_nonzero(occupied)),
            "active_blocks": active_blocks,
            "active_voxels": int(len(self._active_coordinates)),
            "layout": "sparse",
            "integration": dict(self.integration_stats),
        }

    def debug_visibility_points(self, limit=12000):
        self._materialize_voxels()
        limit = max(1, int(limit))
        normalized_band = min(
            1.0, max(self.voxel_size * 1.25, 1e-6) / self.trunc,
        )
        masks = {
            "occupied": np.abs(self._active_tsdf) <= normalized_band,
            "free": self._active_tsdf > normalized_band,
        }
        output = {}
        for label, mask in masks.items():
            indices = np.flatnonzero(mask)
            selection = indices[_sample_evenly(len(indices), limit)]
            output[label] = {
                "points": self._active_coordinates[selection],
                "total": int(len(indices)),
            }
        rng = np.random.default_rng(73)
        count = min(limit * 4, 50000)
        candidates = (
            self.origin
            + rng.random((count, 3))
            * (np.asarray(self.dims) - 1)
            * self.voxel_size
        )
        unknown = (
            self.sample_visibility(candidates) == VIS_UNKNOWN
        )
        unknown_points = candidates[unknown][:limit]
        output["unknown"] = {
            "points": unknown_points,
            "total": self.visibility_summary()["unknown_voxels"],
        }
        return output


def _sample_evenly(count, limit):
    if count <= 0:
        return np.zeros(0, int)
    if count <= limit:
        return np.arange(count, dtype=int)
    return np.linspace(0, count - 1, limit).astype(int)


def _zero_crossing_points(field: np.ndarray, observed: np.ndarray,
                          support: np.ndarray, gradients,
                          origin: np.ndarray, voxel_size: float,
                          min_support: float = 0.35):
    'Interpolate zero crossings along TSDF grid edges.'
    field = np.asarray(field, np.float32)
    observed = np.asarray(observed, bool)
    support = np.asarray(support, np.float32)
    gradients = tuple(np.asarray(component, np.float32) for component in gradients)
    origin = np.asarray(origin, np.float64)
    points, normals = [], []

    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        lower = tuple(lower)
        upper = tuple(upper)

        f0 = field[lower]
        f1 = field[upper]
        delta = f0 - f1
        valid = (
            observed[lower] & observed[upper] &
            (support[lower] >= min_support) &
            (support[upper] >= min_support) &
            (((f0 <= 0.0) & (f1 >= 0.0)) |
             ((f0 >= 0.0) & (f1 <= 0.0))) &
            (np.abs(delta) > 1e-8)
        )
        indices = np.column_stack(np.where(valid))
        if len(indices) == 0:
            continue

        alpha = np.clip(f0[valid] / delta[valid], 0.0, 1.0)
        coordinates = indices.astype(np.float64)
        coordinates[:, axis] += alpha
        points.append(origin + coordinates * voxel_size)

        edge_gradient = np.column_stack([
            component[lower][valid] * (1.0 - alpha) +
            component[upper][valid] * alpha
            for component in gradients
        ])
        magnitude = np.linalg.norm(edge_gradient, axis=1)
        normal = np.zeros_like(edge_gradient)
        nonzero = magnitude > 1e-6
        normal[nonzero] = edge_gradient[nonzero] / magnitude[nonzero, None]
        normals.append(normal)

    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3))
    points = np.vstack(points)
    normals = np.vstack(normals)
    valid_normals = np.linalg.norm(normals, axis=1) > 0.0
    return points[valid_normals], normals[valid_normals]


def _curvature_from_point_neighborhoods(points: np.ndarray,
                                        radius: float) -> np.ndarray:
    'PCA curvature of point positions. SDF-gradient normals remain available, but positional covariance measures surface variation without converting noisy wall normals into false curvature candidates.\n    '
    from scipy.spatial import cKDTree

    if len(points) == 0:
        return np.zeros(0)
    tree = cKDTree(points)
    neigh = tree.query_ball_point(points, radius, workers=kdtree_workers())
    curv = np.zeros(len(points), np.float64)
    batch_size = 4096
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        batch_neigh = neigh[start:stop]
        lengths = np.fromiter((len(idx) for idx in batch_neigh), np.int64)
        if not np.any(lengths >= 6):
            continue
        offsets = np.concatenate([[0], np.cumsum(lengths[:-1])])
        flat_indices = np.concatenate(batch_neigh)
        q = points[flat_indices]
        sums = np.add.reduceat(q, offsets, axis=0)
        outer = np.einsum("ni,nj->nij", q, q)
        sums_outer = np.add.reduceat(outer, offsets, axis=0)
        means = sums / lengths[:, None]
        cov = sums_outer / lengths[:, None, None] - np.einsum(
            "ni,nj->nij", means, means,
        )
        eigenvalues = np.linalg.eigvalsh(cov)
        totals = eigenvalues.sum(axis=1)
        valid = (lengths >= 6) & (totals > 1e-12)
        values = np.zeros(stop - start, np.float64)
        values[valid] = eigenvalues[valid, 0] / totals[valid]
        curv[start:stop] = values
    return curv


# ---------------------------------------------------------------------------
# Pipeline de fusion
# ---------------------------------------------------------------------------
def _scene_bounds(seq: RGBDSequence, fids: List[str], K, depth_trunc, step=8):
    'Estimate scene bounds by back-projecting a subset of frames.'
    mn = np.full(3, np.inf); mx = np.full(3, -np.inf)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for fid in fids[::max(1, len(fids) // 40 or 1)]:
        fr = seq.load(fid)
        if fr is None:
            continue
        H, W = fr.depth.shape
        ys = np.arange(0, H, 12); xs = np.arange(0, W, 12)
        gx, gy = np.meshgrid(xs, ys)
        z = fr.depth[gy, gx]
        m = (z > 0.1) & (z < depth_trunc)
        if not np.any(m):
            continue
        X = (gx[m] - cx) * z[m] / fx
        Y = (gy[m] - cy) * z[m] / fy
        cam = np.column_stack([X, Y, z[m], np.ones(m.sum())])
        world = (fr.pose @ cam.T).T[:, :3]
        mn = np.minimum(mn, world.min(0)); mx = np.maximum(mx, world.max(0))
    return mn, mx


def _trace_frame_sample(frame: RGBDFrame, K, depth_trunc: float,
                        max_points: int = 180):
    'Small world-space RGB measurement cloud for fusion replay. Samples actual integrated observations rather than computing an intermediate isosurface.\n    '
    depth = frame.depth
    height, width = depth.shape
    stride = max(1, int(np.sqrt((height * width) / max(1, max_points))))
    ys = np.arange(stride // 2, height, stride)
    xs = np.arange(stride // 2, width, stride)
    gx, gy = np.meshgrid(xs, ys)
    z = depth[gy, gx]
    valid = (z > 0.1) & (z < depth_trunc)
    if not np.any(valid):
        return {
            "frame_id": str(frame.fid),
            "points": [],
            "colors": [],
            "camera_position": [round(float(x), 4) for x in frame.pose[:3, 3]],
        }
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    px = gx[valid]
    py = gy[valid]
    pz = z[valid]
    camera = np.column_stack([
        (px - cx) * pz / fx,
        (py - cy) * pz / fy,
        pz,
        np.ones(len(pz)),
    ])
    world = (frame.pose @ camera.T).T[:, :3]
    colors = frame.color[py, px]
    if len(world) > max_points:
        keep = np.linspace(0, len(world) - 1, max_points).astype(int)
        world = world[keep]
        colors = colors[keep]
    return {
        "frame_id": str(frame.fid),
        "points": np.round(world, 3).tolist(),
        "colors": colors.astype(np.uint8).tolist(),
        "camera_position": [round(float(x), 4) for x in frame.pose[:3, 3]],
    }


def _select_frame_ids(source_fids, cfg):
    """Select frames while preserving a minimum number on short sequences."""
    requested_stride = max(1, int(cfg.frame_stride))
    effective_stride = requested_stride
    min_frames = max(0, int(getattr(cfg, "min_frames", 0)))
    if min_frames:
        adaptive_stride = max(1, len(source_fids) // min_frames)
        effective_stride = min(requested_stride, adaptive_stride)

    fids = list(source_fids)[::effective_stride]
    if cfg.max_frames:
        fids = fids[: int(cfg.max_frames)]
    return fids, effective_stride


def fuse_sequence(zip_path, cfg: SDFConfig, progress=None, trace=None) -> PointCloud:
    'Fuse an RGB-D sequence into an isosurface cloud.'
    seq = RGBDSequence(zip_path, depth_scale=cfg.depth_scale)
    K = seq.intrinsics()
    source_fids = list(seq.frame_ids)
    fids, effective_frame_stride = _select_frame_ids(source_fids, cfg)
    if not fids:
        raise ValueError('No usable RGB-D frame in ' + str(zip_path))

    mn, mx = _scene_bounds(seq, fids, K, cfg.depth_trunc)
    if not np.isfinite(mn).all():
        raise ValueError('Cannot estimate scene bounds (invalid poses?)')
    try:
        layout = _resolve_volume_layout(cfg, mn, mx)
        common = {
            "backend": cfg.backend,
            "surface_field": getattr(cfg, "surface_field", "raw"),
            "surface_min_support": getattr(
                cfg, "surface_min_support", 0.20,
            ),
            "surface_extraction": getattr(
                cfg, "surface_extraction", "zero_crossing",
            ),
            "surface_band_factor": getattr(
                cfg, "surface_band_factor", 1.5,
            ),
            "surface_rescue_distance_factor": getattr(
                cfg, "surface_rescue_distance_factor", 1.0,
            ),
            "surface_rescue_min_weight": getattr(
                cfg, "surface_rescue_min_weight", 5.0,
            ),
            "gradient_smoothing_sigma": getattr(
                cfg, "gradient_smoothing_sigma", 0.8,
            ),
            "depth_edge_threshold": getattr(
                cfg, "depth_edge_threshold", 0.05,
            ),
            "depth_edge_background_weight": getattr(
                cfg, "depth_edge_background_weight", 0.10,
            ),
            "depth_edge_radius": getattr(cfg, "depth_edge_radius", 1),
            "visibility_surface_band_factor": getattr(
                cfg, "visibility_surface_band_factor", 0.75,
            ),
        }
        if layout == "sparse":
            vol = SparseTSDFVolume(
                mn, mx, cfg.voxel_size, cfg.truncation,
                K=K, depth_trunc=cfg.depth_trunc,
                block_resolution=getattr(
                    cfg, "sparse_block_resolution", 16,
                ),
                block_count=getattr(cfg, "sparse_block_count", 50000),
                visibility_depth_stride=getattr(
                    cfg, "visibility_depth_stride", 2,
                ),
                **common,
            )
        else:
            vol = TSDFVolume(
                mn, mx, cfg.voxel_size, cfg.truncation,
                **common,
            )
    except DenseVolumeLimitError as exc:
        if progress:
            progress(str(exc))
        emit_progress_event(progress, "fusion_rejected", {
            "reason": "dense_volume_limit",
            "dims": list(exc.dims),
            "voxel_count": exc.voxel_count,
            "voxel_size": exc.voxel_size,
            "recommended_voxel_size": exc.recommended_voxel_size,
            "estimated_peak_bytes": exc.estimated_peak_bytes,
            "backend": exc.backend,
        })
        seq.close()
        raise
    if progress:
        progress(
            f"fusion TSDF: volume {layout}, backend {vol.backend}"
        )
        voxel_count = int(np.prod(vol.dims, dtype=np.int64))
        dimensions = " x ".join(str(int(value)) for value in vol.dims)
        if layout == "dense":
            progress(
                f"TSDF volume: {dimensions} = "
                f"{voxel_count / 1_000_000:.1f} M voxels"
            )
        else:
            progress(
                f'domaine TSDF: {dimensions} = {voxel_count / 1000000:.1f} M virtual voxels; only observed blocks are allocated'
            )

    n = len(fids)
    gravity_samples = []
    emit_progress_event(progress, "fusion_started", {
        "total_frames": n,
        "source_total_frames": len(source_fids),
        "backend": vol.backend,
        "volume_layout": layout,
        "voxel_size": float(cfg.voxel_size),
        "surface_field": str(getattr(cfg, "surface_field", "raw")),
        "surface_min_support": float(getattr(
            cfg, "surface_min_support", 0.20,
        )),
        "surface_extraction": str(getattr(
            cfg, "surface_extraction", "zero_crossing",
        )),
        "surface_band_factor": float(getattr(
            cfg, "surface_band_factor", 1.5,
        )),
        "surface_rescue_distance_factor": float(getattr(
            cfg, "surface_rescue_distance_factor", 1.0,
        )),
        "surface_rescue_min_weight": float(getattr(
            cfg, "surface_rescue_min_weight", 5.0,
        )),
        "gradient_smoothing_sigma": float(getattr(
            cfg, "gradient_smoothing_sigma", 0.8,
        )),
        "depth_edge_threshold": float(getattr(
            cfg, "depth_edge_threshold", 0.05,
        )),
        "depth_edge_background_weight": float(getattr(
            cfg, "depth_edge_background_weight", 0.10,
        )),
        "depth_edge_radius": int(getattr(cfg, "depth_edge_radius", 1)),
        "frame_stride": effective_frame_stride,
        "requested_frame_stride": int(cfg.frame_stride),
        "min_frames": int(getattr(cfg, "min_frames", 0)),
        "max_frames": int(cfg.max_frames),
    })
    if trace is not None:
        trace.clear()
        trace.update({
            "backend": vol.backend,
            "volume_layout": layout,
            "total_frames": n,
            "source_total_frames": len(source_fids),
            "frame_stride": effective_frame_stride,
            "requested_frame_stride": int(cfg.frame_stride),
            "surface_field": str(getattr(cfg, "surface_field", "raw")),
            "surface_min_support": float(getattr(
                cfg, "surface_min_support", 0.20,
            )),
            "surface_extraction": str(getattr(
                cfg, "surface_extraction", "zero_crossing",
            )),
            "surface_band_factor": float(getattr(
                cfg, "surface_band_factor", 1.5,
            )),
            "surface_rescue_distance_factor": float(getattr(
                cfg, "surface_rescue_distance_factor", 1.0,
            )),
            "surface_rescue_min_weight": float(getattr(
                cfg, "surface_rescue_min_weight", 5.0,
            )),
            "gradient_smoothing_sigma": float(getattr(
                cfg, "gradient_smoothing_sigma", 0.8,
            )),
            "depth_edge_threshold": float(getattr(
                cfg, "depth_edge_threshold", 0.05,
            )),
            "depth_edge_background_weight": float(getattr(
                cfg, "depth_edge_background_weight", 0.10,
            )),
            "depth_edge_radius": int(getattr(
                cfg, "depth_edge_radius", 1,
            )),
            "min_frames": int(getattr(cfg, "min_frames", 0)),
            "max_frames": int(cfg.max_frames),
            "sample_points_per_frame": 180,
            "frames": [],
        })
    for i, fid in enumerate(fids):
        fr = seq.load(fid)
        if fr is None:
            continue
        camera_down = np.asarray(fr.pose[:3, 1], np.float64)
        if np.isfinite(camera_down).all():
            gravity_samples.append(-camera_down)
        vol.integrate(fr, K, cfg.depth_trunc)
        if trace is not None or callable(getattr(progress, "event", None)):
            frame_trace = _trace_frame_sample(fr, K, cfg.depth_trunc)
            frame_trace["frame_index"] = i + 1
            if trace is not None:
                trace["frames"].append(frame_trace)
            emit_progress_event(progress, "fusion_frame", frame_trace)
        if progress and (i % 10 == 0 or i == n - 1):
            progress(f"fusion TSDF: trame {i+1}/{n}")
    seq.close()
    if gravity_samples:
        gravity_hint = np.median(np.asarray(gravity_samples), axis=0)
        magnitude = np.linalg.norm(gravity_hint)
        if magnitude > 1e-9:
            vol.gravity_up_hint = gravity_hint / magnitude
    vol.to_cpu()
    if trace is not None:
        trace["volume"] = vol.visibility_summary()
    cloud = vol.extract_surface(
        sampling=cfg.iso_sampling,
        max_points=int(getattr(cfg, "max_surface_points", 175000)),
    )
    extraction_stats = dict(getattr(
        vol, "surface_extraction_stats", {},
    ))
    if trace is not None:
        trace["surface_extraction"] = extraction_stats
    emit_progress_event(progress, "surface_extracted", {
        "point_count": int(cloud.size),
        "max_surface_points": int(getattr(
            cfg, "max_surface_points", 175000,
        )),
        **extraction_stats,
    })
    return cloud, vol
