'\nVectorized keypoint constellations using 1-Point RANSAC. Each correspondence provides yaw, scale and translation. Collect geometric and descriptor inliers, refine using primitive-size weights and rank constellations. Seed count is bounded by max_seeds.\n'

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config import PipelineConfig
from matching import Correspondence, KeyPointFeature
from transforms import GroundTransform, refine_transform


@dataclass
class Constellation:
    transform: GroundTransform
    inliers: List[Correspondence]
    quality: float


def _similar(a: GroundTransform, b: GroundTransform) -> bool:
    dth = abs((a.theta - b.theta + np.pi) % (2 * np.pi) - np.pi)
    return (dth < np.radians(10) and abs(a.scale - b.scale) < 0.15
            and np.linalg.norm(a.t - b.t) < 0.2)


def one_point_ransac(corres: List[Correspondence], model_feats, scan_feats,
                     cfg: PipelineConfig, up, diagnostics=None) -> List[Constellation]:
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    rc = cfg.ransac
    n = len(corres)
    if n < rc.min_correspondences:
        if diagnostics is not None:
            diagnostics.update(
                seeds=0, accepted=0,
                rejected_min_correspondences=1,
            )
        return []
    rejected = {
        "empty": 0, "scale": 0, "model_inliers": 0,
        "scan_inliers": 0, "scan_spread": 0,
        "scan_cells": 0, "model_cells": 0,
    }
    maxima = {
        "model_inliers": 0, "scan_inliers": 0,
        "scan_spread": 0.0, "scan_cells": 0, "model_cells": 0,
    }

    # Tableaux précalculés (vectorisation de la collecte d'inliers).
    mpos = np.array([model_feats[c.model_idx].position for c in corres])    # (n,3)
    spos = np.array([scan_feats[c.scan_idx].position for c in corres])      # (n,3)
    ddist = np.array([c.desc_dist for c in corres])                        # (n,)
    msize = np.array([float(model_feats[c.model_idx].size6d.sum()) for c in corres])
    midx = np.array([c.model_idx for c in corres])
    sidx = np.array([c.scan_idx for c in corres])
    desc_ok = ddist < rc.desc_inlier

    def occupied_cells(points, cell_size):
        'Count relative spatial cells, invariant to translation.'
        points = np.asarray(points, float)
        if len(points) == 0:
            return 0
        size = max(float(cell_size), 1e-6)
        cells = np.floor((points - points.min(axis=0)) / size + 1e-9).astype(int)
        return len(np.unique(cells, axis=0))

    def unique_pairs(indices):
        'Keep the best bijective pairs, with unique keypoints on both sides.'
        ordered = sorted(indices, key=lambda index: float(ddist[index]))
        selected = []
        used_model = set()
        used_scan = set()
        for index in ordered:
            model_index = int(midx[index])
            scan_index = int(sidx[index])
            if model_index in used_model or scan_index in used_scan:
                continue
            selected.append(int(index))
            used_model.add(model_index)
            used_scan.add(scan_index)
        return np.asarray(selected, dtype=int)

    def collect(T: GroundTransform):
        rt = T.apply(mpos)
        d = np.linalg.norm(rt - spos, axis=1)
        return unique_pairs(np.where((d < rc.geom_inlier) & desc_ok)[0])

    # Clamp LARGE pendant le raffinement (sécurité numérique seulement) ; le rejet
    # d'scale utilise la plage stricte (rc.scale_min/max) -> un modèle qui veut
    # une scale hors plage est rejected, pas écrasé à la borne.
    refine_bounds = (0.1, 6.0)

    def refine(inl_idx):
        return refine_transform(mpos[inl_idx], spos[inl_idx], msize[inl_idx], up,
                                scale_bounds=refine_bounds)

    def quality(inl_idx) -> float:
        # Q = somme_{Ki distincts} (t_desc - d) * Psize  (meilleur par key point modèle)
        best = {}
        for i in inl_idx:
            mi = midx[i]
            if mi not in best or ddist[i] < ddist[best[mi]]:
                best[mi] = i
        sel = list(best.values())
        return float(np.sum(np.maximum(0.0, rc.desc_inlier - ddist[sel]) * msize[sel]))

    # Germes : correspondences de plus grosse primitive (priorité), limités.
    order = np.argsort(-msize)[: rc.max_seeds]
    results: List[Constellation] = []
    for oi in order:
        T = corres[oi].transform
        inl = collect(T)
        for _ in range(rc.refine_iterations):
            if len(inl) < 2:
                break
            T = refine(inl)
            new = collect(T)
            if len(new) <= len(inl):
                inl = new
                break
            inl = new
        if len(inl) == 0:
            rejected["empty"] += 1
            continue
        # Rejet des constellations dégénérées : scale hors plage (modèle réduit
        # à un point -> "couverture" trompeuse) ou trop peu de key points distincts.
        if not (rc.scale_min <= T.scale <= rc.scale_max):
            rejected["scale"] += 1
            continue
        model_inliers = len(np.unique(midx[inl]))
        scan_inliers = len(np.unique(sidx[inl]))
        maxima["model_inliers"] = max(maxima["model_inliers"], model_inliers)
        maxima["scan_inliers"] = max(maxima["scan_inliers"], scan_inliers)
        if model_inliers < rc.min_inliers:
            rejected["model_inliers"] += 1
            continue
        if scan_inliers < getattr(rc, "min_scan_inliers", rc.min_inliers):
            rejected["scan_inliers"] += 1
            continue
        scan_support = spos[inl]
        if len(scan_support) > 1:
            centered = scan_support - scan_support.mean(axis=0)
            spread = float(2.0 * np.max(np.linalg.norm(centered, axis=1)))
        else:
            spread = 0.0
        maxima["scan_spread"] = max(maxima["scan_spread"], spread)
        if spread < float(getattr(rc, "min_scan_spread", 0.0)):
            rejected["scan_spread"] += 1
            continue
        scan_cells = occupied_cells(
            scan_support, getattr(rc, "scan_cell_size", rc.min_scan_spread),
        )
        maxima["scan_cells"] = max(maxima["scan_cells"], scan_cells)
        if scan_cells < int(getattr(rc, "min_scan_cells", 1)):
            rejected["scan_cells"] += 1
            continue
        model_support = mpos[inl]
        model_cells = occupied_cells(
            model_support, getattr(rc, "model_cell_size", rc.min_scan_spread),
        )
        maxima["model_cells"] = max(maxima["model_cells"], model_cells)
        if model_cells < int(getattr(rc, "min_model_cells", 1)):
            rejected["model_cells"] += 1
            continue
        Q = quality(inl)
        merged = False
        for r in results:
            if _similar(r.transform, T):
                if Q > r.quality:
                    r.transform = T
                    r.inliers = [corres[i] for i in inl]
                    r.quality = Q
                merged = True
                break
        if not merged:
            results.append(Constellation(T, [corres[i] for i in inl], Q))

    results.sort(key=lambda c: -c.quality)
    if diagnostics is not None:
        diagnostics.update({
            "seeds": int(len(order)), "accepted": int(len(results)),
            "rejected": {key: int(value) for key, value in rejected.items()},
            "max_support": {
                key: round(float(value), 6)
                for key, value in maxima.items()
            },
        })
    return results[: rc.top_constellations]
