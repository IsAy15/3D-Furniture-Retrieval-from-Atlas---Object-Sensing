'\nGeometric verification: constrained ICP, forward and reverse coverage, mean surface distance, explained height/thickness and optional visibility/normal checks. Rank admissible poses using the coverage harmonic mean with configured penalties. See docs/PIPELINE.md for metric definitions.\n'

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from constellations import Constellation
from geometry import PointCloud, rotation_about_axis
from parallelism import kdtree_workers
from transforms import GroundTransform, refine_transform
from sdf_fusion import VIS_FREE, VIS_OCCUPIED


@dataclass
class Registration:
    model_name: str
    transform: GroundTransform
    coverage: float            # = couverture avant ; arrière/structure = garde-fous
    mean_surface_dist: float
    quality: float
    cov_forward: float = 0.0
    cov_reverse: float = 0.0


def normal_compatible_coverage(model_cloud, scan_cloud, transform, threshold, angle_deg):
    """Surface coverage requiring unoriented normals to agree; None if unavailable.

    Invalid normals provide no evidence either way, so their close points keep
    distance-only support. Opposite normals agree (RGB-D orientation is arbitrary).
    """
    from geometry import rotation_about_axis
    if angle_deg <= 0 or model_cloud.normals is None or scan_cloud.normals is None:
        return None
    if not model_cloud.size or not scan_cloud.size:
        return 0.0
    distances, indices = scan_cloud.tree.query(transform.apply(model_cloud.points), workers=1)
    model_normals = model_cloud.normals @ rotation_about_axis(transform.up, transform.theta).T
    scan_normals = scan_cloud.normals[indices]
    lengths = np.linalg.norm(model_normals, axis=1) * np.linalg.norm(scan_normals, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-8)
    agreement = np.ones(len(distances), dtype=bool)
    dots = np.abs(np.sum(model_normals[valid] * scan_normals[valid], axis=1)) / lengths[valid]
    agreement[valid] = dots >= np.cos(np.deg2rad(angle_deg))
    return float(np.mean((distances < threshold) & agreement))


def symmetric_coverage_score(coverage: float, cov_reverse: float) -> float:
    'Robust harmonic mean of geometric coverage in both directions.'
    forward = max(0.0, float(coverage))
    reverse = max(0.0, float(cov_reverse))
    if not np.isfinite(forward) or not np.isfinite(reverse):
        return 0.0
    total = forward + reverse
    return 0.0 if total <= 0.0 else 2.0 * forward * reverse / total


def visibility_coverage_score(model_cloud: PointCloud, T: GroundTransform,
                              volume, scan_cloud=None, threshold: float = 0.0):
    'Measure model support only in observed space. Ignore unknown points; occupied cells support the model; observed free cells contradict it beyond surface tolerance. Return support, known fraction, free fraction and known point count.\n    '
    if volume is None or not hasattr(volume, "sample_visibility"):
        return None
    points = T.apply(model_cloud.points)
    if not len(points):
        return None
    labels = np.asarray(volume.sample_visibility(points), np.uint8)
    occupied = labels == VIS_OCCUPIED
    free = labels == VIS_FREE
    if scan_cloud is not None and threshold > 0 and np.any(free):
        distances = scan_cloud.tree.query(points[free], workers=kdtree_workers())[0]
        near_surface = np.zeros(len(points), dtype=bool)
        near_surface[free] = distances < threshold
        occupied |= near_surface
        free &= ~near_surface
    known = occupied | free
    known_count = int(np.count_nonzero(known))
    if known_count == 0:
        return 0.0, 0.0, 0.0, 0
    occupied_count = int(np.count_nonzero(occupied))
    free_count = int(np.count_nonzero(free))
    support = occupied_count / known_count
    return (
        float(support),
        float(known_count / len(points)),
        float(free_count / known_count),
        known_count,
    )


def coverage_score(model_cloud: PointCloud, scan_cloud: PointCloud,
                   T: GroundTransform, threshold: float = 0.05,
                   up=None, floor_band: float = 0.06):
    'Return forward coverage, reverse coverage, mean surface distance, structure height and thickness. Forward is the fraction of model points near the scan. Reverse is the fraction of non-ground scan points in the model footprint near the model. Structure and thickness describe explained scan support.'
    mp = T.apply(model_cloud.points)
    workers = kdtree_workers()
    d_fwd, _ = scan_cloud.tree.query(mp, workers=workers)
    cov_fwd = float((d_fwd < threshold).mean())
    msd = float(d_fwd[d_fwd < threshold].mean()) if (d_fwd < threshold).any() else float("inf")

    lo = mp.min(0) - 2 * threshold
    hi = mp.max(0) + 2 * threshold
    sp = scan_cloud.points
    in_box = np.all((sp >= lo) & (sp <= hi), axis=1)
    sp = sp[in_box]
    if up is not None and len(sp) > 0:
        up = np.asarray(up, float)
        ground = float((mp @ up).min())
        sp = sp[(sp @ up - ground) >= floor_band]      # exclure le sol
    if len(sp) < 10:
        return cov_fwd, 0.0, msd, 0.0, 1.0
    d_rev, _ = cKDTree(mp).query(sp, workers=workers)
    cov_rev = float((d_rev < threshold).mean())
    # Les points de scan réellement expliqués par le modèle.
    expl = sp[d_rev < threshold]
    struct_h = float(np.ptp(expl @ up)) if (up is not None and len(expl) > 0) else 0.0
    # Planéité : plus petite dispersion principale (= épaisseur). Un mur/rebord plat
    # est une feuille mince ; un vrai meuble (pieds + plateau/dossier) est volumineux
    # dans the 3 directions. On rejette the feuilles minces (faux positifs sur murs).
    if len(expl) >= 10:
        sv = np.linalg.svd(expl - expl.mean(0), compute_uv=False)
        thickness = float(sv[-1] / np.sqrt(len(expl)))
    else:
        thickness = 1.0
    return cov_fwd, cov_rev, msd, struct_h, thickness


def group_surface_coverage(model_cloud, scan_cloud, transform, group_points,
                           threshold=.05, up=None, ground=None):
    """Measure an independently defined object region, not the model's crop.

    Use the same two-tolerance padding as coverage_score. The group is a scan
    proposal and is identical across its competing models. Missing groups or
    fewer than ten surface samples do not provide a reliable measurement.
    """
    if group_points is None:
        return None
    points = np.asarray(group_points, float).reshape(-1, 3)
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    lo, hi = points.min(0) - 2 * threshold, points.max(0) + 2 * threshold
    sp = scan_cloud.points
    mask = np.all((sp >= lo) & (sp <= hi), axis=1)
    if up is not None and ground is not None:
        mask &= sp @ np.asarray(up) >= float(ground) + .06
    region = sp[mask]
    if len(region) < 10:
        return None
    distances = cKDTree(transform.apply(model_cloud.points)).query(
        region, workers=kdtree_workers())[0]
    return float(np.mean(distances < threshold)), len(region), (lo, hi)


def model_support_distribution(model_cloud: PointCloud, scan_cloud: PointCloud,
                               T: GroundTransform, threshold: float = 0.05,
                               zone_grid: int = 2,
                               zone_min_support: float = 0.15):
    'Measure support across canonical model zones and the supported spatial extent. A small dense patch must not represent the entire furniture object.\n    '
    points = np.asarray(model_cloud.points, float)
    if len(points) == 0:
        return 0.0, 0.0, 0, 0
    transformed = T.apply(points)
    distances, _ = scan_cloud.tree.query(transformed, workers=kdtree_workers())
    supported = distances < float(threshold)
    if not np.any(supported):
        return 0.0, 0.0, 0, 0

    lo = points.min(axis=0)
    extent = np.ptp(points, axis=0)
    grid = max(1, int(zone_grid))
    normalized = (points - lo) / np.maximum(extent, 1e-9)
    cells = np.minimum(grid - 1, np.floor(normalized * grid).astype(int))
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    zone_count = int(inverse.max(initial=-1) + 1)
    supported_zones = 0
    for zone in range(zone_count):
        members = inverse == zone
        if members.any() and float(supported[members].mean()) >= float(zone_min_support):
            supported_zones += 1
    zone_fraction = supported_zones / max(1, zone_count)

    transformed_extent = np.ptp(transformed, axis=0)
    support_extent = np.ptp(transformed[supported], axis=0)
    relevant = transformed_extent > max(float(threshold), 1e-6)
    ratios = np.clip(
        support_extent[relevant] / np.maximum(transformed_extent[relevant], 1e-9),
        0.0, 1.0,
    )
    extent_ratio = (
        float(np.prod(np.maximum(ratios, 1e-6)) ** (1.0 / len(ratios)))
        if len(ratios) else 1.0
    )
    return float(zone_fraction), extent_ratio, supported_zones, zone_count


def icp_refine(model_pts, scan_cloud: PointCloud, T: GroundTransform, up,
               iters: int = 10, max_dist: float = 0.08,
               scale_bounds=(1.0 / 1.5, 1.5)) -> Optional[GroundTransform]:
    'Dense ICP constrained to yaw, translation and uniform scale. Reject updates outside the admissible scale range rather than clipping to a false boundary optimum.\n    '
    if not (scale_bounds[0] <= T.scale <= scale_bounds[1]):
        return None
    tree = scan_cloud.tree
    sp = scan_cloud.points
    for _ in range(iters):
        mp = T.apply(model_pts)
        d, idx = tree.query(mp, workers=kdtree_workers())
        good = d < max_dist
        if good.sum() < 20:
            break
        w = np.ones(int(good.sum()))
        Tn = refine_transform(model_pts[good], sp[idx[good]], w, up)
        if not (scale_bounds[0] <= Tn.scale <= scale_bounds[1]):
            return None
        if np.allclose(Tn.t, T.t, atol=1e-4) and abs(Tn.theta - T.theta) < 1e-4:
            T = Tn
            break
        T = Tn
    return T


def verify_model(model_name: str, model_cloud: PointCloud, scan_cloud: PointCloud,
                 constellations: List[Constellation], threshold: float = 0.05,
                 top: int = 5, up=None, reverse_gate: float = 0.42,
                 min_structure: float = 0.15, min_thickness: float = 0.06,
                 model_zone_grid: int = 2, model_zone_min_support: float = 0.15,
                 min_model_zone_fraction: float = 0.0,
                 min_support_extent_ratio: float = 0.0,
                 support_locality_weight: float = 0.0,
                 surface_distance_weight: float = 0.05,
                 scale_bounds=(1.0 / 1.5, 1.5),
                 diagnostics=None,
                 coverage_threshold: float = 0.0, group_points=None) -> Optional[Registration]:
    'Choose the best admissible post-ICP registration. Apply safeguards and coverage threshold before ranking harmonic quality with surface/support penalties.\n    '
    # sous-échantillonnage du modèle pour un ICP rapide
    mp_full = model_cloud.points
    if len(mp_full) > 2500:
        mp_sub = mp_full[np.linspace(0, len(mp_full) - 1, 2500).astype(int)]
    else:
        mp_sub = mp_full

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "constellations_available": len(constellations),
            "constellations_tested": min(len(constellations), int(top)),
            "reverse_gate": float(reverse_gate),
            "coverage_threshold": float(coverage_threshold),
            "min_structure": float(min_structure),
            "min_thickness": float(min_thickness),
            "min_model_zone_fraction": float(min_model_zone_fraction),
            "min_support_extent_ratio": float(min_support_extent_ratio),
            "poses": [],
        })

    best: Optional[Registration] = None
    best_score = -1.0
    rejection_counts = {}
    for rank, c in enumerate(constellations[:top], 1):
        pose_trace = {
            "rank": rank,
            "constellation_quality": round(float(c.quality), 4),
            "inliers": len(c.inliers),
        }
        T = (icp_refine(mp_sub, scan_cloud, c.transform, up,
                        scale_bounds=scale_bounds)
             if up is not None else c.transform)
        if T is None:
            rejection_counts["icp_or_scale"] = rejection_counts.get("icp_or_scale", 0) + 1
            pose_trace["status"] = "rejected_icp_or_scale"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        fwd, rev, msd, struct_h, thickness = coverage_score(model_cloud, scan_cloud, T, threshold, up=up)
        zone_fraction, extent_ratio, supported_zones, zone_count = model_support_distribution(
            model_cloud, scan_cloud, T, threshold,
            zone_grid=model_zone_grid,
            zone_min_support=model_zone_min_support,
        )
        pose_trace.update({
            "coverage": round(float(fwd), 4),
            "cov_reverse": round(float(rev), 4),
            "mean_surface_dist_m": round(float(msd), 4) if np.isfinite(msd) else None,
            "structure_height": round(float(struct_h), 4),
            "thickness": round(float(thickness), 4),
            "model_zone_fraction": round(float(zone_fraction), 4),
            "supported_model_zones": int(supported_zones),
            "model_zone_count": int(zone_count),
            "support_extent_ratio": round(float(extent_ratio), 4),
            "theta_deg": round(float(T.theta) * 57.2958, 2),
            "scale": round(float(T.scale), 4),
            "translation": [round(float(x), 4) for x in T.t],
        })
        if rev < reverse_gate:                 # garde-fou anti "plaqué sur mur/sol"
            rejection_counts["reverse_gate"] = rejection_counts.get("reverse_gate", 0) + 1
            pose_trace["status"] = "rejected_reverse_gate"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if up is not None and struct_h < min_structure:   # patch plat horizontal (sol)
            rejection_counts["structure"] = rejection_counts.get("structure", 0) + 1
            pose_trace["status"] = "rejected_structure"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if up is not None and thickness < min_thickness:  # feuille mince (mur/rebord)
            rejection_counts["thickness"] = rejection_counts.get("thickness", 0) + 1
            pose_trace["status"] = "rejected_thickness"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if zone_fraction < min_model_zone_fraction:
            rejection_counts["model_zones"] = rejection_counts.get("model_zones", 0) + 1
            pose_trace["status"] = "rejected_model_zones"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if extent_ratio < min_support_extent_ratio:
            rejection_counts["support_locality"] = rejection_counts.get("support_locality", 0) + 1
            pose_trace["status"] = "rejected_support_locality"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if fwd < coverage_threshold:
            rejection_counts["coverage"] = rejection_counts.get("coverage", 0) + 1
            pose_trace["status"] = "rejected_coverage"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        dist_penalty = (msd / threshold) if np.isfinite(msd) and threshold > 0 else 1.0
        locality_penalty = max(0.0, 1.0 - 0.5 * (zone_fraction + extent_ratio))
        group_support = group_surface_coverage(model_cloud, scan_cloud, T, group_points, threshold, up)
        ranking_reverse = min(rev, group_support[0]) if group_support is not None else rev
        if group_support is not None:
            pose_trace["group_cov_reverse"] = round(group_support[0], 4)
        score = (symmetric_coverage_score(fwd, ranking_reverse)
                 - surface_distance_weight * dist_penalty
                 - support_locality_weight * locality_penalty)
        pose_trace["status"] = "accepted_pose"
        pose_trace["score"] = round(float(score), 4)
        if diagnostics is not None and len(diagnostics["poses"]) < 12:
            diagnostics["poses"].append(pose_trace)
        if score > best_score:
            best_score = score
            best = Registration(model_name, T, fwd, msd, c.quality,
                                cov_forward=fwd, cov_reverse=rev)
    if diagnostics is not None:
        diagnostics["rejection_counts"] = rejection_counts
        diagnostics["status"] = "accepted" if best is not None else "rejected"
        diagnostics["best_score"] = round(float(best_score), 4) if best is not None else None
    return best


def _horizontal_distance(a: np.ndarray, b: np.ndarray, up) -> float:
    up = np.asarray(up, float)
    up = up / (np.linalg.norm(up) + 1e-12)
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d - (d @ up) * up
    return float(np.linalg.norm(d))


def verify_model_candidates(model_name: str, model_cloud: PointCloud, scan_cloud: PointCloud,
                            constellations: List[Constellation], threshold: float = 0.05,
                            top: int = 12, up=None, reverse_gate: float = 0.42,
                            min_structure: float = 0.15, min_thickness: float = 0.06,
                            model_zone_grid: int = 2,
                            model_zone_min_support: float = 0.15,
                            min_model_zone_fraction: float = 0.0,
                            min_support_extent_ratio: float = 0.0,
                            support_locality_weight: float = 0.0,
                            surface_distance_weight: float = 0.05,
                            max_results: int = 1,
                            min_center_distance: float = 0.35,
                            scale_bounds=(1.0 / 1.5, 1.5),
                            diagnostics=None,
                            coverage_threshold: float = 0.0, group_points=None) -> List[Registration]:
    'Return distinct poses for one model. Apply coverage acceptance before ranking and retain sufficiently separated horizontal centers, allowing repeated instances.\n    '
    if max_results <= 1:
        reg = verify_model(model_name, model_cloud, scan_cloud, constellations,
                           threshold=threshold, top=top, up=up,
                           reverse_gate=reverse_gate,
                           min_structure=min_structure,
                           min_thickness=min_thickness,
                           model_zone_grid=model_zone_grid,
                           model_zone_min_support=model_zone_min_support,
                           min_model_zone_fraction=min_model_zone_fraction,
                           min_support_extent_ratio=min_support_extent_ratio,
                           support_locality_weight=support_locality_weight,
                           surface_distance_weight=surface_distance_weight,
                           scale_bounds=scale_bounds,
                           diagnostics=diagnostics,
                           coverage_threshold=coverage_threshold, group_points=group_points)
        return [] if reg is None else [reg]

    mp_full = model_cloud.points
    if len(mp_full) > 2500:
        mp_sub = mp_full[np.linspace(0, len(mp_full) - 1, 2500).astype(int)]
    else:
        mp_sub = mp_full

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "constellations_available": len(constellations),
            "constellations_tested": min(len(constellations), int(top)),
            "reverse_gate": float(reverse_gate),
            "coverage_threshold": float(coverage_threshold),
            "min_structure": float(min_structure),
            "min_thickness": float(min_thickness),
            "min_model_zone_fraction": float(min_model_zone_fraction),
            "min_support_extent_ratio": float(min_support_extent_ratio),
            "poses": [],
        })

    scored = []
    rejection_counts = {}
    for rank, c in enumerate(constellations[:top], 1):
        pose_trace = {
            "rank": rank,
            "constellation_quality": round(float(c.quality), 4),
            "inliers": len(c.inliers),
        }
        T = (icp_refine(mp_sub, scan_cloud, c.transform, up,
                        scale_bounds=scale_bounds)
             if up is not None else c.transform)
        if T is None:
            rejection_counts["icp_or_scale"] = rejection_counts.get("icp_or_scale", 0) + 1
            pose_trace["status"] = "rejected_icp_or_scale"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        fwd, rev, msd, struct_h, thickness = coverage_score(model_cloud, scan_cloud, T, threshold, up=up)
        zone_fraction, extent_ratio, supported_zones, zone_count = model_support_distribution(
            model_cloud, scan_cloud, T, threshold,
            zone_grid=model_zone_grid,
            zone_min_support=model_zone_min_support,
        )
        pose_trace.update({
            "coverage": round(float(fwd), 4),
            "cov_reverse": round(float(rev), 4),
            "mean_surface_dist_m": round(float(msd), 4) if np.isfinite(msd) else None,
            "structure_height": round(float(struct_h), 4),
            "thickness": round(float(thickness), 4),
            "model_zone_fraction": round(float(zone_fraction), 4),
            "supported_model_zones": int(supported_zones),
            "model_zone_count": int(zone_count),
            "support_extent_ratio": round(float(extent_ratio), 4),
            "theta_deg": round(float(T.theta) * 57.2958, 2),
            "scale": round(float(T.scale), 4),
            "translation": [round(float(x), 4) for x in T.t],
        })
        if rev < reverse_gate:
            rejection_counts["reverse_gate"] = rejection_counts.get("reverse_gate", 0) + 1
            pose_trace["status"] = "rejected_reverse_gate"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if up is not None and struct_h < min_structure:
            rejection_counts["structure"] = rejection_counts.get("structure", 0) + 1
            pose_trace["status"] = "rejected_structure"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if up is not None and thickness < min_thickness:
            rejection_counts["thickness"] = rejection_counts.get("thickness", 0) + 1
            pose_trace["status"] = "rejected_thickness"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if zone_fraction < min_model_zone_fraction:
            rejection_counts["model_zones"] = rejection_counts.get("model_zones", 0) + 1
            pose_trace["status"] = "rejected_model_zones"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if extent_ratio < min_support_extent_ratio:
            rejection_counts["support_locality"] = rejection_counts.get("support_locality", 0) + 1
            pose_trace["status"] = "rejected_support_locality"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        if fwd < coverage_threshold:
            rejection_counts["coverage"] = rejection_counts.get("coverage", 0) + 1
            pose_trace["status"] = "rejected_coverage"
            if diagnostics is not None and len(diagnostics["poses"]) < 12:
                diagnostics["poses"].append(pose_trace)
            continue
        dist_penalty = (msd / threshold) if np.isfinite(msd) and threshold > 0 else 1.0
        locality_penalty = max(0.0, 1.0 - 0.5 * (zone_fraction + extent_ratio))
        group_support = group_surface_coverage(model_cloud, scan_cloud, T, group_points, threshold, up)
        ranking_reverse = min(rev, group_support[0]) if group_support is not None else rev
        if group_support is not None:
            pose_trace["group_cov_reverse"] = round(group_support[0], 4)
        score = (symmetric_coverage_score(fwd, ranking_reverse)
                 - surface_distance_weight * dist_penalty
                 - support_locality_weight * locality_penalty)
        pose_trace["status"] = "accepted_pose"
        pose_trace["score"] = round(float(score), 4)
        if diagnostics is not None and len(diagnostics["poses"]) < 12:
            diagnostics["poses"].append(pose_trace)
        scored.append((score, Registration(model_name, T, fwd, msd, c.quality,
                                           cov_forward=fwd, cov_reverse=rev)))

    selected: List[Registration] = []
    for _, reg in sorted(scored, key=lambda x: -x[0]):
        if up is not None:
            if any(_horizontal_distance(reg.transform.t, prev.transform.t, up) < min_center_distance
                   for prev in selected):
                continue
        selected.append(reg)
        if len(selected) >= max_results:
            break
    if diagnostics is not None:
        diagnostics["rejection_counts"] = rejection_counts
        diagnostics["status"] = "accepted" if selected else "rejected"
        diagnostics["accepted_registrations"] = len(selected)
    return selected
