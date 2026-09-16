"""Optional, compact local-UDF proposals for partially observed objects.

This channel supplements the statistical pool. It never removes an existing
candidate and does not read annotations. Exact matching remains authoritative.
"""
from pathlib import Path
import json
import hashlib

import numpy as np

from candidate_index import _top_feature_indices
from descriptors import _grid_offsets, EMPTY, OCCUPIED


def source_fingerprint(db_dir):
    import db_store
    digest = hashlib.sha256()
    for path in db_store.model_files(str(db_dir)):
        stat = Path(path).stat()
        digest.update(f"{Path(path).name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _query_vectors(features, extent=.12, resolution=6):
    features = [features[i] for i in _top_feature_indices(features, 16)]
    occupied, free, lost, heights, confidence = [], [], [], [], []
    for feature in features:
        descriptor = feature.descriptor
        if descriptor.occ is None or not np.isclose(descriptor.extent, extent):
            raise ValueError("Local index and query descriptor extents must agree")
        offsets = _grid_offsets(descriptor.res, descriptor.extent)
        states = descriptor.occ.reshape(-1)
        observed = states == OCCUPIED
        count = max(int(observed.sum()), 1)
        for theta in np.arange(8) * np.pi / 4:
            c, s = np.cos(theta), np.sin(theta)
            rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            rotated = offsets @ rotation.T
            valid = (np.abs(rotated) <= extent + 1e-6).all(1)
            cells = np.clip(np.rint(
                (rotated + extent) / (2 * extent) * (resolution - 1)
            ), 0, resolution - 1).astype(int)
            indices = np.ravel_multi_index(cells.T, (resolution,) * 3)
            occupied.append(np.bincount(
                indices[valid & observed], minlength=resolution ** 3,
            ) / count)
            totals = np.bincount(indices[valid], minlength=resolution ** 3)
            free.append(np.bincount(
                indices[valid & (states == EMPTY)], minlength=resolution ** 3,
            ) / np.maximum(totals, 1))
            lost.append(float((observed & ~valid).sum()) / count)
            heights.append(feature.height)
            confidence.append(descriptor.n_occupied + descriptor.n_unknown)
    return {
        "occupied": np.asarray(occupied, dtype=np.float32).T,
        "free": np.asarray(free, dtype=np.float32).T,
        "lost": np.asarray(lost, dtype=np.float32),
        "height": np.asarray(heights, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "features": len(features),
    }


def _shard_costs(data, query, matching):
    udf = data["udf"].astype(np.float32)
    height = data["height"][:, None]
    distance = udf @ query["occupied"] + .12 * query["lost"][None, :]
    occupied = (udf < .016).astype(np.float32)
    conflict = (occupied @ query["free"]) / np.maximum(
        occupied.sum(1, keepdims=True), 1,
    )
    scale = np.where(height > 1e-3,
                     query["height"][None, :] / np.maximum(height, 1e-9), 1.)
    valid = (
        (scale >= matching.scale_min) & (scale <= matching.scale_max)
        & (query["confidence"][None, :]
           / np.maximum(data["occupancy"][:, None], 1)
           >= matching.confidence_threshold)
    )
    models, starts = np.unique(data["model"], return_index=True)
    result = []
    for values in (distance, distance + .02 * conflict):
        values = np.where(valid, values, np.inf).reshape(
            len(udf), query["features"], 8,
        ).min(2)
        result.append(np.minimum.reduceat(values, starts, axis=0))
    return models, result


def _fuse_cost_rankings(distance, with_free, limit):
    """Keep both complete and robust local support; stable reciprocal ranks."""
    count = distance.shape[1]
    majority = max(1, int(np.ceil(.75 * count)))
    scores = [
        np.minimum(distance, .12).mean(1),
        np.minimum(with_free, .12).mean(1),
        np.minimum(np.sort(with_free, axis=1)[:, :majority], .12).mean(1),
    ]
    supported = np.isfinite(distance).sum(1) >= min(2, count)
    fused = np.zeros(len(distance))
    for score in scores:
        order = np.flatnonzero(supported)
        order = order[np.argsort(score[order], kind="stable")]
        fused[order] += 1. / (60 + np.arange(1, len(order) + 1))
    order = np.flatnonzero(supported)
    return order[np.argsort(-fused[order], kind="stable")][:limit].tolist()


def rank_local_groups(index_dir, names, feature_groups, cfg, limit=500, progress=None):
    """Return one proposal ranking per group from a complete, aligned index."""
    root = Path(index_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["names"] != list(names) or manifest["resolution"] != 6:
        raise ValueError("Local candidate index does not match the model database")
    if not np.allclose(cfg.up_axis, [0, 1, 0]):
        raise ValueError("Local candidate index requires Y-up descriptors")
    shards = [root / f"{start:05d}.npz" for start in range(0, len(names), 1000)]
    if not all(path.is_file() for path in shards):
        raise ValueError("Local candidate index is incomplete")
    queries = [_query_vectors(group) if group else None for group in feature_groups]
    costs = [
        [np.full((len(names), q["features"]), np.inf, dtype=np.float32)
         for _ in range(2)] if q else None for q in queries
    ]
    for shard_number, path in enumerate(shards, 1):
        with np.load(path, allow_pickle=False) as data:
            if not np.allclose(data["extent"], .12):
                raise ValueError("Unsupported model descriptor extent")
            if not len(data["model"]):
                continue
            for q, destination in zip(queries, costs):
                if q is None:
                    continue
                models, values = _shard_costs(data, q, cfg.matching)
                for target, value in zip(destination, values):
                    target[models] = value
        if progress:
            progress(f"Local preselection: {shard_number}/{len(shards)} tranches")
    return [_fuse_cost_rankings(*cost, limit) if cost else [] for cost in costs]


def supplement_pool(pool, group_rankings, local_rankings, extra_limit):
    """Bound extra proposals globally, retaining the entire original pool."""
    result = list(dict.fromkeys(int(i) for i in pool))
    seen = set(result)
    additions = []
    owners = {}
    budget = max(0, int(extra_limit))
    for rank in range(max((len(r) for r in local_rankings), default=0)):
        for group, ranking in enumerate(local_rankings):
            if len(additions) >= budget:
                break
            if rank < len(ranking) and int(ranking[rank]) in owners:
                owners[int(ranking[rank])].add(group)
            if rank < len(ranking) and int(ranking[rank]) not in seen:
                index = int(ranking[rank])
                seen.add(index)
                additions.append(index)
                owners[index] = {group}
        if len(additions) >= budget:
            break
    result.extend(additions)
    updated = [list(r) for r in group_rankings]
    for group, ranking in enumerate(local_rankings):
        proposals = [int(i) for i in ranking if group in owners.get(int(i), set())]
        updated[group] = proposals + [i for i in updated[group] if i not in proposals]
    return result, updated, additions


def supplement_from_database(db_dir, names, groups, cfg, pool, group_rankings,
                             extra_limit=64, progress=None):
    """Use a local index when installed; otherwise preserve legacy retrieval."""
    root = Path(db_dir) / "local_candidate_index"
    metadata = {"available": root.is_dir(), "added": 0, "budget": int(extra_limit)}
    if int(extra_limit) <= 0 or not groups or not root.is_dir():
        return pool, group_rankings, metadata
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        expected = manifest.get("source_fingerprint")
        if expected and expected != source_fingerprint(db_dir):
            raise ValueError("Local candidate index source changed; rebuild the index")
        rankings = rank_local_groups(root, names, groups, cfg, progress=progress)
    except (ValueError, OSError, KeyError) as error:
        metadata["warning"] = str(error)
        if progress:
            progress(f"Local index not used: {error}")
        return pool, group_rankings, metadata
    pool, group_rankings, added = supplement_pool(
        pool, group_rankings, rankings, extra_limit,
    )
    metadata["added"] = len(added)
    metadata["candidate_indices"] = added
    return pool, group_rankings, metadata
